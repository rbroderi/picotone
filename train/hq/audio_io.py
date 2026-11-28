"""Audio decoding that does not trust file extensions.

Sources are a path, optionally with a zip member (`zip_path::member/name.ogg`). The codec is sniffed from the
first bytes (Ogg Vorbis / Ogg FLAC / Opus / native FLAC / MP3 / WAV); libsndfile decodes what it can (it has no
Ogg FLAC), ffmpeg the rest. Everything returns float32 [channels, samples] at the file's own rate.
"""
import io
import os
import shutil
import subprocess
import zipfile
from functools import lru_cache
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 44100
AUDIO_EXT = (".ogg", ".oga", ".opus", ".flac", ".mp3", ".wav", ".m4a", ".aac", ".wma", ".aif", ".aiff")
SEP = "::"


def split_ref(ref):
    if SEP in ref:
        z, m = ref.split(SEP, 1)
        return z, m
    return ref, None


@lru_cache(maxsize=8)
def _zip(path):
    return zipfile.ZipFile(path)


def read_bytes(ref, limit=None):
    path, member = split_ref(ref)
    if member is None:
        with open(path, "rb") as f:
            return f.read() if limit is None else f.read(limit)
    with _zip(path).open(member) as f:
        return f.read() if limit is None else f.read(limit)


def sniff(head):
    """Codec from the first bytes (>= 64 recommended)."""
    if head[:4] == b"OggS":
        if b"\x01vorbis" in head[:64]:
            return "ogg-vorbis"
        if b"\x7fFLAC" in head[:64] or b"fLaC" in head[:64]:
            return "ogg-flac"
        if b"OpusHead" in head[:64]:
            return "ogg-opus"
        if b"Speex" in head[:64]:
            return "ogg-speex"
        return "ogg-other"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "mp3"
    if head[4:8] == b"ftyp":
        return "mp4"
    if head[:4] == b"FORM":
        return "aiff"
    return "unknown"


SNDFILE_OK = {"ogg-vorbis", "ogg-opus", "flac", "wav", "mp3", "aiff"}


def have_ffmpeg():
    return shutil.which("ffmpeg") is not None


def _ffmpeg_decode(data, start_s=0.0, dur_s=None):
    if not have_ffmpeg():
        raise RuntimeError("this file needs ffmpeg (not on PATH)")
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.6f}"]
    cmd += ["-i", "pipe:0"]
    if dur_s is not None:
        cmd += ["-t", f"{dur_s:.6f}"]
    # Probe the channel count/rate first so the raw output can be reshaped.
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                         "stream=sample_rate,channels", "-of", "csv=p=0", "-i", "pipe:0"],
                        input=data, capture_output=True, timeout=120)
    sr, ch = (int(v) for v in pr.stdout.decode().strip().split(",")[:2])
    r = subprocess.run(cmd + ["-f", "f32le", "-"], input=data, capture_output=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace")[-300:])
    x = np.frombuffer(r.stdout, dtype="<f4").reshape(-1, ch).T.copy()
    return x, sr


def info(ref):
    """-> dict(codec, sr, channels, frames, seconds, bytes)."""
    head = read_bytes(ref, 4096)
    codec = sniff(head)
    path, member = split_ref(ref)
    nbytes = _zip(path).getinfo(member).file_size if member else os.path.getsize(path)
    if codec in SNDFILE_OK:
        try:
            src = io.BytesIO(read_bytes(ref)) if member else path
            i = sf.info(src)
            return {"codec": codec, "sr": i.samplerate, "channels": i.channels, "frames": i.frames,
                    "seconds": i.frames / i.samplerate, "bytes": nbytes}
        except Exception:  # noqa: BLE001
            pass
    x, sr = _ffmpeg_decode(read_bytes(ref))
    return {"codec": codec, "sr": sr, "channels": x.shape[0], "frames": x.shape[1], "seconds": x.shape[1] / sr,
            "bytes": nbytes}


def decode(ref, start_s=0.0, dur_s=None, data=None):
    """-> (x float32 [C, T], sr). Reads only the window when the codec allows seeking."""
    path, member = split_ref(ref)
    head = data[:4096] if data is not None else read_bytes(ref, 4096)
    codec = sniff(head)
    if codec in SNDFILE_OK:
        try:
            if data is not None:
                src = io.BytesIO(data)
            elif member and _zip(path).getinfo(member).compress_type == zipfile.ZIP_STORED:
                src = _zip(path).open(member)  # stored member: seekable in place, only the window is read
            elif member:
                src = io.BytesIO(read_bytes(ref))
            else:
                src = path
            with sf.SoundFile(src) as f:
                sr = f.samplerate
                start = int(round(start_s * sr))
                if start:
                    f.seek(min(start, f.frames))
                n = -1 if dur_s is None else int(round(dur_s * sr))
                x = f.read(n, dtype="float32", always_2d=True).T
            return np.ascontiguousarray(x), sr
        except Exception:  # noqa: BLE001
            pass
    return _ffmpeg_decode(data if data is not None else read_bytes(ref), start_s, dur_s)


def resample(x, sr_in, sr_out=SR):
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g, axis=-1).astype(np.float32)


def decode_44k(ref, start_s=0.0, dur_s=None, data=None, margin_s=0.05):
    """Decode a window and resample to 44.1 kHz. For other rates a margin is decoded and trimmed so the
    resampler's edge effects stay outside the window."""
    if dur_s is None:
        x, sr = decode(ref, start_s, None, data)
        return resample(x, sr)
    i0 = max(0.0, start_s - margin_s)
    x, sr = decode(ref, i0, dur_s + (start_s - i0) + margin_s, data)
    if sr == SR:
        y = x
    else:
        y = resample(x, sr)
    a = int(round((start_s - i0) * SR))
    n = int(round(dur_s * SR))
    y = y[:, a:a + n]
    if y.shape[1] < n:
        y = np.pad(y, ((0, 0), (0, n - y.shape[1])))
    return y
