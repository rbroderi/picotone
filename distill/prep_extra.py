"""Extra training material for the q9 fine-tune: clean speech, and rendered drum and piano performances.

    uv run --extra cuda python prep_extra.py --root /data/extra --workers 6

Why: the student's largest gap to stock DAC is speech (13-17% worse log-mel on the five LibriVox eval clips;
training had no speech), then piano, sfx and sharp attacks at q9 (5-8%). This adds two sources:

- `speech`: VCTK 0.92 (CC BY 4.0, 110 English speakers, 48 kHz studio recordings, mic1). Utterances are a few
  seconds long, so each speaker's are joined into ~60 s files with short gaps. Not LibriVox, so no eval reader
  or book can leak in.
- `hits`: drum kits and grand pianos from sfzinstruments on GitHub, rendered into ~45 s stereo performances:
  random grooves with cymbal-heavy voices and sparse lone hits after silence (pre-echo), and piano
  notes/chords/runs with real decays and key releases. Samples are CC0 except Salamander (CC BY 3.0) and
  Headroom (CC BY 4.0).

Stages (each skips work already done, so a re-run resumes):
  1. download into <root>/dl  (VCTK ~11.7 GB, sample libraries ~5.5 GB; resumable)
  2. speech -> <root>/speech/vctk_<speaker>/vctk/NNN-<speaker>.flac   (default 20 h)
  3. hits   -> <root>/hits/<kit>/render/NNNN-<kit>.flac                 (default 6 h drums + 4 h piano)
  4. manifests: prep_corpus.py on each folder -> data/speech.jsonl, data/hits.jsonl (+ .report.json)

Every output FLAC carries artist / title / licence / contact tags, so prep_corpus.py and the credits list pick up
the licence and attribution. The bandwidth test is off for these two sources (`--min-bandwidth 0`): it exists to
catch MP3 transcodes, and these are lossless 48 kHz originals whose quiet top octave (speech, piano) would
otherwise fail it.

Needs ~30 GB free under --root. Options: --stages, --speech-hours, --drum-hours, --piano-hours, --workers.
"""
import argparse
import collections
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
import zlib
from multiprocessing import Pool

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 48000

VCTK = {"url": "https://datashare.ed.ac.uk/server/api/core/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/content",
        "file": "VCTK-Corpus-0.92.zip", "bytes": 11747302977,
        "page": "https://datashare.ed.ac.uk/handle/10283/3443",
        "licence": "https://creativecommons.org/licenses/by/4.0/",
        "credit": "CSTR VCTK Corpus 0.92, Yamagishi, Veaux, MacDonald (University of Edinburgh)"}

# (repo, branch, kind, licence URL, credit). Checked 2026-11-21: all audio is in git (no LFS), flac/wav.
KITS = [
    ("virtuosity_drums", "master", "drums", "https://creativecommons.org/publicdomain/zero/1.0/",
     "Virtuosity Drums, Versilian Studios and Karoryfer Samples"),
    ("karoryfer.big-rusty-drums", "main", "drums", "https://creativecommons.org/publicdomain/zero/1.0/",
     "Big Rusty Drums, Karoryfer Samples"),
    ("karoryfer.swirly-drums", "main", "drums", "https://creativecommons.org/publicdomain/zero/1.0/",
     "Swirly Drums, Karoryfer Samples"),
    ("karoryfer.unruly-drums", "main", "drums", "https://creativecommons.org/publicdomain/zero/1.0/",
     "Unruly Drums, Karoryfer Samples"),
    ("karoryfer.frankensnare", "main", "drums", "https://creativecommons.org/publicdomain/zero/1.0/",
     "Frankensnare, Karoryfer Samples"),
    ("Osiris_Piano", "main", "piano", "https://creativecommons.org/publicdomain/zero/1.0/",
     "Osiris Piano, Versilian Studios and Karoryfer Samples"),
    ("SalamanderGrandPiano", "master", "piano", "https://creativecommons.org/licenses/by/3.0/",
     "Salamander Grand Piano V3, Alexander Holm"),
    ("BengtNilsson.HeadroomPiano", "master", "piano", "https://creativecommons.org/licenses/by/4.0/",
     "Headroom Piano, Bengt Nilsson"),
]
CYMBAL = re.compile(r"crash|ride|hat|hh|cym|china|splash|bell|tamb|shak", re.I)
SKIP_SAMPLE = re.compile(r"noise|release|rel_|_rel|pedal|hammer|harm|resonance", re.I)


# ------------------------------------------------------------------ 1. download

def download(url, dest, size=None):
    if os.path.exists(dest) and (size is None or os.path.getsize(dest) == size):
        print(f"have {dest}", flush=True)
        return
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "picotone-distill-fetch/1.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    for attempt in range(20):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:  # no range support: start again
                    have = 0
                mode = "ab" if have else "wb"
                total = (int(r.headers.get("Content-Length", 0)) + have) or size
                t0, last = time.time(), 0.0
                with open(part, mode) as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                        have += len(b)
                        if time.time() - last > 30:
                            last = time.time()
                            pct = f"{100 * have / total:.1f}%" if total else ""
                            print(f"  {os.path.basename(dest)} {have / 1e9:.2f} GB {pct} "
                                  f"({have / 1e6 / max(1e-3, time.time() - t0):.1f} MB/s this attempt)", flush=True)
            break
        except Exception as e:  # noqa: BLE001  (network drop: resume from what we have)
            print(f"  download error ({e}); retrying in 30 s", flush=True)
            time.sleep(30)
            have = os.path.getsize(part) if os.path.exists(part) else 0
            req = urllib.request.Request(url, headers={"User-Agent": "picotone-distill-fetch/1.0"})
            if have:
                req.add_header("Range", f"bytes={have}-")
    else:
        raise SystemExit(f"download failed: {url}")
    if size is not None and os.path.getsize(part) != size:
        raise SystemExit(f"{part}: {os.path.getsize(part)} bytes, expected {size}; delete it and re-run")
    os.replace(part, dest)
    print(f"done {dest}", flush=True)


def stage_download(a):
    dl = os.path.join(a.root, "dl")
    os.makedirs(dl, exist_ok=True)
    download(VCTK["url"], os.path.join(dl, VCTK["file"]), VCTK["bytes"])
    for repo, branch, *_ in KITS:
        download(f"https://codeload.github.com/sfzinstruments/{repo}/zip/refs/heads/{branch}",
                 os.path.join(dl, f"{repo}.zip"))


# ------------------------------------------------------------------ shared

def to48(x, sr):
    """[T] or [T, C] -> float32 [C, T] at 48 kHz."""
    x = np.asarray(x, np.float32)
    x = x[:, None] if x.ndim == 1 else x
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g, axis=0).astype(np.float32)
    return np.ascontiguousarray(x.T)


def write_flac(path, x, tags, subtype):
    """x [C, T] float32 -> FLAC with Vorbis comments, peak normalised to a random -1..-6 dBFS."""
    rng = random.Random(path)
    peak = float(np.abs(x).max()) or 1.0
    x = x * (10 ** (-rng.uniform(1, 6) / 20) / peak)
    tmp = path + ".tmp.flac"
    sf.write(tmp, x.T, SR, subtype=subtype, format="FLAC")
    from mutagen.flac import FLAC
    f = FLAC(tmp)
    for k, v in tags.items():
        f[k] = v
    f.save()
    os.replace(tmp, path)


def tagset(artist, title, album, licence, contact, credit):
    return {"artist": artist, "title": title, "album": album, "license": licence, "contact": contact,
            "copyright": f"{credit}. Licence: {licence}"}


# ------------------------------------------------------------------ 2. speech (VCTK)

def speech_job(job):
    zpath, spk, members, out_dir, n0, file_s = job
    # Outside the audio tree: prep_corpus.py probes every file under --root.
    marker = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(out_dir))), "state", f"speech_{spk}.done")
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    if os.path.exists(marker):
        return spk, 0
    z = zipfile.ZipFile(zpath)
    rng = random.Random(f"{spk}-{n0}")
    os.makedirs(out_dir, exist_ok=True)
    buf, have, n, made = [], 0, n0, 0

    def flush():
        nonlocal buf, have, n, made
        if not buf:
            return
        path = os.path.join(out_dir, f"{n:03d}-{spk}.flac")
        if not os.path.exists(path):
            write_flac(path, np.concatenate(buf, axis=1),
                       tagset(f"VCTK {spk}", f"VCTK speaker {spk}, part {n}", "VCTK Corpus 0.92",
                              VCTK["licence"], VCTK["page"], VCTK["credit"]), "PCM_16")
        buf, have, n, made = [], 0, n + 1, made + 1

    for m in members:
        x, sr = sf.read(io.BytesIO(z.read(m)), always_2d=False)
        x = to48(x, sr)[:1]
        buf += [x, np.zeros((1, int(SR * rng.uniform(0.15, 0.7))), np.float32)]
        have += x.shape[1]
        if have >= file_s * SR:
            flush()
    if have >= 10 * SR:
        flush()
    open(marker, "w").close()
    return spk, made


def stage_speech(a):
    zpath = os.path.join(a.root, "dl", VCTK["file"])
    z = zipfile.ZipFile(zpath)
    pat = re.compile(r"([ps]\d+)_(\d+)_mic1\.flac$")
    by = {}
    for m in z.namelist():
        g = pat.search(m)
        if g:
            by.setdefault(g.group(1), []).append(m)
    if not by:
        raise SystemExit(f"no *_mic1.flac utterances in {zpath}")
    # Speaker budget: equal seconds per speaker, from the mean utterance length (measured on a sample).
    sample = [m for ms in by.values() for m in ms[:3]][:120]
    mean_s = np.mean([sf.info(io.BytesIO(z.read(m))).duration for m in sample]) + 0.42
    per_spk = a.speech_hours * 3600 / len(by)
    jobs = []
    for spk, ms in sorted(by.items()):
        ms = sorted(ms)
        random.Random(spk).shuffle(ms)
        ms = ms[: max(1, int(per_spk / mean_s))]
        out_dir = os.path.join(a.root, "speech", f"vctk_{spk}", "vctk")
        jobs.append((zpath, spk, ms, out_dir, 0, a.file_seconds))
    print(f"speech: {len(by)} speakers, {sum(len(j[2]) for j in jobs)} utterances "
          f"(~{a.speech_hours:.0f} h, mean {mean_s:.1f} s each with gaps)", flush=True)
    with Pool(a.workers) as pool:
        for i, (spk, made) in enumerate(pool.imap_unordered(speech_job, jobs), 1):
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} speakers", flush=True)
    with open(os.path.join(a.root, "speech", "license.txt"), "w", encoding="utf-8") as f:
        f.write(f"{VCTK['credit']}\n{VCTK['page']}\nLicence: {VCTK['licence']}\n"
                "Utterances joined per speaker, with short silences, by distill/prep_extra.py.\n")


# ------------------------------------------------------------------ 3. hits (rendered drums and piano)

class Samples:
    """A kit's samples (zip members) grouped into voices (their folder), decoded on demand with a size cap."""

    def __init__(self, zpath, cap_bytes=400e6):
        self.zpath, self.z, self.cache, self.size, self.cap = zpath, zipfile.ZipFile(zpath), {}, 0, cap_bytes
        voices = {}
        for m in self.z.namelist():
            if m.lower().endswith((".flac", ".wav", ".ogg")) and not SKIP_SAMPLE.search(m):
                voices.setdefault(os.path.dirname(m), []).append(m)
        self.voices = {k: sorted(v) for k, v in voices.items()}

    def get(self, m):
        x = self.cache.get(m)
        if x is None:
            try:
                y, sr = sf.read(io.BytesIO(self.z.read(m)), always_2d=True)
            except Exception:  # noqa: BLE001
                y, sr = np.zeros((1, 2)), SR
            x = to48(y, sr)
            if x.shape[0] == 1:
                x = np.repeat(x, 2, 0)
            x = x[:2]
            if self.size > self.cap:
                self.cache.clear()
                self.size = 0
            self.cache[m] = x
            self.size += x.nbytes
        return x


def add(out, x, t, gain, pan=0.0, fade=None):
    """Mix x [2, T] into out at sample t, with gain, constant-power pan and an optional release fade."""
    if fade is not None and fade < x.shape[1]:
        rel = min(int(0.25 * SR), x.shape[1] - fade)
        x = x[:, : fade + rel].copy()
        x[:, fade:] *= np.linspace(1, 0, rel, dtype=np.float32) ** 2
    n = min(x.shape[1], out.shape[1] - t)
    if n <= 0:
        return
    th = (pan + 1) * np.pi / 4
    out[0, t:t + n] += gain * np.cos(th) * np.sqrt(2) * x[0, :n]
    out[1, t:t + n] += gain * np.sin(th) * np.sqrt(2) * x[1, :n]


def render_drums(S, rng, secs):
    out = np.zeros((2, int(secs * SR)), np.float32)
    keys = list(S.voices)
    w = np.array([2.0 if CYMBAL.search(k) else 1.0 for k in keys])
    pick = rng.choice(len(keys), size=min(len(keys), int(rng.integers(3, 8))), replace=False, p=w / w.sum())
    voices = [keys[i] for i in pick]
    if rng.random() < 0.3:  # sparse: lone hits after silence, where pre-echo shows
        t = 0.2
        while t < secs - 0.5:
            v = voices[int(rng.integers(len(voices)))]
            add(out, S.get(S.voices[v][int(rng.integers(len(S.voices[v])))]), int(t * SR),
                10 ** (rng.uniform(-18, 0) / 20), rng.uniform(-0.8, 0.8))
            t += rng.uniform(0.4, 3.0)
        return out
    bpm = rng.uniform(70, 190)
    step = 60 / bpm / 4
    pats = {v: (rng.random(16) < rng.uniform(0.1, 0.7), rng.uniform(-0.7, 0.7), rng.uniform(-12, 0)) for v in voices}
    k = 0
    while k * step < secs:
        if rng.random() < 0.03:  # fills and variation
            pats = {v: (p ^ (rng.random(16) < 0.15), pan, g) for v, (p, pan, g) in pats.items()}
        for v, (p, pan, g) in pats.items():
            if p[k % 16]:
                swing = step * 0.12 * rng.random() if k % 2 else 0.0
                add(out, S.get(S.voices[v][int(rng.integers(len(S.voices[v])))]),
                    int((k * step + swing) * SR), 10 ** ((g + rng.uniform(-6, 0)) / 20), pan)
        k += 1
    return out


def render_piano(S, rng, secs):
    out = np.zeros((2, int(secs * SR)), np.float32)
    notes = [m for ms in S.voices.values() for m in ms]
    notes.sort()
    style = rng.choice(["chords", "runs", "sparse", "mixed"])
    t = 0.1
    while t < secs - 0.3:
        if style == "chords" or (style == "mixed" and rng.random() < 0.4):
            n, gap = int(rng.integers(2, 6)), rng.uniform(0.4, 2.5)
        elif style == "runs" or style == "mixed":
            n, gap = 1, rng.uniform(0.06, 0.3)
        else:
            n, gap = int(rng.integers(1, 3)), rng.uniform(1.0, 4.0)
        base = int(rng.integers(len(notes)))
        for j in range(n):
            m = notes[min(len(notes) - 1, max(0, base + int(rng.integers(-12, 13)) * (j > 0)))]
            hold = None if rng.random() < 0.3 else int(rng.uniform(0.15, 3.0) * SR)
            add(out, S.get(m), int((t + j * rng.uniform(0, 0.02)) * SR), 10 ** (rng.uniform(-10, 0) / 20),
                rng.uniform(-0.3, 0.3), hold)
        t += gap
    return out


def hits_job(job):
    zpath, repo, kind, lic, credit, idxs, out_dir, secs = job
    S = Samples(zpath)
    if not S.voices:
        return repo, 0
    os.makedirs(out_dir, exist_ok=True)
    made = 0
    for i in idxs:
        path = os.path.join(out_dir, f"{i:04d}-{repo}.flac")
        if os.path.exists(path):
            continue
        rng = np.random.default_rng([zlib.crc32(repo.encode()), i])
        x = (render_drums if kind == "drums" else render_piano)(S, rng, secs)
        if float(np.abs(x).max()) < 1e-4:
            continue
        write_flac(path, x, tagset(f"{credit} (rendered)", f"{repo} render {i}", repo, lic,
                                   f"https://github.com/sfzinstruments/{repo}", credit), "PCM_24")
        made += 1
    return repo, made


def stage_hits(a):
    jobs = []
    for kind, hours in (("drums", a.drum_hours), ("piano", a.piano_hours)):
        kits = [k for k in KITS if k[2] == kind]
        per_kit = max(1, int(hours * 3600 / a.render_seconds / len(kits)))
        for repo, _branch, _kind, lic, credit in kits:
            zpath = os.path.join(a.root, "dl", f"{repo}.zip")
            out_dir = os.path.join(a.root, "hits", repo, "render")
            # Split each kit over several jobs so the pool stays busy; each job decodes its own samples.
            for c in range(0, per_kit, 25):
                jobs.append((zpath, repo, kind, lic, credit, list(range(c, min(per_kit, c + 25))), out_dir,
                             a.render_seconds))
    print(f"hits: {len(KITS)} kits, {len(jobs)} jobs of up to 25 renders of {a.render_seconds:.0f} s", flush=True)
    done = {}
    with Pool(a.workers) as pool:
        for i, (repo, made) in enumerate(pool.imap_unordered(hits_job, jobs), 1):
            done[repo] = done.get(repo, 0) + made
            if i % 5 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} jobs", flush=True)
    with open(os.path.join(a.root, "hits", "license.txt"), "w", encoding="utf-8") as f:
        for repo, _b, _k, lic, credit in KITS:
            f.write(f"{credit}: https://github.com/sfzinstruments/{repo} ({lic})\n")
        f.write("Rendered into performances by distill/prep_extra.py.\n")
    print("  rendered this run:", json.dumps(done), flush=True)


# ------------------------------------------------------------------ librivox (optional stage)

# The picotone eval's five speech clips are LibriVox 64 kbps MP3s from these items. Their books, authors and
# readers never train. (Authors are also left out at train time: train.py drops every eval clip's `artist`.)
LV_EVAL_ITEMS = ["adventures_holmes", "huck_finn_librivox", "jane_eyre_ver03_0809_librivox",
                 "peter_pan_0707_librivox", "robinson_crusoe_librivox"]
# Names after "Read by" / "Reader:" in an item's description (HTML stripped first). Group recordings list no
# per-chapter readers on archive.org, so for those only the authors and items can be left out.
READ_BY = re.compile(r"\b(?:[Rr]ead by|[Rr]eaders?:)\s+([A-Z][\w'-]*(?: [A-Z][\w'-]*){0,2})")
HTML_TAG = re.compile(r"<[^>]+>")


def eval_readers(text):
    import corpus
    out = {corpus.norm_artist(r) for r in READ_BY.findall(HTML_TAG.sub(" ", text))}
    return {r for r in out if r and "librivox" not in r and "volunteer" not in r and len(r) > 3}


_BR = {1: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],   # MPEG-1 layer III, kbps
       2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]}       # MPEG-2 / 2.5 layer III
_SR = {3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000]}


def mp3_repair(path):
    """Make a byte-prefix MP3 excerpt self-consistent, in place. -> (changed, seconds by frame count).

    The excerpt keeps the chapter's Xing/Info header frame, which states the whole chapter's frame count, so
    libsndfile/mpg123 report the chapter's length, not the excerpt's (found 2026-11-22: 6 min files listed as
    15-38 min, and the probe's windows past the real end judged them silent). This rewrites that header's frame
    count, byte count and seek table for the excerpt, and cuts the torn last frame. (Dropping the header instead
    works for CBR but not VBR: libsndfile then guesses the length from the first frame's bitrate.) A VBRI
    header is dropped. Idempotent."""
    data = open(path, "rb").read()
    pos = 0
    if data[:3] == b"ID3" and len(data) > 10:
        s = data[6:10]
        pos = 10 + ((s[0] & 0x7F) << 21 | (s[1] & 0x7F) << 14 | (s[2] & 0x7F) << 7 | (s[3] & 0x7F))
        pos += 10 if data[5] & 0x10 else 0
    start, frames, out, first, key, xing = pos, 0, [], True, None, None
    spf = sr = 0
    while pos + 4 <= len(data):
        b1, b2 = data[pos + 1], data[pos + 2]
        if data[pos] != 0xFF or (b1 & 0xE0) != 0xE0:
            break
        ver, layer, bri, sri, pad = (b1 >> 3) & 3, (b1 >> 1) & 3, b2 >> 4, (b2 >> 2) & 3, (b2 >> 1) & 1
        if ver == 1 or layer != 1 or bri in (0, 15) or sri == 3:
            break
        if key is None:
            key = (ver, sri)
        elif key != (ver, sri):  # a different stream format mid-file: not a frame boundary
            break
        sr = _SR[ver][sri]
        br = _BR[1 if ver == 3 else 2][bri] * 1000
        flen = (144 if ver == 3 else 72) * br // sr + pad
        spf = 1152 if ver == 3 else 576
        if pos + flen > len(data):  # torn last frame
            break
        fr = data[pos:pos + flen]
        if first and fr[36:40] == b"VBRI":
            pass  # Fraunhofer header frame: drop
        elif first and (b"Xing" in fr[:64] or b"Info" in fr[:64]):
            xing = bytearray(fr)  # patched below, once the audio frames are counted
        else:
            out.append(fr)
            frames += 1
        first = False
        pos += flen
    if not frames:
        raise ValueError("no MP3 frames")
    if xing is not None:
        t = max(xing.find(b"Xing", 0, 64), xing.find(b"Info", 0, 64))
        flags = int.from_bytes(xing[t + 4:t + 8], "big")
        o = t + 8
        total = len(xing) + sum(len(f) for f in out)
        if flags & 1:
            xing[o:o + 4] = frames.to_bytes(4, "big")
            o += 4
        if flags & 2:
            xing[o:o + 4] = total.to_bytes(4, "big")
            o += 4
        if flags & 4:  # seek table: byte position (/256 of the stream) at each 1% of the frames
            offs, acc = [], len(xing)
            for f in out:
                offs.append(acc)
                acc += len(f)
            xing[o:o + 100] = bytes(min(255, offs[min(frames - 1, i * frames // 100)] * 256 // total)
                                    for i in range(100))
        out.insert(0, bytes(xing))
    new = data[:start] + b"".join(out)
    changed = new != data
    if changed:
        with open(path + ".fix", "wb") as f:
            f.write(new)
        os.replace(path + ".fix", path)
    return changed, frames * spf / sr


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "picotone-distill-fetch/1.0"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            print(f"  {url}: {e}; retrying", flush=True)
            time.sleep(10 * (attempt + 1))
    raise SystemExit(f"failed: {url}")


HONORIFICS = {"sir", "dr", "mr", "mrs", "miss", "ms", "rev", "lord", "lady", "jr", "sr", "st", "saint"}
NOT_ONE_AUTHOR = {"various", "anonymous", "unknown", "librivox", "traditional", "none", "n a"}


def display_name(creator):
    """Archive.org creator -> "First Last": drops dates and brackets, and turns "Twain, Mark" (or
    "Twain, Mark, 1835-1910") round."""
    s = creator[0] if isinstance(creator, list) and creator else creator
    s = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", str(s or ""))
    s = re.sub(r"\b(?:b\.|d\.|fl\.|ca\.)?\s*\d{3,4}\??\s*-?\s*(?:\d{3,4}\??)?", " ", s)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) >= 2 and corpus_norm(parts[1]) not in HONORIFICS:
        s = f"{parts[1]} {parts[0]}"
    elif parts:
        s = parts[0]
    return " ".join(s.split())


def corpus_norm(s):
    import corpus
    return corpus.norm_artist(s)


def author_key(creator):
    """(surname, first initial), after display_name and without honorifics: "Sir Arthur Conan Doyle",
    "Doyle, Arthur Conan" and "Doyle, A. C." all give ('doyle', 'a'). ('', '') for no single named author."""
    toks = [t for t in corpus_norm(display_name(creator)).split() if t not in HONORIFICS]
    if not toks or " ".join(toks) in NOT_ONE_AUTHOR or toks[0] in NOT_ONE_AUTHOR:
        return ("", "")
    return (toks[-1], toks[0][0] if len(toks) > 1 else "")


def lv_seconds(s):
    try:
        p = [float(v) for v in str(s).split(":")]
    except ValueError:
        return 0.0
    return sum(v * 60 ** i for i, v in enumerate(reversed(p)))


def lv_text(md):
    d = md.get("description", "")
    return " ".join(d) if isinstance(d, list) else str(d)


def lv_repair_all(out_root):
    """mp3_repair every excerpt already fetched, then check the length libsndfile reports against the frame count
    (that is the length prep_corpus.py records and training samples within)."""
    paths = [os.path.join(d, f) for d, _, fs in os.walk(out_root) for f in fs if f.endswith(".mp3")
             and not f.endswith(".part.mp3")]
    if not paths:
        return
    fixed, bad, tot = 0, [], 0.0
    for p in sorted(paths):
        try:
            ch, secs = mp3_repair(p)
            from mutagen.id3 import ID3
            t = ID3(p)
            if t.getall("APIC"):  # cover art would inflate libsndfile's size-based estimates; not needed
                t.delall("APIC")
                t.save(p)
                ch = True
        except Exception as e:  # noqa: BLE001
            bad.append((p, f"repair failed: {e}"))
            continue
        fixed += ch
        tot += secs
        i = sf.info(p)
        rep = i.frames / i.samplerate
        if abs(rep - secs) > 0.02 * secs + 0.5:
            bad.append((p, f"libsndfile says {rep:.1f} s, frames say {secs:.1f} s"))
    print(f"librivox: {len(paths)} excerpts on disk, {fixed} repaired, {tot / 3600:.1f} h by frame count, "
          f"{len(bad)} with a length mismatch or error", flush=True)
    for p, why in bad[:10]:
        print(f"  {p}: {why}", flush=True)
    if bad:
        raise SystemExit("some excerpts still report the wrong length: stop, and report these lines")


def stage_librivox(a):
    """One 64 kbps MP3 chapter excerpt (the first --lv-minutes, fetched as a byte prefix, not re-encoded) from
    each of many LibriVox books: the same kind of audio as the eval's speech clips, from many readers."""
    import corpus
    from mutagen.id3 import ID3, TALB, TIT2, TPE1, WCOP, WOAR, ID3NoHeaderError
    out_root = os.path.join(a.root, "librivox")
    os.makedirs(out_root, exist_ok=True)
    lv_repair_all(out_root)
    unused = os.path.join(a.root, "librivox_unused")  # outside the folder prep_corpus.py scans
    readers, eval_keys = set(), set()
    for it in LV_EVAL_ITEMS:
        md = get_json(f"https://archive.org/metadata/{it}/metadata").get("result", {})
        eval_keys.add(author_key(md.get("creator", "")))
        readers.update(eval_readers(lv_text(md)))
    if not readers:
        raise SystemExit("found no reader names in the eval items' descriptions; check READ_BY")
    print(f"librivox: leaving out eval items {LV_EVAL_ITEMS}, authors (surname, initial) {sorted(eval_keys)}, "
          f"readers {sorted(readers)}", flush=True)
    need, have, seen_authors, n_ok, skipped = a.lv_hours * 3600, 0.0, collections.Counter(), 0, collections.Counter()
    nbytes = int(a.lv_minutes * 60 * 64000 / 8)

    def refuse(ident, creator):
        """-> why this item can't train, or None. Readers are checked at fetch time (they need the metadata)."""
        if ident in LV_EVAL_ITEMS:
            return "eval item"
        k = author_key(creator)
        if not k[0]:
            return "no single named author"
        if k in eval_keys:
            return "eval author"
        if seen_authors[k] >= 3:  # spread over authors (and so books and readers)
            return "author cap"
        if have >= need:
            return "over the hours budget"
        return None

    def mp3_in(d):
        return sorted(f for f in os.listdir(d) if f.endswith(".mp3") and not f.endswith(".part.mp3"))

    # 1. Excerpts already on disk, in identifier order: a re-run keeps the same set, whatever the search returns.
    #    Ones that can't train (eval author, author cap, over budget) move to librivox_unused/, not deleted.
    for ident in sorted(os.listdir(out_root)):
        d = os.path.join(out_root, ident)
        if not os.path.isdir(d) or not mp3_in(d):
            continue
        p = os.path.join(d, mp3_in(d)[0])
        t = ID3(p)
        creator = str(t["TPE1"].text[0]) if "TPE1" in t else ""
        why = refuse(ident, creator)
        if why:
            skipped[f"on disk, moved to librivox_unused: {why}"] += 1
            os.makedirs(unused, exist_ok=True)
            if os.path.exists(os.path.join(unused, ident)):
                shutil.rmtree(os.path.join(unused, ident))
            shutil.move(d, os.path.join(unused, ident))
            continue
        if creator != display_name(creator):  # "Twain, Mark" -> "Mark Twain", as train.py's eval match expects
            t.delall("TPE1")
            t.add(TPE1(encoding=3, text=display_name(creator)))
            t.save(p)
        seen_authors[author_key(creator)] += 1
        have += a.lv_minutes * 60
        n_ok += 1
    print(f"librivox: {n_ok} excerpts already on disk kept, {have / 3600:.1f} h of {a.lv_hours:g}", flush=True)

    # 2. Fill any shortfall from the search, in a fixed order (identifier, then a seeded shuffle).
    docs = []
    if have < need:
        q = urllib.parse.quote("collection:librivoxaudio AND language:eng AND mediatype:audio")
        docs = get_json(f"https://archive.org/advancedsearch.php?q={q}&fl[]=identifier&fl[]=creator&rows=3000"
                        f"&sort[]=downloads+desc&output=json")["response"]["docs"]
        docs.sort(key=lambda d: d["identifier"])
        random.Random(0).shuffle(docs)
    for d in docs:
        if have >= need:
            break
        ident = d["identifier"]
        creator = d.get("creator", "")
        creator = creator[0] if isinstance(creator, list) else creator
        dest_dir = os.path.join(out_root, ident)
        if os.path.isdir(dest_dir) and mp3_in(dest_dir):
            continue  # already counted above
        why = refuse(ident, creator)
        if why:
            skipped[why] += 1
            continue
        au = author_key(creator)
        creator = display_name(creator)
        m = get_json(f"https://archive.org/metadata/{ident}")
        md = m.get("metadata", {})
        text = lv_text(md)
        if any(f" {r} " in f" {corpus.norm_artist(HTML_TAG.sub(' ', text))} " for r in readers):
            skipped["eval reader"] += 1
            continue
        mp3s = [f for f in m.get("files", []) if f.get("name", "").endswith("_64kb.mp3")
                and lv_seconds(f.get("length", 0)) > a.lv_minutes * 60 + 30]
        if not mp3s:
            skipped["no long 64kb mp3"] += 1
            continue
        f = mp3s[random.Random(ident).randrange(len(mp3s))]
        url = f"https://archive.org/download/{ident}/{urllib.parse.quote(f['name'])}"
        if a.lv_dry_run:
            print(f"  would take {url} ({creator}; {f.get('length')})", flush=True)
            have += a.lv_minutes * 60
            seen_authors[au] += 1
            n_ok += 1
            continue
        old = os.path.join(unused, ident)
        if os.path.isdir(old) and mp3_in(old):  # fetched by an earlier run and set aside: reuse it
            shutil.move(old, dest_dir)
            have += a.lv_minutes * 60
            seen_authors[au] += 1
            n_ok += 1
            continue
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, f["name"])
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "picotone-distill-fetch/1.0",
                                                       "Range": f"bytes=0-{nbytes - 1}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read(nbytes)  # whatever the server sends, keep only the prefix (decoders skip the
            tmp = path[:-4] + ".part.mp3"  # torn last frame; libsndfile wants the .mp3 name)
            with open(tmp, "wb") as fh:
                fh.write(data)
            mp3_repair(tmp)
            try:
                t = ID3(tmp)
            except ID3NoHeaderError:
                t = ID3()
            lic = md.get("licenseurl") or "https://creativecommons.org/publicdomain/mark/1.0/"
            t.delall("TPE1"), t.delall("TALB"), t.delall("TIT2"), t.delall("WCOP"), t.delall("WOAR")
            t.add(TPE1(encoding=3, text=creator))
            t.add(TALB(encoding=3, text=str(md.get("title", ident))))
            t.add(TIT2(encoding=3, text=f"{md.get('title', ident)} ({f['name']}, first {a.lv_minutes:g} min)"))
            t.add(WCOP(url=lic))
            t.add(WOAR(url=f"https://archive.org/details/{ident}"))
            t.save(tmp)
            x, sr = sf.read(tmp, frames=10, start=0)  # decodes: a broken prefix is dropped, not kept
            os.replace(tmp, path)
        except Exception as e:  # noqa: BLE001
            skipped[f"fetch/decode error: {type(e).__name__}"] += 1
            for p in (path[:-4] + ".part.mp3", path):
                if os.path.exists(p):
                    os.remove(p)
            continue
        with open(os.path.join(dest_dir, "license.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"LibriVox recording: {md.get('title', ident)} by {creator}\n"
                     f"https://archive.org/details/{ident}\nLicence: {lic}\n")
        have += a.lv_minutes * 60
        seen_authors[au] += 1
        n_ok += 1
        if n_ok % 10 == 0:
            print(f"  {n_ok} items, {have / 3600:.1f} h of {a.lv_hours:g}", flush=True)
    print(f"librivox: {n_ok} items from {len(seen_authors)} authors, ~{have / 3600:.1f} h; skipped {dict(skipped)}",
          flush=True)
    if have < need * 0.8:
        print("WARNING: less LibriVox audio than asked for", flush=True)
    if not a.lv_dry_run:
        lv_repair_all(out_root)


# ------------------------------------------------------------------ 4. manifests

def stage_manifests(a):
    known = {"speech": [], "hits": ["--heldout-frac", "0"], "librivox": []}
    for src in a.sources.split(","):
        src = src.strip()
        extra = known[src]
        out = os.path.join(a.data_dir, f"{src}.jsonl")
        if a.fresh:  # prep_corpus.py resumes, so rows probed from bad files would otherwise stay
            for p in (out, out + ".report.json", out + ".eval.json"):
                if os.path.exists(p):
                    os.remove(p)
                    print(f"removed {p}", flush=True)
        cmd = [sys.executable, os.path.join(HERE, "prep_corpus.py"), "--root", os.path.join(a.root, src),
               "--source", src, "--out", out, "--workers", str(a.workers), "--min-bandwidth", "0", *extra]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=HERE)
        rows = [json.loads(line) for line in open(out, encoding="utf-8") if line.strip()]
        ok = [r for r in rows if r["keep"] and r["split"] == "train"]
        lic = {}
        for r in ok:
            lic[r["licence"]] = lic.get(r["licence"], 0) + r["seconds"] / 3600
        print(f"{src}: {len(rows)} files, {len(ok)} train+keep = {sum(r['seconds'] for r in ok) / 3600:.1f} h, "
              f"by licence (h): { {k: round(v, 1) for k, v in lic.items()} }", flush=True)
        bad = [r for r in rows if not r["keep"]]
        if bad:
            print(f"  {len(bad)} rejected, e.g. {bad[0]['ref']}: {bad[0]['reasons']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="working folder for downloads and outputs (~30 GB)")
    ap.add_argument("--stages", default="download,speech,hits,manifests")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"), help="where the manifests go")
    ap.add_argument("--sources", default="speech,hits", help="manifests stage: which folders to scan "
                                                             "(speech, hits, librivox)")
    ap.add_argument("--lv-hours", type=float, default=15, help="librivox stage: total hours")
    ap.add_argument("--lv-minutes", type=float, default=6, help="librivox stage: minutes kept per book")
    ap.add_argument("--lv-dry-run", action="store_true", help="librivox stage: list the picks, fetch no audio")
    ap.add_argument("--fresh", action="store_true", help="manifests stage: delete the --sources manifests first, "
                                                         "so every file is probed again")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--speech-hours", type=float, default=20)
    ap.add_argument("--drum-hours", type=float, default=6)
    ap.add_argument("--piano-hours", type=float, default=4)
    ap.add_argument("--file-seconds", type=float, default=60, help="length of each joined speech file")
    ap.add_argument("--render-seconds", type=float, default=45, help="length of each rendered performance")
    a = ap.parse_args()
    a.root = os.path.abspath(a.root)
    stages = {"download": stage_download, "speech": stage_speech, "hits": stage_hits,
              "librivox": stage_librivox, "manifests": stage_manifests}
    for s in a.stages.split(","):
        t0 = time.time()
        print(f"== stage {s}", flush=True)
        stages[s.strip()](a)
        print(f"== stage {s} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
