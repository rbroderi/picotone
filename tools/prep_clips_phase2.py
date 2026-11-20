"""Phase 2: clip the new downloads to WAV and write an interleaved encode queue.

Usage: prep_clips_phase2.py DL_DIR WORK_DIR
Reads manifest2.json, manifest_oga.json and manifest_ccm.json. Regular files give one clip;
zip archives (sound-effect packs, some game music) are read member by member: effects are
joined with short gaps into clips, music members are clipped like tracks. Writes
WORK/wav2/<clip>.wav (48 kHz stereo), WORK/clips2.json and WORK/queue2.txt, whose order
cycles through the genres so that a partial encode stays balanced.
"""
import io
import json
import os
import subprocess
import sys
import zipfile

DL, WORK = sys.argv[1], sys.argv[2]
OUT = os.path.join(WORK, "wav2")
os.makedirs(OUT, exist_ok=True)
AUDIO = (".wav", ".ogg", ".mp3", ".flac")
# seconds per clip: (fit2, test2)
DUR = {"game": (100, 60), "sfx": (90, 60), "speech": (150, 60), "vocal": (90, 60)}
DEFAULT = (75, 60)
RATE = 48000


def decode(args, data=None):
    """Runs ffmpeg and returns 48 kHz stereo s16 PCM bytes (empty on failure)."""
    cmd = ["ffmpeg", "-v", "error"] + args + ["-f", "s16le", "-ar", str(RATE), "-ac", "2", "-"]
    r = subprocess.run(cmd, input=data, capture_output=True)
    return r.stdout if r.returncode == 0 else b""


def write_wav(path, pcm):
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


def clip_file(path, dur):
    pcm = decode(["-ss", "15", "-t", str(dur), "-i", path])
    if len(pcm) < RATE * 4 * 20:
        pcm = decode(["-t", str(dur), "-i", path])
    return pcm


def clips_from_zip(path, group, dur, max_clips):
    out = []
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        return out
    members = sorted(m for m in zf.namelist() if m.lower().endswith(AUDIO) and "__MACOSX" not in m)
    gap = b"\0" * (RATE * 4 // 8)
    buf = b""
    for m in members:
        if len(out) >= max_clips:
            break
        try:
            data = zf.read(m)
        except Exception:  # noqa: BLE001
            continue
        if len(data) > 80_000_000:
            continue
        pcm = decode(["-i", "pipe:0"], data)
        if not pcm:
            continue
        if len(pcm) >= RATE * 4 * 25:
            # a real track: clip it on its own
            start = min(RATE * 4 * 10, max(len(pcm) - RATE * 4 * dur, 0))
            start -= start % 4
            out.append(pcm[start:start + RATE * 4 * dur])
        else:
            buf += pcm[:RATE * 4 * 8] + gap
            if len(buf) >= RATE * 4 * dur:
                out.append(buf[:RATE * 4 * dur])
                buf = b""
    if len(buf) >= RATE * 4 * 10 and len(out) < max_clips:
        out.append(buf)
    return out


entries = []
for name in ("manifest2.json", "manifest_oga.json", "manifest_ccm.json"):
    p = os.path.join(DL, name)
    if os.path.exists(p):
        entries += json.load(open(p, encoding="utf-8"))
clips = []
for e in entries:
    split, group = e["split"], e["group"]
    dur = DUR.get(group, DEFAULT)[0 if split == "fit2" else 1]
    src = os.path.join(DL, e["file"])
    stem = os.path.splitext(e["file"])[0]
    first = os.path.join(OUT, stem + "__c0.wav")
    if e["file"].lower().endswith(".zip"):
        if not os.path.exists(first):
            for i, pcm in enumerate(clips_from_zip(src, group, dur, 8 if split == "fit2" else 3)):
                write_wav(os.path.join(OUT, f"{stem}__c{i}.wav"), pcm)
        i = 0
        while os.path.exists(os.path.join(OUT, f"{stem}__c{i}.wav")):
            clips.append({"name": f"{stem}__c{i}", "split": split, "group": group, "source": e["file"]})
            i += 1
    else:
        if not os.path.exists(first):
            pcm = clip_file(src, dur)
            if len(pcm) < RATE * 4 * 8:
                print("skip (too short or unreadable)", e["file"], flush=True)
                continue
            write_wav(first, pcm)
        clips.append({"name": stem + "__c0", "split": split, "group": group, "source": e["file"]})
json.dump(clips, open(os.path.join(WORK, "clips2.json"), "w"), indent=1)
# interleave genres; test clips first
by = {}
for c in clips:
    by.setdefault((c["split"] != "test2", c["group"]), []).append(c["name"])
queue = []
for is_fit in (False, True):
    groups = [v for (f, _), v in sorted(by.items()) if f == is_fit]
    for i in range(max((len(g) for g in groups), default=0)):
        queue += [g[i] for g in groups if i < len(g)]
with open(os.path.join(WORK, "queue2.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(queue) + "\n")
mins = {}
for c in clips:
    key = (c["split"], c["group"])
    mins[key] = mins.get(key, 0) + os.path.getsize(os.path.join(OUT, c["name"] + ".wav")) / (RATE * 4 * 60)
for k, v in sorted(mins.items()):
    print(k, round(v, 1), "min", flush=True)
print("clips", len(clips), "total hours", round(sum(mins.values()) / 60, 2), flush=True)
