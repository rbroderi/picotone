"""Clip downloaded audio to WAV and encode with encodec-rs at 24 kbps (both models).

Usage: prep_clips.py DL_DIR WORK_DIR EXE WEIGHTS_DIR [split]
Resumable: existing outputs are skipped.
"""
import json
import os
import subprocess
import sys

DL, WORK, EXE, WEIGHTS = sys.argv[1:5]
ONLY = sys.argv[5] if len(sys.argv) > 5 else None
ORDER = sys.argv[6] if len(sys.argv) > 6 else "fwd"
os.makedirs(os.path.join(WORK, "wav"), exist_ok=True)
os.makedirs(os.path.join(WORK, "ecdc"), exist_ok=True)

# (offset s, duration s) per split/group; orchestra uses several windows of a long file.
PLAN = {
    ("fit", "speech"): [(60, 240)],
    ("fit", "piano"): [(8, 45)],
    ("fit", "band"): [(5, 90)],
    ("fit", "electronic"): [(20, 90)],
    ("fit", "orchestraLR"): [(300, 200), (1500, 200), (2700, 200)],
    ("test", "speech"): [(60, 120)],
    ("test", "piano"): [(8, 60)],
    ("test", "band"): [(5, 120)],
    ("test", "electronic"): [(20, 120)],
    ("test", "orchestraLR"): [(600, 120), (2400, 120)],
}
SPEECH_48K_SECONDS = 60  # only the first minute of each speech clip goes to the stereo model


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FAILED", cmd, r.stderr[-400:], flush=True)
        return False
    return True


manifest = json.load(open(os.path.join(DL, "manifest.json"), encoding="utf-8"))
clips = []
done_pairs = set()
for e in manifest:
    split, group = e["split"], e["group"]
    if ONLY and split != ONLY:
        continue
    src = os.path.join(DL, e["file"])
    inputs = ["-i", src]
    filt = []
    stem = os.path.splitext(e["file"])[0]
    if group == "orchestraLR":
        if ".R." in e["file"]:
            continue
        right = src.replace(".L.", ".R.")
        if not os.path.exists(right):
            continue
        inputs = ["-i", src, "-i", right]
        filt = ["-filter_complex", "[0:a][1:a]amerge=inputs=2[a]", "-map", "[a]"]
        stem = stem.replace(".L", "")
    for wi, (off, dur) in enumerate(PLAN[(split, group)]):
        name = f"{stem}__w{wi}"[:150]
        wav = os.path.join(WORK, "wav", name + ".wav")
        if not os.path.exists(wav):
            ok = run(["ffmpeg", "-v", "error", "-y", "-ss", str(off), "-t", str(dur)] + inputs
                     + filt + ["-ar", "48000", "-ac", "2", wav])
            if not ok:
                continue
        if os.path.getsize(wav) < 48000 * 4 * 10:
            # Shorter than 10 s after the offset: take the file from its start instead.
            run(["ffmpeg", "-v", "error", "-y", "-t", str(dur)] + inputs + filt
                + ["-ar", "48000", "-ac", "2", wav])
        clips.append({"name": name, "split": split, "group": group, "source": e["file"],
                      "offset": off, "duration": dur})

if ORDER == "fwd":
    json.dump(clips, open(os.path.join(WORK, f"clips_{ONLY or 'all'}.json"), "w"), indent=1)
print(len(clips), "clips", flush=True)

if ORDER == "rev":
    clips = clips[::-1]
elif ORDER == "mid":
    clips = clips[len(clips) // 2:] + clips[:len(clips) // 2]
for c in clips:
    wav = os.path.join(WORK, "wav", c["name"] + ".wav")
    for model, tag in (("mono", "24k"), ("stereo", "48k")):
        out = os.path.join(WORK, "ecdc", f"{c['name']}.{tag}.ecdc")
        if os.path.exists(out):
            continue
        src = wav
        if tag == "48k" and c["group"] == "speech" and c["split"] == "fit":
            src = os.path.join(WORK, "wav", c["name"] + ".short" + ORDER + ".wav")
            if not run(["ffmpeg", "-v", "error", "-y", "-t", str(SPEECH_48K_SECONDS), "-i", wav,
                        src]):
                continue
        tmp = out + ".tmp" + ORDER
        if run([EXE, "--weights", WEIGHTS, "encode", src, tmp, "-m", model, "-b", "24"]):
            os.replace(tmp, out)
            print("encoded", os.path.basename(out), os.path.getsize(out), flush=True)
print("prep done")
