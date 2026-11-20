"""Job list for the beam-1 against beam-8 comparison (input for pool_list.py).

Usage: beam_jobs.py SCRATCH
Greedy (beam 1) music24-layout encodes of the preset test clips and of the comparison fit
clips (lower codebook counts are prefixes of these), one greedy music12 encode to confirm the
prefix property, then beam-8 music12 and music6 encodes of the fit clips.
"""
import glob
import os
import sys

S = sys.argv[1].replace("\\", "/")
lines = []
for label, wavdir in (("old", "wav"), ("new", "wav2")):
    for f in sorted(glob.glob(f"{S}/work/preset/music24/{label}/*.ecdc")):
        name = os.path.basename(f)[:-5]
        lines.append((f"{S}/work/{wavdir}/{name}.wav", f"{S}/work/preset_b1/music24/{label}/{name}.ecdc",
                      "--preset music24 --beam 1"))
        if label == "old" and len(lines) == 1:
            lines.append((f"{S}/work/{wavdir}/{name}.wav", f"{S}/work/preset_b1/prefixcheck/{name}.ecdc",
                          "--preset music12 --beam 1"))
fit = [l.rstrip("\n").split("\t") for l in open(f"{S}/work/abfit.txt", encoding="utf-8")]
for wav, out, _ in fit:
    lines.append((wav, f"{S}/workG/ecdc/{os.path.basename(out)}", "--preset music24 --beam 1"))
for preset, d in (("music12", "workB12"), ("music6", "workB6")):
    for wav, out, _ in fit:
        lines.append((wav, f"{S}/{d}/ecdc/{os.path.basename(out)}", f"--preset {preset}"))
with open(f"{S}/work/list2.txt", "w", encoding="utf-8", newline="\n") as f:
    for row in lines:
        f.write("\t".join(row) + "\n")
print(len(lines), "jobs")
