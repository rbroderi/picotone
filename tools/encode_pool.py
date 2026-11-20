"""Runs all remaining encodes with a fixed number of concurrent encoder processes.

Usage: encode_pool.py WORK OLD_EXE NEW_EXE WEIGHTS SLOTS
Jobs, in order: the shipping presets (music24, music12, music6: stride 45000, beam 8) on the
old test clips and on a 30-clip subset of the new test clips, then every fit clip of
WORK/queue2.txt that has no 24 kbps 48 kHz encode yet. Each encoder runs with two threads;
at most SLOTS run at once. Outputs appear atomically; existing ones are skipped.
"""
import os
import subprocess
import sys
import time

import lib

WORK, OLD_EXE, NEW_EXE, WEIGHTS = sys.argv[1:5]
SLOTS = int(sys.argv[5])
jobs = []
tests = {"old": ("wav", lib.load_files("48k", "test")), "new": ("wav2", lib.load_files("48k", "test2"))}
subset = tests["new"][1]
tests["new"] = ("wav2", subset[::max(len(subset) // 30, 1)][:30])
for preset in ("music24", "music12", "music6"):
    for label, (wavdir, files) in tests.items():
        d = os.path.join(WORK, "preset", preset, label)
        os.makedirs(d, exist_ok=True)
        for f in files:
            name = f["name"].replace(".48k.ecdc", "")
            jobs.append((os.path.join(d, name + ".ecdc"),
                         [NEW_EXE, "--weights", WEIGHTS, "--threads", "2", "encode",
                          os.path.join(WORK, wavdir, name + ".wav"), None, "--preset", preset]))
for name in open(os.path.join(WORK, "queue2.txt"), encoding="utf-8").read().split():
    if name.startswith("fit2"):
        jobs.append((os.path.join(WORK, "ecdc2", name + ".48k.ecdc"),
                     [OLD_EXE, "--weights", WEIGHTS, "--threads", "2", "encode",
                      os.path.join(WORK, "wav2", name + ".wav"), None, "-m", "stereo", "-b", "24"]))
jobs = [j for j in jobs if not os.path.exists(j[0]) and os.path.exists(j[1][6])]
print(len(jobs), "jobs", flush=True)
running = []
done = failed = 0
t0 = time.time()
while jobs or running:
    for item in list(running):
        proc, out, tmp = item
        if proc.poll() is None:
            continue
        running.remove(item)
        if proc.returncode == 0 and os.path.exists(tmp):
            os.replace(tmp, out)
            done += 1
        else:
            failed += 1
            print("FAILED", os.path.basename(out), flush=True)
        if (done + failed) % 20 == 0:
            print(f"{done} done, {failed} failed, {len(jobs)} queued, "
                  f"{(time.time() - t0) / 60:.0f} min", flush=True)
    while jobs and len(running) < SLOTS:
        out, cmd = jobs.pop(0)
        if os.path.exists(out):
            continue
        tmp = out + ".pooltmp"
        cmd = [tmp if c is None else c for c in cmd]
        running.append((subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
                        out, tmp))
    time.sleep(1.0)
print("POOL DONE", done, "done", failed, "failed", flush=True)
