"""Runs listed encodes with a fixed number of concurrent encoder processes.

Usage: encode_list.py EXE WEIGHTS SLOTS LIST.txt
LIST.txt lines are "input.wav<TAB>output.ecdc<TAB>extra encoder arguments". Each encoder
runs with two threads; outputs appear atomically and existing ones are skipped.
"""
import os
import subprocess
import sys
import time

EXE, WEIGHTS, SLOTS, LIST = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
jobs = []
for line in open(LIST, encoding="utf-8"):
    wav, out, extra = line.rstrip("\n").split("\t")
    if not os.path.exists(out) and os.path.exists(wav):
        jobs.append((wav, out, extra.split()))
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
    while jobs and len(running) < SLOTS:
        wav, out, extra = jobs.pop(0)
        tmp = out + ".pooltmp"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cmd = [EXE, "--weights", WEIGHTS, "--threads", "2", "encode", wav, tmp] + extra
        running.append((subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
                        out, tmp))
    time.sleep(1.0)
print(f"POOL DONE {done} done {failed} failed in {(time.time() - t0) / 60:.0f} min", flush=True)
