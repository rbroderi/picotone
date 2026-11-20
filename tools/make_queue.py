"""Writes WORK/queue2.txt: a capped test2 set first, then fit2 clips cycling through genres.

Usage: make_queue.py WORK_DIR
The test cap keeps the new test set near an hour; the uncapped rest of test2 is not encoded.
"""
import os
import sys

WORK = sys.argv[1]
CAP = {"game": 12, "sfx": 8}
names = sorted(n[:-4] for n in os.listdir(os.path.join(WORK, "wav2")) if n.endswith(".wav"))
by = {}
for n in names:
    split, group = n.split("__")[:2]
    by.setdefault((split != "test2", group), []).append(n)
queue = []
for is_fit in (False, True):
    groups = []
    for (fit, group), clips in sorted(by.items()):
        if fit != is_fit:
            continue
        if not is_fit:
            # spread the kept test clips over different sources
            step = max(len(clips) // CAP.get(group, 6), 1)
            clips = clips[::step][:CAP.get(group, 6)]
        groups.append(clips)
    for i in range(max((len(g) for g in groups), default=0)):
        queue += [g[i] for g in groups if i < len(g)]
with open(os.path.join(WORK, "queue2.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(queue) + "\n")
print(len(queue), "clips queued;", sum(1 for q in queue if q.startswith("test2")), "test2")
