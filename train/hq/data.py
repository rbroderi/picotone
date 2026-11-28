"""Streaming training batches from manifests, without loading the corpus.

Batch k is a pure function of (seed, k): its RNG picks `tracks_per_batch` tracks (source by --mix weight,
track by duration within the source), decodes one random window (`chunk_s`) of each, resamples to 44.1 kHz,
and cuts the batch's crops from those windows (random channel, random sample offset, silent crops redrawn).
Workers take every W-th batch, so any worker count gives the same stream and `--resume` at step k continues the
exact sequence. Only the windows in flight are in memory.
"""
import math
import os

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

import audio_io
import corpus

SR = audio_io.SR


def parse_path_map(specs):
    """['OLD=NEW', ...] -> [(old, new)], longest prefix first."""
    out = []
    for s in specs:
        old, sep, new = s.partition("=")
        if not sep or not old:
            raise SystemExit(f"--path-map wants OLD=NEW, got {s!r}")
        out.append((old, new))
    return sorted(out, key=lambda m: -len(m[0]))


def remap_ref(ref, maps):
    """Rewrite a manifest ref recorded on another machine: swap the first matching prefix (case-insensitive,
    either slash direction), then use '/' throughout when running on a POSIX system. A zip member ('::') keeps
    its own path."""
    if not maps:
        return ref
    path, sep, member = ref.partition(audio_io.SEP)
    norm = path.replace("\\", "/").lower()
    for old, new in maps:
        o = old.replace("\\", "/").rstrip("/").lower()
        if norm == o or norm.startswith(o + "/"):
            path = new.rstrip("/\\") + path[len(o):]
            break
    if os.sep == "/":
        path = path.replace("\\", "/")
    return path + sep + member


# Placeholder creator names that say nothing about who made a track: never used to exclude anything.
GENERIC_ARTISTS = {"", "various", "various artists", "unknown", "anonymous", "n/a", "none"}


def norm_artist(name):
    return " ".join((name or "").split()).lower()


def eval_artists(eval_files):
    """The creators of every eval clip (normalised), minus placeholder names."""
    import json

    out = set()
    for p in eval_files:
        for e in json.load(open(p, encoding="utf-8")):
            out.add(norm_artist(e.get("artist")))
    return out - GENERIC_ARTISTS


def load_sources(manifests, licences, mix, mix_floor, log=print, path_map=(), exclude_artists=()):
    """manifests: list of jsonl paths. -> list of dict(name, rows, hours, weight).

    exclude_artists: normalised creator names whose tracks never train (the eval set's creators). Each
    manifest holds out its own eval creators, but a creator can turn up in another manifest too: the Open
    Goldberg Variations are in the picotone eval set and also in the Jamendo dump (found 2026-11-21)."""
    by = {}
    dropped = {}
    excl = set(exclude_artists) - GENERIC_ARTISTS
    for p in manifests:
        for r in corpus.read_jsonl(p):
            if path_map:
                r["ref"] = remap_ref(r["ref"], path_map)
            if r.get("split") != "train" or not r.get("keep"):
                continue
            if r.get("licence", "unknown") not in licences:
                continue
            who = norm_artist(r.get("artist") or r.get("creator"))
            if who in excl:
                d = dropped.setdefault((r["source"], who), [0, 0.0])
                d[0] += 1
                d[1] += r.get("seconds", 0) or 0
                continue
            by.setdefault(r["source"], []).append(r)
    for (src, who), (n, sec) in sorted(dropped.items()):
        log(f"source {src}: {n} tracks ({sec / 3600:.2f} h) by eval creator {who!r} left out of training")
    if not by:
        raise SystemExit("no training tracks pass the split/quality/licence filters")
    for name, rows in by.items():  # catch a wrong or missing --path-map now, not as silent skipped windows
        missing = [r["ref"] for r in rows[:: max(1, len(rows) // 20)]
                   if not os.path.exists(audio_io.split_ref(r["ref"])[0])]
        if missing:
            raise SystemExit(f"source {name}: {len(missing)} of the sampled files are missing, e.g. {missing[0]} "
                             f"(manifests made on another machine need --path-map OLD=NEW)")
    src = [{"name": n, "rows": sorted(rows, key=lambda r: r["id"]),
            "hours": sum(r["seconds"] for r in rows) / 3600} for n, rows in sorted(by.items())]
    total = sum(s["hours"] for s in src)
    if mix and mix != "auto":
        w = dict(kv.split("=") for kv in mix.split(","))
        for s in src:
            s["weight"] = float(w.get(s["name"], 0.0))
    else:
        for s in src:
            s["weight"] = s["hours"] / total
        if len(src) > 1:  # duration-proportional, but no source under mix_floor
            low = [s for s in src if s["weight"] < mix_floor]
            high = [s for s in src if s["weight"] >= mix_floor]
            if low and high:
                rest = 1 - mix_floor * len(low)
                hsum = sum(s["weight"] for s in high)
                for s in low:
                    s["weight"] = mix_floor
                for s in high:
                    s["weight"] *= rest / hsum
    z = sum(s["weight"] for s in src)
    for s in src:
        s["weight"] /= z
        s["cum"] = np.cumsum([r["seconds"] for r in s["rows"]])
        log(f"source {s['name']}: {len(s['rows'])} tracks, {s['hours']:.1f} h, sampling weight {s['weight']:.3f}")
    return [s for s in src if s["weight"] > 0]


class BatchStream(IterableDataset):
    def __init__(self, sources, batch, crop, n_q_choices, seed=0, start_step=0, tracks_per_batch=8,
                 chunk_s=12.0, silence_db=-45.0):
        super().__init__()
        self.sources = [{"name": s["name"], "weight": s["weight"], "cum": s["cum"],
                         "rows": [(r["ref"], r["seconds"]) for r in s["rows"]]} for s in sources]
        self.w = np.array([s["weight"] for s in self.sources])
        self.batch, self.crop, self.nq = batch, crop, list(n_q_choices)
        self.seed, self.start = seed, start_step
        self.G = max(1, min(tracks_per_batch, batch))
        self.chunk_s = max(chunk_s, crop / SR + 1.0)
        self.silence = 10 ** (silence_db / 20)

    def make(self, step):
        rng = np.random.default_rng([self.seed, step])
        per = math.ceil(self.batch / self.G)
        xs = []
        while len(xs) < self.batch:
            s = self.sources[rng.choice(len(self.sources), p=self.w)]
            i = int(np.searchsorted(s["cum"], rng.uniform(0, s["cum"][-1]), side="right"))
            ref, dur = s["rows"][min(i, len(s["rows"]) - 1)]
            win = min(self.chunk_s, dur)
            t0 = float(rng.uniform(0, max(0.0, dur - win)))
            try:
                w = audio_io.decode_44k(ref, t0, win)
            except Exception:  # noqa: BLE001  (a broken file costs one window, not the run)
                continue
            if w.shape[1] < self.crop:
                w = np.pad(w, ((0, 0), (0, self.crop - w.shape[1])))
            for _ in range(min(per, self.batch - len(xs))):
                for _attempt in range(4):
                    c = int(rng.integers(w.shape[0]))
                    o = int(rng.integers(0, w.shape[1] - self.crop + 1))
                    x = w[c, o:o + self.crop]
                    if np.sqrt(np.mean(x ** 2)) > self.silence:
                        break
                xs.append(x)
        x = torch.from_numpy(np.stack(xs)[:, None].astype(np.float32))
        nq = torch.from_numpy(rng.choice(self.nq, size=self.batch)).long()
        return {"step": step, "x": x, "n_q": nq}

    def __iter__(self):
        wi = get_worker_info()
        wid, n = (wi.id, wi.num_workers) if wi else (0, 1)
        k = self.start + wid
        while True:
            yield self.make(k)
            k += n


def loader(ds, workers):
    return torch.utils.data.DataLoader(ds, batch_size=None, num_workers=workers, persistent_workers=workers > 0,
                                       prefetch_factor=4 if workers > 0 else None, pin_memory=torch.cuda.is_available())


def load_eval(eval_files, max_clips=0, log=print, path_map=()):
    """-> list of dict(name, set, x [C, T] float32 44.1 kHz, T multiple of 512)."""
    import json

    out = []
    for p in eval_files:
        if max_clips and len(out) >= max_clips:
            break
        for e in json.load(open(p, encoding="utf-8")):
            e["ref"] = remap_ref(e["ref"], path_map)
            x = audio_io.decode_44k(e["ref"], e["offset"], e["dur"])
            T = x.shape[1] // 512 * 512
            out.append({**e, "x": np.ascontiguousarray(x[:, :T])})
            if max_clips and len(out) >= max_clips:
                break
    log(f"eval: {len(out)} clips, {sum(e['x'].shape[1] for e in out) / SR:.0f} s, sets "
        f"{sorted({e['set'] for e in out})}")
    return out
