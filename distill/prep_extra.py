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
import io
import json
import os
import random
import re
import subprocess
import sys
import time
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


# ------------------------------------------------------------------ 4. manifests

def stage_manifests(a):
    for src, extra in (("speech", []), ("hits", ["--heldout-frac", "0"])):
        out = os.path.join(a.data_dir, f"{src}.jsonl")
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
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--speech-hours", type=float, default=20)
    ap.add_argument("--drum-hours", type=float, default=6)
    ap.add_argument("--piano-hours", type=float, default=4)
    ap.add_argument("--file-seconds", type=float, default=60, help="length of each joined speech file")
    ap.add_argument("--render-seconds", type=float, default=45, help="length of each rendered performance")
    a = ap.parse_args()
    a.root = os.path.abspath(a.root)
    stages = {"download": stage_download, "speech": stage_speech, "hits": stage_hits,
              "manifests": stage_manifests}
    for s in a.stages.split(","):
        t0 = time.time()
        print(f"== stage {s}", flush=True)
        stages[s.strip()](a)
        print(f"== stage {s} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
