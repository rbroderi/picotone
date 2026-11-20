"""Shared corpus tools: licence parsing, the quality probe, artist splits and manifest IO."""
import hashlib
import json
import os
import re

import numpy as np
from scipy.signal import welch

import audio_io

# ------------------------------------------------------------------ licences

LICENCE_CLASSES = ("pd", "cc0", "by", "by-sa", "fal", "by-nd", "by-nc", "by-nc-sa", "by-nc-nd", "other", "unknown")
LICENCE_PRESETS = {
    "strict": ("pd", "cc0"),                     # the original picotone training rule
    "permissive": ("pd", "cc0", "by"),           # attribution only; credits are in the manifest
    "sharealike": ("pd", "cc0", "by", "by-sa", "fal"),   # default; FAL is copyleft like BY-SA
    "commercial": ("pd", "cc0", "by", "by-sa", "fal", "by-nd"),
    "all": LICENCE_CLASSES,
}
DEFAULT_LICENCES = "sharealike"
LICENCE_URLS = {"pd": "https://creativecommons.org/publicdomain/mark/1.0/",
                "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
                "by": "https://creativecommons.org/licenses/by/4.0/",
                "by-sa": "https://creativecommons.org/licenses/by-sa/4.0/",
                "fal": "https://artlibre.org/licence/lal/en/",
                "by-nd": "https://creativecommons.org/licenses/by-nd/4.0/"}
_CC = re.compile(r"creativecommons\.org/(licenses|publicdomain)/([a-z+\-]+)(?:/([\d.]+))?", re.I)


def licence_class(text):
    """Licence class and canonical URL from any text (URL, tag, license.txt)."""
    if not text:
        return "unknown", ""
    t = str(text)
    m = _CC.search(t)
    if m:
        kind, name = m.group(1).lower(), m.group(2).lower()
        url = "https://" + m.group(0).rstrip("/") + "/"
        if kind == "publicdomain":
            return ("cc0" if name == "zero" else "pd"), url
        if name == "publicdomain":
            return "pd", url
        if name in ("by", "by-sa", "by-nd", "by-nc", "by-nc-sa", "by-nc-nd"):
            return name, url
        if name == "by-nd-nc":
            return "by-nc-nd", url
        return "other", url
    low = t.lower()
    if "art libre" in low or "free art licen" in low or "artlibre.org" in low:
        return "fal", "https://artlibre.org/licence/lal/en/"
    if "public domain" in low:
        return "pd", ""
    return "unknown", ""


def parse_licences(arg):
    """'permissive' / 'strict' / ... or a comma list of classes."""
    arg = arg or DEFAULT_LICENCES
    out = set()
    for part in arg.split(","):
        part = part.strip().lower()
        if part in LICENCE_PRESETS:
            out.update(LICENCE_PRESETS[part])
        elif part in LICENCE_CLASSES:
            out.add(part)
        elif part:
            raise SystemExit(f"unknown licence class/preset {part!r}; classes {LICENCE_CLASSES}, "
                             f"presets {tuple(LICENCE_PRESETS)}")
    return out


# ------------------------------------------------------------------ quality probe

def bandwidth_hz(x, sr, drop_db=60.0):
    """Effective bandwidth: the highest frequency at which the smoothed long-term spectrum is within
    `drop_db` of the 1-4 kHz reference level. MP3/Vorbis low-pass and band-limited transcodes show a cliff
    of 60-100 dB, natural roll-off rarely reaches it below 18 kHz."""
    m = x.mean(axis=0) if x.ndim == 2 else x
    if m.size < 4096 or float(np.abs(m).max()) < 1e-4:
        return 0.0
    f, p = welch(m, sr, nperseg=4096, noverlap=2048)
    db = 10 * np.log10(p + 1e-20)
    k = 11
    sm = np.convolve(db, np.ones(k) / k, mode="same")
    ref = float(np.median(db[(f >= 1000) & (f <= 4000)]))
    ok = np.nonzero((sm > ref - drop_db) & (f < sr / 2 - 100))[0]
    return float(f[ok[-1]]) if ok.size else 0.0


def probe(ref, windows=3, window_s=8.0, drop_db=60.0):
    """-> info + bandwidth (min over windows, so a single bright moment can't hide a low-passed file) + rms."""
    inf = audio_io.info(ref)
    dur = inf["seconds"]
    data = audio_io.read_bytes(ref) if audio_io.SEP in ref else None
    bws, rms = [], []
    for i in range(windows):
        start = max(0.0, dur * (i + 1) / (windows + 1) - window_s / 2)
        x, sr = audio_io.decode(ref, start, min(window_s, dur), data)
        if x.size == 0:
            continue
        r = float(np.sqrt((x ** 2).mean()))
        rms.append(r)
        if r > 1e-3:  # skip near-silent windows for the bandwidth
            bws.append(bandwidth_hz(x, sr, drop_db))
    inf["bandwidth_hz"] = float(np.median(bws)) if bws else 0.0
    inf["bandwidth_min_hz"] = float(min(bws)) if bws else 0.0
    inf["rms_db"] = float(20 * np.log10(max(np.mean(rms) if rms else 0.0, 1e-9)))
    inf["kbps"] = inf["bytes"] * 8 / max(dur, 1e-3) / 1000
    return inf


LOSSLESS = {"flac", "ogg-flac", "wav", "aiff"}


def judge(inf, min_bw, min_kbps, min_seconds=5.0, drop_low_kbps=False):
    """-> (keep, reasons, flags). Bandwidth is the main test; bitrate is a flag (a lossless file can hold a
    low-bitrate transcode, and a container's bitrate says little), a reject only with drop_low_kbps."""
    reasons, flags = [], []
    if inf["seconds"] < min_seconds:
        reasons.append("short")
    if inf["rms_db"] < -50:
        reasons.append("silent")
    if inf["bandwidth_hz"] < min_bw:
        reasons.append(f"bandwidth<{min_bw:.0f}")
    if inf["sr"] < 44100:
        flags.append(f"sr{inf['sr']}")
    if inf["codec"] not in LOSSLESS and inf["kbps"] < min_kbps:
        (reasons if drop_low_kbps else flags).append(f"kbps<{min_kbps:.0f}")
    return not reasons, reasons, flags


# ------------------------------------------------------------------ splits and IO

def norm_artist(a):
    return re.sub(r"[^a-z0-9]+", " ", (a or "").lower()).strip()


def artist_heldout(source, artist, frac, seed=0):
    h = hashlib.sha1(f"{seed}|{source}|{norm_artist(artist)}".encode()).hexdigest()
    return int(h[:8], 16) / 2 ** 32 < frac


def track_id(source, ref):
    return hashlib.sha1(f"{source}|{ref}".encode()).hexdigest()[:16]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def report(rows, title=""):
    """Hours by split/keep, codec, sample rate, licence and reject reason."""
    from collections import defaultdict

    def table(key, only_keep=None):
        d = defaultdict(lambda: [0, 0.0])
        for r in rows:
            if only_keep is not None and r["keep"] != only_keep:
                continue
            k = key(r)
            d[k][0] += 1
            d[k][1] += r["seconds"] / 3600
        return dict(sorted(d.items(), key=lambda kv: -kv[1][1]))

    out = {
        "title": title,
        "all": table(lambda r: "all"),
        "split_x_keep": table(lambda r: f"{r['split']}/{'pass' if r['keep'] else 'fail'}"),
        "codec": table(lambda r: r["codec"]),
        "sample_rate": table(lambda r: str(r["sr"])),
        "channels": table(lambda r: str(r.get("channels", "?"))),
        "zip_member": table(lambda r: r.get("zip", "not in a zip")),
        "kbps_lossy_passing": table(lambda r: "lossless" if r["codec"] in LOSSLESS else
                                    f"{int(r.get('kbps', 0) // 64 * 64)}-{int(r.get('kbps', 0) // 64 * 64 + 63)} kbps",
                                    True),
        "licence_passing": table(lambda r: r["licence"], True),
        "licence_all": table(lambda r: r["licence"]),
        "reject_reason": table(lambda r: ",".join(x.split("<")[0] for x in r["reasons"]) or "-", False),
        "flags_passing": table(lambda r: ",".join(f.split("<")[0] for f in r["flags"]) or "-", True),
    }
    lines = [f"== {title}"]
    for k, v in out.items():
        if k == "title":
            continue
        lines.append(f"  {k}: " + "; ".join(f"{a} {n} files {h:.1f} h" for a, (n, h) in v.items()))
    return out, "\n".join(lines)


def write_report(rows, path, title):
    out, text = report(rows, title)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(text, flush=True)
    return out
