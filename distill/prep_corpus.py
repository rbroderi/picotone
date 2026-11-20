"""Scan a folder of audio (recursively, zip archives included) into a training manifest.

    uv run --extra cuda python prep_corpus.py --root D:\\jamendo --source jamendo --out data\\jamendo.jsonl --workers 14

Every track gets: codec (sniffed from bytes, not the extension), sample rate, channels, duration, container
bitrate, measured effective bandwidth, artist/album/title, licence class + URL, an attribution URL, keep/reject
with reasons, and its split (train, or `heldout` for a deterministic ~2% of artists).

Licences come from, in order: the file's tags (LICENSE / COPYRIGHT / WCOP / comment), then a `license*.txt`
or `readme*.txt` beside it (the Jamendo 2012 dump has both in each album folder). The filter itself is applied
at train time (`train.py --licences`), so changing policy needs no rescan.

Resumable: tracks already in --out are skipped. Writes <out>.report.json and <out>.eval.json (fixed held-out
excerpts for evaluation).
"""
import argparse
import os
import re
import sys
import zipfile
from multiprocessing import Pool

import audio_io
import corpus

SIDE_NAMES = re.compile(r"^(licen[sc]e|readme|copying)[^/\\]*\.txt$", re.I)
URL = re.compile(r"https?://[^\s\"'<>]+")
SKIP_EXT = (".txt", ".jpg", ".jpeg", ".png", ".gif", ".pdf", ".nfo", ".m3u", ".pls", ".cue", ".log", ".xml",
            ".html", ".htm", ".db", ".ini", ".sfv", ".md5", ".zip", ".torrent", ".json", ".sqlite")


def walk(roots):
    """Yields (ref, sidecar-text) for every candidate audio file. Sidecars are read per directory."""
    for root in roots:
        for d, _, files in os.walk(root):
            side = ""
            for f in files:
                if SIDE_NAMES.match(f):
                    try:
                        side += open(os.path.join(d, f), encoding="utf-8", errors="replace").read(20000) + "\n"
                    except OSError:
                        pass
            for f in sorted(files):
                p = os.path.join(d, f)
                low = f.lower()
                if low.endswith(".zip"):
                    yield from walk_zip(p)
                elif not low.endswith(SKIP_EXT):
                    yield p, side


def walk_zip(path):
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        print("bad zip", path, flush=True)
        return
    by_dir = {}
    for n in z.namelist():
        if n.endswith("/") or "__MACOSX" in n:
            continue
        by_dir.setdefault(os.path.dirname(n), []).append(n)
    for d, names in sorted(by_dir.items()):
        side = ""
        for n in names:
            if SIDE_NAMES.match(os.path.basename(n)):
                side += z.read(n)[:20000].decode("utf-8", errors="replace") + "\n"
        for n in sorted(names):
            if not n.lower().endswith(SKIP_EXT):
                yield f"{path}{audio_io.SEP}{n}", side


def tags(ref):
    """artist/album/title/licence text/url from tags, via mutagen (works on zip members too)."""
    import io

    import mutagen

    out = {}
    try:
        mf = mutagen.File(io.BytesIO(audio_io.read_bytes(ref)) if audio_io.SEP in ref else ref)
    except Exception:  # noqa: BLE001
        return out
    if mf is None or mf.tags is None:
        return out
    t = mf.tags

    def get(*keys):
        for k in keys:
            try:
                v = t.get(k)
            except Exception:  # noqa: BLE001
                v = None
            if v:
                v = v[0] if isinstance(v, list) else v
                v = getattr(v, "text", v)
                v = v[0] if isinstance(v, list) and v else v
                v = getattr(v, "url", v)
                if str(v).strip():
                    return str(v).strip()
        return ""

    out["artist"] = get("artist", "ARTIST", "TPE1", "albumartist", "TPE2")
    out["album"] = get("album", "ALBUM", "TALB")
    out["title"] = get("title", "TITLE", "TIT2")
    out["lic_text"] = " ".join(filter(None, [get("license", "LICENSE", "WCOP"), get("copyright", "COPYRIGHT", "TCOP"),
                                             get("comment", "COMMENT", "description")]))
    out["url"] = get("contact", "CONTACT", "website", "WOAR", "WOAF", "url")
    return out


def guess_from_path(ref):
    """Artist/album/title from a Jamendo-style path: <Artist>.zip::<Artist>/<Album>/<NN>-<Title>.ogg."""
    path, member = audio_io.split_ref(ref)
    parts = re.split(r"[\\/]", member if member else path)
    title = re.sub(r"^\d+[\s_.-]*", "", os.path.splitext(parts[-1])[0]).replace("_", " ")
    album = parts[-2].replace("_", " ") if len(parts) >= 2 else ""
    artist = parts[-3].replace("_", " ") if len(parts) >= 3 else (
        os.path.splitext(os.path.basename(path))[0].replace("_", " ") if member else "")
    return artist, album, title


def one(job):
    ref, side, a = job
    row = {"ref": ref, "source": a.source, "id": corpus.track_id(a.source, ref)}
    try:
        inf = corpus.probe(ref, a.probe_windows, a.probe_seconds, a.bw_drop_db)
    except Exception as e:  # noqa: BLE001
        row.update({"keep": False, "reasons": [f"decode-error: {str(e)[:120]}"], "flags": [], "seconds": 0.0,
                    "codec": "error", "sr": 0, "split": "error", "licence": "unknown"})
        return row
    tg = tags(ref)
    g_artist, g_album, g_title = guess_from_path(ref)
    lic, lic_url = corpus.licence_class(tg.get("lic_text", ""))
    if lic == "unknown":
        lic, lic_url = corpus.licence_class(side)
    if lic == "unknown" and a.default_licence:
        lic, lic_url = corpus.licence_class(a.default_licence)
    urls = URL.findall(tg.get("url", "") + " " + side)
    url = next((u for u in urls if "creativecommons" not in u and "artlibre" not in u), "")
    if not url and a.url_base:
        path, member = audio_io.split_ref(ref)
        rel = os.path.relpath(path, a.root[0]).replace("\\", "/")
        url = a.url_base.rstrip("/") + "/" + rel + ("/" + member if member else "")
    keep, reasons, flags = corpus.judge(inf, a.min_bandwidth, a.min_kbps, a.min_seconds, a.drop_low_kbps)
    artist = tg.get("artist") or g_artist
    path, member = audio_io.split_ref(ref)
    if member:  # stored members are read in place; deflated ones are read whole for every training window
        ct = audio_io._zip(path).getinfo(member).compress_type
        inf["zip"] = "stored" if ct == zipfile.ZIP_STORED else f"compressed:{ct}"
    row.update(inf)
    row.update({"artist": artist, "album": tg.get("album") or g_album, "title": tg.get("title") or g_title,
                "licence": lic, "licence_url": lic_url, "url": url, "keep": keep, "reasons": reasons,
                "flags": flags})
    held = corpus.norm_artist(artist) in a.exclude or corpus.artist_heldout(a.source, artist, a.heldout_frac,
                                                                           a.seed)
    row["split"] = "heldout" if held else "train"
    return row


def eval_excerpts(rows, n, dur, name):
    """Deterministic held-out excerpts: n passing held-out tracks spread over artists, 40% into the track."""
    cand = [r for r in rows if r["split"] == "heldout" and r["keep"] and r["seconds"] >= dur + 4]
    cand.sort(key=lambda r: r["id"])
    seen, out = set(), []
    for r in cand:
        a = corpus.norm_artist(r.get("artist"))
        if a in seen:
            continue
        seen.add(a)
        out.append({"name": f"{name}_{len(out):02d}_{r['id']}", "set": name, "ref": r["ref"],
                    "offset": round(r["seconds"] * 0.4, 2), "dur": dur, "licence": r["licence"],
                    "artist": r.get("artist", ""), "title": r.get("title", "")})
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True, help="folder to scan (repeatable)")
    ap.add_argument("--source", required=True, help="source name used by --mix, e.g. jamendo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--min-bandwidth", type=float, default=18000, help="Hz; main quality test")
    ap.add_argument("--bw-drop-db", type=float, default=60, help="level under the 1-4 kHz reference that "
                                                                 "counts as 'no content'")
    ap.add_argument("--min-kbps", type=float, default=160, help="lossy files under this are flagged")
    ap.add_argument("--drop-low-kbps", action="store_true", help="reject (not only flag) low-bitrate lossy files")
    ap.add_argument("--min-seconds", type=float, default=5)
    ap.add_argument("--probe-windows", type=int, default=3)
    ap.add_argument("--probe-seconds", type=float, default=8)
    ap.add_argument("--heldout-frac", type=float, default=0.02, help="fraction of artists held out")
    ap.add_argument("--exclude-artist", action="append", default=[], help="always held out (repeatable)")
    ap.add_argument("--default-licence", default="", help="licence URL assumed when a track has none")
    ap.add_argument("--url-base", default="", help="attribution URL prefix when a track has no URL, e.g. "
                                                   "https://archive.org/download/jamendo_complete_zip_2012_05_20")
    ap.add_argument("--eval-n", type=int, default=30, help="held-out excerpts for evaluation")
    ap.add_argument("--eval-seconds", type=float, default=10)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new tracks (tests)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.exclude = {corpus.norm_artist(x) for x in a.exclude_artist}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = {r["ref"] for r in corpus.read_jsonl(a.out)} if os.path.exists(a.out) else set()
    jobs = []
    for ref, side in walk(a.root):
        if ref in done:
            continue
        jobs.append((ref, side, a))
        if a.limit and len(jobs) >= a.limit:
            break
    print(f"{len(done)} tracks already in {a.out}, {len(jobs)} to probe with {a.workers} workers", flush=True)
    buf, n = [], 0
    with Pool(a.workers) as pool:
        for row in pool.imap_unordered(one, jobs, chunksize=4):
            buf.append(row)
            n += 1
            if len(buf) >= 200:
                corpus.append_jsonl(a.out, buf)
                buf = []
                print(f"  {n}/{len(jobs)}", flush=True)
    corpus.append_jsonl(a.out, buf)
    rows = corpus.read_jsonl(a.out)
    corpus.write_report(rows, a.out + ".report.json", f"{a.source}: {a.out}")
    ev = eval_excerpts(rows, a.eval_n, a.eval_seconds, a.source)
    import json

    json.dump(ev, open(a.out + ".eval.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{len(ev)} held-out eval excerpts -> {a.out}.eval.json")


if __name__ == "__main__":
    sys.exit(main())
