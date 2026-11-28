"""The existing picotone public-domain / CC0 corpus (about 24 h of clips) -> a training manifest.

    uv run --extra cuda python prep_pd.py --pd-root D:\\picotone_pd --out data\\pd.jsonl --workers 14

--pd-root holds what the picotone fetch/clip tools produced (copy it from the build machine, or re-fetch):
    dl\\manifest.json, dl\\manifest2.json, dl\\manifest_oga.json      (licence + creator per download)
    work\\clips_fit.json, work\\clips_test.json, work\\clips2.json    (clip lists)
    work\\wav\\*.wav, work\\wav2\\*.wav                                (48 kHz stereo clips)

Rules:
- licence from the download manifest (the corpus also has CC-BY items; `train.py --licences` decides);
- the same quality probe as prep_corpus.py (these clips were cut from MP3s, so band-limited ones are dropped);
- clips of the picotone test sets (`test` = old, `test2` = new) are held out, and so is every other clip by a
  creator who appears in either test set (`excluded-creator`);
- writes <out>.eval.json: the 53 clips of picotone's music24 old/new test sets (data/picotone_testclips.json),
  10 s from 10 s in, so the trained HQ decoder, stock DAC and picotone are scored on the same audio.
"""
import argparse
import json
import os
from multiprocessing import Pool

import corpus

HERE = os.path.dirname(os.path.abspath(__file__))


def one(job):
    c, wav, man, a = job
    row = {"ref": wav, "source": a.source, "id": corpus.track_id(a.source, c["name"]), "name": c["name"],
           "group": c["group"], "orig_split": c["split"]}
    try:
        inf = corpus.probe(wav, a.probe_windows, a.probe_seconds, a.bw_drop_db)
    except Exception as e:  # noqa: BLE001
        row.update({"keep": False, "reasons": [f"decode-error: {str(e)[:120]}"], "flags": [], "seconds": 0.0,
                    "codec": "error", "sr": 0, "split": "error", "licence": "unknown"})
        return row
    lic, lic_url = corpus.licence_class(man.get("licence", ""))
    keep, reasons, flags = corpus.judge(inf, a.min_bandwidth, a.min_kbps, a.min_seconds)
    row.update(inf)
    # Bitrate of a decoded WAV says nothing about its source; record the source's own extension instead.
    row["flags"] = [f for f in flags if not f.startswith("kbps")] + [f"src:{os.path.splitext(c['source'])[1]}"]
    row.update({"artist": man.get("creator", ""), "title": man.get("title", ""), "album": man.get("item", ""),
                "licence": lic, "licence_url": lic_url or man.get("licence", ""),
                "url": man.get("page") or man.get("url", ""), "keep": keep, "reasons": reasons})
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pd-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="pd")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--min-bandwidth", type=float, default=18000)
    ap.add_argument("--bw-drop-db", type=float, default=60)
    ap.add_argument("--min-kbps", type=float, default=160)
    ap.add_argument("--min-seconds", type=float, default=5)
    ap.add_argument("--probe-windows", type=int, default=3)
    ap.add_argument("--probe-seconds", type=float, default=8)
    ap.add_argument("--eval-offset", type=float, default=10)
    ap.add_argument("--eval-seconds", type=float, default=10)
    ap.add_argument("--limit", type=int, default=0, help="probe only this many training clips (tests)")
    a = ap.parse_args()
    dl, work = os.path.join(a.pd_root, "dl"), os.path.join(a.pd_root, "work")
    man = {}
    for m in ("manifest.json", "manifest2.json", "manifest_oga.json", "manifest_ccm.json"):
        p = os.path.join(dl, m)
        if os.path.exists(p):
            for e in json.load(open(p, encoding="utf-8")):
                man[e["file"]] = e
    clips = []
    for f, sub in (("clips_fit.json", "wav"), ("clips_test.json", "wav"), ("clips2.json", "wav2")):
        p = os.path.join(work, f)
        if os.path.exists(p):
            for c in json.load(open(p, encoding="utf-8")):
                c["wav"] = os.path.join(work, sub, c["name"] + ".wav")
                clips.append(c)
    if not clips or not man:
        raise SystemExit(f"no picotone corpus under {a.pd_root}: expected dl\\manifest*.json and "
                         f"work\\clips_*.json + work\\wav*\\ (see the top of this file)")
    test_creators = {corpus.norm_artist(man.get(c["source"], {}).get("creator", ""))
                     for c in clips if c["split"] in ("test", "test2")}
    test_creators.discard("")
    print(f"{len(clips)} clips, {len(man)} downloads, {len(test_creators)} test-set creators", flush=True)
    jobs, missing = [], 0
    for c in clips:
        if not os.path.exists(c["wav"]):
            missing += 1
            continue
        jobs.append((c, c["wav"], man.get(c["source"], {}), a))
    if missing:
        print(f"WARNING: {missing} clip WAVs missing under {work}", flush=True)
    if a.limit:
        tests = [j for j in jobs if j[0]["split"] in ("test", "test2")]
        jobs = [j for j in jobs if j[0]["split"] not in ("test", "test2")][: a.limit] + tests
    with Pool(a.workers) as pool:
        rows = list(pool.imap_unordered(one, jobs, chunksize=4))
    for r in rows:
        if r["orig_split"] in ("test", "test2"):
            r["split"] = "heldout"
        elif corpus.norm_artist(r.get("artist", "")) in test_creators:
            r["split"] = "excluded-creator"
        else:
            r["split"] = "train"
    rows.sort(key=lambda r: r["name"])
    if os.path.exists(a.out):
        os.remove(a.out)
    corpus.append_jsonl(a.out, rows)
    corpus.write_report(rows, a.out + ".report.json", f"{a.source}: {a.out}")
    by_name = {r["name"]: r for r in rows}
    ev = []
    for t in json.load(open(os.path.join(HERE, "data", "picotone_testclips.json"), encoding="utf-8")):
        r = by_name.get(t["name"])
        if r is None:
            print("WARNING: test clip missing", t["name"])
            continue
        ev.append({"name": f"{t['set']}_{t['index']:02d}_{t['group']}", "set": t["set"], "ref": r["ref"],
                   "offset": a.eval_offset, "dur": a.eval_seconds, "licence": r["licence"],
                   "artist": r.get("artist", ""), "title": r.get("title", ""), "quality_pass": r["keep"]})
    json.dump(ev, open(a.out + ".eval.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{len(ev)} picotone test-set excerpts -> {a.out}.eval.json")


if __name__ == "__main__":
    main()
