"""Download public-domain / CC0 audio from archive.org and write a manifest.

Usage: fetch_audio.py OUT_DIR
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "audio-research-script/1.0"}
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)


def get(url):
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            print("retry", url, e, flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def secs(f):
    v = str(f.get("length", "0") or "0")
    if ":" in v:
        parts = [float(p) for p in v.split(":")]
        t = 0.0
        for p in parts:
            t = t * 60 + p
        return t
    return float(v)


def gold(split_test):
    def pick(files):
        out = []
        for f in sorted(files, key=lambda f: f["name"]):
            n = f["name"]
            if not n.endswith(".flac"):
                continue
            is_test = any(s in n for s in ("26Variatio25", "28Variatio27", "- 08 Variatio 7",
                                           "23Variatio22"))
            if is_test == split_test:
                out.append(f)
        return out
    return pick


def chopin(split_test):
    def pick(files):
        out = []
        for f in sorted(files, key=lambda f: f["name"]):
            n = f["name"]
            if not n.endswith(".mp3") or secs(f) < 90:
                continue
            is_test = not (n.startswith("Prelude") or n.startswith("Waltz")
                           or n.startswith("Mazurka") or n.startswith("Etude"))
            if is_test == split_test:
                out.append(f)
        return out[:4] if split_test else out[:24]
    return pick


def ext(e, names=None, limit=99):
    def pick(files):
        out = [f for f in sorted(files, key=lambda f: f["name"]) if f["name"].endswith(e)
               and (names is None or any(s in f["name"] for s in names)) and secs(f) >= 30]
        return out[:limit]
    return pick


def chapter(files):
    mp3s = [f for f in files if f["name"].endswith("_64kb.mp3") and 400 < secs(f) < 2400]
    if not mp3s:
        mp3s = [f for f in files if f["name"].endswith(".mp3")
                and 3_000_000 < int(f.get("size", 0)) < 20_000_000]
    mp3s = sorted(mp3s, key=lambda f: f["name"])
    # A chapter from the middle, so readers differ from the usual chapter-1 reader.
    return mp3s[len(mp3s) // 2:len(mp3s) // 2 + 1]


ITEMS = [
    ("OpenGoldbergVariations", gold(False), "fit", "piano"),
    ("OpenGoldbergVariations", gold(True), "test", "piano"),
    ("musopen-chopin", chopin(False), "fit", "piano"),
    ("musopen-chopin", chopin(True), "test", "piano"),
    ("musopen-brahms-symphony-premix", ext(".mp3", ["No._1", "No._2"]), "fit", "orchestraLR"),
    ("musopen-brahms-symphony-premix", ext(".mp3", ["No._4"]), "test", "orchestraLR"),
    ("WorldClassMarches", ext(".mp3"), "fit", "band"),
    ("StraussWaltz", ext(".mp3"), "fit", "band"),
    ("TurkeyInTheStraw", ext(".mp3"), "fit", "band"),
    ("GeorgeM.CohanMedley", ext(".mp3"), "fit", "band"),
    ("MarineCorpsBirthdayBallMusic", ext(".mp3", limit=6), "fit", "band"),
    ("MusicOfRichardStrauss", ext(".mp3", ["03Guntram", "05Parade", "06LoveScene"]), "test",
     "band"),
    ("KingChampagne", ext(".mp3"), "test", "band"),
    ("TheFourSeasonsWinter", ext(".mp3"), "test", "band"),
    ("GT585", ext(".mp3", limit=8), "fit", "electronic"),
    ("gt373TheRogueSources-Honey", ext(".mp3", limit=5), "fit", "electronic"),
    ("gt536Maelstrom-ThePassage", ext(".mp3", limit=4), "fit", "electronic"),
    ("gt577Noisesurfer-RainTearDrops", ext(".mp3", limit=3), "test", "electronic"),
]
for ident in ["art_of_war_librivox", "tom_sawyer_librivox", "moby_dick_librivox",
              "dracula_librivox", "pride_and_prejudice_librivox",
              "count_monte_cristo_0711_librivox", "secret_garden_librivox",
              "odyssey_butler_librivox", "grimms_english_librivox", "uncle_toms_cabin_librivox",
              "tale_two_cities_librivox", "treasure_island_ap_librivox", "emma_solo_librivox",
              "swiss_family_robinson_librivox", "great_expectations_mfs_0812_librivox",
              "bleak_house_cl_librivox", "anthem_librivox", "romeo_and_juliet_librivox",
              "game_of_life_0911_librivox", "memoirs_holmes_0709_librivox"]:
    ITEMS.append((ident, chapter, "fit", "speech"))
for ident in ["adventures_holmes", "huck_finn_librivox", "jane_eyre_ver03_0809_librivox",
              "peter_pan_0707_librivox", "robinson_crusoe_librivox"]:
    ITEMS.append((ident, chapter, "test", "speech"))

manifest = []
for ident, pick, split, group in ITEMS:
    raw = get("https://archive.org/metadata/" + ident)
    if raw is None:
        print("NO METADATA", ident, flush=True)
        continue
    meta = json.loads(raw)
    lic = str(meta.get("metadata", {}).get("licenseurl", ""))
    if "publicdomain" not in lic:
        print("SKIP licence", ident, lic, flush=True)
        continue
    chosen = pick(meta.get("files", []))
    if not chosen:
        print("NO FILE", ident, flush=True)
    for f in chosen:
        name = f["name"]
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in ident + "__" + name)
        path = os.path.join(OUT, split + "__" + group + "__" + safe)
        url = "https://archive.org/download/" + ident + "/" + urllib.parse.quote(name)
        if not os.path.exists(path):
            data = get(url)
            if data is None:
                print("FAILED", url, flush=True)
                continue
            with open(path, "wb") as fh:
                fh.write(data)
        manifest.append({"file": os.path.basename(path), "url": url, "licence": lic,
                         "split": split, "group": group, "item": ident,
                         "title": meta["metadata"].get("title"),
                         "creator": meta["metadata"].get("creator")})
        print("ok", os.path.basename(path), os.path.getsize(path) // 1000, "kB", lic, flush=True)
        with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1)
print("done", len(manifest))
