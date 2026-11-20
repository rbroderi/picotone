"""Phase 2 data: downloads PD / CC0 / CC-BY audio from several sources, with a licence record.

Usage: fetch_audio_phase2.py OUT_DIR
Sources: archive.org (Kevin MacLeod discography, netlabel releases by genre, LibriVox),
ccMixter (CC-BY uploads by tag) and Kenney (CC0 sound effects). Only items whose own
metadata states public domain, CC0 or CC-BY (no NC / ND / SA) are taken. Appends to
OUT_DIR/manifest2.json after every file, so it can be re-run to resume.
Splits: "fit2" builds tables; "test2" is held out and comes from creators / items that
never appear in fit2 (enforced by creator name below).
"""
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile

UA = {"User-Agent": "audio-research-script/1.0"}
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
MANIFEST = os.path.join(OUT, "manifest2.json")
manifest = json.load(open(MANIFEST, encoding="utf-8")) if os.path.exists(MANIFEST) else []
have = {m["url"] for m in manifest}
MAX_BYTES = 40_000_000


def get(url, limit=None, tries=3):
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read(limit) if limit else r.read()
        except Exception as e:  # noqa: BLE001
            print("retry", url[:100], str(e)[:80], flush=True)
            time.sleep(3 * (attempt + 1))
    return None


def allowed(lic):
    """PD, CC0 or CC-BY only."""
    lic = (lic or "").lower()
    if "publicdomain" in lic:
        return True
    m = re.search(r"creativecommons\.org/licenses/([a-z-]+)/", lic)
    return bool(m) and m.group(1) == "by"


def safe(text):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)[:70]


def save(url, split, group, lic, source, creator, title, item, ext=None):
    if url in have:
        return True
    ext = ext or os.path.splitext(urllib.parse.urlparse(url).path)[1].lower() or ".mp3"
    name = f"{split}__{group}__{safe(item)}__{hashlib.sha1(url.encode()).hexdigest()[:8]}{ext}"
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        data = get(url)
        if not data or len(data) < 50_000:
            print("FAILED", url[:120], flush=True)
            return False
        with open(path, "wb") as f:
            f.write(data)
    manifest.append({"file": name, "url": url, "licence": lic, "split": split, "group": group,
                     "source": source, "creator": creator, "title": title, "item": item})
    have.add(url)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print("ok", name, os.path.getsize(path) // 1000, "kB", flush=True)
    return True


def ia_search(query, rows):
    params = [("q", query), ("fl[]", "identifier"), ("fl[]", "creator"), ("fl[]", "licenseurl"),
              ("fl[]", "title"), ("sort[]", "downloads desc"), ("rows", str(rows)),
              ("output", "json")]
    raw = get("https://archive.org/advancedsearch.php?" + urllib.parse.urlencode(params))
    return json.loads(raw)["response"]["docs"] if raw else []


def ia_files(ident):
    raw = get("https://archive.org/metadata/" + ident)
    if not raw:
        return None, []
    meta = json.loads(raw)
    files = [f for f in meta.get("files", []) if f["name"].lower().endswith(".mp3")
             and 1_500_000 < int(f.get("size", 0) or 0) < MAX_BYTES]
    return meta.get("metadata", {}), sorted(files, key=lambda f: f["name"])


def ia_url(ident, name):
    return "https://archive.org/download/" + ident + "/" + urllib.parse.quote(name)


def creator_of(doc):
    c = doc.get("creator") or doc.get("identifier")
    return (c[0] if isinstance(c, list) else str(c)).strip().lower()


# ---------------------------------------------------------------- Kevin MacLeod (CC-BY 4.0)
def macleod(limit):
    ident = "KevinMacLeod_2019-04_Discography"
    meta, files = ia_files(ident)
    lic = str((meta or {}).get("licenseurl", ""))
    if not allowed(lic):
        print("SKIP macleod licence", lic, flush=True)
        return
    step = max(len(files) // limit, 1)
    for f in files[::step][:limit]:
        save(ia_url(ident, f["name"]), "fit2", "game", lic, "archive.org", "kevin macleod",
             f["name"], ident)


# ---------------------------------------------------------------- netlabel releases by genre
GENRES = {
    "rock": "rock OR punk OR metal OR indie", "pop": "pop OR synthpop OR indiepop",
    "folk": "folk OR acoustic OR singer-songwriter OR country",
    "jazz": "jazz OR blues OR funk OR soul", "hiphop": '"hip hop" OR hiphop OR rap',
    "ambient": "ambient OR drone OR soundscape", "electronic": "techno OR house OR idm OR chiptune",
    "vocal": "vocal OR songs OR choir OR singing",
    "world": "world OR latin OR reggae OR classical OR orchestral OR soundtrack",
}


def netlabels(per_genre_fit, per_genre_test, tracks_per_item):
    seen_items = set()
    fit_creators, test_creators = set(), set()
    for genre, terms in GENRES.items():
        q = ("collection:netlabels AND mediatype:audio AND (licenseurl:(*licenses\\/by\\/*) OR "
             "licenseurl:(*publicdomain\\/zero*)) AND subject:(" + terms + ")")
        docs = ia_search(q, (per_genre_fit + per_genre_test) * 3)
        n_fit = n_test = 0
        for i, doc in enumerate(docs):
            ident, lic = doc["identifier"], str(doc.get("licenseurl", ""))
            who = creator_of(doc)
            # GT releases often have no creator; treat each such item as its own creator.
            if ident in seen_items or not allowed(lic):
                continue
            want_test = (i % 5 == 4) and n_test < per_genre_test
            if want_test and who in fit_creators:
                want_test = False
            if not want_test and (who in test_creators or n_fit >= per_genre_fit):
                if n_test >= per_genre_test or who in fit_creators:
                    continue
                want_test = True
            split = "test2" if want_test else "fit2"
            meta, files = ia_files(ident)
            if not files or not allowed(str((meta or {}).get("licenseurl", ""))):
                continue
            seen_items.add(ident)
            got = 0
            step = max(len(files) // tracks_per_item, 1)
            for f in files[::step][:tracks_per_item]:
                got += save(ia_url(ident, f["name"]), split, genre, lic, "archive.org netlabels",
                            who, f"{doc.get('title')} / {f['name']}", ident)
            if got:
                if want_test:
                    n_test += 1
                    test_creators.add(who)
                else:
                    n_fit += 1
                    fit_creators.add(who)
            if n_fit >= per_genre_fit and n_test >= per_genre_test:
                break
        print("netlabels", genre, "fit items", n_fit, "test items", n_test, flush=True)


# ---------------------------------------------------------------- ccMixter (CC-BY)
def ccmixter(tags, per_tag_fit, per_tag_test):
    fit_users, test_users = set(), set()
    for tag in tags:
        raw = get("http://ccmixter.org/api/query?" + urllib.parse.urlencode(
            {"f": "json", "lic": "by", "limit": str((per_tag_fit + per_tag_test) * 3), "tags": tag,
             "sort": "rank"}))
        try:
            rows = json.loads(raw) if raw else []
        except ValueError:
            rows = []
        n_fit = n_test = 0
        for i, row in enumerate(rows):
            lic, user = row.get("license_url", ""), str(row.get("user_name", "")).lower()
            files = [f for f in row.get("files", []) if str(f.get("download_url", "")).endswith(".mp3")]
            if not allowed(lic) or not files or user in ("",):
                continue
            want_test = (i % 5 == 4 and user not in fit_users and n_test < per_tag_test)
            if not want_test and (user in test_users or n_fit >= per_tag_fit):
                continue
            ok = save(files[0]["download_url"], "test2" if want_test else "fit2",
                      "vocal" if tag in ("vocals", "acappella", "female_vocals", "male_vocals") else tag,
                      lic, "ccMixter", user, row.get("upload_name"), row.get("file_page_url", ""))
            if ok and want_test:
                n_test += 1
                test_users.add(user)
            elif ok:
                n_fit += 1
                fit_users.add(user)
        print("ccmixter", tag, "fit", n_fit, "test", n_test, flush=True)


# ---------------------------------------------------------------- LibriVox (public domain)
def librivox(rows, skip_ids):
    docs = ia_search("collection:librivoxaudio AND mediatype:audio", rows)
    n = 0
    for doc in docs:
        ident, lic = doc["identifier"], str(doc.get("licenseurl", ""))
        if ident in skip_ids or "publicdomain" not in lic:
            continue
        meta, files = ia_files(ident)
        files = [f for f in files if f["name"].endswith("_64kb.mp3")]
        if not files:
            continue
        f = files[len(files) // 2]
        n += save(ia_url(ident, f["name"]), "fit2", "speech", lic, "LibriVox via archive.org",
                  ident, f["name"], ident)
    print("librivox", n, flush=True)


# ---------------------------------------------------------------- Kenney (CC0 sound effects)
KENNEY = [("rpg-audio", "fit2"), ("impact-sounds", "fit2"), ("interface-sounds", "fit2"),
          ("sci-fi-sounds", "fit2"), ("digital-audio", "fit2"), ("ui-audio", "fit2"),
          ("casino-audio", "fit2"), ("music-jingles", "fit2"), ("voiceover-pack", "fit2"),
          ("voiceover-pack-fighter", "fit2")]


def kenney():
    for slug, split in KENNEY:
        page = get("https://kenney.nl/assets/" + slug)
        if not page:
            continue
        text = page.decode("utf-8", "replace")
        m = re.search(r'href="(https://kenney\.nl/media/pages/assets/[^"]+\.zip)"', text)
        cc0 = "Creative Commons CC0" in text or "CC0" in text
        if not m or not cc0:
            print("kenney: no zip or no CC0 statement for", slug, flush=True)
            continue
        save(m.group(1), split, "sfx", "https://creativecommons.org/publicdomain/zero/1.0/",
             "kenney.nl", "kenney", slug, "https://kenney.nl/assets/" + slug, ext=".zip")


if __name__ == "__main__":
    which = sys.argv[2:] or ["kenney", "macleod", "ccmixter", "netlabels", "librivox"]
    if "kenney" in which:
        kenney()
    if "macleod" in which:
        macleod(170)
    if "ccmixter" in which:
        ccmixter(["vocals", "acappella", "hip_hop", "pop", "rock", "folk", "jazz", "female_vocals",
                  "male_vocals", "ambient", "soundtrack"], 14, 3)
    if "netlabels" in which:
        netlabels(36, 5, 2)
    if "librivox" in which:
        old = json.load(open(os.path.join(OUT, "manifest.json"), encoding="utf-8"))
        librivox(70, {m["item"] for m in old})
    print("done", len(manifest), flush=True)
