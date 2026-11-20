"""Phase 2 data: CC0 game music and sound effects from OpenGameArt.org.

Usage: fetch_opengameart.py OUT_DIR
Uses the site's advanced search restricted to the CC0 licence, then checks that each art
page itself states CC0. Authors are split by a hash of their name: about a quarter go to
"test2" and never appear in "fit2". Appends to OUT_DIR/manifest_oga.json.
"""
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "audio-research-script/1.0"}
OUT = sys.argv[1]
MANIFEST = os.path.join(OUT, "manifest_oga.json")
manifest = json.load(open(MANIFEST, encoding="utf-8")) if os.path.exists(MANIFEST) else []
have = {m["url"] for m in manifest}
done_pages = {m["item"] for m in manifest}
CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"
AUDIO = (".mp3", ".ogg", ".wav", ".flac", ".zip")


def get(url):
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            print("retry", url[:100], str(e)[:80], flush=True)
            time.sleep(4 * (attempt + 1))
    return None


def safe(text):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)[:60]


def run(art_type, group, pages, max_bytes):
    for page in range(pages):
        url = ("https://opengameart.org/art-search-advanced?keys=&field_art_type_tid%5B%5D="
               f"{art_type}&field_art_licenses_tid%5B%5D=4&sort_by=count&sort_order=DESC&page={page}")
        raw = get(url)
        if not raw:
            continue
        links = sorted(set(re.findall(r'href="(/content/[^"#?]+)"', raw.decode("utf-8", "replace"))))
        for link in links:
            page_url = "https://opengameart.org" + link
            if page_url in done_pages:
                continue
            time.sleep(1.0)
            art = get(page_url)
            if not art:
                continue
            text = art.decode("utf-8", "replace")
            if "CC0" not in text:
                continue
            author = re.search(r'class="username"[^>]*>([^<]+)<', text)
            author = html.unescape(author.group(1)).strip().lower() if author else "unknown"
            files = [html.unescape(f) for f in re.findall(
                r'href="(https?://opengameart\.org/sites/default/files/[^"]+)"', text)]
            files = [f for f in files if f.lower().endswith(AUDIO)]
            if not files:
                continue
            split = "test2" if int(hashlib.sha1(author.encode()).hexdigest(), 16) % 4 == 0 else "fit2"
            # prefer a directly playable file over an archive
            files.sort(key=lambda f: f.lower().endswith(".zip"))
            f = files[0]
            if f in have:
                continue
            data = get(urllib.parse.quote(f, safe=":/%"))
            if not data or len(data) < 20_000 or len(data) > max_bytes:
                continue
            ext = os.path.splitext(f)[1].lower()
            name = f"{split}__{group}__oga_{safe(link.split('/')[-1])}__{hashlib.sha1(f.encode()).hexdigest()[:8]}{ext}"
            with open(os.path.join(OUT, name), "wb") as fh:
                fh.write(data)
            title = re.search(r"<title>([^<|]+)", text)
            manifest.append({"file": name, "url": f, "licence": CC0, "split": split, "group": group,
                             "source": "OpenGameArt.org", "creator": author,
                             "title": html.unescape(title.group(1)).strip() if title else link,
                             "item": page_url})
            have.add(f)
            done_pages.add(page_url)
            with open(MANIFEST, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=1)
            print("ok", name, len(data) // 1000, "kB", author, flush=True)


run(12, "game", 8, 60_000_000)
run(13, "sfx", 6, 60_000_000)
print("done", len(manifest), flush=True)
