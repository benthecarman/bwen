#!/usr/bin/env python3
"""Stage 01 — parse the raw Twitter/X archive into flat JSON.

Twitter exports tweets as JS files: `window.YTD.tweets.part0 = [ ... ]`. Large
accounts split this across `tweets.js`, `tweets-part1.js`, etc., and older exports
use `tweet.js`. We glob all variants and concatenate.

If clean.include_likes is set, also parses like*.js (liked tweets) — these are used
only to enrich theme discovery and are never trained on.

Output: data/raw/tweets.json  (a plain JSON array of the raw tweet objects)
        data/raw/likes.json   (when include_likes — liked tweet objects)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from common import archive_dir, base_argparser, data_dir, load_config

PREFIX_RE = re.compile(r"^\s*window\.YTD\.[\w.]+\s*=\s*", re.DOTALL)


def find_files(adir: Path, pattern: str) -> list[Path]:
    return sorted(p for p in adir.glob("*.js") if re.match(pattern, p.name))


def parse_file(path: Path, key: str) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    text = PREFIX_RE.sub("", text, count=1).strip()
    data = json.loads(text)
    # Each entry wraps the object under `key` (e.g. "tweet" or "like").
    return [entry.get(key, entry) for entry in data]


def parse_group(adir: Path, pattern: str, key: str, label: str) -> list[dict]:
    files = find_files(adir, pattern)
    if not files:
        return []
    print(f"[parse] {label}: {len(files)} file(s): {', '.join(f.name for f in files)}")
    items: list[dict] = []
    for f in files:
        part = parse_file(f, key)
        print(f"[parse]   {f.name}: {len(part)}")
        items.extend(part)
    return items


def main() -> int:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config)
    adir = archive_dir(cfg)
    out = data_dir(cfg) / "raw" / "tweets.json"
    likes_out = out.parent / "likes.json"

    # Skip only when EVERY configured output already exists. If a user first ran with
    # include_likes:false (so likes.json was never written) and later enables likes,
    # tweets.json exists but likes.json doesn't — skipping on tweets.json alone would
    # leave likes.json missing forever, and stage 02 would silently omit all likes.
    expected = [out] + ([likes_out] if cfg["filter"]["include_likes"] else [])
    if all(p.exists() for p in expected) and not args.force:
        for p in expected:
            print(f"[skip] {p} exists (use --force to recompute)")
        return 0

    tweets = parse_group(adir, r"^tweets?(-part\d+)?\.js$", "tweet", "tweets")
    if not tweets:
        print(f"[error] no tweets*.js found in {adir}", file=sys.stderr)
        return 1
    if args.limit:
        tweets = tweets[: args.limit]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tweets, ensure_ascii=False), encoding="utf-8")
    print(f"[parse] wrote {len(tweets)} tweets -> {out}")

    if cfg["filter"]["include_likes"]:
        likes = parse_group(adir, r"^like(-part\d+)?\.js$", "like", "likes")
        likes_out.write_text(json.dumps(likes, ensure_ascii=False), encoding="utf-8")
        print(f"[parse] wrote {len(likes)} likes -> {likes_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
