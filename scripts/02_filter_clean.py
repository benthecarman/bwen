#!/usr/bin/env python3
"""Stage 02 — filter and clean tweets into a tidy JSONL.

- Drops non-target languages and tweets that are only a URL/mention.
- Retweets: dropped by default; with clean.include_retweets they are kept tagged
  `is_own: false` so they enrich theme discovery (stage 03) ONLY — stages 04 and 06
  filter to is_own=true, so a retweet is never labeled or trained on (it's not your words).
- Expands t.co links to their real URLs (or strips trailing media links).
- De-dupes near-identical text.
- Keeps a small, useful set of fields.

Output: data/clean/tweets.jsonl
"""
from __future__ import annotations

import html
import json
import random
import re

from common import base_argparser, data_dir, load_config, maybe_skip, read_jsonl, write_jsonl

URL_RE = re.compile(r"https?://t\.co/\w+")
ANY_URL_RE = re.compile(r"https?://\S+")
RT_PREFIX_RE = re.compile(r"^RT @\w+:\s*")
MENTION_ONLY_RE = re.compile(r"^(?:@\w+\s*)+$")
LEADING_MENTIONS_RE = re.compile(r"^(?:@\w+\s+)+")
WS_RE = re.compile(r"\s+")


def build_url_map(tweet: dict) -> dict[str, str | None]:
    """Map each t.co url -> expanded_url (urls) or None (media, to strip)."""
    m: dict[str, str | None] = {}
    ents = tweet.get("entities", {})
    for u in ents.get("urls", []):
        if u.get("url"):
            m[u["url"]] = u.get("expanded_url")
    for media in ents.get("media", []) + tweet.get("extended_entities", {}).get("media", []):
        if media.get("url"):
            m[media["url"]] = None  # media links carry no text value -> strip
    return m


def clean_text(tweet: dict, cfg: dict) -> str:
    text = RT_PREFIX_RE.sub("", tweet.get("full_text", ""))  # drop "RT @user:" if present
    if cfg["clean"]["expand_urls"]:
        url_map = build_url_map(tweet)

        def repl(match: re.Match) -> str:
            url = match.group(0)
            if url in url_map:
                expanded = url_map[url]
                return "" if expanded is None and cfg["clean"]["strip_trailing_media_url"] else (expanded or url)
            return url

        text = URL_RE.sub(repl, text)
    if cfg["clean"]["strip_urls"]:
        # A generated URL is always a hallucination; drop them all for a voice model.
        text = ANY_URL_RE.sub("", text)
    text = html.unescape(text)
    text = WS_RE.sub(" ", text).strip()
    if cfg["clean"]["strip_leading_mentions"]:
        text = LEADING_MENTIONS_RE.sub("", text).strip()
    return text


def main() -> int:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config)
    ddir = data_dir(cfg)
    src = ddir / "raw" / "tweets.json"
    out = ddir / "clean" / "tweets.jsonl"

    if maybe_skip(out, args.force):
        return 0

    raw = json.loads(src.read_text(encoding="utf-8"))
    if args.limit:
        raw = raw[: args.limit]

    langs = set(cfg["clean"]["languages"] or [])
    include_rts = cfg["clean"]["include_retweets"]
    min_chars = cfg["clean"]["min_chars"]
    drop_replies_others = cfg["clean"]["drop_replies_to_others"]
    acct_id = str(cfg["account_id"])

    kept: list[dict] = []
    seen: set[str] = set()
    counts = {"rt": 0, "lang": 0, "empty": 0, "short": 0, "reply_other": 0, "dup": 0}

    for t in raw:
        full = t.get("full_text", "")
        is_retweet = full.startswith("RT @")
        if is_retweet and not include_rts:
            counts["rt"] += 1
            continue
        if langs and t.get("lang") not in langs:
            counts["lang"] += 1
            continue

        reply_to_user = t.get("in_reply_to_user_id_str")
        is_reply = (reply_to_user is not None) and not is_retweet
        is_self_reply = is_reply and reply_to_user == acct_id
        if drop_replies_others and is_reply and not is_self_reply:
            counts["reply_other"] += 1
            continue

        text = clean_text(t, cfg)
        if not text or MENTION_ONLY_RE.match(text):
            counts["empty"] += 1
            continue
        if len(text) < min_chars:
            counts["short"] += 1
            continue

        key = WS_RE.sub(" ", text.lower())
        if key in seen:
            counts["dup"] += 1
            continue
        seen.add(key)

        kept.append({
            "id": t.get("id_str") or t.get("id"),
            "text": text,
            "is_own": not is_retweet,   # retweets are context-only: themes yes, training no
            "favorite_count": int(t.get("favorite_count", 0) or 0),
            "retweet_count": int(t.get("retweet_count", 0) or 0),
            "is_reply": is_reply,
            "is_self_reply": is_self_reply,
            "in_reply_to_screen_name": t.get("in_reply_to_screen_name"),
            "created_at": t.get("created_at"),
        })

    # Fold in liked tweets (context-only) for theme discovery. Capped so a large likes
    # set doesn't swamp your own tweets; never labeled or trained on (is_own=false).
    n_like = 0
    if cfg["clean"]["include_likes"]:
        likes_path = ddir / "raw" / "likes.json"
        if not likes_path.exists():
            print("[clean] include_likes set but data/raw/likes.json missing — run stage 01 first")
        else:
            likes_raw = json.loads(likes_path.read_text(encoding="utf-8"))
            cap = cfg["clean"]["max_likes"]  # 0 = no cap (use all likes)
            if args.limit:
                cap = min(cap, args.limit) if cap else args.limit
            if cap and len(likes_raw) > cap:
                likes_raw = random.Random(cfg["train"]["seed"]).sample(likes_raw, cap)
            for lk in likes_raw:
                ft = lk.get("fullText")
                if not ft:
                    continue
                text = clean_text({"full_text": ft}, cfg)
                if not text or MENTION_ONLY_RE.match(text) or len(text) < min_chars:
                    continue
                key = WS_RE.sub(" ", text.lower())
                if key in seen:
                    continue
                seen.add(key)
                kept.append({
                    "id": lk.get("tweetId"), "text": text, "is_own": False,
                    "favorite_count": 0, "retweet_count": 0, "is_reply": False,
                    "is_self_reply": False, "in_reply_to_screen_name": None,
                    "created_at": None,
                })
                n_like += 1
            print(f"[clean] folded in {n_like} liked tweets (themes-only)")

    n = write_jsonl(out, kept)
    n_notown = sum(1 for r in kept if not r["is_own"])
    n_rt = n_notown - n_like
    print(f"[clean] kept {n} ({n - n_notown} own + {n_rt} retweets + {n_like} likes) -> {out}")
    print(f"[clean] dropped: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
