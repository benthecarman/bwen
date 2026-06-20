#!/usr/bin/env python3
"""Stage 02 — filter and clean tweets into a tidy JSONL.

- Drops non-target languages and tweets that are only a URL/mention.
- Retweets: dropped by default; with filter.include_retweets they are kept tagged
  `is_own: false` so they enrich theme discovery (stage 03) ONLY — stages 04 and 06
  filter to is_own=true, so a retweet is never labeled or trained on (it's not your words).
- Expands t.co links to their real URLs (or strips trailing media links).
- Folds repeats of the same text (ignoring case/punctuation/emoji) into a dup_count
  instead of dropping them, so a recurring catchphrase can be emphasized downstream.
- Keeps a small, useful set of fields.

Output: data/filtered/tweets.jsonl
"""
from __future__ import annotations

import html
import json
import random
import re

from common import (base_argparser, data_dir, load_config, maybe_skip, read_jsonl,
                    require_file, write_jsonl)

URL_RE = re.compile(r"https?://t\.co/\w+")
ANY_URL_RE = re.compile(r"https?://\S+")
RT_PREFIX_RE = re.compile(r"^RT @\w+:\s*")
MENTION_ONLY_RE = re.compile(r"^(?:@\w+\s*)+$")
LEADING_MENTIONS_RE = re.compile(r"^(?:@\w+\s+)+")
WS_RE = re.compile(r"\s+")
# Anything that isn't a word char or whitespace — i.e. punctuation and emoji. \w is
# Unicode-aware so accented letters/other scripts are kept; only symbols are dropped.
PUNCT_EMOJI_RE = re.compile(r"[^\w\s]", re.UNICODE)


def dedup_key(text: str) -> str:
    """Normalized key for folding exact-ish repeats into one dup_count.

    Lowercases, drops punctuation/emoji, and collapses whitespace so a catchphrase
    and its variants ('fuck the bears', 'Fuck the bears!', 'fuck the bears 🐻') all
    map to the same key and accumulate. The original text is kept for display.
    """
    return WS_RE.sub(" ", PUNCT_EMOJI_RE.sub(" ", text.lower())).strip()


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
    if cfg["filter"]["expand_urls"]:
        url_map = build_url_map(tweet)

        def repl(match: re.Match) -> str:
            url = match.group(0)
            if url in url_map:
                expanded = url_map[url]
                return "" if expanded is None and cfg["filter"]["strip_trailing_media_url"] else (expanded or url)
            return url

        text = URL_RE.sub(repl, text)
    if cfg["filter"]["strip_urls"]:
        # A generated URL is always a hallucination; drop them all for a voice model.
        text = ANY_URL_RE.sub("", text)
    text = html.unescape(text)
    text = WS_RE.sub(" ", text).strip()
    if cfg["filter"]["strip_leading_mentions"]:
        text = LEADING_MENTIONS_RE.sub("", text).strip()
    return text


def main() -> int:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config)
    ddir = data_dir(cfg)
    src = ddir / "raw" / "tweets.json"
    out = ddir / "filtered" / "tweets.jsonl"

    if maybe_skip(out, args.force):
        return 0

    require_file(src, "stage 01 (just parse)")
    raw = json.loads(src.read_text(encoding="utf-8"))
    if args.limit:
        raw = raw[: args.limit]

    langs = set(cfg["filter"]["languages"] or [])
    include_rts = cfg["filter"]["include_retweets"]
    min_chars = cfg["filter"]["min_chars"]
    drop_replies_others = cfg["filter"]["drop_replies_to_others"]
    acct_id = str(cfg["account_id"])

    kept: list[dict] = []
    seen: dict[str, int] = {}   # normalized text -> index in `kept`, to fold dups into a count
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

        key = dedup_key(text)
        if key in seen:
            # Don't discard exact repeats — fold them into a count so a phrase you tweet
            # over and over (a catchphrase) can be emphasized in the voice layer later.
            kept[seen[key]]["dup_count"] += 1
            counts["dup"] += 1
            continue
        seen[key] = len(kept)

        kept.append({
            "id": t.get("id_str") or t.get("id"),
            "text": text,
            "dup_count": 1,             # times this exact text was tweeted (>=1)
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
    if cfg["filter"]["include_likes"]:
        likes_path = ddir / "raw" / "likes.json"
        if not likes_path.exists():
            print("[filter] include_likes set but data/raw/likes.json missing — run stage 01 first")
        else:
            likes_raw = json.loads(likes_path.read_text(encoding="utf-8"))
            cap = cfg["filter"]["max_likes"]  # 0 = no cap (use all likes)
            if not cap and not args.limit:
                print(f"[filter] WARNING: max_likes:0 (uncapped) — folding in all "
                      f"{len(likes_raw)} likes. A large likes set swamps your own tweets "
                      f"in theme discovery and makes clustering slow; set a cap (e.g. 5000).")
            if args.limit:
                cap = min(cap, args.limit) if cap else args.limit
            if cap and len(likes_raw) > cap:
                likes_raw = random.Random(cfg["train"]["seed"]).sample(likes_raw, cap)
            # Likes carry only fullText — no `entities` — so build_url_map can't map their
            # t.co links and expand_urls is a no-op for them (only own tweets can expand).
            # Harmless under the default strip_urls:true; surface it only if a user relies
            # on expansion without stripping, so the inconsistency isn't silent.
            if cfg["filter"]["expand_urls"] and not cfg["filter"]["strip_urls"]:
                print("[filter] note: liked tweets lack URL entities — their t.co links "
                      "won't be expanded (only your own tweets can be).")
            for lk in likes_raw:
                ft = lk.get("fullText")
                if not ft:
                    continue
                text = clean_text({"full_text": ft}, cfg)
                if not text or MENTION_ONLY_RE.match(text) or len(text) < min_chars:
                    continue
                key = dedup_key(text)
                if key in seen:
                    continue
                seen[key] = len(kept)
                kept.append({
                    # likes are themes-only and never trained, so dup_count stays 1 here
                    "id": lk.get("tweetId"), "text": text, "dup_count": 1, "is_own": False,
                    "favorite_count": 0, "retweet_count": 0, "is_reply": False,
                    "is_self_reply": False, "in_reply_to_screen_name": None,
                    "created_at": None,
                })
                n_like += 1
            print(f"[filter] folded in {n_like} liked tweets (themes-only)")

    n = write_jsonl(out, kept)
    n_notown = sum(1 for r in kept if not r["is_own"])
    n_rt = n_notown - n_like
    print(f"[filter] kept {n} ({n - n_notown} own + {n_rt} retweets + {n_like} likes) -> {out}")
    # `dup` here is exact repeats folded into dup_count (emphasis), not discarded rows.
    print(f"[filter] dropped: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
