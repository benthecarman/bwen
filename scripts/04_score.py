#!/usr/bin/env python3
"""Stage 04 — score candidates and select a cluster-balanced shortlist to label.

- Heuristic score: engagement (favorites + retweets) + a length sweet-spot factor.
- Optional LLM score (filter only, never writes training text): the top heuristic
  pool is scored 1-5 for opinion-density and voice via a local Ollama model.
- Selects `shortlist_size` tweets, round-robin across discovered clusters so opinion
  coverage is broad rather than dominated by one loud topic.

Output: data/candidates_scored.jsonl  (all candidates + scores)
        data/shortlist.jsonl          (the subset to hand-label in stage 05)
"""
from __future__ import annotations

import math
from collections import defaultdict

from tqdm import tqdm

from common import (apply_subject_edits, base_argparser, data_dir, load_config,
                    maybe_skip, ollama_generate, read_jsonl, write_jsonl)

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "opinion_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "voice_score": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["opinion_score", "voice_score"],
}

SCORE_PROMPT = (
    "Rate this tweet on two axes, 1-5 each. opinion_score: how strongly it expresses "
    "a personal stance/opinion (5 = sharp take, 1 = pure fact/logistics). voice_score: "
    "how distinctive/personality-rich the writing is (5 = very characterful, 1 = generic). "
    'Reply as JSON {{"opinion_score": n, "voice_score": n}}.\n\nTweet: {text}'
)


def heuristic_score(r: dict) -> float:
    eng = math.log1p(r.get("favorite_count", 0) + 2 * r.get("retweet_count", 0))
    n = len(r["text"])
    # Sweet spot ~40-220 chars: substantial but punchy.
    length_factor = min(n, 220) / 220 if n >= 25 else n / 50
    return eng + length_factor


def llm_score(text: str, model: str) -> tuple[int, int]:
    import json
    resp = ollama_generate(model, SCORE_PROMPT.format(text=text), fmt=SCORE_SCHEMA,
                           options={"temperature": 0})
    d = json.loads(resp)
    return int(d["opinion_score"]), int(d["voice_score"])


def combined(r: dict) -> float:
    if "opinion_score" in r:
        return r["opinion_score"] + r["voice_score"] + 0.25 * r["heuristic"]
    return r["heuristic"]


def main() -> int:
    p = base_argparser(__doc__)
    args = p.parse_args()
    cfg = load_config(args.config)
    ddir = data_dir(cfg)
    rows = read_jsonl(ddir / "candidates.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    scored_out = ddir / "candidates_scored.jsonl"
    shortlist_out = ddir / "shortlist.jsonl"
    if maybe_skip(shortlist_out, args.force):
        return 0

    # Pick up any renames/merges made to subjects.txt after stage 03 so the edited
    # theme labels flow into the shortlist, the labeling UI, and eval.jsonl.
    n_edit = apply_subject_edits(rows, ddir)
    if n_edit:
        print(f"[score] applied {n_edit} subjects.txt edit(s) to candidate subjects")

    for r in rows:
        r["heuristic"] = round(heuristic_score(r), 4)

    # Retweets shape theme discovery (stage 03) but are not your words: never label
    # or train on them. Only own tweets are scored and eligible for the shortlist.
    own = [r for r in rows if r.get("is_own", True)]
    n_ctx = len(rows) - len(own)
    if n_ctx:
        print(f"[score] excluding {n_ctx} context-only tweets (retweets/likes) from shortlist")

    scfg = cfg["score"]
    if scfg["use_llm"]:
        pool_n = scfg["llm_score_pool"]
        pool = sorted(own, key=lambda r: r["heuristic"], reverse=True)[:pool_n]
        print(f"[score] LLM-scoring top {len(pool)} by heuristic with {scfg['llm_model']}...")
        for r in tqdm(pool, desc="llm-score"):
            try:
                op, vo = llm_score(r["text"], scfg["llm_model"])
                r["opinion_score"], r["voice_score"] = op, vo
            except Exception as e:  # noqa: BLE001
                print(f"[score] skip {r['id']}: {e}")

    write_jsonl(scored_out, rows)

    # Round-robin across clusters by combined score (own tweets only). HDBSCAN labels
    # off-theme tweets as cluster -1 (noise), often a large fraction of points; giving
    # it an equal rotation slot would pack the shortlist with off-theme tweets, so we
    # exclude it from the rotation and only backfill from it if the real clusters can't
    # fill the target.
    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for r in own:
        by_cluster[r.get("cluster", -1)].append(r)
    for lst in by_cluster.values():
        lst.sort(key=combined, reverse=True)

    target = scfg["shortlist_size"]
    balance = scfg["balance_across_clusters"]
    shortlist: list[dict] = []
    if balance:
        clusters = sorted(c for c in by_cluster if c != -1)
        cursors = {c: 0 for c in clusters}
        while len(shortlist) < target and any(cursors[c] < len(by_cluster[c]) for c in clusters):
            for c in clusters:
                if cursors[c] < len(by_cluster[c]):
                    shortlist.append(by_cluster[c][cursors[c]])
                    cursors[c] += 1
                    if len(shortlist) >= target:
                        break
        if len(shortlist) < target and -1 in by_cluster:
            backfill = by_cluster[-1][: target - len(shortlist)]
            if backfill:
                print(f"[score] real clusters under target; backfilling {len(backfill)} from noise")
            shortlist.extend(backfill)
    else:
        shortlist = sorted(own, key=combined, reverse=True)[:target]

    write_jsonl(shortlist_out, shortlist)
    n_theme_clusters = len([c for c in by_cluster if c != -1])
    print(f"[score] wrote {len(rows)} scored -> {scored_out}")
    print(f"[score] wrote {len(shortlist)} shortlist (across {n_theme_clusters} clusters) -> {shortlist_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
