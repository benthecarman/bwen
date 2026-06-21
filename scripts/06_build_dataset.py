#!/usr/bin/env python3
"""Stage 06 — assemble the training set from real tweets only.

Two kinds of examples (both 100% your real words — no synthetic text):
  - chat: hand-labeled instruction pairs  {system persona, user=your prompt, assistant=your tweet}
  - text: voice layer of raw tweets (no prompt) to reinforce style for free

Templating is deferred to stage 07 (which has the tokenizer), so this stage stays
dependency-light and the output is human-inspectable.

Output: data/train.jsonl  (rows tagged {"type": "chat"|"text", ...})
        data/eval.jsonl   (held-out instruction pairs for stage 09)
"""
from __future__ import annotations

import random
import re

from common import base_argparser, data_dir, load_config, read_jsonl, state_dir, write_jsonl

_WORD_RE = re.compile(r"\w+")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(text.lower()))


def _near_dup(tokens: frozenset[str], refs: list[frozenset[str]], thresh: float) -> bool:
    """True if `tokens` has token-set Jaccard >= thresh with any reference.

    Stage 02 only drops exact duplicates, so a lightly-reworded near-duplicate of an
    eval reference can otherwise survive into the voice layer and contaminate eval.
    """
    if not tokens:
        return False
    for r in refs:
        union = len(tokens | r)
        if union and len(tokens & r) / union >= thresh:
            return True
    return False


def main() -> int:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config)
    ddir = data_dir(cfg)
    dcfg = cfg["dataset"]
    seed = cfg["train"]["seed"]
    persona = dcfg["persona"].format(handle=cfg["handle"])

    labeled_path = state_dir(cfg) / "labeled.jsonl"
    labeled = read_jsonl(labeled_path) if labeled_path.exists() else []
    if not labeled:
        print("[build] no labeled.jsonl yet — run stage 05 first.")
        return 1
    if args.limit:
        labeled = labeled[: args.limit]

    rng = random.Random(seed)
    rng.shuffle(labeled)
    holdout = min(dcfg["eval_holdout"], max(0, len(labeled) - 1))
    eval_rows = labeled[:holdout]
    train_pairs = labeled[holdout:]

    rows: list[dict] = []
    for r in train_pairs:
        rows.append({
            "type": "chat",
            "messages": [
                {"role": "system", "content": persona},
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["completion"]},
            ],
        })

    # Voice layer: raw tweets, excluding ones already used as completions.
    if dcfg["voice_layer"]:
        used = {str(r["id"]) for r in labeled}
        scored = read_jsonl(ddir / "candidates_scored.jsonl")
        pool = [r for r in scored
                if r.get("is_own", True) and str(r["id"]) not in used
                and r.get("heuristic", 0) >= dcfg["voice_min_score"]]
        pool.sort(key=lambda r: r.get("heuristic", 0), reverse=True)
        pool = pool[: dcfg["voice_pool_size"]]
        # Exclude near-duplicates of eval references (id exclusion above only catches
        # exact reuse) so a reworded copy of a held-out tweet can't leak into training.
        eval_refs = [_tokens(r["completion"]) for r in eval_rows]
        thresh = dcfg["voice_dedup_threshold"]
        kept_pool = [r for r in pool if not _near_dup(_tokens(r["text"]), eval_refs, thresh)]
        n_leak = len(pool) - len(kept_pool)
        # A tweet repeated verbatim (a catchphrase) carries dup_count > 1 from stage 02.
        # Emphasize it by emitting it that many times. cap controls the ceiling:
        #   0 -> unlimited (use the full dup_count), 1 -> no emphasis (one copy each),
        #   >1 -> capped so one phrase can't swamp the voice layer.
        cap = dcfg["voice_max_repeat"]
        n_emph = 0
        for r in kept_pool:
            dc = max(1, int(r.get("dup_count", 1)))
            reps = dc if cap == 0 else (1 if cap <= 1 else min(dc, cap))
            n_emph += reps - 1
            for _ in range(reps):
                rows.append({"type": "text", "text": r["text"]})
        print(f"[build] voice layer: {len(kept_pool)} raw tweets"
              + (f" (+{n_emph} repeated for emphasis)" if n_emph else "")
              + (f" ({n_leak} dropped as eval near-duplicates)" if n_leak else ""))

    rng.shuffle(rows)
    write_jsonl(ddir / "train.jsonl", rows)
    write_jsonl(ddir / "eval.jsonl",
                [{"prompt": r["prompt"], "reference": r["completion"], "subject": r.get("subject")}
                 for r in eval_rows])

    n_chat = sum(1 for r in rows if r["type"] == "chat")
    n_text = sum(1 for r in rows if r["type"] == "text")
    print(f"[build] train.jsonl: {n_chat} chat pairs + {n_text} voice tweets = {len(rows)} rows")
    print(f"[build] eval.jsonl: {len(eval_rows)} held-out pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
