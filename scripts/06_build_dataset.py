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

from common import base_argparser, data_dir, load_config, read_jsonl, write_jsonl


def main() -> int:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config)
    ddir = data_dir(cfg)
    dcfg = cfg["dataset"]
    seed = cfg["train"]["seed"]
    persona = dcfg["persona"].format(handle=cfg["handle"])

    labeled = read_jsonl(ddir / "labeled.jsonl") if (ddir / "labeled.jsonl").exists() else []
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
        for r in pool:
            rows.append({"type": "text", "text": r["text"]})
        print(f"[build] voice layer: {len(pool)} raw tweets")

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
