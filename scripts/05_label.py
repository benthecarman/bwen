#!/usr/bin/env python3
"""Stage 05 — hand-write a prompt for each shortlisted tweet.

This is the heart of the "no synthetic data" approach: the COMPLETION is your real
tweet; YOU write the prompt (the question/instruction it answers). You also supply the
missing context replies lack — since you remember what you were responding to.

Terminal tool, resumable. For each tweet:
  - type a prompt + Enter to save it
  - Enter on an empty line to skip (won't be shown again)
  - `b` to redo the previous tweet, `q` to save and quit

Draws from the full balanced candidate pool (not just the fixed shortlist), so a skip is
topped up by the next-best candidate — skipping never shrinks the set you can label.

Output: data/labeled.jsonl  ({id, prompt, completion, subject})
        data/label_skip.json (ids you skipped, so they don't reappear)
"""
from __future__ import annotations

import json
from pathlib import Path

from common import (apply_subject_edits, balanced_order, base_argparser, data_dir,
                    load_config, read_jsonl, require_file)


def main() -> int:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config)
    ddir = data_dir(cfg)
    require_file(ddir / "candidates_scored.jsonl", "stage 04 (just score)")
    rows = read_jsonl(ddir / "candidates_scored.jsonl")
    # Own tweets only (retweets/likes are context for themes, never labeled), ordered by
    # the same balanced round-robin as the shortlist — so we top up in priority order.
    own = [r for r in rows if r.get("is_own", True)]
    # Honor edits made to subjects.txt after scoring, so labels shown here reflect renames.
    apply_subject_edits(own, ddir)
    pool = balanced_order(own)
    if args.limit:
        pool = pool[: args.limit]

    labeled_path = ddir / "labeled.jsonl"
    skip_path = ddir / "label_skip.json"
    labeled = read_jsonl(labeled_path) if labeled_path.exists() else []
    done_ids = {str(r["id"]) for r in labeled}
    skipped = set(json.loads(skip_path.read_text())) if skip_path.exists() else set()

    todo = [r for r in pool if str(r["id"]) not in done_ids and str(r["id"]) not in skipped]
    target = cfg["dataset"]["label_target"]

    print(f"\n{'='*70}\n  {len(labeled)} labeled · target {target} · {len(todo)} candidates left"
          f"\n  Commands: <text>=save · <empty>=skip · b=back · q=quit\n{'='*70}\n")

    fav = lambda r: r.get("favorite_count", 0)
    # Undo stack of (action, id). Both saving and skipping advance `i`, so `b` must
    # know which the previous action was: undoing a save pops labeled, undoing a skip
    # un-skips. Keying off labeled alone would wrongly drop a good label after a skip.
    history: list[tuple[str, str]] = []
    i = 0
    while i < len(todo):
        r = todo[i]
        print(f"[{len(labeled)} done] cluster={r.get('subject','?')} "
              f"♥{fav(r)} op={r.get('opinion_score','-')} vo={r.get('voice_score','-')}")
        print(f"  TWEET: {r['text']}\n")
        try:
            ans = input("  Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[label] interrupted — progress saved.")
            break

        if ans == "q":
            break
        if ans == "b":
            if not history:
                print("  (nothing to undo)\n")
                continue
            action, last_id = history.pop()
            if action == "save":
                labeled.pop()
                done_ids.discard(last_id)
                # rewrite file without the popped row
                with open(labeled_path, "w") as f:
                    for row in labeled:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:  # "skip"
                skipped.discard(last_id)
                skip_path.write_text(json.dumps(sorted(skipped)))
            i -= 1
            print(f"  (reverted previous {action})\n")
            continue
        if ans == "":
            skipped.add(str(r["id"]))
            skip_path.write_text(json.dumps(sorted(skipped)))
            history.append(("skip", str(r["id"])))
            i += 1
            continue

        row = {"id": str(r["id"]), "prompt": ans, "completion": r["text"],
               "subject": r.get("subject")}
        labeled.append(row)
        done_ids.add(str(r["id"]))
        with open(labeled_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        history.append(("save", str(r["id"])))
        i += 1
        hit = "  🎯 target reached — keep going or q to stop" if len(labeled) >= target else ""
        print(f"  saved ({len(labeled)}/{target}){hit}\n")

    print(f"\n[label] {len(labeled)} labeled total -> {labeled_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
