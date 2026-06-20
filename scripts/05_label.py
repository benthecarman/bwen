#!/usr/bin/env python3
"""Stage 05 — hand-write a prompt for each shortlisted tweet.

This is the heart of the "no synthetic data" approach: the COMPLETION is your real
tweet; YOU write the prompt (the question/instruction it answers). You also supply the
missing context replies lack — since you remember what you were responding to.

Terminal tool, resumable. For each tweet:
  - type a prompt + Enter to save it
  - Enter on an empty line to skip (won't be shown again)
  - `b` to redo the previous tweet, `q` to save and quit

Output: data/labeled.jsonl  ({id, prompt, completion, subject})
        data/label_skip.json (ids you skipped, so they don't reappear)
"""
from __future__ import annotations

import json
from pathlib import Path

from common import apply_subject_edits, base_argparser, data_dir, load_config, read_jsonl


def main() -> int:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config)
    ddir = data_dir(cfg)
    shortlist = read_jsonl(ddir / "shortlist.jsonl")
    if args.limit:
        shortlist = shortlist[: args.limit]
    # Honor edits made to subjects.txt after the shortlist was written, so the labels
    # shown here (and recorded in labeled.jsonl) reflect any renames/merges.
    apply_subject_edits(shortlist, ddir)

    labeled_path = ddir / "labeled.jsonl"
    skip_path = ddir / "label_skip.json"
    labeled = read_jsonl(labeled_path) if labeled_path.exists() else []
    done_ids = {str(r["id"]) for r in labeled}
    skipped = set(json.loads(skip_path.read_text())) if skip_path.exists() else set()

    todo = [r for r in shortlist if str(r["id"]) not in done_ids and str(r["id"]) not in skipped]
    target = cfg["dataset"]["label_target"]

    print(f"\n{'='*70}\n  {len(labeled)} labeled so far · {len(todo)} remaining in shortlist"
          f"\n  Commands: <text>=save · <empty>=skip · b=back · q=quit\n{'='*70}\n")

    fav = lambda r: r.get("favorite_count", 0)
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
            if labeled and i > 0:
                last = labeled.pop()
                done_ids.discard(str(last["id"]))
                i -= 1
                # rewrite file without the popped row
                with open(labeled_path, "w") as f:
                    for row in labeled:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print("  (reverted previous)\n")
            continue
        if ans == "":
            skipped.add(str(r["id"]))
            skip_path.write_text(json.dumps(sorted(skipped)))
            i += 1
            continue

        row = {"id": str(r["id"]), "prompt": ans, "completion": r["text"],
               "subject": r.get("subject")}
        labeled.append(row)
        done_ids.add(str(r["id"]))
        with open(labeled_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        i += 1
        print(f"  saved ({len(labeled)}/{target})\n")

    print(f"\n[label] {len(labeled)} labeled total -> {labeled_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
