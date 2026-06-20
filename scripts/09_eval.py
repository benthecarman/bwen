#!/usr/bin/env python3
"""Stage 09 — eval scorecard: base vs. finetuned, side by side.

Runs the held-out eval prompts (and any extra prompts in eval_prompts.txt) through
both the base Ollama model and your finetuned model, writing a Markdown scorecard so
you have a consistent signal when you tweak data / epochs / LoRA rank.

Output: runs/<timestamp>.md
"""
from __future__ import annotations

from datetime import datetime

from common import (REPO_ROOT, base_argparser, data_dir, load_config, ollama_generate,
                    read_jsonl, think_off)


def gen(model: str, prompt: str, system: str) -> str:
    try:
        # Greedy decoding: the scorecard's job is a consistent signal across data/epoch/
        # LoRA tweaks, so it must be deterministic — sampling would conflate noise with
        # real changes. (Day-to-day "what does it sound like" sampling lives in `ollama run`.)
        # think_off disables Qwen3 reasoning for the base model too (its Modelfile has no
        # baked SYSTEM); it's idempotent, so the tuned model never gets a doubled /no_think.
        return ollama_generate(model, prompt, system=think_off(system),
                               options={"temperature": 0}).strip()
    except Exception as e:  # noqa: BLE001
        return f"[error: {e}]"


def main() -> int:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config)
    ddir = data_dir(cfg)
    persona = cfg["dataset"]["persona"].format(handle=cfg["handle"])
    base = cfg["eval"]["base_model_tag"]
    tuned = cfg["export"]["ollama_model_name"]

    prompts: list[dict] = []
    eval_path = ddir / "eval.jsonl"
    if eval_path.exists():
        prompts.extend(read_jsonl(eval_path))
    extra = REPO_ROOT / "eval_prompts.txt"
    if extra.exists():
        for line in extra.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append({"prompt": line, "reference": None})
    if args.limit:
        prompts = prompts[: args.limit]

    runs = REPO_ROOT / cfg["paths"]["runs_dir"]
    runs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = runs / f"{stamp}.md"

    lines = [f"# Eval {stamp}", "",
             f"- base: `{base}`  ·  tuned: `{tuned}`",
             f"- persona: {persona}", "", "---", ""]
    for i, p in enumerate(prompts, 1):
        lines.append(f"### {i}. {p['prompt']}")
        if p.get("reference"):
            lines.append(f"> **your real tweet:** {p['reference']}")
        lines.append(f"\n**base:** {gen(base, p['prompt'], persona)}\n")
        lines.append(f"**tuned:** {gen(tuned, p['prompt'], persona)}\n")
        lines.append("---\n")
        print(f"[eval] {i}/{len(prompts)} done")

    out.write_text("\n".join(lines))
    print(f"[eval] wrote scorecard -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
