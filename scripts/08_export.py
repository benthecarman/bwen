#!/usr/bin/env python3
"""Stage 08 — merge LoRA, export to GGUF, and register the model with Ollama.

Requires the train extras (uses Unsloth's GGUF export, which shells out to llama.cpp).
Writes a Modelfile from config (persona + thinking disabled) and runs `ollama create`.

Output: data/model/gguf/*.gguf  and an Ollama model named cfg.export.ollama_model_name
"""
from __future__ import annotations

import re
import subprocess

from pathlib import Path

from common import REPO_ROOT, base_argparser, load_config, system_prompt, think_off


def find_gguf(out_dir: Path, quant: str) -> Path | None:
    """Find a produced .gguf under out_dir. Unsloth writes to a sibling '<dir>_gguf'
    folder and names files with uppercase quant (Q4_K_M), so search recursively and
    match the quant case-insensitively."""
    ggufs = list(out_dir.rglob("*.gguf"))
    matched = [g for g in ggufs if quant.lower() in g.name.lower()]
    return sorted(matched or ggufs)[0] if ggufs else None


def main() -> int:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config)
    tc, ec = cfg["train"], cfg["export"]
    out_dir = REPO_ROOT / tc["output_dir"]
    lora_dir = out_dir / "lora"
    quant = ec["gguf_quant"]

    gguf_path = find_gguf(out_dir, quant)
    if gguf_path and not args.force:
        print(f"[export] reusing existing GGUF {gguf_path} (--force to rebuild)")
    else:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(lora_dir),
            max_seq_length=tc["max_seq_len"],
            load_in_4bit=tc["load_in_4bit"],
            dtype=None,
        )
        gguf_dir = out_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        print(f"[export] writing GGUF ({quant}) -> {gguf_dir} (~10 min)")
        model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=quant)
        gguf_path = find_gguf(out_dir, quant)
        if not gguf_path:
            print("[export] ERROR: no .gguf produced")
            return 1

    persona = system_prompt(cfg)
    modelfile = out_dir / "Modelfile"
    # Build on Unsloth's generated Modelfile (next to the gguf): it carries the correct
    # Qwen3 chat TEMPLATE and stop tokens. Without TEMPLATE, Ollama feeds the raw prompt
    # and the model just continues it instead of answering. Fix FROM to the absolute gguf
    # path and inject our persona SYSTEM, keeping Unsloth's template + params.
    src = gguf_path.parent / "Modelfile"
    if not src.exists():
        print(f"[export] WARNING: {src} missing — Modelfile will lack the chat template")
    kept = [ln for ln in (src.read_text().splitlines() if src.exists() else [])
            if not ln.strip().startswith(("FROM ", "SYSTEM"))]
    lines = [f"FROM {gguf_path}", *kept, f'SYSTEM """{think_off(persona)}"""']
    text = "\n".join(lines) + "\n"
    # Prefill an empty think block at the assistant generation prompt so Qwen3 answers
    # directly instead of emitting <think> reasoning (matches enable_thinking=False used
    # in training; the template's /no_think text alone is ignored by Unsloth's template).
    text, n = re.subn(r"(<\|im_start\|>assistant\n)(\{\{ end \}\})",
                      r"\1<think>\n\n</think>\n\n\2", text)
    print(f"[export] disabled thinking at {n} generation prompt(s)")
    modelfile.write_text(text)
    name = ec["ollama_model_name"]
    print(f"[export] ollama create {name} -f {modelfile}")
    subprocess.run(["ollama", "create", name, "-f", str(modelfile)], check=True)
    print(f"[export] done. Try:  ollama run {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
