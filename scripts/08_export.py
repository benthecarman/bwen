#!/usr/bin/env python3
"""Stage 08 — merge LoRA, export to GGUF, and register the model with Ollama.

Requires the train extras (uses Unsloth's GGUF export, which shells out to llama.cpp).
Writes a Modelfile from config (persona + thinking disabled) and runs `ollama create`.

Output: data/model/gguf/*.gguf  and an Ollama model named cfg.export.ollama_model_name
"""
from __future__ import annotations

import subprocess

from common import REPO_ROOT, base_argparser, load_config


def main() -> int:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config)
    tc, ec = cfg["train"], cfg["export"]
    out_dir = REPO_ROOT / tc["output_dir"]
    lora_dir = out_dir / "lora"
    gguf_dir = out_dir / "gguf"
    quant = ec["gguf_quant"]

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(lora_dir),
        max_seq_length=tc["max_seq_len"],
        load_in_4bit=False,
        dtype=None,
    )
    gguf_dir.mkdir(parents=True, exist_ok=True)
    print(f"[export] writing GGUF ({quant}) -> {gguf_dir}")
    model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=quant)

    # Locate the produced gguf.
    ggufs = sorted(gguf_dir.glob(f"*{quant}*.gguf")) or sorted(gguf_dir.glob("*.gguf"))
    if not ggufs:
        print("[export] ERROR: no .gguf produced")
        return 1
    gguf_path = ggufs[0]

    persona = cfg["dataset"]["persona"].format(handle=cfg["handle"])
    modelfile = out_dir / "Modelfile"
    modelfile.write_text(
        f"FROM {gguf_path}\n"
        f'SYSTEM """{persona} /no_think"""\n'
        "PARAMETER temperature 0.8\n"
        "PARAMETER top_p 0.9\n"
    )
    name = ec["ollama_model_name"]
    print(f"[export] ollama create {name} -f {modelfile}")
    subprocess.run(["ollama", "create", name, "-f", str(modelfile)], check=True)
    print(f"[export] done. Try:  ollama run {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
