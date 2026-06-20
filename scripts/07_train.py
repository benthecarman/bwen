#!/usr/bin/env python3
"""Stage 07 — LoRA finetune with Unsloth.

Reads data/train.jsonl (chat pairs + raw voice tweets), applies the Qwen3 chat
template with thinking disabled to the chat rows, and trains a LoRA adapter.

Requires the heavy extras:  uv sync --extra train
On the RTX 5060 Ti (Blackwell / sm_120) torch must be a cu128 build — see README.
"""
from __future__ import annotations

from common import REPO_ROOT, base_argparser, data_dir, load_config, read_jsonl


def build_texts(rows: list[dict], tokenizer) -> list[str]:
    texts: list[str] = []
    eos = tokenizer.eos_token or ""
    for r in rows:
        if r["type"] == "chat":
            texts.append(tokenizer.apply_chat_template(
                r["messages"], tokenize=False, add_generation_prompt=False,
                enable_thinking=False))
        else:  # raw voice tweet
            texts.append(r["text"] + eos)
    return texts


def main() -> int:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config)
    tc = cfg["train"]

    import torch
    print(f"[train] torch {torch.__version__} · cuda avail={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[train] gpu: {torch.cuda.get_device_name(0)} "
              f"(capability {torch.cuda.get_device_capability(0)})")
    else:
        print("[train] WARNING: CUDA not available — training will be unusably slow.")

    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=tc["base_model"],
        max_seq_length=tc["max_seq_len"],
        load_in_4bit=False,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=tc["lora_rank"],
        lora_alpha=tc["lora_alpha"],
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=tc["seed"],
    )

    rows = read_jsonl(data_dir(cfg) / "train.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    ds = Dataset.from_dict({"text": build_texts(rows, tokenizer)})
    print(f"[train] {len(ds)} examples")

    out_dir = REPO_ROOT / tc["output_dir"]
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=tc["max_seq_len"],
            per_device_train_batch_size=tc["batch_size"],
            gradient_accumulation_steps=tc["grad_accum"],
            num_train_epochs=tc["epochs"],
            learning_rate=float(tc["learning_rate"]),
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            logging_steps=5,
            optim="adamw_8bit",
            seed=tc["seed"],
            output_dir=str(out_dir / "checkpoints"),
            report_to="none",
        ),
    )
    trainer.train()

    model.save_pretrained(str(out_dir / "lora"))
    tokenizer.save_pretrained(str(out_dir / "lora"))
    print(f"[train] saved LoRA -> {out_dir / 'lora'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
