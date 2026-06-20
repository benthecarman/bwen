#!/usr/bin/env python3
"""Stage 07 — LoRA finetune with Unsloth.

Reads data/train.jsonl (chat pairs + raw voice tweets), applies the Qwen3 chat
template with thinking disabled to the chat rows, and trains a LoRA adapter.

Chat rows are masked so the loss is computed only on the assistant turn (your real
tweet) — not the persona or the hand-written prompt, which aren't your voice. Raw
voice tweets carry no prompt and are trained in full.

Requires the heavy extras:  uv sync --extra train
On the RTX 5060 Ti (Blackwell / sm_120) torch must be a cu128 build — see README.
"""
from __future__ import annotations

from common import REPO_ROOT, base_argparser, data_dir, load_config, read_jsonl


def build_examples(rows: list[dict], tokenizer, max_len: int) -> list[dict]:
    """Tokenize each row into {input_ids, labels}.

    chat rows: mask everything before the assistant turn so loss falls only on YOUR
    tweet, not the persona boilerplate or the scaffolding prompt you hand-wrote.
    text rows: train on the whole raw tweet — it's all your voice, nothing to mask.
    """
    eos = tokenizer.eos_token or ""
    examples: list[dict] = []
    for r in rows:
        if r["type"] == "chat":
            # Full templated conversation, and the prefix up to where the assistant
            # content starts (same messages minus the assistant turn, + gen prompt).
            full = tokenizer.apply_chat_template(
                r["messages"], tokenize=False, add_generation_prompt=False,
                enable_thinking=False)
            prefix = tokenizer.apply_chat_template(
                r["messages"][:-1], tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            ids = tokenizer(full, add_special_tokens=False,
                            truncation=True, max_length=max_len)["input_ids"]
            n_prefix = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
            labels = list(ids)
            for i in range(min(n_prefix, len(labels))):
                labels[i] = -100   # mask system + user prompt
        else:  # raw voice tweet — train fully
            ids = tokenizer(r["text"] + eos, add_special_tokens=False,
                            truncation=True, max_length=max_len)["input_ids"]
            labels = list(ids)
        examples.append({"input_ids": ids, "labels": labels})
    return examples


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
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
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
    ds = Dataset.from_list(build_examples(rows, tokenizer, tc["max_seq_len"]))
    n_chat = sum(1 for r in rows if r["type"] == "chat")
    print(f"[train] {len(ds)} examples ({n_chat} chat w/ masked prompts + "
          f"{len(ds) - n_chat} voice)")

    # The dataset is already tokenized with masked labels, so the collator only pads.
    # DataCollatorForSeq2Seq pads labels with -100 (ignored by the loss) and builds the
    # attention mask. Because the data is fully prepared we use the plain HF Trainer
    # rather than TRL's SFTTrainer — SFTTrainer exists to tokenize/format raw datasets,
    # which we don't need, and its dataset-prep gating varies across TRL versions.
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

    out_dir = REPO_ROOT / tc["output_dir"]
    trainer = Trainer(
        model=model,
        train_dataset=ds,
        data_collator=collator,
        args=TrainingArguments(
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
