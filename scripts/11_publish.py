#!/usr/bin/env python3
"""Stage 11 — publish the dataset and finetuned model to the Hugging Face Hub.

Reproduces the public layout straight from the pipeline outputs — nothing is
hand-massaged, so re-running after a retrain keeps the Hub in sync:

  dataset repo  <- pairs.jsonl  (chat rows from train.jsonl; subject recovered from
                                 the hand-labels, since train.jsonl folds it into the prompt)
                   voice.jsonl  (raw-voice text rows from train.jsonl)
                   eval.jsonl   (held-out instruction pairs, copied as-is)
  model repo    <- <repo name>.<QUANT>.gguf  (the exported GGUF, renamed to match the repo)
                   Modelfile                 (FROM rewritten to the uploaded GGUF name)
                   lora/                      (the LoRA adapter from stage 07)

Repo IDs come from config (publish.hf_dataset_repo / publish.hf_model_repo) — set them in
config.yaml; a blank repo skips that side. Authenticate first with `hf auth login` (or set
HF_TOKEN); repos are created on first push. Use --dry-run to build the upload files without
pushing, and --dataset-only / --model-only to publish just one side.
"""
from __future__ import annotations

from pathlib import Path

from common import (REPO_ROOT, base_argparser, data_dir, load_config,
                    read_jsonl, require_file, state_dir, write_jsonl)


def build_dataset_files(ddir: Path, sdir: Path) -> list[tuple[Path, str]]:
    """Project the pipeline outputs into the published dataset schema.

    Returns (local_path, path_in_repo) pairs. pairs.jsonl/voice.jsonl are derived from
    train.jsonl's type-tagged rows; subject is looked up from the hand-labels because the
    builder (stage 06) folds it into the chat prompt rather than keeping a column.
    """
    train = read_jsonl(require_file(ddir / "train.jsonl", "06 build (just data)"))

    subject = {}
    labeled = sdir / "labeled.jsonl"
    if labeled.exists():
        for r in read_jsonl(labeled):
            subject[(r["prompt"], r["completion"])] = r.get("subject")

    pairs, voice = [], []
    for r in train:
        if r["type"] == "chat":
            m = {x["role"]: x["content"] for x in r["messages"]}
            pairs.append({"prompt": m["user"], "completion": m["assistant"],
                          "subject": subject.get((m["user"], m["assistant"]))})
        elif r["type"] == "text":
            voice.append({"text": r["text"]})

    write_jsonl(ddir / "pairs.jsonl", pairs)
    write_jsonl(ddir / "voice.jsonl", voice)
    print(f"[publish] dataset: {len(pairs)} pairs + {len(voice)} voice rows")

    files = [(ddir / "pairs.jsonl", "pairs.jsonl"), (ddir / "voice.jsonl", "voice.jsonl")]
    eval_path = ddir / "eval.jsonl"
    if eval_path.exists():
        files.append((eval_path, "eval.jsonl"))   # already {prompt, reference, subject}
    return files


def build_model_files(cfg: dict, out_dir: Path, gguf_name: str) -> tuple[list[tuple[Path, str]], Path]:
    """Locate the exported GGUF, rewrite the Modelfile's FROM to the uploaded name.

    Returns (files, lora_dir): files are (local_path, path_in_repo) pairs for single files;
    the LoRA adapter is a directory, uploaded separately by the caller.
    """
    quant = cfg["export"]["gguf_quant"]
    ggufs = list(out_dir.rglob("*.gguf"))
    matched = [g for g in ggufs if quant.lower() in g.name.lower()] or ggufs
    if not matched:
        raise SystemExit(f"[publish] no .gguf under {out_dir} — run `just export` first")
    gguf = sorted(matched)[0]

    src = require_file(out_dir / "Modelfile", "08 export (just export)")
    lines = src.read_text().splitlines(keepends=True)
    rewritten = [f"FROM ./{gguf_name}\n" if ln.startswith("FROM ") else ln for ln in lines]
    modelfile = out_dir / "Modelfile.hf"
    modelfile.write_text("".join(rewritten))
    print(f"[publish] model: {gguf.name} -> {gguf_name}, Modelfile FROM ./{gguf_name}")

    return [(gguf, gguf_name), (modelfile, "Modelfile")], out_dir / "lora"


def main() -> int:
    p = base_argparser(__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="build the upload files but do not push to the Hub")
    p.add_argument("--dataset-only", action="store_true", help="publish only the dataset repo")
    p.add_argument("--model-only", action="store_true", help="publish only the model repo")
    args = p.parse_args()
    cfg = load_config(args.config)

    pc = cfg["publish"]
    ddir, sdir = data_dir(cfg), state_dir(cfg)
    out_dir = REPO_ROOT / cfg["train"]["output_dir"]

    do_dataset = not args.model_only
    do_model = not args.dataset_only

    api = None
    if not args.dry_run:
        from huggingface_hub import HfApi
        api = HfApi()

    if do_dataset:
        repo = pc["hf_dataset_repo"]
        if not repo:
            print("[publish] publish.hf_dataset_repo is blank — skipping dataset")
        else:
            files = build_dataset_files(ddir, sdir)
            if args.dry_run:
                for _, dest in files:
                    print(f"[publish] would upload datasets/{repo}/{dest}")
            else:
                api.create_repo(repo, repo_type="dataset", exist_ok=True)
                for path, dest in files:
                    api.upload_file(path_or_fileobj=str(path), path_in_repo=dest,
                                    repo_id=repo, repo_type="dataset")
                    print(f"[publish] uploaded datasets/{repo}/{dest}")

    if do_model:
        repo = pc["hf_model_repo"]
        if not repo:
            print("[publish] publish.hf_model_repo is blank — skipping model")
        else:
            gguf_name = f"{repo.split('/')[-1]}.{cfg['export']['gguf_quant'].upper()}.gguf"
            files, lora_dir = build_model_files(cfg, out_dir, gguf_name)
            if args.dry_run:
                for _, dest in files:
                    print(f"[publish] would upload {repo}/{dest}")
                if lora_dir.exists():
                    print(f"[publish] would upload {repo}/lora/ (from {lora_dir})")
            else:
                api.create_repo(repo, repo_type="model", exist_ok=True)
                for path, dest in files:
                    api.upload_file(path_or_fileobj=str(path), path_in_repo=dest, repo_id=repo)
                    print(f"[publish] uploaded {repo}/{dest}")
                if lora_dir.exists():
                    api.upload_folder(folder_path=str(lora_dir), path_in_repo="lora", repo_id=repo)
                    print(f"[publish] uploaded {repo}/lora/")

    print("[publish] done." + (" (dry run — nothing pushed)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
