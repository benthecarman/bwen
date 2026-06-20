"""Shared helpers: config loading, paths, and a thin Ollama client.

Nothing in here is person-specific — all such values come from config.yaml.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load config with config.example.yaml as the single source of default values.

    Defaults come from config.example.yaml; the user's config.yaml (or --config path)
    is deep-merged on top. So scripts never carry their own fallback defaults, and a
    user config may omit any key it doesn't want to override.
    """
    with open(REPO_ROOT / "config.example.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg_path = Path(path) if path else REPO_ROOT / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    else:
        print(f"[warn] {cfg_path.name} not found; using config.example.yaml defaults. "
              f"Copy it to config.yaml and edit.", file=sys.stderr)

    # Lets the dry-run (and tests) redirect all artifacts to a throwaway dir so they
    # never poison the real data/ outputs that later stages would skip-reuse.
    if os.environ.get("BWEN_DATA_DIR"):
        cfg["paths"]["data_dir"] = os.environ["BWEN_DATA_DIR"]

    cfg.setdefault("_root", str(REPO_ROOT))
    return cfg


def data_dir(cfg: dict) -> Path:
    d = REPO_ROOT / cfg["paths"]["data_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def archive_dir(cfg: dict) -> Path:
    return REPO_ROOT / cfg["paths"]["archive_dir"]


# ---- JSONL helpers ----

def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---- Ollama ----

def ollama_embed(model: str, texts: list[str]) -> list[list[float]]:
    """Batch-embed via the Ollama /api/embed endpoint."""
    resp = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": model, "input": texts},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def ollama_generate(model: str, prompt: str, *, fmt: str | dict | None = None,
                    system: str | None = None, options: dict | None = None) -> str:
    payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
    if fmt is not None:
        payload["format"] = fmt
    if system is not None:
        payload["system"] = system
    if options:
        payload["options"] = options
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()["response"]


# ---- CLI ----

def base_argparser(description: str) -> argparse.ArgumentParser:
    """Standard flags shared by every stage: --config, --limit, --force."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--limit", type=int, default=None,
                   help="process at most N rows (dry-run / fast iteration)")
    p.add_argument("--force", action="store_true",
                   help="recompute even if the output already exists")
    return p


def maybe_skip(out_path: Path, force: bool) -> bool:
    """Return True if we should skip because output exists and not --force."""
    if out_path.exists() and not force:
        print(f"[skip] {out_path} exists (use --force to recompute)")
        return True
    return False
