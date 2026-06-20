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


def require_file(path: Path, prev_stage: str) -> Path:
    """Exit with a 'run <stage> first' hint instead of a raw FileNotFoundError."""
    if not path.exists():
        raise SystemExit(f"[error] {path} not found — run {prev_stage} first.")
    return path


def ollama_preflight() -> None:
    """Fail fast with a friendly message if the Ollama server isn't reachable.

    Otherwise the first embed/generate call dies deep in a stage with a raw
    ConnectionError, which doesn't hint that the fix is to start `ollama serve`.
    """
    try:
        requests.get(OLLAMA_HOST, timeout=5)
    except requests.RequestException:
        raise SystemExit(
            f"[error] can't reach Ollama at {OLLAMA_HOST}. Is it running? "
            f"Start it with `ollama serve` (or set OLLAMA_HOST).")


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


# ---- Persona / Qwen3 thinking ----

NO_THINK = "/no_think"  # Qwen3 directive: answer directly instead of emitting <think> blocks


def think_off(persona: str) -> str:
    """Persona system string with Qwen3 thinking disabled.

    Defined once so the `/no_think` suffix has a single source of truth across stages
    (the export Modelfile and the eval requests) and can't accidentally double up.
    """
    return persona if persona.rstrip().endswith(NO_THINK) else f"{persona} {NO_THINK}"


# ---- Subjects (hand-editable theme labels) ----

def load_subject_edits(ddir: Path) -> dict[int, str]:
    """Map cluster id -> (possibly hand-edited) subject from data/subjects.txt.

    Stage 03 writes one line per non-noise cluster in sorted cluster-id order, derived
    from the full candidate set; we rebuild that order from candidates.jsonl so the
    mapping is stable even when applied to a subset (e.g. the shortlist). Returns {} if
    subjects.txt is absent or its line count no longer matches the clusters (a user
    added/removed lines) — callers then keep the labels baked into candidates.jsonl.
    """
    subjects_path = ddir / "subjects.txt"
    cand_path = ddir / "candidates.jsonl"
    if not subjects_path.exists() or not cand_path.exists():
        return {}
    subjects = [l.strip() for l in subjects_path.read_text().splitlines() if l.strip()]
    clusters = sorted({int(r["cluster"]) for r in read_jsonl(cand_path)
                       if int(r.get("cluster", -1)) != -1})
    if len(subjects) != len(clusters):
        print(f"[subjects] subjects.txt has {len(subjects)} lines but {len(clusters)} "
              f"non-noise clusters — ignoring edits (rename/merge lines, don't add/remove)")
        return {}
    return dict(zip(clusters, subjects))


def apply_subject_edits(rows: list[dict], ddir: Path) -> int:
    """Overwrite each row's `subject` with the edited label for its cluster.

    Returns the number of rows whose subject changed. Rows in a noise cluster (-1) or a
    cluster absent from the map are left untouched.
    """
    mapping = load_subject_edits(ddir)
    if not mapping:
        return 0
    n = 0
    for r in rows:
        c = int(r.get("cluster", -1))
        if c in mapping and r.get("subject") != mapping[c]:
            r["subject"] = mapping[c]
            n += 1
    return n


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
