#!/usr/bin/env python3
"""Stage 10 — "ask my tweets": RAG over your own tweets, answered in your voice.

Retrieval grounds the finetuned model in what you *actually* tweeted, instead of the
model's frozen impression of you. For each question:
  1. embed it with the same model used for the corpus (Ollama nomic-embed-text),
  2. cosine-search your own tweets (data/embeddings.npy, aligned to candidates.jsonl),
  3. feed the top matches to the tuned model as grounding, answered in your persona.

Usage:
  just ask "what do you think about covenants"   # one-shot
  just ask                                        # interactive REPL
"""
from __future__ import annotations

import numpy as np

from common import (base_argparser, data_dir, load_config, ollama_embed, ollama_generate,
                    ollama_preflight, read_jsonl, require_file, system_prompt, think_off)


def normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-9, None)


def answer(query: str, emb: np.ndarray, texts: list[str], cfg: dict) -> tuple[str, list[str]]:
    rcfg = cfg["rag"]
    q = normalize(np.asarray(ollama_embed(cfg["themes"]["embed_model"], [query])[0], dtype=np.float32))
    sims = emb @ q
    idx = np.argsort(-sims)[: rcfg["top_k"]]
    hits = [texts[i] for i in idx]
    context = "\n".join(f"- {t}" for t in hits)
    prompt = (
        "You've tweeted these before (most relevant first):\n"
        f"{context}\n\n"
        f"Grounded in what you actually tweeted above, answer this in your own voice: {query}"
    )
    persona = system_prompt(cfg)
    out = ollama_generate(cfg["export"]["ollama_model_name"], prompt,
                          system=think_off(persona), options={"temperature": 0.7})
    return out.strip(), hits


def main() -> int:
    p = base_argparser(__doc__)
    p.add_argument("query", nargs="*", help="question to ask (omit for interactive mode)")
    args = p.parse_args()
    cfg = load_config(args.config)
    ddir = data_dir(cfg)

    require_file(ddir / "candidates.jsonl", "stage 03 (just themes)")
    require_file(ddir / "embeddings.npy", "stage 03 (just themes)")
    rows = read_jsonl(ddir / "candidates.jsonl")
    emb = np.load(ddir / "embeddings.npy")
    if emb.shape[0] != len(rows):
        print("[ask] embeddings and candidates out of sync — rerun `just themes --force`")
        return 1
    # Retrieve only your own tweets (retweets/likes are context, not your words).
    keep = [i for i, r in enumerate(rows) if r.get("is_own", True)]
    emb = normalize(emb[keep].astype(np.float32))
    texts = [rows[i]["text"] for i in keep]
    print(f"[ask] {len(texts)} of your tweets indexed · model {cfg['export']['ollama_model_name']}")
    ollama_preflight()

    def run(q: str) -> None:
        out, hits = answer(q, emb, texts, cfg)
        print(f"\n{out}\n")
        if cfg["rag"]["show_sources"]:
            print("  sources:")
            for t in hits:
                print(f"    · {t[:120]}")
            print()

    if args.query:
        run(" ".join(args.query))
        return 0
    print("Ask your tweets (blank line or Ctrl-D to quit).")
    while True:
        try:
            q = input("\nask> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        run(q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
