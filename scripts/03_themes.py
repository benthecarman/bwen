#!/usr/bin/env python3
"""Stage 03 — discover themes via embeddings + clustering.

- Embeds every cleaned tweet with a local Ollama embedding model (cached to .npy).
- Clusters (HDBSCAN or k-means) to auto-discover your real topics from the data.
- Optionally asks the LLM to name each cluster -> subjects.txt (review/edit after).
- Ranks each tweet by distance to its cluster centroid (near = most representative).

Output: data/candidates.jsonl  (each clean tweet + cluster, subject, centroid_dist)
        data/embeddings.npy     (cached embedding matrix)
        data/subjects.txt       (one discovered theme per line)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import (base_argparser, data_dir, load_config, maybe_skip, ollama_embed,
                    ollama_generate, ollama_preflight, read_jsonl, require_file, write_jsonl)


def get_embeddings(texts: list[str], model: str, cache: Path, force: bool) -> np.ndarray:
    meta = cache.with_suffix(".key.json")
    # Key on the model too: a different embed_model must invalidate the cache even
    # when the tweet text is unchanged, or we'd cluster old vectors under new config.
    key = hashlib.sha1(("\x00".join([model, *texts])).encode()).hexdigest()
    if cache.exists() and meta.exists() and not force:
        if json.loads(meta.read_text()) == key:
            print(f"[themes] reusing cached embeddings {cache}")
            return np.load(cache)
        print("[themes] cache stale (model or text changed); re-embedding")

    vecs: list[list[float]] = []
    batch = 64
    for i in tqdm(range(0, len(texts), batch), desc="embedding"):
        vecs.extend(ollama_embed(model, texts[i:i + batch]))
    arr = np.asarray(vecs, dtype=np.float32)
    np.save(cache, arr)
    meta.write_text(json.dumps(key))
    return arr


def normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def reduce(emb: np.ndarray, cfg: dict) -> np.ndarray:
    """PCA reduction — HDBSCAN/k-means cluster poorly on raw high-dim embeddings."""
    n = cfg["themes"]["reduce_dims"]
    if not n or n >= emb.shape[1]:
        return emb
    from sklearn.decomposition import PCA
    n = min(n, emb.shape[0] - 1)
    return PCA(n_components=n, random_state=42).fit_transform(emb)


def cluster(emb: np.ndarray, cfg: dict) -> np.ndarray:
    emb = reduce(emb, cfg)
    algo = cfg["themes"]["algorithm"]
    if algo == "kmeans":
        from sklearn.cluster import KMeans
        k = cfg["themes"]["kmeans_k"]
        return KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(emb)
    import hdbscan
    mcs = cfg["themes"]["hdbscan_min_cluster_size"]
    return hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean").fit_predict(emb)


def centroid_distances(emb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    dist = np.zeros(len(emb), dtype=np.float32)
    for lab in set(labels):
        if lab == -1:
            # HDBSCAN noise isn't a coherent cluster — a centroid over it is meaningless,
            # so leave noise points at distance 0 rather than ranking noise by noise.
            continue
        idx = np.where(labels == lab)[0]
        centroid = emb[idx].mean(axis=0)
        dist[idx] = np.linalg.norm(emb[idx] - centroid, axis=1)
    return dist


def name_clusters(rows, labels, dist, model, reps) -> dict[int, str]:
    names: dict[int, str] = {}
    for lab in sorted(set(labels)):
        if lab == -1:
            names[lab] = "misc/noise"
            continue
        idx = np.where(labels == lab)[0]
        idx = idx[np.argsort(dist[idx])][:reps]
        examples = "\n".join(f"- {rows[i]['text'][:200]}" for i in idx)
        prompt = (
            "Below are example tweets from one cluster. Reply with a short topic "
            "label of 2-4 words (no punctuation, no quotes) describing their common "
            f"theme.\n\n{examples}\n\nLabel:"
        )
        try:
            label = ollama_generate(model, prompt, options={"temperature": 0}).strip()
            label = label.splitlines()[0].strip().strip('"').strip()[:40]
        except Exception as e:  # noqa: BLE001
            label = f"cluster_{lab}"
            print(f"[themes] naming failed for cluster {lab}: {e}")
        names[lab] = label or f"cluster_{lab}"
        print(f"[themes]   cluster {lab} ({len(np.where(labels == lab)[0])}): {names[lab]}")
    return names


def main() -> int:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config)
    ddir = data_dir(cfg)
    require_file(ddir / "filtered" / "tweets.jsonl", "stage 02 (just filter)")
    rows = read_jsonl(ddir / "filtered" / "tweets.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    out = ddir / "candidates.jsonl"
    if maybe_skip(out, args.force):
        return 0
    ollama_preflight()

    texts = [r["text"] for r in rows]
    # A --limit run embeds only a subset; keep it out of embeddings.npy so it can't
    # clobber the full-archive cache the README suggests reusing for RAG.
    cache = ddir / (f"embeddings.limit{args.limit}.npy" if args.limit else "embeddings.npy")
    emb = get_embeddings(texts, cfg["themes"]["embed_model"], cache, args.force)
    emb = normalize(emb)

    print("[themes] clustering...")
    labels = cluster(emb, cfg)
    n_clusters = len({l for l in labels if l != -1})
    n_noise = int((labels == -1).sum())
    print(f"[themes] {n_clusters} clusters, {n_noise} noise points")

    dist = centroid_distances(emb, labels)

    names: dict[int, str] = {}
    if cfg["themes"]["name_clusters"]:
        names = name_clusters(rows, labels, dist, cfg["score"]["llm_model"],
                              cfg["themes"]["reps_per_cluster"])
        subjects = [names[l] for l in sorted(set(labels)) if l != -1]
        subjects_path = ddir / "subjects.txt"
        subjects_path.write_text("\n".join(subjects) + "\n")
        print(f"[themes] wrote {len(subjects)} subjects -> {subjects_path} (review/edit it)")

    for r, lab, d in zip(rows, labels, dist):
        r["cluster"] = int(lab)
        r["subject"] = names.get(int(lab), f"cluster_{int(lab)}")
        r["centroid_dist"] = float(d)

    write_jsonl(out, rows)
    print(f"[themes] wrote {len(rows)} candidates -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
