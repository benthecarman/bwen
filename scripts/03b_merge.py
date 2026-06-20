#!/usr/bin/env python3
"""Stage 03b — compress fine clusters into higher-level themes.

UMAP + HDBSCAN produces many fine, overlapping clusters (e.g. "ethereum scam",
"shitcoin debate", "scam alerts"). This pass merges clusters whose centroids are
close (agglomerative, cosine) down to a target number of themes, then LLM-names each
merged group from its member sub-labels + representative tweets.

Fast to re-run while tuning `themes.merge.target` — it reuses the embeddings and
clusters from stage 03 and never re-embeds or re-clusters.

Output: rewrites data/candidates.jsonl with `theme` + `theme_id` (keeps the fine
        `cluster` / `subject`), and writes data/themes.yaml (the compressed list,
        editable for review).
"""
from __future__ import annotations

import collections

import numpy as np
import yaml

from common import (base_argparser, data_dir, load_config, ollama_generate,
                    ollama_preflight, read_jsonl, require_file, write_jsonl)


def normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def cluster_centroids(emb: np.ndarray, labels: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Return (ordered cluster ids excluding noise, matrix of their unit centroids)."""
    ids = sorted({int(l) for l in labels if l != -1})
    cents = np.vstack([emb[labels == c].mean(axis=0) for c in ids])
    return ids, normalize(cents)


def merge_clusters(cents: np.ndarray, mcfg: dict) -> np.ndarray:
    """Agglomerative merge of cluster centroids -> a merged-group id per cluster.

    No target count: clusters closer than `distance_threshold` (cosine) collapse
    together, so the number of themes emerges naturally from how similar they are.
    """
    if len(cents) < 2:
        return np.zeros(len(cents), dtype=int)
    from sklearn.cluster import AgglomerativeClustering
    # complete linkage: a group forms only when ALL its clusters are within threshold,
    # so it resists chaining (average linkage collapses into one mega-theme on dense data).
    model = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="complete",
                                    distance_threshold=mcfg["distance_threshold"])
    return model.fit_predict(cents)


def name_theme(member_subjects, example_texts, model) -> str:
    subjects = ", ".join(dict.fromkeys(member_subjects))  # dedup, keep order
    examples = "\n".join(f"- {t[:160]}" for t in example_texts)
    prompt = (
        "These sub-topics and example tweets all belong to one broader theme:\n"
        f"sub-topics: {subjects}\n\nexamples:\n{examples}\n\n"
        "Reply with a single short higher-level theme label of 2-4 words "
        "(no punctuation, no quotes)."
    )
    label = ollama_generate(model, prompt, options={"temperature": 0}).strip()
    return label.splitlines()[0].strip().strip('"').strip()[:40]


def main() -> int:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config)
    ddir = data_dir(cfg)
    mcfg = cfg["themes"]["merge"]
    if not mcfg["enabled"]:
        print("[merge] themes.merge.enabled is false — nothing to do.")
        return 0

    # Match stage 03's limit-aware cache name so a --limit dry-run lines up.
    cache = ddir / (f"embeddings.limit{args.limit}.npy" if args.limit else "embeddings.npy")
    require_file(ddir / "candidates.jsonl", "stage 03 (just themes)")
    require_file(cache, "stage 03 (just themes)")
    rows = read_jsonl(ddir / "candidates.jsonl")
    emb = normalize(np.load(cache))
    if emb.shape[0] != len(rows):
        print(f"[merge] embeddings ({emb.shape[0]}) and candidates ({len(rows)}) are out of "
              f"sync — rerun `just themes --force` first.")
        return 1

    labels = np.array([r["cluster"] for r in rows])
    ids, cents = cluster_centroids(emb, labels)
    if not ids:
        print("[merge] no clusters to merge (all noise).")
        return 1

    groups = merge_clusters(cents, mcfg)
    cluster_to_group = {c: int(g) for c, g in zip(ids, groups)}
    n_groups = len(set(groups))
    print(f"[merge] {len(ids)} clusters -> {n_groups} themes")

    ollama_preflight()
    model = cfg["score"]["llm_model"]
    reps = cfg["themes"]["naming"]["reps_per_cluster"]

    # Build per-group members and name each from its sub-labels + central tweets.
    group_members = collections.defaultdict(list)  # group_id -> [cluster_id]
    for c, g in cluster_to_group.items():
        group_members[g].append(c)

    group_name: dict[int, str] = {}
    group_size: dict[int, int] = {}
    group_subjects: dict[int, list[str]] = {}
    subj_by_cluster = {r["cluster"]: r.get("subject", "") for r in rows}

    for g, member_clusters in sorted(group_members.items()):
        mask = np.isin(labels, member_clusters)
        idx = np.where(mask)[0]
        group_size[g] = len(idx)
        center = normalize(emb[idx].mean(axis=0)[None, :])[0]
        nearest = idx[np.argsort(-(emb[idx] @ center))][:reps]
        member_subjects = sorted({subj_by_cluster[c] for c in member_clusters})
        group_subjects[g] = member_subjects
        try:
            group_name[g] = name_theme(member_subjects,
                                       [rows[i]["text"] for i in nearest], model) or f"theme_{g}"
        except Exception as e:  # noqa: BLE001
            group_name[g] = member_subjects[0] if member_subjects else f"theme_{g}"
            print(f"[merge] naming failed for theme {g}: {e}")
        print(f"[merge]   theme {g} ({group_size[g]} tweets, {len(member_clusters)} clusters): {group_name[g]}")

    for r in rows:
        g = cluster_to_group.get(r["cluster"])
        r["theme_id"] = g if g is not None else -1
        r["theme"] = group_name.get(g, "misc/noise") if g is not None else "misc/noise"

    write_jsonl(ddir / "candidates.jsonl", rows)

    themes_path = ddir / "themes.yaml"
    themes = [{
        "name": group_name[g],
        "theme_id": g,
        "tweets": group_size[g],
        "clusters": len(group_members[g]),
        "subtopics": [s for s in group_subjects[g] if s],
    } for g in sorted(group_name, key=lambda x: -group_size[x])]
    with open(themes_path, "w") as f:
        f.write(f"# {n_groups} themes from {len(ids)} fine clusters. Edit freely — rename,\n"
                f"# delete, or move subtopics between themes; this file is for your review.\n")
        yaml.safe_dump(themes, f, sort_keys=False, allow_unicode=True, width=100)
    print(f"[merge] wrote {n_groups} themes -> {themes_path} (review/merge it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
