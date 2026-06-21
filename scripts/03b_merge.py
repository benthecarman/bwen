#!/usr/bin/env python3
"""Stage 03b — consolidate the fine clusters into higher-level themes.

Stage 03 produces hundreds of fine, overlapping labels. We group them deterministically
and let the LLM only *name* the groups (it's reliable at naming, flaky at free-generating
a whole taxonomy):

  1. Represent each fine cluster by its label + a few central example tweets, and embed
     that (label text alone is too thin — when most labels share a dominant-subject prefix
     it collapses otherwise-unrelated topics together).
  2. Agglomeratively group those vectors (cosine, complete linkage, a distance threshold).
     The number of themes emerges from similarity — no target count.
  3. LLM names each group from its member labels + example tweets.

Reuses only stage 03's candidates.jsonl; re-embeds the ~N labels (cheap), never re-clusters
the full tweet set.

Output: rewrites data/candidates.jsonl with `theme` + `theme_id` (keeps the fine
        `cluster` / `subject`), and writes data/themes.yaml (editable, for review).
"""
from __future__ import annotations

import collections
import json
import re

import numpy as np
import yaml

from common import (base_argparser, data_dir, load_config, ollama_embed,
                    ollama_generate, ollama_preflight, read_jsonl, require_file, write_jsonl)


def gather_clusters(rows: list[dict], examples_per_label: int) -> dict[int, dict]:
    """For each fine cluster (excluding noise): its label, size, and N central tweets."""
    by_cluster: dict[int, list[dict]] = collections.defaultdict(list)
    for r in rows:
        if r["cluster"] != -1:
            by_cluster[r["cluster"]].append(r)
    clusters = {}
    for cid, members in by_cluster.items():
        members.sort(key=lambda r: r.get("centroid_dist", 0.0))  # most central first
        clusters[cid] = {
            "label": members[0].get("subject", f"cluster_{cid}"),
            "count": len(members),
            "examples": [m["text"] for m in members[:examples_per_label]],
        }
    return clusters


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)


def _agglom(emb: np.ndarray, threshold: float) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="complete",
                                   distance_threshold=threshold).fit_predict(emb)


# Tightening schedule + caps for the recursive split (sensible internals, not config).
_SPLIT_FACTOR = 0.8   # multiply the threshold each time we descend into an oversized theme
_MIN_THRESHOLD = 0.2  # don't split below this (avoids shattering tiny distinctions)
_MAX_DEPTH = 4


def group_clusters(clusters: dict[int, dict], embed_model: str, threshold: float,
                   max_share: float) -> dict[int, int]:
    """Embed each cluster's label+examples, then group with a recursive agglomerative
    split: a single global threshold can't serve uneven density (the dominant topic
    stays one blob while sparse topics split fine), so any theme over `max_share` of all
    tweets is re-clustered at a tighter threshold until every theme is under the cap.
    Returns {cluster_id: group_id}.
    """
    cids = list(clusters)
    reprs = [f"{clusters[c]['label']}: " + " | ".join(clusters[c]["examples"]) for c in cids]
    vecs: list[list[float]] = []
    for i in range(0, len(reprs), 128):
        vecs.extend(ollama_embed(embed_model, reprs[i:i + 128]))
    emb = _normalize(np.asarray(vecs, dtype=np.float32))
    counts = [clusters[c]["count"] for c in cids]
    total = sum(counts)
    cap = max(1, int(total * max_share)) if max_share > 0 else total + 1  # 0 = no splitting

    def split(idxs: list[int], thr: float, depth: int) -> list[list[int]]:
        size = sum(counts[i] for i in idxs)
        if size <= cap or len(idxs) < 2 or thr < _MIN_THRESHOLD or depth >= _MAX_DEPTH:
            return [idxs]
        buckets: dict[int, list[int]] = collections.defaultdict(list)
        for i, lab in zip(idxs, _agglom(emb[idxs], thr)):
            buckets[int(lab)].append(i)
        if len(buckets) == 1:                       # cohesive at thr — tighten and retry
            return split(idxs, thr * _SPLIT_FACTOR, depth + 1)
        out: list[list[int]] = []
        for grp in buckets.values():
            out.extend(split(grp, thr * _SPLIT_FACTOR, depth + 1))
        return out

    groups = split(list(range(len(cids))), threshold, 0)
    return {cids[i]: gid for gid, grp in enumerate(groups) for i in grp}


def name_group(member_labels: list[str], example_texts: list[str], model: str,
               taken: list[str]) -> str:
    subjects = ", ".join(dict.fromkeys(member_labels))
    examples = "\n".join(f"- {t[:160]}" for t in example_texts)
    distinct = ""
    if taken:
        distinct = ("\n\nAlready-used names (yours must be different from every one):\n"
                    + ", ".join(taken))
    prompt = (
        "These sub-topics and example tweets all belong to one theme:\n"
        f"sub-topics: {subjects}\n\nexamples:\n{examples}{distinct}\n\n"
        "Reply with a single short theme label of 2-4 words. Name the SPECIFIC sub-topic "
        "that sets this group apart, drawn from its sub-topics above. Avoid broad umbrella "
        "terms that would fit many different groups, and don't use a vague or abstract word "
        "just to differ. It must differ from every already-used name. "
        "No punctuation, no quotes."
    )
    label = ollama_generate(model, prompt, options={"temperature": 0, "num_predict": 32}).strip()
    return label.splitlines()[0].strip().strip('"').strip()[:40]


def main() -> int:
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config)
    ddir = data_dir(cfg)
    mcfg = cfg["themes"]["merge"]
    if not mcfg["enabled"]:
        print("[merge] themes.merge.enabled is false — nothing to do.")
        return 0

    require_file(ddir / "candidates.jsonl", "stage 03 (just themes)")
    rows = read_jsonl(ddir / "candidates.jsonl")
    clusters = gather_clusters(rows, mcfg["examples_per_label"])
    if not clusters:
        print("[merge] no clusters to consolidate (all noise).")
        return 1

    ollama_preflight()
    model = cfg["score"]["llm_model"]

    print(f"[merge] grouping {len(clusters)} clusters by label+example similarity...")
    cluster_to_group = group_clusters(clusters, cfg["themes"]["embed_model"],
                                      mcfg["distance_threshold"], mcfg["max_theme_share"])
    members: dict[int, list[int]] = collections.defaultdict(list)
    for cid, g in cluster_to_group.items():
        members[g].append(cid)
    print(f"[merge] {len(clusters)} clusters -> {len(members)} groups; naming...")

    # Name largest theme first, passing the names already taken so each new name is
    # distinct — the dominant theme gets the clean name, overlapping siblings differentiate.
    group_size = {g: sum(clusters[c]["count"] for c in mc) for g, mc in members.items()}
    group_name: dict[int, str] = {}
    taken: list[str] = []
    for g in sorted(members, key=lambda g: -group_size[g]):
        member_clusters = sorted(members[g], key=lambda c: -clusters[c]["count"])
        labels = [clusters[c]["label"] for c in member_clusters]
        examples = [t for c in member_clusters[:5] for t in clusters[c]["examples"][:2]]
        try:
            name = name_group(labels, examples[:8], model, taken) or labels[0]
        except Exception as e:  # noqa: BLE001
            name = labels[0]
            print(f"[merge] naming failed for group {g}: {e}")
        group_name[g] = name
        taken.append(name)
        print(f"[merge]   {group_size[g]:5d} tweets, {len(member_clusters):2d} clusters: {name}")

    # Resolve duplicate names (independent naming can collide) by appending the group id.
    # Case-insensitive so two names differing only in case don't both stand.
    counts = collections.Counter(n.lower() for n in group_name.values())
    for g, name in group_name.items():
        if counts[name.lower()] > 1:
            group_name[g] = f"{name} ({g})"

    # id themes largest-first by cluster size, then append any keyword themes.
    size_by_group: collections.Counter = collections.Counter()
    for g, member_clusters in members.items():
        for c in member_clusters:
            size_by_group[group_name[g]] += clusters[c]["count"]
    name_to_id = {name: i for i, name in
                  enumerate(sorted(size_by_group, key=lambda n: -size_by_group[n]))}

    # Keyword themes: pull every matching own tweet into a named theme, overriding its
    # semantic assignment. An entity that scatters across topics (a recurring ticker,
    # name, project, catchphrase) never forms a cluster; this surfaces it as one theme.
    # Word-boundary match (alphanumeric boundaries) so a keyword like "dell" hits
    # "$dell" / "Dell" but not "odell"; matching is case-insensitive.
    kw_patterns = {
        name: re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(w.lower()) for w in kws)
                         + r")(?![a-z0-9])")
        for name, kws in (mcfg.get("keyword_themes") or {}).items() if kws}
    for name in kw_patterns:
        name_to_id.setdefault(name, len(name_to_id))

    for r in rows:
        g = cluster_to_group.get(r["cluster"]) if r["cluster"] != -1 else None
        name = group_name.get(g) if g is not None else None
        if r.get("is_own", True):
            low = r["text"].lower()
            for kname, pat in kw_patterns.items():
                if pat.search(low):
                    name = kname
                    break
        r["theme"] = name or "misc/noise"
        r["theme_id"] = name_to_id[name] if name else -1
    write_jsonl(ddir / "candidates.jsonl", rows)

    # themes.yaml from the FINAL per-tweet assignments, so keyword themes appear too.
    size_by_name: collections.Counter = collections.Counter()
    subs: dict[str, set] = collections.defaultdict(set)
    for r in rows:
        if r["theme_id"] == -1:
            continue
        size_by_name[r["theme"]] += 1
        if r.get("subject"):
            subs[r["theme"]].add(r["subject"])
    themes = [{
        "name": name,
        "theme_id": name_to_id[name],
        "tweets": size_by_name[name],
        "subtopics": sorted(s for s in subs[name] if s),
    } for name in sorted(size_by_name, key=lambda n: -size_by_name[n])]

    themes_path = ddir / "themes.yaml"
    with open(themes_path, "w") as f:
        f.write(f"# {len(themes)} themes consolidated from {len(clusters)} fine clusters.\n"
                f"# Edit freely — rename, delete, or move subtopics; this is for review.\n")
        yaml.safe_dump(themes, f, sort_keys=False, allow_unicode=True, width=100)
    print(f"[merge] {len(clusters)} clusters -> {len(themes)} themes -> {themes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
