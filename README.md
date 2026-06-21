# bwen — finetune a model that talks like you

Finetune a small model to write in **your voice** and hold **your opinions**, trained on
**only your real tweets — no synthetic / AI-written training text.** Iterate fast on
`qwen3:1.7b`, then re-run on a bigger model by changing one config value.

- **How it works (full methodology):** [docs/PROCESS.md](docs/PROCESS.md)
- **Example model:** [benthecarman/bwen-14b](https://huggingface.co/benthecarman/bwen-14b)
- **Example dataset:** [benthecarman/bwen-dataset](https://huggingface.co/datasets/benthecarman/bwen-dataset)

## How it works

bwen turns your Twitter/X archive into a training set made entirely of your real tweets, then
LoRA-finetunes a small model on it:

1. **Parse & filter** — pull your tweets from the archive; drop retweets, links, and non-English;
   clean URLs and the leading `@mentions` on replies.
2. **Discover themes** — embed every tweet and cluster them (UMAP + HDBSCAN) to map your real
   topics, then consolidate those into a few dozen named themes.
3. **Score & shortlist** — rank tweets by engagement plus an LLM opinion/voice score, and pick a
   theme-balanced shortlist so coverage is broad.
4. **Hand-label** — for each shortlisted tweet you write a short prompt (the question or situation
   it answers); the tweet itself is the target. A few hundred pairs.
5. **Build the dataset** — your prompt→tweet pairs as chat examples, plus a *voice layer* of raw
   tweets (no prompt) to reinforce style.
6. **Train, export, evaluate** — LoRA/QLoRA finetune (loss falls only on your tweet, not the
   prompt), export to GGUF for Ollama, and compare base vs. tuned.

Two things hold throughout: every completion is your real words — there's **no synthetic text**,
the LLM only ever filters, scores, and names — and the prompt is just the *trigger* while the
tweet carries the voice and opinions. Full details in **[docs/PROCESS.md](docs/PROCESS.md)**.

## Setup

```bash
uv venv --python 3.12          # torch/unsloth don't ship 3.14 wheels
uv pip install -e .            # data-pipeline + training stack (torch cu128, unsloth, ...)
cp config.example.yaml config.yaml             # then edit: handle, account_id, models
cp eval_prompts.example.txt eval_prompts.txt   # then edit with prompts from your domain
cp Modelfile.example Modelfile                 # then edit with your handle

# Pull every Ollama model the pipeline uses (names must match your config.yaml):
ollama pull nomic-embed-text   # themes.embed_model — clustering (stage 03)
ollama pull qwen2.5:7b         # score.llm_model    — scorer/cluster-namer (stages 03-04)
ollama pull qwen3:1.7b         # eval.base_model_tag — untuned base for the eval (stage 09)
```

Pulling these up front avoids a connection/404 error mid-run. If you change any of
`themes.embed_model`, `score.llm_model`, or `eval.base_model_tag` in `config.yaml`,
pull the model you switched to instead.

Get your archive: X → Settings → *Download an archive of your data*. Unzip so the
`data/` folder is at `twitter-archive/data/` (or point `paths.archive_dir` at it).

## Run the data pipeline

```bash
just dry-run        # smoke-test the data stages (01-04, 06) on a 200-tweet sample first
just discover       # 01 parse → 02 filter → 03 themes → 03b merge  (stop to review themes)
just all            # 01 parse → 02 filter → 03 themes → 03b merge → 04 score  (full)
$EDITOR data/themes.yaml    # review/merge the compressed themes
just label          # hand-write prompts for the shortlist (resumable)
just data           # build train.jsonl (chat pairs + voice layer) + eval.jsonl
```

Each stage writes a file and **skips if its output exists** (use `--force` to recompute;
embeddings are cached). Every stage takes `--limit N` for fast iteration.

## Train, export, evaluate

```bash
just train                 # LoRA finetune (needs a CUDA GPU)
just export                # merge → GGUF → `ollama create`
just eval                  # base vs. tuned scorecard -> runs/<timestamp>.md
ollama run bwen
```

> **GPU notes:** `just train` prints the detected torch/CUDA/GPU at startup. If it reports
> `cuda avail=False` or an `sm_XXX` / unsupported-architecture error, your torch build doesn't
> match your GPU — install the matching CUDA wheel, e.g. for a recent (Blackwell-class) GPU:
> `uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch`.
> A ~1.7B model trains in 16-bit on a modest GPU; larger models (8B+) need `train.load_in_4bit:
> true` (QLoRA) to fit ~16 GB of VRAM.

## Pipeline

| Stage | Script | Output |
|---|---|---|
| 01 parse | `scripts/01_parse.py` | `data/raw/tweets.json` |
| 02 filter | `scripts/02_filter_clean.py` | `data/filtered/tweets.jsonl` |
| 03 themes | `scripts/03_themes.py` | `data/candidates.jsonl`, `data/subjects.txt`, `data/embeddings.npy` |
| 03b merge | `scripts/03b_merge.py` | `data/themes.yaml` (+ `theme`/`theme_id` on candidates) |
| 04 score | `scripts/04_score.py` | `data/candidates_scored.jsonl`, `data/shortlist.jsonl` |
| 05 label | `scripts/05_label.py` | `data/labeled.jsonl` |
| 06 build | `scripts/06_build_dataset.py` | `data/train.jsonl`, `data/eval.jsonl` |
| 07 train | `scripts/07_train.py` | `data/model/lora/` |
| 08 export | `scripts/08_export.py` | `data/model/gguf/*.gguf` + Ollama model |
| 09 eval | `scripts/09_eval.py` | `runs/<timestamp>.md` |

## Knobs worth tuning (all in `config.yaml`)

- `score.shortlist_size` / `dataset.label_target` — how many pairs you hand-write.
- `dataset.voice_pool_size` — size of the raw-tweet voice layer.
- `train.epochs`, `learning_rate`, `lora_rank` — start low; watch for overfit on a small set.
- `themes.cluster` — `algorithm` (`hdbscan`|`kmeans`), `hdbscan.min_cluster_size`, `kmeans.k` — cluster granularity.
- `themes.reduce.method` — `pca` (default) | `umap` | `none`. UMAP gives tighter, better-separated
  topic clusters with less noise. Tune with `themes.reduce.umap.neighbors` (smaller = finer/more
  topics) and `umap.min_dist`.
  If `hdbscan` won't build (it's historically friction-prone on new Python / NumPy),
  set `themes.cluster.algorithm: kmeans` — it uses only scikit-learn and needs no extra wheel.
- `themes.merge` — stage 03b consolidates the fine clusters into broad themes: it groups them
  by label+example similarity, then the LLM names each group. Tune `distance_threshold` (lower =
  more, finer themes; higher = fewer, broader) and `examples_per_label`. `max_theme_share`
  recursively re-splits any theme bigger than that fraction of all tweets (so a dominant topic
  doesn't stay one giant blob; `0` disables splitting). `enabled: false` to skip.
- `filter.include_retweets` / `filter.include_likes` (+ `max_likes`) — let retweets and
  liked tweets enrich theme discovery (denser clusters, broader topic map). They're tagged
  `is_own:false` and never labeled or trained on — training stays your words only.
- `train.base_model` — swap to an ~8B model to scale up; everything else is unchanged.

## Scaling up / phase 2

Once the technique is proven on 1.7B, set `train.base_model` to a larger model (e.g.
`unsloth/Qwen3-8B`) and rerun 06–09. If the model sounds like you but gets *positions*
wrong, add RAG over `data/embeddings.npy` (retrieve your real tweets at inference) — voice
from the finetune, opinions grounded in real quotes.
