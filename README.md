# bwen — finetune a model that talks like you

Finetune a small model to write in **your voice** and hold **your opinions**, trained on
**only your real tweets — no synthetic / AI-written training text.** Iterate fast on
`qwen3:1.7b`, then re-run on a bigger model by changing one config value.

## Why this works (and the prior approach didn't)

- **Curate a small, high-signal subset** instead of dumping 30k+ tweets. Embeddings +
  clustering surface your real themes and most representative tweets per theme.
- **You hand-write the prompts** for a few hundred examples. The *completion* is your real
  tweet; you supply the *prompt*. This kills generic AI-prompt slop **and** solves the
  missing-context problem — your archive has your replies but not what you replied to, so
  you reconstruct the intent from memory.
- A **voice layer** of raw tweets (no prompts) reinforces style for free.
- An **LLM is used only as a filter/scorer**, never to write training text.

## Setup

```bash
uv venv --python 3.12          # torch/unsloth don't ship 3.14 wheels
uv pip install -e .            # light data-pipeline deps
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
just discover       # 01 parse → 02 filter → 03 themes  (stop to review subjects)
just all            # 01 parse → 02 filter → 03 themes → 04 score  (full)
$EDITOR data/subjects.txt   # review/merge the auto-discovered themes
just label          # hand-write prompts for the shortlist (resumable)
just data           # build train.jsonl (chat pairs + voice layer) + eval.jsonl
```

Each stage writes a file and **skips if its output exists** (use `--force` to recompute;
embeddings are cached). Every stage takes `--limit N` for fast iteration.

## Train, export, evaluate

```bash
uv sync --extra train      # heavy stack: torch (cu128), unsloth, trl, transformers
just train                 # LoRA finetune on the RTX 5060 Ti
just export                # merge → GGUF → `ollama create`
just eval                  # base vs. tuned scorecard -> runs/<timestamp>.md
ollama run bwen
```

> **Blackwell GPU (RTX 5060 Ti / sm_120):** torch must be a CUDA 12.8 build. If
> `just train` reports `cuda avail=False` or an sm_120 error, install the cu128 wheel:
> `uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch`

## Pipeline

| Stage | Script | Output |
|---|---|---|
| 01 parse | `scripts/01_parse.py` | `data/raw/tweets.json` |
| 02 filter | `scripts/02_filter_clean.py` | `data/filtered/tweets.jsonl` |
| 03 themes | `scripts/03_themes.py` | `data/candidates.jsonl`, `data/subjects.txt`, `data/embeddings.npy` |
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
- `themes.algorithm` / `hdbscan_min_cluster_size` / `kmeans_k` — cluster granularity.
  If `hdbscan` won't build (it's historically friction-prone on new Python / NumPy),
  set `themes.algorithm: kmeans` — it uses only scikit-learn and needs no extra wheel.
- `filter.include_retweets` / `filter.include_likes` (+ `max_likes`) — let retweets and
  liked tweets enrich theme discovery (denser clusters, broader topic map). They're tagged
  `is_own:false` and never labeled or trained on — training stays your words only.
- `train.base_model` — swap to an ~8B model to scale up; everything else is unchanged.

## Scaling up / phase 2

Once the technique is proven on 1.7B, set `train.base_model` to a larger model (e.g.
`unsloth/Qwen3-8B`) and rerun 06–09. If the model sounds like you but gets *positions*
wrong, add RAG over `data/embeddings.npy` (retrieve your real tweets at inference) — voice
from the finetune, opinions grounded in real quotes.
