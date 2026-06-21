# How bwen works — the full process

This explains the end-to-end method behind [bwen](../README.md): turning a personal
Twitter/X archive into a small model that writes in your voice and holds your opinions,
trained on **only your real tweets — no synthetic / AI-written training text.**

The trained model and dataset built with this pipeline:
- Model: https://huggingface.co/benthecarman/bwen-14b
- Dataset: https://huggingface.co/datasets/benthecarman/bwen-dataset

---

## 1. The idea (and why a naive attempt fails)

The goal is a model that (a) *sounds* like you and (b) *thinks* like you. The naive approach —
dump all 30k tweets into a finetune with AI-generated prompts — fails two ways:

1. **Too large and generic.** Averaging over everything (retweets, one-word replies, links)
   washes out the distinctive voice into a bland mean.
2. **AI-generated prompts.** If a model invents the prompts, you train on *its* idea of what
   you'd be asked, and the instruction style leaks into the output, diluting your voice.

bwen fixes both:

- **Curate a small, high-signal subset** instead of the whole firehose.
- **You hand-write the prompts** for a few hundred examples. The *completion* is your real
  tweet; you supply the *prompt*. This also solves a hidden problem: your archive contains only
  *your* tweets, not the tweets you replied to (82% of tweets are replies). When you write the
  prompt, you reconstruct the missing context from memory — nothing is fetched or invented.
- **A "voice layer"** of raw tweets (no prompts) reinforces style for free.
- **The LLM is only ever a filter/scorer**, never a writer of training text.

Two jobs, kept separate: the **completions** teach voice + opinions (your real words); the
**prompts** teach the model how to be *triggered* to produce them.

---

## 2. Stack

- A local **CUDA GPU** (~16 GB VRAM is enough; larger bases need QLoRA), **Ollama**. On a very
  recent GPU, PyTorch must match the CUDA arch (e.g. a CUDA 12.8 build for Blackwell-class cards).
- Python pinned to **3.12** (torch/unsloth lack 3.14 wheels).
- Embeddings + cluster naming run on local Ollama models; finetuning via **Unsloth** (LoRA/QLoRA).
- Everything is config-driven (`config.yaml`) and run via `just`. No value is hardcoded — point it
  at your own archive and go.

---

## 3. The pipeline, stage by stage

Each stage writes a file and is independently re-runnable (cached/skippable), so you can iterate
on any step without redoing the rest.

### 01 — parse (`scripts/01_parse.py`)
Strip the `window.YTD.tweets.part0 = ` JS wrapper from the archive and emit flat JSON. Globs split
exports (`tweets-part1.js`, …) so large accounts work unchanged.

### 02 — filter (`scripts/02_filter_clean.py`)
Drop retweets, non-target languages, and link-only/one-word tweets. Clean the text: strip t.co and
all URLs (a *generated* URL is always a hallucination), drop the leading `@a @b` reply-addressing
(it's addressing, not voice), de-dupe.

**The `is_own` guardrail:** retweets and likes are *other people's words*. Optionally fold them in
(`include_retweets` / `include_likes`, capped by `max_likes`) tagged `is_own: false` — they enrich
**theme discovery only**. Stages 04 and 06 filter them out, so training stays *your words only*.

### 03 — themes (`scripts/03_themes.py`)
Discover your real topics so the shortlist covers them broadly instead of drowning in your loudest
subject. Embed every tweet (Ollama `nomic-embed-text`) → reduce → cluster → name.

- **Reduction matters.** PCA (linear) flattens the embedding manifold, so density clustering dumps
  most tweets into "noise". **UMAP** (nonlinear, neighborhood-preserving) gives tight, separated
  clusters — on the real archive it cut noise from ~86% to ~18%. `themes.reduce.method` = `umap`.
- Cluster with **HDBSCAN** (or k-means), then the LLM names each cluster from its most central
  tweets → `subjects.txt`.

### 03b — merge (`scripts/03b_merge.py`)
UMAP+HDBSCAN produces hundreds of fine, overlapping clusters. Consolidate them into a few dozen
higher-level themes — but **deterministically**, with the LLM only naming:

1. Represent each fine cluster by its **label + a few example tweets**, and embed that. (Label text
   alone is too thin — a shared dominant-subject prefix collapses unrelated topics together.)
2. **Agglomeratively group** those vectors (cosine, complete linkage, `distance_threshold`). The
   theme count emerges from similarity; there's no target.
3. **Recursively split** any theme over `max_theme_share` of all tweets at a tighter threshold, so a
   dominant topic can't stay one giant blob while sparse topics split fine.
4. **Name each group largest-first**, passing the names already taken so each is distinct, and asking
   for the *specific facet* (not a broad umbrella). Domain-agnostic prompt — the tweets supply the
   topic.

(Earlier attempts — geometric centroid-merge, and free-form LLM taxonomy generation — were dropped:
the former fused embedding-adjacent but unrelated topics; the latter was unreliable run-to-run and
enumerated near-duplicate names. Deterministic grouping + LLM naming is the stable combination.)

### 04 — score (`scripts/04_score.py`)
Rank candidates so you label the strongest material first:
- **Heuristic**: engagement (favorites + retweets) + a length sweet-spot.
- **LLM scorer** (filter only): rate each tweet 1–5 for *opinion density* and *voice* via a local model.
- **Balanced shortlist**: round-robin across themes by combined score, so coverage is broad. Noise
  is held out of the rotation and only used to backfill.

### 05 — label (`scripts/05_label.py`)
The heart of it. A resumable terminal tool: for each tweet it shows the text, theme, engagement,
scores, and a link to the tweet (open it in-thread for context). You type the prompt; the completion
is the tweet. It draws from the **full balanced pool**, so skipping is topped up by the next-best
candidate — skips never shrink the set. Target ~150–300 pairs.

*Prompt-writing philosophy* (what makes or breaks the result): write the **trigger that elicits the
tweet**, not a paraphrase of it. For a sincere opinion, "what's your take on X?" For a comeback, the
*situation* it answers. For hype, the *mood*. Let the completion carry the voice; the prompt only
sets the moment. Casual grammar is fine (matches how you'll actually prompt it).

### 06 — build dataset (`scripts/06_build_dataset.py`)
Assemble training data from real tweets only:
- **Instruction pairs** from your labels → chat format (`system` persona + `user` prompt +
  `assistant` = your tweet), with Qwen3 thinking disabled.
- **Voice layer**: a few thousand raw tweets as plain completions (no prompt).
- A held-out eval split.

### 07 — train (`scripts/07_train.py`)
LoRA finetune with Unsloth. Key detail: **prompt masking** — for chat pairs, loss is computed only on
the assistant turn (your tweet), not the persona/prompt, so the model learns *your words*, not the
scaffolding. Voice tweets are trained in full. Set `load_in_4bit` (QLoRA) to fit larger bases.

### 08 — export (`scripts/08_export.py`)
Merge the LoRA, convert to **GGUF**, and register an Ollama model. Three gotchas this handles:
- Unsloth writes the GGUF to a sibling `<dir>_gguf` folder with an uppercase quant in the name →
  found recursively/case-insensitively, and reused so you don't redo the ~10-min conversion.
- The Ollama Modelfile **must carry the chat template + stop tokens**, or Ollama feeds the raw prompt
  and the model just *continues* it instead of answering → built on Unsloth's generated Modelfile.
- Qwen3 thinks by default; the template ignores `/no_think`, so an **empty `<think></think>` is
  prefilled** at the assistant turn (matching `enable_thinking=False` in training) → it answers directly.

### 09 — eval (`scripts/09_eval.py`)
Run a fixed prompt set (held-out pairs + `eval_prompts.txt`) through the base model and the tuned one,
side by side → `runs/<timestamp>.md`. This is the iteration signal.

### 10 — ask / RAG (`scripts/10_ask.py`)
Optional. The finetune captures *voice* well but its *opinions* are frozen at the archive's date and
skewed toward your loudest takes. Retrieval fixes that: embed the question with the same model used
for the corpus, cosine-search your own tweets (`embeddings.npy`, filtered to `is_own`), and feed the
top matches to the tuned model as grounding. The voice comes from the finetune; the positions come
from your real quotes. Reuses stage 03's embeddings, so there's no extra setup — `just ask "..."`.

---

## 4. Model size & scaling

The pipeline works with any Unsloth-supported base (Llama, Mistral, Gemma, Qwen3, …) — change
`train.base_model` and rerun; nothing else changes. (The thinking-mode handling above is
Qwen3-specific and harmlessly no-ops on non-reasoning models.) A small base is cheap to iterate
with and a larger one is more coherent — a good workflow is to settle your data/prompts on a small
model, then rerun the train→export→eval stages on a bigger one.

This project went **1.7B → 14B**. On a 16 GB GPU, 14B needs **QLoRA** (`load_in_4bit: true`, small
batch + grad accumulation).
The catch is *export*: merging 14B back to 16-bit for GGUF needs ~28 GB, so on a low-RAM box add
swap to let it spill to disk. The 14B was markedly more coherent and on-message than the 1.7B (e.g.
it knew DLCs are oracle contracts on Bitcoin where the base model hallucinated "Digital Locker
Contracts"), while keeping the blunt, no-hedging voice.

---

## 5. Design principles (the lessons)

- **No synthetic training text.** Completions are real tweets; prompts are hand-written by you.
- **Separate voice from triggering.** Voice/opinions live in completions and the voice layer;
  prompts only learn to summon them.
- **LLM as filter/namer, never writer.** It scores, clusters, and names — it never produces training
  text.
- **Deterministic where possible.** Theme grouping is geometric + reproducible; the LLM only names.
- **Generic, not hardcoded.** Prompts and config carry no domain terms — the *data* supplies the
  domain, so anyone can run this on their own archive.
- **Curate small and high-signal.** A few hundred sharp pairs + a voice layer beat dumping everything.

---

## 6. Reproduce it

1. Get your X archive (Settings → *Download an archive of your data*), unzip so `data/` sits at
   `twitter-archive/data/`.
2. `just setup`, then `cp config.example.yaml config.yaml` and edit handle/account_id/models.
3. `just dry-run` → `just all` → review `data/themes.yaml` → `just label` → `just data`.
4. `uv sync --extra train`, then `just train` → `just export` → `just eval`.

See the [README](../README.md) for the command reference and tunable knobs. To start from the exact
data used here instead of your own, see the
[dataset](https://huggingface.co/datasets/benthecarman/bwen-dataset).
