# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

bwen is a config-driven pipeline that finetunes a small model to write in a person's voice and hold
their opinions, trained **only on their real tweets — no synthetic / AI-written training text**. It's
pure Python (uv + `pyproject.toml`), orchestrated by `just`. See `README.md` for the command reference
and `docs/PROCESS.md` for the full methodology — prefer reading those over rediscovering the design.

## Commands

```bash
just setup                 # uv venv (Python 3.12) + install deps
just dry-run               # smoke-test stages 01–06 on a 200-tweet sample in a throwaway dir
just all                   # parse → filter → themes → 03b merge → score  (data pipeline)
just label                 # interactive: hand-write a prompt per shortlisted tweet (resumable)
just data                  # build train.jsonl + eval.jsonl
uv sync --extra train      # heavy stack (torch cu128, unsloth, trl, transformers) — before training
just train                 # LoRA/QLoRA finetune (needs a CUDA GPU)
just export                # merge → GGUF → `ollama create`
just eval                  # base-vs-tuned scorecard → runs/<timestamp>.md
just ask "..."             # stage 10: RAG over the tweets, answered in the tuned voice
just publish               # stage 11: push dataset + model to HF (repos from config.publish)
just clean                 # wipe data/ (regenerable); state/ is kept
```

- **Iterate / run one stage:** every stage takes `--limit N` (sample) and `--force` (ignore cached
  output), e.g. `.venv/bin/python scripts/04_score.py --limit 50 --force` or `just score --force`.
- **There is no test suite.** Validate changes with `just dry-run`, a `--limit` run of the touched
  stage, and a syntax check: `.venv/bin/python -c "import ast; ast.parse(open('scripts/NN_x.py').read())"`.
- **Long ops** (train ~1h for 14B, export's GGUF merge, the LLM scoring pass) should be run in the
  background. Export of a large model merges to ~28 GB; on a low-RAM box add swap first.

## Architecture

Numbered stages in `scripts/` (`01_parse` → `10_ask`), each reading one file and writing the next, so
any stage is independently re-runnable. `common.py` holds the shared helpers every stage imports:
`load_config`, `data_dir`/`state_dir`, `read_jsonl`/`write_jsonl`, `ollama_embed`/`ollama_generate`,
`balanced_order`/`combined_score`, `think_off`. The README has the stage→output table.

**Config is the single source of truth.** `load_config` loads `config.example.yaml` as the default
layer and deep-merges the user's (gitignored) `config.yaml` over it. **Do not add inline defaults in
scripts** — read `cfg[...]` directly; the example file holds every default, so a missing key should
surface as a `KeyError`, not be silently defaulted.

**`data/` vs `state/`.** `data/` is fully regenerable and wiped by `just clean`. `state/` holds what you
can't regenerate — hand-labels (`labeled.jsonl`), skips (`label_skip.json`) — plus the expensive LLM
`score_cache.json`. Personal/large paths are gitignored: `twitter-archive/`, `data/`, `state/`, `runs/`,
`config.yaml`, `eval_prompts.txt`, `Modelfile`.

## Conventions that matter (easy to violate)

- **No synthetic training text.** Completions are always real tweets; prompts are hand-written by the
  user. The LLM is only ever a filter/scorer/namer — it must never generate training text.
- **Keep it generic.** No domain-specific terms (e.g. "bitcoin", "crypto") in prompts, code, config
  defaults, or the `*.example` files. The *data* passed into a prompt supplies the domain, so the tool
  works on anyone's archive. This has regressed repeatedly — guard it.
- **`is_own` guardrail.** Retweets/likes (`filter.include_retweets`/`include_likes`) enrich theme
  discovery only; they're tagged `is_own: false` and stages 04/06 exclude them, so training stays the
  user's own words.
- **Qwen3 thinking is the only model-specific code:** `enable_thinking=False` (07), the `<think></think>`
  prefill regex in the generated Modelfile, and `/no_think` (08). These harmlessly no-op on non-Qwen
  bases — the pipeline otherwise works with any Unsloth-supported model via `train.base_model`.
- **GGUF/Ollama export is finicky:** Unsloth writes the GGUF to a sibling `<dir>_gguf` folder with an
  uppercase quant in the filename, and the Ollama Modelfile must carry the chat `TEMPLATE` + stop tokens
  (or the model just continues the prompt). `08_export.py` already handles both — don't regress them.
- **Editing a script mid-run:** `just all` launches stages as separate processes in sequence. Editing a
  stage whose process hasn't started yet can be read mid-write (this has caused a null-byte corruption);
  only edit stages that have already started or finished.

## Git

- Default branch is `main`. Commit messages follow the cbea rules and are **enforced by a commit-msg
  hook** (subject ≤50 chars, body wrapped at **≤72**, imperative, no issue/PR links) — commits are
  rejected otherwise. Watch multibyte chars (an em-dash counts toward the 72). End every commit with the
  `Co-Authored-By:` trailer from the global config.
- **Stage files explicitly** (`git add scripts/foo.py ...`), never `git add -A` — untracked junk
  (`node_modules/`, scratch files) lives in the working dir and must not be committed.
- Published artifacts: GitHub `benthecarman/bwen`, HF model `benthecarman/bwen-14b`, HF dataset
  `benthecarman/bwen-dataset`.
