# One-command runner. Override the python via `just py=python <recipe>`.
py := ".venv/bin/python"
s  := "scripts"

# List available recipes.
default:
    @just --list

# Create venv + install light deps.
setup:
    uv venv --python 3.12
    uv pip install -e .

# 01 — parse the raw archive. Pass flags through, e.g. `just parse --limit 200`.
parse *args:
    {{py}} {{s}}/01_parse.py {{args}}

# 02 — filter + clean.
clean *args:
    {{py}} {{s}}/02_filter_clean.py {{args}}

# 03 — embed + cluster + name themes.
themes *args:
    {{py}} {{s}}/03_themes.py {{args}}

# 04 — score + select shortlist.
score *args:
    {{py}} {{s}}/04_score.py {{args}}

# 05 — hand-write prompts (resumable).
label *args:
    {{py}} {{s}}/05_label.py {{args}}

# 06 — build train.jsonl + eval.jsonl.
data *args:
    {{py}} {{s}}/06_build_dataset.py {{args}}

# 07 — LoRA finetune (needs `uv sync --extra train`).
train *args:
    {{py}} {{s}}/07_train.py {{args}}

# 08 — merge + GGUF + ollama create.
export *args:
    {{py}} {{s}}/08_export.py {{args}}

# 09 — base-vs-tuned eval scorecard.
eval *args:
    {{py}} {{s}}/09_eval.py {{args}}

# Data pipeline up to the shortlist (then hand-label, then build + train).
all: parse clean themes score
    @echo "Next: just label  ->  just data  ->  just train (uv sync --extra train first)  ->  just export  ->  just eval"

# Fast smoke test of the data stages on a small sample. Writes to a throwaway dir
# (BWEN_DATA_DIR) so it never poisons the real data/ artifacts that later stages reuse.
# Stage 05 is interactive, so we seed a stub labeled.jsonl to still exercise stage 06.
dry-run:
    #!/usr/bin/env bash
    set -euo pipefail
    export BWEN_DATA_DIR=.dryrun
    trap 'rm -rf "$BWEN_DATA_DIR"' EXIT   # clean up even if a stage fails
    {{py}} {{s}}/01_parse.py --limit 200 --force
    {{py}} {{s}}/02_filter_clean.py --limit 200 --force
    {{py}} {{s}}/03_themes.py --force
    {{py}} {{s}}/04_score.py --force
    # Stand in for the interactive stage 05: turn a few shortlist rows into labeled pairs.
    {{py}} -c "import json,os; d=os.environ['BWEN_DATA_DIR']; rows=[json.loads(l) for l in open(d+'/shortlist.jsonl')][:5]; open(d+'/labeled.jsonl','w').write(''.join(json.dumps({'id':str(r['id']),'prompt':'test prompt','completion':r['text'],'subject':r.get('subject')})+chr(10) for r in rows))"
    {{py}} {{s}}/06_build_dataset.py
    echo "dry-run OK (isolated; real data/ untouched)"
