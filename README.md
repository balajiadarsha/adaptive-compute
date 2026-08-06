# adaptive-compute

Studies how constraining a reasoning model's "thinking" token budget affects
safety behavior. Reasoning models (e.g. L1-Qwen, DeepScaleR,
DeepSeek-R1-Distill) are prompted with `Think for N tokens.` at varying
budgets, and their responses to harmful/benign prompts from
[UWNSL/SafeChain](https://huggingface.co/datasets/UWNSL/SafeChain) or
[allenai/wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix)
(WildGuardTest split) are scored with
[WildGuard](https://github.com/allenai/wildguard) for attack success rate,
refusal rate, and false refusal rate.

## Layout

- `eval/eval_safechain.py` — two-stage generate/evaluate script. Generates
  responses with vLLM and saves them to a parquet file, then (separately)
  runs WildGuard classification and computes safety metrics.
- `eval/sweep_budget.py` — runs `eval_safechain.py` across a list of token
  budgets, aggregates results into an Excel summary and a plot.
- `plots/` — standalone plotting/report scripts.
- `scripts/jobs/` — PBS (`qsub`) job scripts for running sweeps on the ALCF
  Polaris cluster.

## Setup

Install dependencies (see [requirements.md](requirements.md)):

```bash
pip install -r requirements.md
```

The `wildguard` dependency is installed from a local editable path
(`file:///../wildguard`), so a `wildguard/` checkout must exist as a sibling
of this directory (i.e. inside the parent `safety-eval/` repo).

## Running

### Single evaluation

Run generation and evaluation in one go:

```bash
cd eval
python eval_safechain.py --dataset safechain --n 100 --max-tokens 2048
```

Or split into two stages (useful since generation needs a GPU but WildGuard
scoring doesn't):

```bash
# Stage 1: generate responses, save to parquet
python eval_safechain.py --generate --dataset wildguard --n 100 \
  --max-tokens 2048 --parquet generations.parquet

# Stage 2: run WildGuard on the saved parquet
python eval_safechain.py --evaluate --parquet generations.parquet \
  --output results.json
```

Use `--think-budget N` to constrain reasoning to `N` tokens, and
`--model-id` to evaluate a different model (default:
`l3lab/L1-Qwen-1.5B-Exact`).

### Budget sweep

Sweep several token budgets and produce a summary Excel file plus plots:

```bash
cd eval
python sweep_budget.py --dataset safechain --n 500 \
  --budgets 256 512 1024 2048 4096
```

Add `--no-baseline` to skip the unconstrained (no budget) run, or
`--evaluate-only` to re-score existing parquets in `--out-dir` without
regenerating.

### Cluster (PBS) jobs

`scripts/jobs/` contains per-model, per-dataset PBS job scripts for the ALCF
Polaris cluster (hardcoded cluster paths and modules — not runnable
locally). Submit all sweeps at once with:

```bash
bash scripts/jobs/submit_all.sh            # SafeChain
bash scripts/jobs/submit_all_wildguard.sh  # WildGuardMix
```

## Requirements

See [requirements.md](requirements.md) for pinned Python dependencies.
