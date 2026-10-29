# adaptive-compute

![Approach: reasoning budget and precision as two compute knobs, evaluated for skill, safety and FLOPs](plots/approach_figure.png)

Studies how constraining a reasoning model's "thinking" token budget affects
safety behavior and task skill. Reasoning models (e.g. L1-Qwen, DeepScaleR,
DeepSeek-R1-Distill) are prompted with `Think for N tokens.` at varying
budgets, and evaluated two ways:

- **Safety**: responses to harmful/benign prompts from
  [UWNSL/SafeChain](https://huggingface.co/datasets/UWNSL/SafeChain) or
  [allenai/wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix)
  (WildGuardTest split) are scored with
  [WildGuard](https://github.com/allenai/wildguard) for attack success rate,
  refusal rate, and false refusal rate.
- **Skill**: responses to math/science/reasoning benchmarks (AIME, GPQA,
  LSAT) are scored by exact-matching the model's extracted final answer
  against each dataset's gold label — so budget-vs-safety tradeoffs can be
  read against budget-vs-capability tradeoffs.

Separately, the repo also studies how **post-training quantization**
(GPTQ weight-only int8/int4/int3, and W8A8/W4A8 weight-activation) affects
the same safety/skill metrics, to see whether compute savings from budget
constraints and from quantization compose.

## Features

Each feature is one compute knob. Both feed the same reporting steps
([Reporting](#reporting-tables-and-compute-plots)), so every configuration
gets skill, safety and compute (FLOPs) numbers measured on the same
responses. The job scripts in `scripts/jobs/` `cd` into this checkout on
Perlmutter and use the `reason` (vLLM) and `flatquant` conda envs. Change
those paths if you run elsewhere.

### Length control (reasoning budget)

Appends `Think for N tokens.` to every prompt (`--think-budget N`). Leaving
the flag out runs the model with no budget prompt ("No limit"). Each response
records `token_count` and `think_token_count`, so cost always pairs with the
metric.

| Step | Run | Writes |
|---|---|---|
| 1. Skill (generate + score) | `scripts/jobs/run_skill_grid.sh` → `eval/eval_skill.py` | `eval/results/<dir>/<model>-budget<N>/<dataset>.json` |
| 2. Safety generation | `scripts/jobs/run_safety_grid.sh` → `eval/eval_safety.py --generate` | `.../<model>-budget<N>/{safechain,wildguard,strongreject}.parquet` |
| 3. Safety scoring | `eval/score_safety.py` (WildGuard + StrongREJECT judges) | a `.json` next to each parquet |
| 4. Tables, plots | see [Reporting](#reporting-tables-and-compute-plots) | `context/` |

```bash
# 1. Skill: AIME/GPQA/MATH/GSM8K at four budgets, sampled (T=0.6, top-p 0.95).
#    CAP_MODE "default" = per-dataset caps; "32k" + "none" = the No-limit run.
TEMPERATURE=0.6 TOP_P=0.95 bash scripts/jobs/run_skill_grid.sh \
    l3lab/L1-Qwen-1.5B-Exact L1-Qwen-1.5B-Exact 0 results/full_eval_t06 default 512 1024 2048 4096
TEMPERATURE=0.6 TOP_P=0.95 bash scripts/jobs/run_skill_grid.sh \
    l3lab/L1-Qwen-1.5B-Exact L1-Qwen-1.5B-Exact 0 results/full_eval_32k_t06 32k none

# 2. Safety generation (8192-token cap; "32k none" for No limit).
bash scripts/jobs/run_safety_grid.sh \
    l3lab/L1-Qwen-1.5B-Exact L1-Qwen-1.5B-Exact results/safety_t06 default 512 1024 2048 4096

# 3. Score every parquet in one process (skips ones already scored).
cd eval && python score_safety.py results/safety_t06/L1-Qwen-1.5B-Exact-budget*/*.parquet
```

On Slurm, `scripts/jobs/submit_safety_grid.sh` submits the whole safety
grid. It runs one generation job per model group and chains a scoring job
after each. All steps resume: rerunning skips finished outputs.

### Quantization

Post-training quantization of the same models, evaluated through the same
grids. GPTQ and AWQ produce checkpoints that vLLM loads directly. FlatQuant
produces learned transform matrices, which its own HF-`generate` evaluators
apply (slower; run in the `flatquant` env).

| Method | Precision | Quantize with | Evaluate with |
|---|---|---|---|
| GPTQ | INT8 / INT4 / INT3 weights | `quantize/quantize_model.py` | `run_skill_grid.sh`, `run_safety_grid.sh` (checkpoint as model) |
| AWQ | W4A16 weights | `quantize/quantize_model_awq.py` | same as GPTQ |
| FlatQuant | W8A8KV8, W4A4KV4 (weights, activations, KV cache) | `scripts/jobs/flatquant_calibrate.sh` | `scripts/jobs/run_flatquant_grid.sh` → `eval/eval_skill_flatquant.py`; `scripts/jobs/run_flatquant_safety.sh` → `eval/eval_safety_flatquant.py` |
| W8A8 / W4A8 (legacy) | weights + activations | `quantize/quantize_model_activation.py` | `eval/sweep_quantization.py` |

```bash
# 1. Quantize (calibrated on SafeChain prompts disjoint from the eval split).
cd quantize
python quantize_model.py --model-id l3lab/L1-Qwen-1.5B-Exact --bits 4 \
    --calib-dataset safechain --calib-samples 256 \
    --output-dir ./checkpoints/L1-Qwen-1.5B-Exact-gptq-int4
python quantize_model_awq.py --model-id l3lab/L1-Qwen-1.5B-Exact --scheme W4A16_ASYM \
    --output-dir ./checkpoints/L1-Qwen-1.5B-Exact-awq-w4a16_asym
bash ../scripts/jobs/flatquant_calibrate.sh l3lab/L1-Qwen-1.5B-Exact   # W8A8KV8 + W4A4KV4 matrices
cd ..

# 2. Evaluate GPTQ/AWQ: pass the checkpoint as the model (combines with any budget).
bash scripts/jobs/run_skill_grid.sh ../quantize/checkpoints/L1-Qwen-1.5B-Exact-gptq-int4 \
    L1-Qwen-1.5B-Exact-gptq-int4 0 results/full_eval default 512 1024 2048 4096
bash scripts/jobs/run_safety_grid.sh ../quantize/checkpoints/L1-Qwen-1.5B-Exact-gptq-int4 \
    L1-Qwen-1.5B-Exact-gptq-int4 results/safety_t06 default 512 1024 2048 4096

#    Evaluate FlatQuant: <model> <scheme> <budget> <out dir> <batch size>.
TEMPERATURE=0.6 TOP_P=0.95 bash scripts/jobs/run_flatquant_grid.sh \
    l3lab/L1-Qwen-1.5B-Exact w4a4kv4 1024 results/full_eval_t06 16
bash scripts/jobs/run_flatquant_safety.sh l3lab/L1-Qwen-1.5B-Exact w4a4kv4 1024 results/safety_t06 16

# 3. Score safety as for length control, then build the tables and plots.
cd eval && python score_safety.py results/safety_t06/L1-Qwen-1.5B-Exact-*-budget*/*.parquet
```

GPTQ only runs on CUDA (see `quantize/KNOWN_ISSUES.md` for the Aurora/XPU
failure). The FlatQuant evaluators checkpoint every batch and resume a
half-finished dataset.

### Reporting: tables and compute plots

Run from `eval/` once results exist. Missing runs show as `--` or are left
out, so these also work as status reports while jobs are still running.
Outputs go to `context/`, which git ignores.

| Script | Produces |
|---|---|
| `make_budget_table.py` | skill Acc / Len table per model, budget and precision (`.md`, `.tex`) |
| `make_safety_table.py` | safety tables (ASR, FRR, StrongREJECT; answer / trace) for length control and quantization |
| `plot_acc_vs_flops.py` | accuracy vs mean FLOPs per problem, per dataset + average |
| `plot_safety_vs_flops.py` | ASR, FRR and StrongREJECT vs mean FLOPs |
| `make_plot_data_xlsx.py` | `plot_data.xlsx` with every plotted point (run after the two plot scripts) |

FLOPs come from `utils/flops.py`, an analytical, precision-independent count
of prefill, decode, lm_head and attention operations. Quantization therefore
changes compute only through the number of tokens a model generates.

```bash
cd eval
python make_budget_table.py && python make_safety_table.py
python plot_acc_vs_flops.py && python plot_safety_vs_flops.py && python make_plot_data_xlsx.py
```

## Setup

Install dependencies (see [requirements.md](requirements.md), or
[requirements-xpu.md](requirements-xpu.md) for the Aurora/Intel XPU
environment):

```bash
pip install -r requirements.md
```

The `wildguard` dependency is installed from a local editable path
(`file:///../wildguard`), so a `wildguard/` checkout must exist as a sibling
of this directory (i.e. inside the parent `safety-eval/` repo).

## Running

### Safety evaluation

Run generation and evaluation in one go:

```bash
cd eval
python eval_safety.py --dataset safechain --n 100 --max-tokens 2048
```

Or split into two stages (useful since generation needs a GPU but WildGuard
scoring doesn't):

```bash
# Stage 1: generate responses, save to parquet
python eval_safety.py --generate --dataset wildguard --n 100 \
  --max-tokens 2048 --parquet generations.parquet

# Stage 2: run WildGuard on the saved parquet
python eval_safety.py --evaluate --parquet generations.parquet \
  --output results.json
```

Use `--think-budget N` to constrain reasoning to `N` tokens, `--model-id` to
evaluate a different model (default: `l3lab/L1-Qwen-1.5B-Exact`), and
`--dataset {safechain,wildguard}` to pick the benchmark. On a multi-GPU
node, `--tensor-parallel-size` shards one model replica across GPUs (default
4), and `--num-dp-workers` runs multiple replicas in parallel, sharding the
dataset across them (needs `num-dp-workers * tensor-parallel-size` GPUs).

### Skill evaluation

Same two-stage flow, against AIME / GPQA / LSAT instead of SafeChain /
WildGuardMix:

```bash
cd eval
python eval_skill.py --dataset gpqa --n 100 --max-tokens 4096

# or split into stages:
python eval_skill.py --generate --dataset aime --n 100 \
  --max-tokens 32768 --think-budget 2048 --parquet generations.parquet
python eval_skill.py --evaluate --parquet generations.parquet \
  --output results.json
```

`--dataset` accepts `aime`, `gpqa`, or `lsat`; all other flags
(`--think-budget`, `--model-id`, `--tensor-parallel-size`,
`--num-dp-workers`, etc.) match `eval_safety.py`. Scoring is local
exact-match against each dataset's gold answer — no WildGuard/external
evaluator model is used.

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

### Quantization

Produce a GPTQ weight-only checkpoint (int8/int4/int3):

```bash
cd quantize
python quantize_model.py --model-id l3lab/L1-Qwen-1.5B-Exact --bits 4 \
  --calib-dataset safechain --calib-samples 256 \
  --output-dir ./checkpoints/L1-Qwen-1.5B-Exact-gptq-int4
```

Or a weight-activation checkpoint (W8A8/W4A8) via `llmcompressor`:

```bash
cd quantize
python quantize_model_activation.py --model-id l3lab/L1-Qwen-1.5B-Exact \
  --scheme W8A8 --calib-dataset safechain --calib-samples 256 \
  --output-dir ./checkpoints/L1-Qwen-1.5B-Exact-w8a8
```

Either checkpoint directory can then be passed directly as `--model-id` to
`eval_safety.py` / `eval_skill.py` — vLLM auto-detects the quantization
format from the checkpoint's config. `--calib-dataset` accepts
`{safechain,wildguard}`, matching the safety datasets, since calibration
prompts are drawn from the same distribution used for evaluation.

To compare a base model against several quantized variants in one run, use
`eval/sweep_quantization.py`, either against pre-built checkpoints:

```bash
cd eval
python sweep_quantization.py \
  --bf16-model-id l3lab/L1-Qwen-1.5B-Exact \
  --int8-model-id ../quantize/checkpoints/L1-Qwen-1.5B-Exact-gptq-int8 \
  --int4-model-id ../quantize/checkpoints/L1-Qwen-1.5B-Exact-gptq-int4 \
  --dataset safechain --n 500
```

or letting it quantize on the fly with `--quantize int8 int4 int3 w8a8 w4a8`
instead of passing pre-built checkpoint paths.

**Note:** GPTQ (`quantize_model.py`) currently fails on Aurora's Intel XPU
nodes due to an upstream `gptqmodel`/XPU incompatibility — see
[quantize/KNOWN_ISSUES.md](quantize/KNOWN_ISSUES.md) for the full debugging
trail. It works on CUDA (e.g. Polaris). `quantize_model_activation.py`
(`llmcompressor`) is the untested fallback for weight-activation
quantization on XPU.

### Cluster (PBS) jobs

`scripts/jobs/` contains per-model, per-dataset PBS job scripts. Most target
the ALCF Polaris (CUDA) cluster (hardcoded cluster paths and modules — not
runnable locally); files suffixed `_aurora.sh` target the ALCF Aurora
(Intel XPU) cluster instead. Submit all sweeps at once with:

```bash
bash scripts/jobs/submit_all.sh            # SafeChain
bash scripts/jobs/submit_all_wildguard.sh  # WildGuardMix
```

## Requirements

See [requirements.md](requirements.md) for pinned Python dependencies on
CUDA clusters, or [requirements-xpu.md](requirements-xpu.md) for the
Aurora/Intel XPU environment.
