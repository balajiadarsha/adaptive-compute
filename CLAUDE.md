# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo studies

Two independent axes of "adaptive compute" for 1.5B reasoning models
(`l3lab/L1-Qwen-1.5B-Exact`, `agentica-org/DeepScaleR-1.5B-Preview`,
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`), measured against the same two
metric families:

- **Safety** — responses to SafeChain / WildGuardMix prompts, classified by
  WildGuard for attack success rate, refusal rate, false refusal rate.
- **Skill** — responses to AIME / GPQA / LSAT, scored by local exact-match
  against each dataset's gold answer (no external evaluator model).

The two axes:
1. **Reasoning token budget** — models are prompted with `Think for N
   tokens.` at varying budgets to see the safety-vs-capability tradeoff as
   thinking is constrained.
2. **Post-training quantization** — GPTQ weight-only (int8/int4/int3) and
   W8A8/W4A8 weight-activation quantization, to see whether compute savings
   from quantization compose with those from budget constraints.

Every eval records both `token_count` (total output) and
`think_token_count` (tokens inside `<think>...</think>`; if unclosed by
`max_tokens`, the whole generation counts as thinking) per response, so
token cost and task metrics are always paired.

## Repo location matters

This directory is a subdirectory of a parent `safety-eval` checkout — PBS
job scripts reference `${PROJ_DIR}/adaptive-compute/eval` and the
`wildguard` pip dependency is installed from `file:///../wildguard`, i.e. a
sibling checkout one level up from this repo root. `eval_safety.py` and
`sweep_quantization.py` (which classifies with WildGuard) will fail to
import without that sibling checkout present.

## Setup

```bash
pip install -r requirements.md   # yes, .md — it's a real pip requirements file
```

Use `requirements-xpu.md` instead on Aurora/Intel XPU nodes. Known pip
conflict: `vllm>=0.27` pulls `protobuf 6.x`, older than what `gptqmodel`
wants (`>=7.34`); pip warns but doesn't fail, and it only matters if
`quantize_model.py` runs in the same env as eval.

`eval_skill.py` reads `aime`/`gpqa`/`lsat` from local disk
(`data/aime_2025`, `data/gpqa_diamond`, `data/agieval_lsat_lr` — see
`LOCAL_DATASET_PATH` in `eval/eval_skill.py`) via `datasets.load_from_disk`
rather than the HF Hub, since compute nodes on this cluster have
unreliable/slow Hub access. GPQA (`Idavidrein/gpqa`) is itself a gated HF
dataset, so populating `data/gpqa_diamond` (via
`datasets.Dataset.save_to_disk`, run once from a node with Hub access and
an authenticated `HF_TOKEN` with accepted terms on that dataset page) is a
prerequisite before `--dataset gpqa` will load.

## Commands

All `eval/*.py` scripts are run from inside `eval/`. All take `--model-id`
(defaults to `l3lab/L1-Qwen-1.5B-Exact`), `--think-budget N` (omit for
unconstrained reasoning), `--tensor-parallel-size` (GPUs per model
replica), and `--num-dp-workers` (parallel model replicas, each getting
`--tensor-parallel-size` GPUs, sharding the dataset across replicas —
requires `num-dp-workers * tensor-parallel-size` GPUs on the node).

```bash
cd eval

# Safety eval (SafeChain or WildGuardMix), one-shot:
python eval_safety.py --dataset safechain --n 100 --max-tokens 2048

# Skill eval (aime / gpqa / lsat), one-shot:
python eval_skill.py --dataset aime --n 100 --max-tokens 32768

# Two-stage form (either script): generation needs GPU, scoring often doesn't
python eval_safety.py --generate --dataset wildguard --n 100 --max-tokens 2048 --parquet gen.parquet
python eval_safety.py --evaluate --parquet gen.parquet --output results.json

# Budget sweep across several --think-budget values, with Excel + plot output:
python sweep_budget.py --dataset safechain --n 500 --budgets 256 512 1024 2048 4096

# Quantization comparison (BF16 vs int8/int4/int3/W8A8/W4A8) on a safety dataset:
python sweep_quantization.py --bf16-model-id l3lab/L1-Qwen-1.5B-Exact \
  --int8-model-id ../quantize/checkpoints/L1-Qwen-1.5B-Exact-gptq-int8 \
  --dataset safechain --n 500
# or quantize on the fly: --quantize int8 int4 int3 w8a8 w4a8
```

```bash
cd quantize

# GPTQ weight-only (CUDA only — fails on Aurora XPU, see KNOWN_ISSUES.md):
python quantize_model.py --model-id l3lab/L1-Qwen-1.5B-Exact --bits 4 \
  --calib-dataset safechain --calib-samples 256 \
  --output-dir ./checkpoints/L1-Qwen-1.5B-Exact-gptq-int4

# W8A8/W4A8 weight-activation (llmcompressor):
python quantize_model_activation.py --model-id l3lab/L1-Qwen-1.5B-Exact \
  --scheme W8A8 --calib-dataset safechain --calib-samples 256 \
  --output-dir ./checkpoints/L1-Qwen-1.5B-Exact-w8a8
```

A produced checkpoint directory can be passed straight back in as
`--model-id` to `eval_safety.py` / `eval_skill.py` — vLLM auto-detects the
quantization format from the checkpoint config, no extra flags needed.

Cluster jobs: `bash scripts/jobs/submit_all.sh` (SafeChain) and
`bash scripts/jobs/submit_all_wildguard.sh` (WildGuardMix) submit the
per-model PBS sweeps. There is no local test suite — correctness is
checked by running a small `--n` eval and inspecting the JSON output.

## Architecture

**Two-stage generate/evaluate split.** Both `eval_safety.py` and
`eval_skill.py` follow the same shape: `--generate` runs vLLM and writes a
parquet (prompt, response, token_count, think_token_count, think_budget,
dataset); `--evaluate` reads that parquet and scores it, needing no GPU for
skill eval (local regex/exact-match) but still needing a GPU for safety
eval (WildGuard itself runs via vLLM). Omitting both flags runs both
stages in one process. This split exists so the two independently
scriptable stages (generation is GPU-bound and slow, scoring often isn't)
can be pipelined or rerun independently — `sweep_budget.py`'s
`--evaluate-only` flag depends on this.

**Data-parallel generation.** `generate_responses_data_parallel` spawns
`num_dp_workers` separate processes (via `torch.multiprocessing`, spawn
context), each pinned to a `CUDA_VISIBLE_DEVICES` slice of
`tensor_parallel_size` GPUs and given a round-robin shard of the dataset
(not contiguous, so each worker sees a representative mix if the source
data is ordered by label/category), then reassembles results back into
original item order via `shard_indices`. This is a separate path from
plain `tensor_parallel_size`-only generation (`generate_responses`), used
when you want multiple independent model replicas instead of one model
sharded across all GPUs.

**Dataset loaders reshape everything to a common schema** so the rest of
the pipeline is dataset-agnostic:
- Safety (`utils/dataset_utils.py`): `{instruction, label}` where `label`
  is one of `vanilla_harmful` / `adversarial_harmful` / `vanilla_benign` /
  `adversarial_benign` (SafeChain and WildGuardMix both get mapped here).
- Skill (`eval/eval_skill.py`): `{instruction, answer, kind, choices}`
  where `kind` is `"numeric"` (AIME, matched via `\boxed{...}` → int) or
  `"mcq"` (GPQA/LSAT, matched via `\boxed{...}` → letter A-H; GPQA also
  shuffles its 4 answer choices per-example with a seeded RNG so the
  correct letter isn't positionally biased).
- `utils/dataset_utils.py` is shared between `eval_safety.py` and the
  `quantize/` scripts, since GPTQ/W8A8 calibration prompts are drawn from
  the same SafeChain/WildGuardMix distribution used for eval.

**Think-token counting** (`_think_token_counts` in both eval scripts):
finds `</think>` in the decoded response and re-encodes only the prefix up
to and including it; if the tag never appears (budget/max-tokens truncated
generation before the model closed its thinking), the entire response
counts as thinking tokens. This function is duplicated between
`eval_safety.py` and `eval_skill.py` rather than shared.

**Sweep scripts orchestrate the eval scripts as subprocesses**, not by
importing them — `sweep_budget.py` and `sweep_quantization.py` both shell
out to `eval_safety.py` per configuration via `subprocess.run`, then
aggregate the resulting `results.json` files into a pandas DataFrame,
Excel workbook, and matplotlib figure. Follow this same subprocess +
aggregate pattern for any new multi-run sweep rather than importing
`generate_responses`/`classify_safety` directly.

**Cluster job scripts are not portable.** `scripts/jobs/*.sh` (Polaris/PBS)
hardcode ALCF paths (`/lus/eagle/projects/argonne_tpc/...`), proxy
settings, and `module load` calls; `*_aurora.sh` variants target Aurora's
Intel XPU nodes instead, using `requirements-xpu.md`. GPTQ quantization
(`quantize_model.py`, via `gptqmodel`) does not currently work on Aurora
XPU — see `quantize/KNOWN_ISSUES.md` for the debugging trail;
`quantize_model_activation.py` (`llmcompressor`) is the untested XPU
fallback for weight-activation schemes.
