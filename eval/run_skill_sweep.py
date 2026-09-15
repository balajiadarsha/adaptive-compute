"""
Run eval_skill.py across a matrix of {model} x {dataset} and aggregate
accuracy / output-token / think-token stats into one summary table.

Precisions with a portable checkpoint (BF16, GPTQ, AWQ) are just model IDs
or checkpoint directories passed via --models -- eval_skill.py loads them
through vLLM identically either way. FlatQuant (W8A8KV8 / W4A4KV4) is not a
vLLM-loadable checkpoint, so those precisions are evaluated separately via
--flatquant-schemes, which shells out to eval_skill_flatquant.py (HF
transformers generation, no vLLM) under a *different* python interpreter --
the dedicated `flatquant` conda env -- since it depends on the vendored
FlatQuant package and that env's own transformers/torch versions.

Usage:
    python run_skill_sweep.py --n 50 --out-dir results/skill_eval_20260820

    python run_skill_sweep.py --n 50 --out-dir results/precision_sweep \
        --models deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
                 ../quantize/checkpoints/DeepSeek-R1-Distill-Qwen-1.5B-gptq-int8 \
                 ../quantize/checkpoints/DeepSeek-R1-Distill-Qwen-1.5B-gptq-int4 \
                 ../quantize/checkpoints/DeepSeek-R1-Distill-Qwen-1.5B-gptq-int3 \
                 ../quantize/checkpoints/DeepSeek-R1-Distill-Qwen-1.5B-awq-w4a16_asym \
        --flatquant-schemes w8a8kv8 w4a4kv4 \
        --datasets aime gpqa math gsm8k
"""
import argparse
import json
import os
import subprocess
import sys

import pandas as pd

SCRIPT = os.path.join(os.path.dirname(__file__), "eval_skill.py")
FLATQUANT_SCRIPT = os.path.join(os.path.dirname(__file__), "eval_skill_flatquant.py")

# The flatquant conda env's interpreter -- eval_skill_flatquant.py imports
# the vendored quantize/flatquant package and needs that env's pinned
# transformers/torch, not this repo's usual `reason` env.
DEFAULT_FLATQUANT_PYTHON = os.path.expanduser("~/.conda/envs/flatquant/bin/python")

MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "agentica-org/DeepScaleR-1.5B-Preview",
    "l3lab/L1-Qwen-1.5B-Exact",
]

DATASETS = ("aime", "gpqa", "lsat", "math", "gsm8k")
FLATQUANT_SCHEMES = ("w8a8kv8", "w4a4kv4")


# eval_skill.py hardcodes max_model_len=32768; the AIME budget must leave
# headroom for the prompt itself or vLLM raises a context-length error.
DEFAULT_MAX_TOKENS = {"aime": 30000, "gpqa": 8192, "lsat": 4096, "math": 8192, "gsm8k": 4096}

# FlatQuant generates with plain HF `model.generate()` (no continuous
# batching/PagedAttention) at batch_size=4, which makes AIME's 30000-token
# budget impractically slow -- empirically, a single 4-prompt batch didn't
# finish within a 1-hour interactive allocation. Cap AIME lower for FlatQuant
# specifically so this is at least tractable; every other precision (vLLM,
# fast) still gets the full 30000 for aime, so those results aren't affected.
FLATQUANT_MAX_TOKENS = {**DEFAULT_MAX_TOKENS, "aime": 8192}


def run(cmd: list[str], label: str, python_exe: str = sys.executable):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print("  " + " ".join(cmd))
    result = subprocess.run([python_exe] + cmd)
    if result.returncode != 0:
        print(f"ERROR: {label} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def slug(model_id: str) -> str:
    return model_id.split("/")[-1]


def split_think_answer(response: str) -> tuple[str, str]:
    """Split a raw model response into its <think>...</think> content and the
    remainder. Mirrors eval_skill.py's think-token accounting: if the think
    block is unclosed (generation was truncated before </think>), the whole
    response counts as thinking and there is no separate answer text."""
    idx = response.find("</think>")
    if idx == -1:
        return response.strip(), ""
    think = response[:idx].replace("<think>", "", 1).strip()
    answer = response[idx + len("</think>"):].strip()
    return think, answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=MODELS,
                        help="Model IDs / checkpoint dirs to evaluate via vLLM. Pass with no "
                             "values (--models) to run only --flatquant-schemes.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip (model, dataset) pairs whose results.json already exists")
    parser.add_argument("--flatquant-schemes", nargs="+", default=[], choices=FLATQUANT_SCHEMES,
                        help="Also evaluate these FlatQuant precisions via eval_skill_flatquant.py "
                             "(runs under --flatquant-python, not this process' interpreter)")
    parser.add_argument("--flatquant-python", type=str, default=DEFAULT_FLATQUANT_PYTHON,
                        help="Python interpreter (flatquant conda env) for --flatquant-schemes runs")
    parser.add_argument("--flatquant-batch-size", type=int, default=4,
                        help="HF generate() batch size for FlatQuant runs (no vLLM batching)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    response_rows = []

    for model_id in args.models:
        model_slug = slug(model_id)
        model_dir = os.path.join(args.out_dir, model_slug)
        os.makedirs(model_dir, exist_ok=True)

        for dataset in args.datasets:
            parquet = os.path.join(model_dir, f"{dataset}.parquet")
            output = os.path.join(model_dir, f"{dataset}.json")
            label = f"{model_slug} | {dataset}"

            if args.skip_existing and os.path.exists(output):
                print(f"Skipping {label} (results exist at {output})")
            else:
                run(
                    [SCRIPT,
                     "--dataset", dataset,
                     "--model-id", model_id,
                     "--n", str(args.n),
                     "--max-tokens", str(DEFAULT_MAX_TOKENS[dataset]),
                     "--seed", str(args.seed),
                     "--tensor-parallel-size", str(args.tensor_parallel_size),
                     "--parquet", parquet,
                     "--output", output],
                    label,
                )

            if not os.path.exists(output):
                print(f"Warning: missing results for {label}, skipping in summary")
                continue

            with open(output) as f:
                r = json.load(f)

            token_counts = [d["token_count"] for d in r["data"] if d["token_count"] is not None]
            mean_output_tokens = round(sum(token_counts) / len(token_counts), 1) if token_counts else None

            for d in r["data"]:
                think, answer = split_think_answer(d["response"])
                response_rows.append({
                    "model": model_id,
                    "dataset": dataset,
                    "question": d["instruction"],
                    "think": think,
                    "answer": answer,
                    "think_token_count": d["think_token_count"],
                    "total_output_tokens": d["token_count"],
                })

            rows.append({
                "model": model_id,
                "dataset": dataset,
                "n": r["metrics"]["total"],
                "accuracy_%": r["metrics"]["accuracy_%"],
                "accuracy_%_incl_parse_errors": r["metrics"]["accuracy_%_incl_parse_errors"],
                "parse_errors": r["metrics"]["parse_errors"],
                "mean_output_tokens": mean_output_tokens,
                "mean_think_tokens": r["token_budget_stats"].get("mean_tokens"),
            })

    for scheme in args.flatquant_schemes:
        model_slug = f"flatquant-{scheme}"
        model_dir = os.path.join(args.out_dir, model_slug)
        os.makedirs(model_dir, exist_ok=True)

        for dataset in args.datasets:
            parquet = os.path.join(model_dir, f"{dataset}.parquet")
            output = os.path.join(model_dir, f"{dataset}.json")
            label = f"{model_slug} | {dataset}"

            if args.skip_existing and os.path.exists(output):
                print(f"Skipping {label} (results exist at {output})")
            else:
                run(
                    [FLATQUANT_SCRIPT,
                     "--dataset", dataset,
                     "--scheme", scheme,
                     "--n", str(args.n),
                     "--max-tokens", str(FLATQUANT_MAX_TOKENS[dataset]),
                     "--seed", str(args.seed),
                     "--batch-size", str(args.flatquant_batch_size),
                     "--parquet", parquet,
                     "--output", output],
                    label,
                    python_exe=args.flatquant_python,
                )

            if not os.path.exists(output):
                print(f"Warning: missing results for {label}, skipping in summary")
                continue

            with open(output) as f:
                r = json.load(f)

            token_counts = [d["token_count"] for d in r["data"] if d["token_count"] is not None]
            mean_output_tokens = round(sum(token_counts) / len(token_counts), 1) if token_counts else None

            for d in r["data"]:
                think, answer = split_think_answer(d["response"])
                response_rows.append({
                    "model": model_slug,
                    "dataset": dataset,
                    "question": d["instruction"],
                    "think": think,
                    "answer": answer,
                    "think_token_count": d["think_token_count"],
                    "total_output_tokens": d["token_count"],
                })

            rows.append({
                "model": model_slug,
                "dataset": dataset,
                "n": r["metrics"]["total"],
                "accuracy_%": r["metrics"]["accuracy_%"],
                "accuracy_%_incl_parse_errors": r["metrics"]["accuracy_%_incl_parse_errors"],
                "parse_errors": r["metrics"]["parse_errors"],
                "mean_output_tokens": mean_output_tokens,
                "mean_think_tokens": r["token_budget_stats"].get("mean_tokens"),
            })

    responses_df = pd.DataFrame(response_rows)
    responses_path = os.path.join(args.out_dir, "responses.parquet")
    responses_df.to_parquet(responses_path, index=False)
    print(f"\nPer-response data saved to {responses_path} ({len(responses_df)} rows)")

    df = pd.DataFrame(rows)
    print(f"\n{'=' * 70}")
    print("  SWEEP SUMMARY")
    print(f"{'=' * 70}")
    print(df.to_string(index=False))

    csv_path = os.path.join(args.out_dir, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")

    excel_path = os.path.join(args.out_dir, "summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Summary", index=False)
        ws = writer.sheets["Summary"]
        for col_idx, col in enumerate(df.columns, 1):
            max_len = max(len(str(col)), df[col].fillna("").astype(str).str.len().max())
            ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = max_len + 2
    print(f"Summary saved to {excel_path}")


if __name__ == "__main__":
    main()
