"""
Compare a base model against its post-training-quantized variants
(BF16, int8, int4, int3 weight-only GPTQ; W8A8, W4A8 weight-activation;
AWQ weight-only) on UWNSL/SafeChain or allenai/wildguardmix, evaluated with
WildGuard.

Quantized checkpoints can either already exist (see ../quantize/quantize_model.py
for int8/int4/int3, ../quantize/quantize_model_activation.py for W8A8/W4A8,
../quantize/quantize_model_awq.py for AWQ -- pass them via --int8-model-id
etc.), or be produced by this script itself via --quantize, which saves the
resulting checkpoints to --out-dir/checkpoints before generation/evaluation
runs.

Usage (pre-built checkpoints):
    python sweep_quantization.py \
        --bf16-model-id l3lab/L1-Qwen-1.5B-Exact \
        --int8-model-id ./checkpoints/L1-Qwen-1.5B-Exact-gptq-int8 \
        --int4-model-id ./checkpoints/L1-Qwen-1.5B-Exact-gptq-int4 \
        --int3-model-id ./checkpoints/L1-Qwen-1.5B-Exact-gptq-int3 \
        --w8a8-model-id ./checkpoints/L1-Qwen-1.5B-Exact-w8a8 \
        --w4a8-model-id ./checkpoints/L1-Qwen-1.5B-Exact-w4a8 \
        --awq-model-id ./checkpoints/L1-Qwen-1.5B-Exact-awq-w4a16_asym \
        --dataset safechain --n 500

Usage (quantize then evaluate in one go):
    python sweep_quantization.py \
        --bf16-model-id l3lab/L1-Qwen-1.5B-Exact \
        --quantize int8 int4 int3 w8a8 w4a8 awq \
        --dataset safechain --n 500
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from dataset_utils import DATASETS  # noqa: E402


SCRIPT = os.path.join(os.path.dirname(__file__), "eval_safety.py")
QUANTIZE_GPTQ_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "quantize", "quantize_model.py")
QUANTIZE_ACT_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "quantize", "quantize_model_activation.py")
QUANTIZE_AWQ_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "quantize", "quantize_model_awq.py")

PRECISIONS = ("bf16", "int8", "int4", "int3", "w8a8", "w4a8", "awq")
QUANTIZABLE_PRECISIONS = ("int8", "int4", "int3", "w8a8", "w4a8", "awq")
GPTQ_BITS = {"int8": 8, "int4": 4, "int3": 3}
ACTIVATION_SCHEMES = {"w8a8": "W8A8", "w4a8": "W4A8"}
AWQ_SCHEMES = {"awq": "W4A16_ASYM"}


def run(cmd: list[str], label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print("  " + " ".join(cmd))
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"ERROR: {label} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def quantize_stage(precisions: list[str], bf16_model_id: str, calib_dataset: str,
                    calib_samples: int, seed: int, out_dir: str) -> dict[str, str]:
    """Quantize bf16_model_id to each requested precision and save the
    resulting checkpoints under out_dir/checkpoints. Returns a dict mapping
    precision -> checkpoint path."""
    checkpoints_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    checkpoint_paths = {}
    for precision in precisions:
        output_dir = os.path.join(checkpoints_dir, precision)
        if precision in GPTQ_BITS:
            cmd = [QUANTIZE_GPTQ_SCRIPT,
                   "--model-id", bf16_model_id,
                   "--bits", str(GPTQ_BITS[precision]),
                   "--calib-dataset", calib_dataset,
                   "--calib-samples", str(calib_samples),
                   "--seed", str(seed),
                   "--output-dir", output_dir]
        elif precision in AWQ_SCHEMES:
            cmd = [QUANTIZE_AWQ_SCRIPT,
                   "--model-id", bf16_model_id,
                   "--scheme", AWQ_SCHEMES[precision],
                   "--calib-dataset", calib_dataset,
                   "--calib-samples", str(calib_samples),
                   "--seed", str(seed),
                   "--output-dir", output_dir]
        else:
            cmd = [QUANTIZE_ACT_SCRIPT,
                   "--model-id", bf16_model_id,
                   "--scheme", ACTIVATION_SCHEMES[precision],
                   "--calib-dataset", calib_dataset,
                   "--calib-samples", str(calib_samples),
                   "--seed", str(seed),
                   "--output-dir", output_dir]
        run(cmd, f"Quantize | precision={precision} | model={bf16_model_id}")
        checkpoint_paths[precision] = output_dir
    return checkpoint_paths


def generate_stage(model_ids: dict[str, str], n: int, max_tokens: int, seed: int,
                    out_dir: str, dataset: str, think_budget: int | None):
    for precision, model_id in model_ids.items():
        parquet = os.path.join(out_dir, f"gen_{precision}.parquet")
        cmd = [SCRIPT,
               "--generate",
               "--dataset", dataset,
               "--model-id", model_id,
               "--n", str(n),
               "--max-tokens", str(max_tokens),
               "--parquet", parquet,
               "--seed", str(seed)]
        if think_budget is not None:
            cmd += ["--think-budget", str(think_budget)]
        run(cmd, f"Generate | precision={precision} | model={model_id}")


def evaluate_stage(model_ids: dict[str, str], out_dir: str) -> list[tuple[str, str]]:
    result_files = []
    for precision in model_ids:
        parquet = os.path.join(out_dir, f"gen_{precision}.parquet")
        output = os.path.join(out_dir, f"results_{precision}.json")
        run(
            [SCRIPT, "--evaluate", "--parquet", parquet, "--output", output],
            f"Evaluate | precision={precision}",
        )
        result_files.append((precision, output))
    return result_files


def summarize(result_files: list[tuple[str, str]], out_dir: str) -> pd.DataFrame:
    rows = []
    per_label_rows = []

    for precision, path in result_files:
        if not os.path.exists(path):
            print(f"Warning: result file not found: {path}")
            continue
        with open(path) as f:
            r = json.load(f)

        harmful = r["metrics"]["harmful_prompts"]
        benign = r["metrics"]["benign_prompts"]

        rows.append({
            "precision": precision,
            "model": r["meta"]["model"],
            "attack_success_%": harmful.get("attack_success_rate_%"),
            "refusal_%": harmful.get("refusal_rate_%"),
            "false_refusal_%": benign.get("false_refusal_rate_%"),
            "benign_response_harmful_%": benign.get("response_flagged_harmful_%"),
        })

        for label, stats in r["metrics"]["per_label"].items():
            per_label_rows.append({
                "precision": precision,
                "label": label,
                "total": stats.get("total"),
                "response_harmful_%": stats.get("response_harmful_%"),
                "refusal_%": stats.get("refusal_%"),
                "parse_errors": stats.get("parse_errors"),
            })

    df = pd.DataFrame(rows)
    df_per_label = pd.DataFrame(per_label_rows)

    print(f"\n{'='*60}")
    print("  QUANTIZATION SWEEP SUMMARY")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    excel_path = os.path.join(out_dir, "quantization_sweep_results.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Summary", index=False)
        df_per_label.to_excel(writer, sheet_name="Per-Label", index=False)
        for sheet_name, sheet_df in [("Summary", df), ("Per-Label", df_per_label)]:
            ws = writer.sheets[sheet_name]
            for col_idx, col in enumerate(sheet_df.columns, 1):
                max_len = max(len(str(col)), sheet_df[col].fillna("").astype(str).str.len().max())
                ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = max_len + 2

    print(f"Excel saved to {excel_path}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compare BF16/int8/int4/int3/W8A8/W4A8/AWQ checkpoints on safety evals")
    parser.add_argument("--dataset", type=str, default="safechain", choices=("safechain", "wildguard"))
    parser.add_argument("--bf16-model-id", type=str, default=None,
                        help="Base BF16 model ID or path (unquantized)")
    parser.add_argument("--int8-model-id", type=str, default=None,
                        help="Path to the GPTQ int8 checkpoint")
    parser.add_argument("--int4-model-id", type=str, default=None,
                        help="Path to the GPTQ int4 checkpoint")
    parser.add_argument("--int3-model-id", type=str, default=None,
                        help="Path to the GPTQ int3 checkpoint")
    parser.add_argument("--w8a8-model-id", type=str, default=None,
                        help="Path to the W8A8 (weight-activation) checkpoint")
    parser.add_argument("--w4a8-model-id", type=str, default=None,
                        help="Path to the W4A8 (weight-activation) checkpoint")
    parser.add_argument("--awq-model-id", type=str, default=None,
                        help="Path to the AWQ (weight-only) checkpoint")
    parser.add_argument("--quantize", type=str, default=None, nargs="+",
                        choices=QUANTIZABLE_PRECISIONS,
                        help="Quantize --bf16-model-id to these precisions and save the "
                             "resulting checkpoints to --out-dir/checkpoints before "
                             "generation/evaluation, instead of pointing at pre-built "
                             "checkpoints with --int8-model-id etc.")
    parser.add_argument("--calib-dataset", type=str, default=None, choices=DATASETS,
                        help="Dataset to draw quantization calibration prompts from "
                             "(default: same as --dataset)")
    parser.add_argument("--calib-samples", type=int, default=256,
                        help="Number of calibration prompts for --quantize (default: 256)")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--think-budget", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Skip generation, only run WildGuard on existing parquets in --out-dir")
    args = parser.parse_args()

    if args.quantize and not args.bf16_model_id:
        print("Error: --quantize requires --bf16-model-id (the base model to quantize)")
        sys.exit(1)

    model_ids = {
        "bf16": args.bf16_model_id,
        "int8": args.int8_model_id,
        "int4": args.int4_model_id,
        "int3": args.int3_model_id,
        "w8a8": args.w8a8_model_id,
        "w4a8": args.w4a8_model_id,
        "awq": args.awq_model_id,
    }
    model_ids = {k: v for k, v in model_ids.items() if v is not None}
    if not model_ids and not args.quantize:
        print("Error: provide at least one of --bf16-model-id/--int8-model-id/--int4-model-id/"
              "--int3-model-id/--w8a8-model-id/--w4a8-model-id/--awq-model-id, or --quantize")
        sys.exit(1)

    out_dir = args.out_dir or f"quant_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print(f"Dataset: {args.dataset}")

    if args.quantize:
        quantized_paths = quantize_stage(
            precisions=args.quantize,
            bf16_model_id=args.bf16_model_id,
            calib_dataset=args.calib_dataset or args.dataset,
            calib_samples=args.calib_samples,
            seed=args.seed,
            out_dir=out_dir,
        )
        for precision, path in quantized_paths.items():
            model_ids.setdefault(precision, path)

    print(f"Precisions: {list(model_ids)}")

    if not args.evaluate_only:
        generate_stage(
            model_ids=model_ids,
            n=args.n,
            max_tokens=args.max_tokens,
            seed=args.seed,
            out_dir=out_dir,
            dataset=args.dataset,
            think_budget=args.think_budget,
        )

    result_files = evaluate_stage(model_ids=model_ids, out_dir=out_dir)
    summarize(result_files, out_dir)


if __name__ == "__main__":
    main()
