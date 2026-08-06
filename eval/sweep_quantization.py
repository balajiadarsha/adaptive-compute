"""
Compare a base model against its post-training-quantized variants
(BF16, int8, int4, int3 weight-only GPTQ; W8A8, W4A8 weight-activation)
on UWNSL/SafeChain or allenai/wildguardmix, evaluated with WildGuard.

Quantized checkpoints must already exist (see ../quantize/quantize_model.py
for int8/int4/int3, ../quantize/quantize_model_activation.py for W8A8/W4A8).
This script only runs eval_safety.py against each --model-id and summarizes
the results; it does not quantize anything itself.

Usage:
    python sweep_quantization.py \
        --bf16-model-id l3lab/L1-Qwen-1.5B-Exact \
        --int8-model-id ./checkpoints/L1-Qwen-1.5B-Exact-gptq-int8 \
        --int4-model-id ./checkpoints/L1-Qwen-1.5B-Exact-gptq-int4 \
        --int3-model-id ./checkpoints/L1-Qwen-1.5B-Exact-gptq-int3 \
        --w8a8-model-id ./checkpoints/L1-Qwen-1.5B-Exact-w8a8 \
        --w4a8-model-id ./checkpoints/L1-Qwen-1.5B-Exact-w4a8 \
        --dataset safechain --n 500
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd


SCRIPT = os.path.join(os.path.dirname(__file__), "eval_safety.py")

PRECISIONS = ("bf16", "int8", "int4", "int3", "w8a8", "w4a8")


def run(cmd: list[str], label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print("  " + " ".join(cmd))
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"ERROR: {label} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


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
        description="Compare BF16/int8/int4/int3/W8A8/W4A8 checkpoints on safety evals")
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
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--think-budget", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Skip generation, only run WildGuard on existing parquets in --out-dir")
    args = parser.parse_args()

    model_ids = {
        "bf16": args.bf16_model_id,
        "int8": args.int8_model_id,
        "int4": args.int4_model_id,
        "int3": args.int3_model_id,
        "w8a8": args.w8a8_model_id,
        "w4a8": args.w4a8_model_id,
    }
    model_ids = {k: v for k, v in model_ids.items() if v is not None}
    if not model_ids:
        print("Error: provide at least one of --bf16-model-id/--int8-model-id/--int4-model-id/"
              "--int3-model-id/--w8a8-model-id/--w4a8-model-id")
        sys.exit(1)

    out_dir = args.out_dir or f"quant_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print(f"Dataset: {args.dataset}")
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
