"""
Retroactive total-compute-cost report: sums analytical FLOPs (see
utils/flops.py) over every ALREADY-COMPLETED precision-sweep parquet file,
using each response's real prompt length (re-tokenized) and token_count.
Requires zero new GPU runs -- this is a pure CPU pass over existing data.

Quantization does not change the algorithmic FLOP count (same architecture,
same number of multiply-adds); it changes which kernel executes the matmul
and therefore achieved throughput, not this total-compute figure. So the
same per-token formula applies to every precision of a given model.

Directory-naming tables copied from
results/precision_sweep/collate_all_results.py so labeling stays
consistent with the existing accuracy tables.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from flops import QWEN2_1_5B, total_flops_for_response  # noqa: E402

DEEPSEEK_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
L1_MODEL_ID = "l3lab/L1-Qwen-1.5B-Exact"

DEEPSEEK_PRECISIONS = [
    ("BF16", ""),
    ("GPTQ-int8", "-gptq-int8"),
    ("GPTQ-int4", "-gptq-int4"),
    ("GPTQ-int3", "-gptq-int3"),
    ("AWQ-W4A16", "-awq-w4a16_asym"),
]
DEEPSEEK_FLATQUANT = [
    ("FlatQuant-W8A8KV8", "flatquant-w8a8kv8"),
    ("FlatQuant-W4A4KV4", "flatquant-w4a4kv4"),
]
L1_PRECISIONS = [
    ("BF16", ""),
    ("GPTQ-int8", "-gptq-int8"),
    ("GPTQ-int4", "-gptq-int4"),
    ("AWQ-W4A16", "-awq-w4a16_asym"),
]
BUDGETS = [512, 1024, 2048, 4096]
DATASETS = ["aime", "gpqa", "math", "gsm8k"]


def iter_runs():
    """Yield (model_label, tokenizer_model_id, precision, budget, dirname) for
    every planned run, matching collate_all_results.py's tables exactly."""
    for precision, suffix in DEEPSEEK_PRECISIONS:
        yield "DeepSeek-R1", DEEPSEEK_MODEL_ID, precision, None, f"DeepSeek-R1-Distill-Qwen-1.5B{suffix}"
    for precision, dirname in DEEPSEEK_FLATQUANT:
        yield "DeepSeek-R1", DEEPSEEK_MODEL_ID, precision, None, dirname
    for precision, suffix in L1_PRECISIONS:
        for budget in BUDGETS:
            yield "L1", L1_MODEL_ID, precision, budget, f"L1-Qwen-1.5B-Exact{suffix}-budget{budget}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/precision_sweep")
    ap.add_argument("--out-dir", default="results/flops_report")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer_cache = {}  # model_id -> AutoTokenizer
    prompt_len_cache = {}  # (model_id, prompt) -> token length

    def get_tokenizer(model_id):
        if model_id not in tokenizer_cache:
            from transformers import AutoTokenizer
            print(f"Loading tokenizer for {model_id}...")
            tokenizer_cache[model_id] = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        return tokenizer_cache[model_id]

    def prompt_len(model_id, prompt):
        key = (model_id, prompt)
        if key not in prompt_len_cache:
            tok = get_tokenizer(model_id)
            prompt_len_cache[key] = len(tok.encode(prompt))
        return prompt_len_cache[key]

    rows = []
    cfg = QWEN2_1_5B

    for model_label, model_id, precision, budget, dirname in iter_runs():
        for ds in DATASETS:
            path = os.path.join(args.results_dir, dirname, f"{ds}.parquet")
            if not os.path.exists(path):
                continue
            df = pd.read_parquet(path, columns=["prompt", "response", "token_count"])
            total_flops = 0.0
            total_backbone = 0.0
            total_lm_head = 0.0
            total_attn = 0.0
            total_output_tokens = 0
            total_prompt_tokens = 0
            for _, row in df.iterrows():
                p_len = prompt_len(model_id, row["prompt"])
                o_len = int(row["token_count"])
                r = total_flops_for_response(cfg, p_len, o_len)
                total_flops += r["total_flops"]
                total_backbone += r["backbone_flops"]
                total_lm_head += r["lm_head_flops"]
                total_attn += r["attention_flops"]
                total_output_tokens += o_len
                total_prompt_tokens += p_len
            rows.append({
                "model": model_label,
                "precision": precision,
                "budget": budget,
                "dataset": ds,
                "n_responses": len(df),
                "total_prompt_tokens": total_prompt_tokens,
                "total_output_tokens": total_output_tokens,
                "total_backbone_flops": total_backbone,
                "total_lm_head_flops": total_lm_head,
                "total_attention_flops": total_attn,
                "total_flops": total_flops,
            })
            print(f"  {model_label} | {precision} | budget={budget} | {ds}: "
                  f"{len(df)} responses, {total_flops:.3e} total FLOPs")

    if not rows:
        print("No completed parquet files found -- nothing to report.")
        return

    df_long = pd.DataFrame(rows).sort_values(
        ["model", "precision", "budget", "dataset"], na_position="first"
    ).reset_index(drop=True)

    long_path = os.path.join(args.out_dir, "flops_by_run.csv")
    df_long.to_csv(long_path, index=False)
    print(f"\nWrote {long_path} ({len(df_long)} rows)")

    # Roll up per (model, precision, budget) across datasets.
    rollup = df_long.groupby(["model", "precision", "budget"], dropna=False).agg(
        n_responses=("n_responses", "sum"),
        total_prompt_tokens=("total_prompt_tokens", "sum"),
        total_output_tokens=("total_output_tokens", "sum"),
        total_flops=("total_flops", "sum"),
    ).reset_index()

    xlsx_path = os.path.join(args.out_dir, "flops_summary.xlsx")
    with pd.ExcelWriter(xlsx_path) as writer:
        df_long.to_excel(writer, sheet_name="by_run", index=False)
        rollup.to_excel(writer, sheet_name="rollup_by_precision", index=False)
        for sheet_name, sheet_df in (("by_run", df_long), ("rollup_by_precision", rollup)):
            ws = writer.sheets[sheet_name]
            for i, col in enumerate(sheet_df.columns):
                max_cell_len = max((len(str(v)) for v in sheet_df[col]), default=0)
                width = max(max_cell_len, len(col)) + 2
                ws.column_dimensions[chr(65 + i)].width = min(width, 40)
    print(f"Wrote {xlsx_path}")


if __name__ == "__main__":
    main()
