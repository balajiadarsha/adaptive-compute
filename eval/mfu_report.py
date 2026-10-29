"""
Combines measured throughput (tok/s, from bench_throughput_vllm.py and
bench_throughput_flatquant.py) with the analytical FLOPs/token formula
(utils/flops.py) to produce achieved-FLOP/s and Model FLOPs Utilization
(MFU%) for every measured precision, plus the retroactive total-FLOPs-per-
completed-eval-run figure from flops_report.py.

Hardware peak reference: NVIDIA A100 (both PCIe-40GB and SXM4-80GB variants
share the same GA100 tensor-core peak specs; only memory bandwidth/power
cap differ between them). BF16/FP16 dense tensor-core peak = 312 TFLOPS.

CRITICAL: GPTQ/AWQ are WEIGHT-ONLY quantization in vLLM -- the matmul is
dequantized to bf16 before the GEMM runs (e.g. via Marlin kernels), so
their compute roofline is the BF16 peak, NOT a hypothetical INT4/INT8 peak.
Their throughput benefit at low batch size comes from reduced HBM traffic
(smaller weights), i.e. they are typically memory-bandwidth-bound, not
compute-bound. FlatQuant's activation quantization here is confirmed
simulated fake-quant (see bench_throughput_flatquant.py's docstring) --
also executes in bf16, so it also uses the bf16 roofline, not int8/int4.
Every execution-kernel-precision entry below is therefore "bf16" today;
the int8 peak is kept in the table only so a future reader who verifies a
REAL low-precision kernel exists can flip one entry, with this comment
explaining why it isn't flipped now.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from flops import QWEN2_1_5B, total_flops_for_response  # noqa: E402

A100_PEAK_TFLOPS = {
    "bf16": 312.0,
    "fp16": 312.0,
    "tf32": 156.0,
    "fp32": 19.5,
    "int8": 624.0,  # TOPS -- only meaningful for a REAL int8 GEMM kernel; unused today, see module docstring
}

# Every precision measured in this repo executes its GEMM in bf16 (weight-
# only dequant for GPTQ/AWQ; confirmed fake-quant-to-bf16 for FlatQuant).
EXECUTION_KERNEL_PRECISION = {
    "BF16": "bf16",
    "GPTQ-int8": "bf16",
    "GPTQ-int4": "bf16",
    "AWQ-W4A16": "bf16",
    "FlatQuant-W8A8KV8": "bf16",
    "FlatQuant-W4A4KV4": "bf16",
}

# Physical-plausibility thresholds (see verification plan): if these are
# crossed, print a loud warning -- it almost always means a units bug
# (e.g. FLOPs computed per-1-op instead of per-2-op, or TFLOPS/TOPS mixup)
# rather than a genuine result, since 1.5B-model small-batch decode is
# normally deep in the memory-bound regime (single-digit-% MFU).
MFU_IMPLAUSIBLE_PCT = 90.0
MFU_RATIO_IMPLAUSIBLE = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--throughput-csv", default="results/throughput_sweep/throughput_summary.csv")
    ap.add_argument("--flatquant-csv", default="results/throughput_sweep/flatquant_throughput.csv")
    ap.add_argument("--flops-summary", default="results/flops_report/flops_summary.xlsx")
    ap.add_argument("--out", default="results/throughput_sweep/mfu_summary.csv")
    args = ap.parse_args()

    cfg = QWEN2_1_5B
    frames = []

    if os.path.exists(args.throughput_csv):
        frames.append(pd.read_csv(args.throughput_csv))
    if os.path.exists(args.flatquant_csv):
        fq_df = pd.read_csv(args.flatquant_csv)
        # Align columns with the vLLM throughput CSV (no requests_per_second/
        # total_tokens_per_second for FlatQuant -- HF generate() doesn't
        # report those the way vllm bench throughput does).
        frames.append(fq_df)

    if not frames:
        print("No throughput CSVs found -- run bench_throughput_vllm.py / "
              "bench_throughput_flatquant.py first.")
        return

    df = pd.concat(frames, ignore_index=True)

    def compute_row(row):
        flops = total_flops_for_response(cfg, int(row["input_len"]), int(row["output_len"]))["total_flops"]
        # achieved FLOP/s = FLOPs-per-request * requests/s, equivalently
        # per-request-FLOPs / (elapsed_time / num_prompts)
        requests_per_s = row["num_prompts"] / row["elapsed_time_s"]
        achieved_flops_s = flops * requests_per_s
        kernel = EXECUTION_KERNEL_PRECISION.get(row["precision"], "bf16")
        peak_flops_s = A100_PEAK_TFLOPS[kernel] * 1e12
        mfu_pct = 100 * achieved_flops_s / peak_flops_s
        return pd.Series({
            "flops_per_request": flops,
            "achieved_tflops_s": achieved_flops_s / 1e12,
            "execution_kernel": kernel,
            "hw_peak_tflops": A100_PEAK_TFLOPS[kernel],
            "mfu_pct": mfu_pct,
        })

    df = pd.concat([df, df.apply(compute_row, axis=1)], axis=1)

    # Physical-plausibility cross-check.
    warnings = []
    if (df["mfu_pct"] > MFU_IMPLAUSIBLE_PCT).any():
        bad = df[df["mfu_pct"] > MFU_IMPLAUSIBLE_PCT]
        warnings.append(
            f"WARNING: {len(bad)} row(s) have MFU% > {MFU_IMPLAUSIBLE_PCT}% -- physically implausible "
            f"for small-batch decode on a 1.5B model. Likely a units bug (FLOPs-per-op convention or "
            f"TFLOPS/TOPS mixup). Rows:\n{bad[['model','precision','length_preset','mfu_pct']].to_string(index=False)}"
        )
    bf16_mfu = df[df["precision"] == "BF16"].set_index(["model", "length_preset"])["mfu_pct"]
    for _, row in df.iterrows():
        key = (row["model"], row["length_preset"])
        if row["precision"] != "BF16" and key in bf16_mfu.index:
            ratio = row["mfu_pct"] / bf16_mfu.loc[key] if bf16_mfu.loc[key] > 0 else float("inf")
            if ratio > MFU_RATIO_IMPLAUSIBLE:
                warnings.append(
                    f"WARNING: {row['model']} {row['precision']} @ {row['length_preset']} has MFU "
                    f"{ratio:.1f}x higher than BF16 at the same length preset -- GPTQ/AWQ/FlatQuant "
                    f"execute on the SAME bf16 kernel as BF16, so MFU should not be drastically higher "
                    f"(tok/s can be higher due to memory-bandwidth savings, but MFU shouldn't blow past "
                    f"BF16's by more than ~{MFU_RATIO_IMPLAUSIBLE}x). Check for a units bug."
                )

    # Join with retroactive total-FLOPs-per-completed-eval-run, rolled up
    # per (model, precision) across datasets/budgets -- NaN where no
    # completed accuracy run exists for that precision (no fabrication).
    if os.path.exists(args.flops_summary):
        rollup = pd.read_excel(args.flops_summary, sheet_name="rollup_by_precision")
        rollup_total = rollup.groupby(["model", "precision"])["total_flops"].sum().reset_index()
        rollup_total = rollup_total.rename(columns={"total_flops": "total_flops_completed_eval_run"})
        df = df.merge(rollup_total, on=["model", "precision"], how="left")

    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(df)} rows)")
    print(df.to_string(index=False))

    if warnings:
        print("\n" + "=" * 70)
        for w in warnings:
            print(w)
        print("=" * 70)
    else:
        print("\nNo physical-plausibility warnings triggered.")


if __name__ == "__main__":
    main()
