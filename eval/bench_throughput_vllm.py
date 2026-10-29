"""
Orchestrates `vllm bench throughput` across every vLLM-servable precision
of DeepSeek-R1-Distill-Qwen-1.5B and L1-Qwen-1.5B-Exact (BF16, GPTQ-int8,
GPTQ-int4, AWQ-W4A16 -- NOT GPTQ-int3, which is never runnable in vLLM for
this repo, confirmed in earlier eval runs) at a few representative
(input_len, output_len) presets, and aggregates the results into one CSV.

Deliberately does NOT touch eval_skill.py's accuracy-eval path -- this is a
separate, standalone throughput measurement using vLLM's own built-in
`vllm bench throughput` CLI, so it carries zero risk to the already-
completed accuracy-sweep results.

Length presets were derived empirically from the real per-dataset mean
prompt/output token lengths already observed in the completed accuracy
runs (see eval/flops_report.py's output), not guessed:
  short  ~ GSM8K   (mean prompt ~90,  mean output ~1300)
  medium ~ MATH/GPQA (mean prompt ~200, mean output ~2900)
  long   ~ AIME    (mean prompt ~220, mean output ~6900)
"""
import argparse
import json
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

LENGTH_PRESETS = {
    "short": (90, 1300),
    "medium": (200, 2900),
    "long": (220, 6900),
}

MODELS = [
    ("DeepSeek-R1", "BF16", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("DeepSeek-R1", "GPTQ-int8", "../quantize/checkpoints/DeepSeek-R1-Distill-Qwen-1.5B-gptq-int8"),
    ("DeepSeek-R1", "GPTQ-int4", "../quantize/checkpoints/DeepSeek-R1-Distill-Qwen-1.5B-gptq-int4"),
    ("DeepSeek-R1", "AWQ-W4A16", "../quantize/checkpoints/DeepSeek-R1-Distill-Qwen-1.5B-awq-w4a16_asym"),
    ("L1", "BF16", "l3lab/L1-Qwen-1.5B-Exact"),
    ("L1", "GPTQ-int8", "../quantize/checkpoints/L1-Qwen-1.5B-Exact-gptq-int8"),
    ("L1", "GPTQ-int4", "../quantize/checkpoints/L1-Qwen-1.5B-Exact-gptq-int4"),
    ("L1", "AWQ-W4A16", "../quantize/checkpoints/L1-Qwen-1.5B-Exact-awq-w4a16_asym"),
    ("DeepSeek-R1-7B", "BF16", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    ("DeepSeek-R1-7B", "GPTQ-int8", "../quantize/checkpoints/DeepSeek-R1-Distill-Qwen-7B-gptq-int8"),
    ("DeepSeek-R1-7B", "GPTQ-int4", "../quantize/checkpoints/DeepSeek-R1-Distill-Qwen-7B-gptq-int4"),
    # GPTQ-int3 included for completeness, but expected to fail the same way
    # it does for the 1.5B models -- vLLM's GPTQ kernels generally only
    # support 4/8-bit; run_one() returns None on failure and the
    # orchestrator skips to the next entry, so this won't block the rest.
    ("DeepSeek-R1-7B", "GPTQ-int3", "../quantize/checkpoints/DeepSeek-R1-Distill-Qwen-7B-gptq-int3"),
    ("DeepSeek-R1-7B", "AWQ-W4A16", "../quantize/checkpoints/DeepSeek-R1-Distill-Qwen-7B-awq-w4a16_asym"),
]

VLLM_BIN = os.path.expanduser("~/.conda/envs/reason/bin/vllm")


def run_one(model_id, input_len, output_len, num_prompts, out_json, tensor_parallel_size, gpu_mem_util):
    cmd = [
        VLLM_BIN, "bench", "throughput",
        "--model", model_id,
        "--trust-remote-code",
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--num-prompts", str(num_prompts),
        "--seed", "42",
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--output-json", out_json,
    ]
    print("  $ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        print(f"  ERROR: exit code {result.returncode}")
        return None
    with open(out_json) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/throughput_sweep")
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    ap.add_argument("--presets", nargs="+", default=list(LENGTH_PRESETS.keys()),
                     choices=list(LENGTH_PRESETS.keys()))
    ap.add_argument("--models", nargs="+", default=None,
                     help="Subset of precision labels to run, e.g. BF16 GPTQ-int8. Default: all.")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for model_label, precision, model_id in MODELS:
        if args.models and precision not in args.models:
            continue
        for preset_name in args.presets:
            input_len, output_len = LENGTH_PRESETS[preset_name]
            slug = f"{model_label}_{precision}_{preset_name}".replace("/", "_")
            out_json = os.path.join(args.out_dir, f"{slug}.json")

            if args.skip_existing and os.path.exists(out_json):
                print(f"Skipping {model_label} | {precision} | {preset_name} (exists at {out_json})")
                with open(out_json) as f:
                    result = json.load(f)
            else:
                print(f"\n=== {model_label} | {precision} | {preset_name} "
                      f"(input_len={input_len}, output_len={output_len}) ===")
                result = run_one(model_id, input_len, output_len, args.num_prompts,
                                  out_json, args.tensor_parallel_size, args.gpu_memory_utilization)
                if result is None:
                    continue

            rows.append({
                "model": model_label,
                "precision": precision,
                "length_preset": preset_name,
                "input_len": input_len,
                "output_len": output_len,
                "num_prompts": args.num_prompts,
                "elapsed_time_s": result["elapsed_time"],
                "requests_per_second": result["requests_per_second"],
                "total_tokens_per_second": result["tokens_per_second"],
                # vllm bench throughput's --output-json only exposes combined
                # (prompt+output) total_num_tokens/tokens_per_second, not an
                # output-only figure -- but --random-output-len forces each
                # request to generate exactly that many tokens (no early EOS
                # on synthetic random prompts), so output_len*num_prompts is
                # the exact output token count, not an approximation.
                "output_tokens_per_second": (output_len * args.num_prompts) / result["elapsed_time"],
            })

    if not rows:
        print("No throughput results produced.")
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "throughput_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path} ({len(df)} rows)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
