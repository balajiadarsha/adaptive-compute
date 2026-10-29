"""
Controlled, dedicated throughput benchmark for FlatQuant-W8A8KV8/W4A4KV4 on
DeepSeek-R1-Distill-Qwen-1.5B -- the vLLM-based `bench_throughput_vllm.py`
does not apply here since the `flatquant` conda env has no vLLM at all
(confirmed) and eval_skill_flatquant.py uses raw HF `.generate()` in a
manual batched loop, no continuous batching/PagedAttention.

MANDATORY CAVEAT (do not drop when reporting these numbers): FlatQuant's
activation/weight quantization here is confirmed SIMULATED fake-quant
(quantize/flatquant/flatquant/quant_utils.py: ActivationQuantizer.forward
-> fake_quant -> sym_quant_dequant/asym_quant_dequant) -- values are
quantized then immediately dequantized back to bf16/fp16 BEFORE the matmul
runs. The GEMM always executes in the original floating-point precision,
never on a real int8/int4 tensor-core kernel. Combined with using plain HF
.generate() (no continuous batching), any tok/s number from this script
measures the CURRENT NUMERICAL-ACCURACY SIMULATION HARNESS's speed, not
FlatQuant's fundamental hardware potential. Every output row is labeled
accordingly -- never compare these numbers to the vLLM-measured BF16/GPTQ/
AWQ throughput as if they were apples-to-apples hardware comparisons.

Must run under the `flatquant` conda env:
  ~/.conda/envs/flatquant/bin/python bench_throughput_flatquant.py
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_skill_flatquant as fq  # noqa: E402
import flatquant.utils as futils  # noqa: E402

LENGTH_PRESETS = {
    "short": (90, 1300),
    "medium": (200, 2900),
    "long": (220, 6900),
}

DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def build_synthetic_prompts(tokenizer, input_len, num_prompts):
    """Build num_prompts distinct prompts of exactly input_len tokens each,
    by repeating a filler sentence and truncating -- matches
    bench_throughput_vllm.py's synthetic (not real-dataset) methodology so
    the two are comparable in spirit (fixed length, not real content)."""
    filler = "The quick brown fox jumps over the lazy dog. " * (input_len // 8 + 4)
    ids = tokenizer.encode(filler)[:input_len]
    base_prompt = tokenizer.decode(ids)
    # Vary a trailing index so prompts aren't byte-identical (avoids any
    # accidental cache effects), while keeping token length ~constant.
    return [f"{base_prompt} [{i}]" for i in range(num_prompts)]


@torch.no_grad()
def bench_one(model, tokenizer, input_len, output_len, num_prompts, batch_size):
    prompts = build_synthetic_prompts(tokenizer, input_len, num_prompts)
    total_output_tokens = 0
    total_wall_time_s = 0.0
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(futils.DEV)
        in_len = inputs["input_ids"].shape[1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            **inputs,
            max_new_tokens=output_len,
            min_new_tokens=output_len,  # force exact output_len for a clean tok/s measurement
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        total_wall_time_s += time.perf_counter() - t0
        total_output_tokens += out.shape[0] * (out.shape[1] - in_len)
        print(f"    batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size} done")
    return total_output_tokens, total_wall_time_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--schemes", nargs="+", default=list(fq.SCHEMES), choices=list(fq.SCHEMES))
    ap.add_argument("--presets", nargs="+", default=list(LENGTH_PRESETS.keys()),
                     choices=list(LENGTH_PRESETS.keys()))
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/throughput_sweep")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    rows = []
    for scheme in args.schemes:
        matrix_path = fq.DEFAULT_MATRIX_DIR[scheme]
        print(f"\n=== Loading FlatQuant {scheme} ===")
        model = fq._load_flatquant_model(scheme, args.model_id, matrix_path, args.seed)
        model.eval()

        for preset_name in args.presets:
            input_len, output_len = LENGTH_PRESETS[preset_name]
            print(f"\n--- {scheme} | {preset_name} (input_len={input_len}, output_len={output_len}) ---")
            total_output_tokens, total_wall_time_s = bench_one(
                model, tokenizer, input_len, output_len, args.num_prompts, args.batch_size,
            )
            tok_s = total_output_tokens / total_wall_time_s
            print(f"    -> {tok_s:.1f} output tok/s [SIMULATED fake-quant, not a real int8/int4 kernel]")
            rows.append({
                "model": "DeepSeek-R1",
                "precision": f"FlatQuant-{scheme.upper()}",
                "length_preset": preset_name,
                "input_len": input_len,
                "output_len": output_len,
                "num_prompts": args.num_prompts,
                "elapsed_time_s": total_wall_time_s,
                "output_tokens_per_second": tok_s,
                "note": "SIMULATED fake-quant (dequant-to-bf16-before-matmul); "
                        "not a real int8/int4 kernel -- do not compare directly to vLLM BF16/GPTQ/AWQ tok/s",
            })

        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "flatquant_throughput.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path} ({len(df)} rows)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
