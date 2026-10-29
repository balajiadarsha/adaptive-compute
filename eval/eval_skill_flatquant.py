"""
Evaluate a FlatQuant-quantized (W8A8KV8 / W4A4KV4) Qwen2-architecture model
(DeepSeek-R1-Distill-Qwen-1.5B by default; L1-Qwen-1.5B/7B-Exact too) on the same skill benchmarks as eval_skill.py (aime, gpqa, lsat, math, gsm8k),
producing the identical results.json / parquet schema so it can be
aggregated by run_skill_sweep.py alongside the vLLM-based precisions.

FlatQuant's quantized model is not a portable checkpoint vLLM can load --
it's a set of learned transform matrices (flat_matrices.pth) that only make
sense once reloaded into a live model built by the vendored FlatQuant
package (../quantize/flatquant). So this script reconstructs the quantized
model in-process (model_utils.get_model -> apply_flatquant_to_model ->
load_flat_matrices -> reparameterize_model -> rtn_fwrd, mirroring
quantize/flatquant/main.py's own reload path) and generates with plain HF
`model.generate()` instead of vLLM -- no continuous batching/PagedAttention,
so this is much slower per-token than eval_skill.py's vLLM path.

Must run in the dedicated `flatquant` conda env (it imports the vendored
flatquant package, and that env's transformers/torch versions are what the
calibration run itself used), NOT the `reason` env eval_skill.py runs in.

Usage:
    conda activate flatquant
    python eval_skill_flatquant.py --dataset aime --scheme w8a8kv8 --n 50 --max-tokens 8192
"""
import argparse
import json
import os
import sys
from datetime import datetime

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
import eval_skill  # noqa: E402

FLATQUANT_DIR = os.path.join(os.path.dirname(__file__), "..", "quantize", "flatquant")
sys.path.insert(0, FLATQUANT_DIR)
import flatquant.model_utils as model_utils  # noqa: E402
import flatquant.flat_utils as flat_utils  # noqa: E402
import flatquant.utils as futils  # noqa: E402
import gptq_utils  # noqa: E402


SCHEMES = ("w8a8kv8", "w4a4kv4")

# Must exactly match the args used for the calibration run that produced
# each scheme's flat_matrices.pth (see quantize/flatquant calibration
# commands) -- apply_flatquant_to_model builds its module structure from
# these fields, and load_flat_matrices' state_dict load will silently
# mismatch shapes/behavior if they drift from what was calibrated.
FLATQUANT_PRESETS = {
    "w8a8kv8": {"w_bits": 8, "a_bits": 8, "k_bits": 8, "v_bits": 8},
    "w4a4kv4": {"w_bits": 4, "a_bits": 4, "k_bits": 4, "v_bits": 4},
}

def default_matrix_dir(model_id: str, scheme: str) -> str:
    # Mirrors main.py's exp_dir layout: outputs/<model basename>/w{w}a{a}/exp
    bits = FLATQUANT_PRESETS[scheme]
    return os.path.join(
        FLATQUANT_DIR, "outputs", model_id.rstrip("/").split("/")[-1],
        f"w{bits['w_bits']}a{bits['a_bits']}", "exp",
    )


def _build_flatquant_args(scheme: str, model_id: str, matrix_path: str, seed: int) -> argparse.Namespace:
    bits = FLATQUANT_PRESETS[scheme]
    return argparse.Namespace(
        model=model_id, hf_token=None, seed=seed,
        a_bits=bits["a_bits"], a_groupsize=-1, a_asym=False,
        w_bits=bits["w_bits"], w_groupsize=-1, w_asym=False,
        gptq=False, gptq_mse=False, percdamp=0.01, act_order=False,
        cali_trans=True, add_diag=True, lwc=True, lac=True,
        resume=False, save_matrix=False, reload_matrix=True, matrix_path=matrix_path,
        diag_init="sq_style", diag_alpha=0.3, warmup=False, deactive_amp=False,
        direct_inv=False, separate_vtrans=False,
        q_bits=16, q_asym=False, q_groupsize=-1,
        k_bits=bits["k_bits"], k_asym=True, k_groupsize=128,
        v_bits=bits["v_bits"], v_asym=True, v_groupsize=128,
        distribute_model=False, quantized_save=False,
    )


def _load_flatquant_model(scheme: str, model_id: str, matrix_path: str, seed: int):
    args = _build_flatquant_args(scheme, model_id, matrix_path, seed)
    futils.seed_everything(seed=seed)

    print(f"\nLoading {model_id} and reconstructing FlatQuant {scheme}...")
    model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    model = apply_flatquant_to_model(args, model)
    flat_utils.load_flat_matrices(args, model, path=matrix_path)
    flat_utils.reparameterize_model(model)

    if args.w_bits < 16:
        gptq_utils.rtn_fwrd(model, futils.DEV, args)

    model.to(futils.DEV)
    return model


@torch.no_grad()
def generate_responses_flatquant(
    items: list[dict],
    model_id: str,
    scheme: str,
    matrix_path: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    think_budget: int | None,
    batch_size: int,
    seed: int,
    checkpoint_path: str | None = None,
) -> tuple[list[str], list[str], list[int], list[int]]:
    """HF `.generate()` (no vLLM continuous batching) makes a single dataset
    run much slower than the vLLM-based precisions -- an AIME run at
    max_tokens=30000 can take longer than this cluster's interactive
    allocations last. checkpoint_path, when given, is a small JSON file
    (prompts + responses/token_counts completed so far) rewritten after each
    batch; a restarted run with the same checkpoint_path skips straight to
    the first unfinished batch instead of regenerating from scratch."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompts = eval_skill._build_prompts(items, tokenizer, think_budget)

    responses: list[str | None] = [None] * len(prompts)
    token_counts: list[int | None] = [None] * len(prompts)
    start_idx = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
        except json.JSONDecodeError:
            print(f"Checkpoint {checkpoint_path} is corrupted (partial write), ignoring it")
            ckpt = {}
        if ckpt.get("prompts") == prompts:
            done = len(ckpt["responses"])
            responses[:done] = ckpt["responses"]
            token_counts[:done] = ckpt["token_counts"]
            start_idx = done
            print(f"Resuming from checkpoint {checkpoint_path}: {done}/{len(prompts)} already generated")
        else:
            print(f"Checkpoint {checkpoint_path} doesn't match current prompts, ignoring it")

    model = _load_flatquant_model(scheme, model_id, matrix_path, seed)

    print(f"Generating responses for {len(prompts)} prompts (batch_size={batch_size})...")
    for i in range(start_idx, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(futils.DEV)
        input_len = inputs["input_ids"].shape[1]
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        for j, row in enumerate(out):
            gen_ids = row[input_len:].tolist()
            if tokenizer.eos_token_id in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(tokenizer.eos_token_id) + 1]
            responses[i + j] = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            token_counts[i + j] = len(gen_ids)
        done = min(i + batch_size, len(prompts))
        print(f"  {done}/{len(prompts)}")
        if checkpoint_path:
            # Write to a temp file and rename over the real path so a kill
            # (SIGKILL from a job/session timeout, mid-write) can never leave
            # a truncated, unparseable checkpoint -- os.replace is atomic.
            tmp_path = checkpoint_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump({
                    "prompts": prompts,
                    "responses": responses[:done],
                    "token_counts": token_counts[:done],
                }, f)
            os.replace(tmp_path, checkpoint_path)

    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    think_token_counts = eval_skill._think_token_counts(
        list(zip(responses, token_counts)), tokenizer
    )
    return prompts, responses, token_counts, think_token_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate", action="store_true")
    mode.add_argument("--evaluate", action="store_true")

    parser.add_argument("--dataset", type=str, default="aime", choices=eval_skill.DATASETS)
    parser.add_argument("--scheme", type=str, required=True, choices=SCHEMES)
    parser.add_argument("--model-id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--matrix-path", type=str, default=None,
                        help="Directory containing flat_matrices.pth (default: the vendored "
                             "FlatQuant repo's outputs/<model>/w{w}a{a}/exp for this model and scheme)")
    parser.add_argument("--parquet", type=str, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--think-budget", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="HF generate() batch size (no vLLM continuous batching here)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    matrix_path = args.matrix_path or default_matrix_dir(args.model_id, args.scheme)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parquet_path = args.parquet or f"generations_flatquant_{args.scheme}_{timestamp}.parquet"
    output_path = args.output or f"results_flatquant_{args.scheme}_{timestamp}.json"

    if args.evaluate:
        if not os.path.exists(parquet_path):
            print(f"Error: parquet file not found: {parquet_path}")
            sys.exit(1)
        items, prompts, responses, token_counts, think_token_counts, think_budget, dataset = \
            eval_skill.load_parquet(parquet_path)
        if args.think_budget is not None:
            think_budget = args.think_budget
    else:
        dataset = args.dataset
        print(f"Loading {eval_skill.DATASET_HF_ID[dataset]} dataset...")
        items = eval_skill.load_dataset(dataset, args.n, seed=args.seed)
        print(f"Loaded {len(items)} samples")

        think_budget = args.think_budget
        prompts, responses, token_counts, think_token_counts = generate_responses_flatquant(
            items,
            model_id=args.model_id,
            scheme=args.scheme,
            matrix_path=matrix_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            think_budget=think_budget,
            batch_size=args.batch_size,
            seed=args.seed,
            checkpoint_path=parquet_path + ".ckpt.json",
        )
        eval_skill.save_parquet(items, prompts, responses, token_counts, parquet_path,
                                 think_budget=think_budget, think_token_counts=think_token_counts,
                                 dataset=dataset)

        if args.generate:
            print("Generation complete. Run with --evaluate to score against gold answers.")
            return

    print("\nScoring responses against gold answers...")
    scores = [eval_skill.score_response(item, resp) for item, resp in zip(items, responses)]
    metrics = eval_skill.compute_metrics(items, scores)
    token_stats = eval_skill.compute_token_budget_stats(think_token_counts, think_budget)

    print("\n=== Results ===")
    print(json.dumps(metrics, indent=2))
    if token_stats:
        print("\n=== Token Budget Stats ===")
        print(json.dumps(token_stats, indent=2))

    output = {
        "meta": {
            "model": f"{args.model_id} (flatquant-{args.scheme})",
            "dataset": eval_skill.DATASET_HF_ID[dataset],
            "parquet": parquet_path,
            "n_samples": len(items),
            "think_budget": think_budget,
            "timestamp": datetime.now().isoformat(),
        },
        "metrics": metrics,
        "token_budget_stats": token_stats,
        "data": [
            {
                "instruction": item["instruction"],
                "kind": item["kind"],
                "gold_answer": item["answer"],
                "prompt": prompt,
                "response": resp,
                "token_count": tc,
                "think_token_count": ttc,
                "budget_delta": (ttc - think_budget) if (think_budget is not None and ttc is not None) else None,
                "score": score,
            }
            for item, prompt, resp, tc, ttc, score in zip(items, prompts, responses, token_counts, think_token_counts, scores)
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
