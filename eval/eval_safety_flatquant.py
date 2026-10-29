"""
Generate safety responses (safechain / wildguard / strongreject) from a
FlatQuant-quantized model, writing the same parquet schema as
`eval_safety.py --generate` so score_safety.py judges both alike.

Generation reuses eval_skill_flatquant.generate_responses_flatquant (HF
generate on the reconstructed fake-quant model, checkpointed per batch); its
prompt builder is identical to eval_safety._build_prompts. Run in the
`flatquant` conda env.

Usage:
    python eval_safety_flatquant.py --dataset strongreject --scheme w8a8kv8 \
        --model-id l3lab/L1-Qwen-1.5B-Exact --think-budget 1024 --max-tokens 8192 \
        --temperature 0.6 --top-p 0.95 --parquet out/strongreject.parquet
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import eval_safety  # noqa: E402
import eval_skill_flatquant  # noqa: E402
from dataset_utils import DATASETS, DATASET_HF_ID, SAFECHAIN_CALIB_SAMPLES, load_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--scheme", type=str, required=True, choices=eval_skill_flatquant.SCHEMES)
    parser.add_argument("--model-id", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--matrix-path", type=str, default=None)
    parser.add_argument("--parquet", type=str, required=True)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--think-budget", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--safechain-offset", type=int, default=SAFECHAIN_CALIB_SAMPLES)
    args = parser.parse_args()

    print(f"Loading {DATASET_HF_ID[args.dataset]} dataset...")
    items = load_dataset(args.dataset, args.n, seed=args.seed, offset=args.safechain_offset)
    print(f"Loaded {len(items)} samples")

    prompts, responses, token_counts, think_token_counts = eval_skill_flatquant.generate_responses_flatquant(
        items,
        model_id=args.model_id,
        scheme=args.scheme,
        matrix_path=args.matrix_path or eval_skill_flatquant.default_matrix_dir(args.model_id, args.scheme),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        think_budget=args.think_budget,
        batch_size=args.batch_size,
        seed=args.seed,
        checkpoint_path=args.parquet + ".ckpt.json",
    )
    eval_safety.save_parquet(items, prompts, responses, token_counts, args.parquet,
                             think_budget=args.think_budget, think_token_counts=think_token_counts,
                             dataset=args.dataset)


if __name__ == "__main__":
    main()
