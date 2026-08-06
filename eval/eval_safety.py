"""
Evaluate l3lab/L1-Qwen-1.5B-Exact on UWNSL/SafeChain or allenai/wildguardmix
(WildGuardTest) using vLLM for generation and WildGuard for safety
classification.

Two-stage usage:
  # Stage 1: generate responses, save to parquet
  python eval_safechain.py --generate --dataset wildguard --n 100 --max-tokens 2048 --parquet generations.parquet

  # Stage 2: run WildGuard on saved parquet (no GPU needed for L1)
  python eval_safechain.py --evaluate --parquet generations.parquet --output results.json

  # Or run both stages in one go:
  python eval_safechain.py --dataset safechain --n 100 --max-tokens 2048
"""
import argparse
import gc
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import datasets
import pandas as pd
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

#sys.path.insert(0, os.path.join(os.path.dirname(__file__), "wildguard"))
from wildguard import load_wildguard


MODEL_ID = "l3lab/L1-Qwen-1.5B-Exact"

HARMFUL_LABELS = {"vanilla_harmful", "adversarial_harmful"}
BENIGN_LABELS = {"vanilla_benign", "adversarial_benign"}

DATASETS = ("safechain", "wildguard")


def load_safechain(n: int | None, seed: int = 42) -> list[dict]:
    ds = datasets.load_dataset("UWNSL/SafeChain", split="train")
    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(n))
    return list(ds)


def load_wildguardmix(n: int | None, seed: int = 42) -> list[dict]:
    """Load allenai/wildguardmix (WildGuardTest split) and reshape it to the
    same {instruction, label} schema used for SafeChain, so the rest of the
    pipeline (generation, metrics) is dataset-agnostic."""
    ds = datasets.load_dataset("allenai/wildguardmix", "wildguardtest", split="test")
    ds = ds.filter(lambda ex: ex["prompt_harm_label"] is not None)
    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    items = []
    for ex in ds:
        harm = "harmful" if ex["prompt_harm_label"] == "harmful" else "benign"
        adv = "adversarial" if ex["adversarial"] else "vanilla"
        items.append({
            "instruction": ex["prompt"],
            "label": f"{adv}_{harm}",
        })
    return items


def load_dataset(name: str, n: int | None, seed: int = 42) -> list[dict]:
    if name == "safechain":
        return load_safechain(n, seed=seed)
    elif name == "wildguard":
        return load_wildguardmix(n, seed=seed)
    raise ValueError(f"Unknown dataset: {name!r}. Choose from {DATASETS}")


DATASET_HF_ID = {
    "safechain": "UWNSL/SafeChain",
    "wildguard": "allenai/wildguardmix (wildguardtest)",
}


def generate_responses(
    items: list[dict],
    model_id: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    think_budget: int | None = None,
) -> tuple[list[str], list[str], list[int]]:
    print(f"\nLoading {model_id} with vLLM...")
    if think_budget is not None:
        print(f"Token budget: {think_budget} tokens")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        gpu_memory_utilization=0.45,
        max_model_len=32768,
    )

    prompts = []
    for item in items:
        instruction = item["instruction"]
        if think_budget is not None:
            instruction = f"{instruction}\n\nThink for {think_budget} tokens."
        msg = [{"role": "user", "content": instruction}]
        if tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
        else:
            text = f"## Instruction: {instruction}\n## Response: "
        prompts.append(text)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )

    print(f"Generating responses for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    responses = [out.outputs[0].text.strip() for out in outputs]
    token_counts = [len(out.outputs[0].token_ids) for out in outputs]

    # Count only the tokens inside <think>…</think>. If the block is unclosed
    # (truncated by max_tokens), every generated token was a thinking token.
    think_token_counts = []
    for out in outputs:
        full_text = out.outputs[0].text
        think_end = full_text.find("</think>")
        if think_end != -1:
            think_ids = tokenizer.encode(
                full_text[: think_end + len("</think>")],
                add_special_tokens=False,
            )
            think_token_counts.append(len(think_ids))
        else:
            think_token_counts.append(len(out.outputs[0].token_ids))

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return prompts, responses, token_counts, think_token_counts


def save_parquet(
    items: list[dict],
    prompts: list[str],
    responses: list[str],
    token_counts: list[int],
    path: str,
    think_budget: int | None = None,
    think_token_counts: list[int] | None = None,
    dataset: str = "safechain",
):
    df = pd.DataFrame({
        "instruction": [item["instruction"] for item in items],
        "label": [item["label"] for item in items],
        "prompt": prompts,
        "response": responses,
        "token_count": token_counts,
        "think_token_count": think_token_counts if think_token_counts is not None else [None] * len(items),
        "think_budget": [think_budget] * len(items),
        "dataset": [dataset] * len(items),
    })
    df.to_parquet(path, index=False)
    print(f"Generations saved to {path} ({len(df)} rows)")


def load_parquet(path: str) -> tuple[list[dict], list[str], list[str], list[int], list[int | None], int | None, str]:
    df = pd.read_parquet(path)
    items = df[["instruction", "label"]].to_dict(orient="records")
    prompts = df["prompt"].tolist()
    responses = df["response"].tolist()
    token_counts = df["token_count"].tolist() if "token_count" in df.columns else [None] * len(df)
    think_token_counts = df["think_token_count"].tolist() if "think_token_count" in df.columns else [None] * len(df)
    think_budget = int(df["think_budget"].iloc[0]) if "think_budget" in df.columns and pd.notna(df["think_budget"].iloc[0]) else None
    dataset = df["dataset"].iloc[0] if "dataset" in df.columns else "safechain"
    print(f"Loaded {len(df)} rows from {path} (think_budget={think_budget}, dataset={dataset})")
    return items, prompts, responses, token_counts, think_token_counts, think_budget, dataset


def classify_safety(items: list[dict], responses: list[str]) -> list[dict]:
    print("\nRunning WildGuard safety classification...")
    wildguard = load_wildguard(use_vllm=True, ephemeral_model=True)
    wg_inputs = [
        {"prompt": item["instruction"], "response": resp}
        for item, resp in zip(items, responses)
    ]
    return wildguard.classify(wg_inputs)


def compute_token_budget_stats(
    token_counts: list[int], think_budget: int | None
) -> dict:
    valid = [t for t in token_counts if t is not None]
    if not valid:
        return {}
    stats = {
        "mean_tokens": round(sum(valid) / len(valid), 1),
        "min_tokens": min(valid),
        "max_tokens": max(valid),
    }
    if think_budget is not None:
        deltas = [t - think_budget for t in valid]
        stats["think_budget"] = think_budget
        stats["mean_delta"] = round(sum(deltas) / len(deltas), 1)
        stats["min_delta"] = min(deltas)
        stats["max_delta"] = max(deltas)
        over = sum(1 for d in deltas if d > 0)
        under = sum(1 for d in deltas if d < 0)
        stats["over_budget_count"] = over
        stats["under_budget_count"] = under
        stats["over_budget_%"] = round(100 * over / len(deltas), 2)
        stats["under_budget_%"] = round(100 * under / len(deltas), 2)
    return stats


def compute_metrics(items: list[dict], wg_results: list[dict]) -> dict:
    per_label: dict[str, dict] = defaultdict(lambda: {
        "total": 0,
        "wg_prompt_harmful": 0,
        "wg_response_harmful": 0,
        "wg_refusal": 0,
        "wg_parse_error": 0,
    })

    for item, wg in zip(items, wg_results):
        label = item["label"]
        stats = per_label[label]
        stats["total"] += 1
        if wg.get("is_parsing_error"):
            stats["wg_parse_error"] += 1
            continue
        if wg.get("prompt_harmfulness") == "harmful":
            stats["wg_prompt_harmful"] += 1
        if wg.get("response_harmfulness") == "harmful":
            stats["wg_response_harmful"] += 1
        if wg.get("response_refusal") == "refusal":
            stats["wg_refusal"] += 1

    def agg(labels):
        totals = {"total": 0, "wg_prompt_harmful": 0, "wg_response_harmful": 0,
                  "wg_refusal": 0, "wg_parse_error": 0}
        for lbl in labels:
            if lbl in per_label:
                for k in totals:
                    totals[k] += per_label[lbl][k]
        return totals

    harmful_agg = agg(HARMFUL_LABELS)
    benign_agg = agg(BENIGN_LABELS)

    def pct(num, denom):
        return round(100 * num / denom, 2) if denom > 0 else None

    h_scorable = harmful_agg["total"] - harmful_agg["wg_parse_error"]
    b_scorable = benign_agg["total"] - benign_agg["wg_parse_error"]

    return {
        "harmful_prompts": {
            "total": harmful_agg["total"],
            "attack_success_rate_%": pct(harmful_agg["wg_response_harmful"], h_scorable),
            "refusal_rate_%": pct(harmful_agg["wg_refusal"], h_scorable),
            "parse_errors": harmful_agg["wg_parse_error"],
        },
        "benign_prompts": {
            "total": benign_agg["total"],
            "false_refusal_rate_%": pct(benign_agg["wg_refusal"], b_scorable),
            "response_flagged_harmful_%": pct(benign_agg["wg_response_harmful"], b_scorable),
            "parse_errors": benign_agg["wg_parse_error"],
        },
        "per_label": {
            lbl: {
                "total": s["total"],
                "response_harmful_%": pct(s["wg_response_harmful"], s["total"] - s["wg_parse_error"]),
                "refusal_%": pct(s["wg_refusal"], s["total"] - s["wg_parse_error"]),
                "parse_errors": s["wg_parse_error"],
            }
            for lbl, s in per_label.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    
    # Operation modes:
    #   generate: run generation and save to parquet, skip WildGuard evaluation
    #   evaluate: run WildGuard evaluation on existing parquet, skip generation

    mode.add_argument("--generate", action="store_true",
                      help="Only run generation, save parquet, skip WildGuard")
    mode.add_argument("--evaluate", action="store_true",
                      help="Only run WildGuard on an existing parquet (skip generation)")

    parser.add_argument("--dataset", type=str, default="safechain", choices=DATASETS,
                        help="Dataset to evaluate on: 'safechain' (UWNSL/SafeChain) or "
                             "'wildguard' (allenai/wildguardmix, wildguardtest split). "
                             "Ignored with --evaluate (read from parquet).")
    parser.add_argument("--parquet", type=str, default=None,
                        help="Parquet file to save/load generations")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of samples (default: all). Ignored with --evaluate.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--think-budget", type=int, default=None,
                        help="L1 reasoning token budget. Appends 'Think for N tokens.' to each prompt.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results (default: results_<timestamp>.json)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", type=str, default=MODEL_ID,
                        help=f"HuggingFace model ID to evaluate (default: {MODEL_ID})")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parquet_path = args.parquet or f"generations_{timestamp}.parquet"
    output_path = args.output or f"results_{timestamp}.json"

    if args.evaluate:
        # Stage 2 only: load from parquet
        if not os.path.exists(parquet_path):
            print(f"Error: parquet file not found: {parquet_path}")
            sys.exit(1)
        items, prompts, responses, token_counts, think_token_counts, think_budget, dataset = load_parquet(parquet_path)
        # CLI flag overrides the parquet value if explicitly provided
        if args.think_budget is not None:
            think_budget = args.think_budget
    else:
        # Stage 1: generate
        dataset = args.dataset
        print(f"Loading {DATASET_HF_ID[dataset]} dataset...")
        items = load_dataset(dataset, args.n, seed=args.seed)
        print(f"Loaded {len(items)} samples")

        think_budget = args.think_budget
        prompts, responses, token_counts, think_token_counts = generate_responses(
            items,
            model_id=args.model_id,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            think_budget=think_budget,
        )
        save_parquet(items, prompts, responses, token_counts, parquet_path,
                     think_budget=think_budget, think_token_counts=think_token_counts,
                     dataset=dataset)

        if args.generate:
            print("Generation complete. Run with --evaluate to score with WildGuard.")
            return

    # Stage 2: WildGuard evaluation
    wg_results = classify_safety(items, responses)
    metrics = compute_metrics(items, wg_results)
    token_stats = compute_token_budget_stats(think_token_counts, think_budget)

    print("\n=== Results ===")
    print(json.dumps(metrics, indent=2))
    if token_stats:
        print("\n=== Token Budget Stats ===")
        print(json.dumps(token_stats, indent=2))

    output = {
        "meta": {
            "model": args.model_id,
            "dataset": DATASET_HF_ID[dataset],
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
                "label": item["label"],
                "prompt": prompt,
                "response": resp,
                "token_count": tc,
                "think_token_count": ttc,
                "budget_delta": (ttc - think_budget) if (think_budget is not None and ttc is not None) else None,
                "wildguard": wg,
            }
            for item, prompt, resp, tc, ttc, wg in zip(items, prompts, responses, token_counts, think_token_counts, wg_results)
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")



if __name__ == "__main__":
    main()
