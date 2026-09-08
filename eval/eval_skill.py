"""
Evaluate l3lab/L1-Qwen-1.5B-Exact on math/science reasoning benchmarks
(AIME, GPQA, LSAT) using vLLM for generation. These datasets carry their own
gold answers, so scoring is done by extracting the model's final answer and
exact-matching it against the dataset label -- no external evaluator model
is needed.

Two-stage usage:
  # Stage 1: generate responses, save to parquet
  python eval_skill.py --generate --dataset gpqa --n 100 --max-tokens 4096 --parquet generations.parquet

  # Stage 2: score an existing parquet (no GPU needed)
  python eval_skill.py --evaluate --parquet generations.parquet --output results.json

  # Or run both stages in one go:
  python eval_skill.py --dataset aime --n 100 --max-tokens 32768 --think-budget 2048
"""
import argparse
import gc
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import datasets
import pandas as pd
import torch
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


MODEL_ID = "l3lab/L1-Qwen-1.5B-Exact"

DATASETS = ("aime", "gpqa", "lsat")

DATASET_HF_ID = {
    "aime": "MathArena/aime_2025",
    "gpqa": "Idavidrein/gpqa (gpqa_diamond)",
    "lsat": "hails/agieval-lsat-lr",
}

LETTERS = "ABCDEFGH"


def _flashinfer_comm_broken() -> bool:
    """vLLM's torch.compile backend init imports flashinfer.comm; on this
    cluster's flashinfer build that import raises
    "TypeError: type 'array.array' is not subscriptable". Detect that here
    so enforce_eager (which disables CUDA graph capture) is only forced on
    environments that actually hit it."""
    try:
        import flashinfer.comm  # noqa: F401
        return False
    except Exception:
        return True


_ENFORCE_EAGER = _flashinfer_comm_broken()

# Datasets are pre-downloaded to local disk (compute nodes on this cluster
# have unreliable/slow access to the HF Hub for dataset downloads) via
# `datasets.Dataset.save_to_disk`. Loaders read from here with
# `datasets.load_from_disk` instead of hitting the Hub.
LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOCAL_DATASET_PATH = {
    "aime": os.path.join(LOCAL_DATA_DIR, "aime_2025"),
    "gpqa": os.path.join(LOCAL_DATA_DIR, "gpqa_diamond"),
    "lsat": os.path.join(LOCAL_DATA_DIR, "agieval_lsat_lr"),
}


# ---------------------------------------------------------------------------
# Dataset loading. Each loader reshapes its source into the common schema:
#   instruction: str        -- the fully-formed question to prompt the model with
#   answer: str              -- gold answer, comparable via the dataset's `kind`
#   kind: "numeric" | "mcq"  -- selects the answer-extraction/scoring strategy
#   choices: dict | None     -- {letter: text} for mcq, else None
# ---------------------------------------------------------------------------

def load_aime(n: int | None, seed: int = 42) -> list[dict]:
    ds = datasets.load_from_disk(LOCAL_DATASET_PATH["aime"])
    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    items = []
    for ex in ds:
        items.append({
            "instruction": (
                f"{ex['problem']}\n\n"
                "Give your final answer as a single integer, written as \\boxed{answer}."
            ),
            "answer": str(int(ex["answer"])),
            "kind": "numeric",
            "choices": None,
        })
    return items


def load_gpqa(n: int | None, seed: int = 42) -> list[dict]:
    ds = datasets.load_from_disk(LOCAL_DATASET_PATH["gpqa"])
    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    import random
    rng = random.Random(seed)

    items = []
    for ex in ds:
        options = [
            ex["Correct Answer"],
            ex["Incorrect Answer 1"],
            ex["Incorrect Answer 2"],
            ex["Incorrect Answer 3"],
        ]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [options[i] for i in order]
        correct_letter = LETTERS[order.index(0)]

        choices = {LETTERS[i]: text for i, text in enumerate(shuffled)}
        options_block = "\n".join(f"{letter}) {text}" for letter, text in choices.items())
        instruction = (
            f"{ex['Question']}\n\n{options_block}\n\n"
            "Give your final answer as a single letter, written as \\boxed{answer}."
        )
        items.append({
            "instruction": instruction,
            "answer": correct_letter,
            "kind": "mcq",
            "choices": choices,
        })
    return items


def load_lsat(n: int | None, seed: int = 42) -> list[dict]:
    ds = datasets.load_from_disk(LOCAL_DATASET_PATH["lsat"])
    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    items = []
    for ex in ds:
        # AGIEval encodes options inline as "(A)...(B)..." within `query`, and
        # `gold` is a list of correct option indices (0-indexed).
        gold_idx = ex["gold"][0] if isinstance(ex["gold"], list) else ex["gold"]
        answer = LETTERS[gold_idx]
        instruction = (
            f"{ex['query']}\n\n"
            "Give your final answer as a single letter, written as \\boxed{answer}."
        )
        items.append({
            "instruction": instruction,
            "answer": answer,
            "kind": "mcq",
            "choices": None,
        })
    return items


def load_dataset(name: str, n: int | None, seed: int = 42) -> list[dict]:
    if name == "aime":
        return load_aime(n, seed=seed)
    elif name == "gpqa":
        return load_gpqa(n, seed=seed)
    elif name == "lsat":
        return load_lsat(n, seed=seed)
    raise ValueError(f"Unknown dataset: {name!r}. Choose from {DATASETS}")


# ---------------------------------------------------------------------------
# Prompting / generation (mirrors eval_safety.py)
# ---------------------------------------------------------------------------

def _build_prompts(items: list[dict], tokenizer, think_budget: int | None) -> list[str]:
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
    return prompts


def _think_token_counts(outputs, tokenizer) -> list[int]:
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
    return think_token_counts


def generate_responses(
    items: list[dict],
    model_id: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    think_budget: int | None = None,
    tensor_parallel_size: int = 4,
) -> tuple[list[str], list[str], list[int], list[int]]:
    print(f"\nLoading {model_id} with vLLM (tensor_parallel_size={tensor_parallel_size})...")
    if think_budget is not None:
        print(f"Token budget: {think_budget} tokens")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        gpu_memory_utilization=0.45,
        max_model_len=32768,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=_ENFORCE_EAGER,
    )

    prompts = _build_prompts(items, tokenizer, think_budget)

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
    think_token_counts = _think_token_counts(outputs, tokenizer)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return prompts, responses, token_counts, think_token_counts


def _dp_worker(
    rank: int,
    gpu_ids: list[str],
    items: list[dict],
    model_id: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    think_budget: int | None,
    tensor_parallel_size: int,
    result_queue,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    print(f"[dp-worker {rank}] GPUs={gpu_ids} items={len(items)}")
    prompts, responses, token_counts, think_token_counts = generate_responses(
        items,
        model_id=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        think_budget=think_budget,
        tensor_parallel_size=tensor_parallel_size,
    )
    result_queue.put((rank, prompts, responses, token_counts, think_token_counts))


def generate_responses_data_parallel(
    items: list[dict],
    model_id: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    num_dp_workers: int,
    tensor_parallel_size: int = 1,
    think_budget: int | None = None,
) -> tuple[list[str], list[str], list[int], list[int]]:
    """Run `num_dp_workers` independent vLLM engines in parallel, each pinned
    to its own slice of `tensor_parallel_size` GPUs, sharding `items` across
    them. Requires num_dp_workers * tensor_parallel_size GPUs on the node."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    all_gpu_ids = visible.split(",") if visible else [str(i) for i in range(torch.cuda.device_count())]
    needed = num_dp_workers * tensor_parallel_size
    if needed > len(all_gpu_ids):
        raise ValueError(
            f"num_dp_workers ({num_dp_workers}) * tensor_parallel_size ({tensor_parallel_size}) "
            f"= {needed} exceeds available GPUs ({len(all_gpu_ids)}: {all_gpu_ids})"
        )

    # Round-robin shard so each worker gets a representative slice of the
    # dataset (matters if items are ordered by label/category).
    shards: list[list[dict]] = [[] for _ in range(num_dp_workers)]
    shard_indices: list[list[int]] = [[] for _ in range(num_dp_workers)]
    for i, item in enumerate(items):
        w = i % num_dp_workers
        shards[w].append(item)
        shard_indices[w].append(i)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = []
    for rank in range(num_dp_workers):
        gpu_ids = all_gpu_ids[rank * tensor_parallel_size:(rank + 1) * tensor_parallel_size]
        p = ctx.Process(
            target=_dp_worker,
            args=(rank, gpu_ids, shards[rank], model_id, max_tokens, temperature,
                  top_p, think_budget, tensor_parallel_size, result_queue),
        )
        p.start()
        procs.append(p)

    results_by_rank = {}
    for _ in procs:
        rank, prompts, responses, token_counts, think_token_counts = result_queue.get()
        results_by_rank[rank] = (prompts, responses, token_counts, think_token_counts)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"DP worker exited with code {p.exitcode}")

    # Reassemble in original item order.
    n = len(items)
    prompts: list[str | None] = [None] * n
    responses: list[str | None] = [None] * n
    token_counts: list[int | None] = [None] * n
    think_token_counts: list[int | None] = [None] * n
    for rank, (p_, r_, tc_, ttc_) in results_by_rank.items():
        for local_i, orig_i in enumerate(shard_indices[rank]):
            prompts[orig_i] = p_[local_i]
            responses[orig_i] = r_[local_i]
            token_counts[orig_i] = tc_[local_i]
            think_token_counts[orig_i] = ttc_[local_i]

    return prompts, responses, token_counts, think_token_counts


def save_parquet(
    items: list[dict],
    prompts: list[str],
    responses: list[str],
    token_counts: list[int],
    path: str,
    think_budget: int | None = None,
    think_token_counts: list[int] | None = None,
    dataset: str = "aime",
):
    df = pd.DataFrame({
        "instruction": [item["instruction"] for item in items],
        "answer": [item["answer"] for item in items],
        "kind": [item["kind"] for item in items],
        "choices": [json.dumps(item["choices"]) if item["choices"] is not None else None for item in items],
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
    items = []
    for _, row in df.iterrows():
        choices = json.loads(row["choices"]) if row.get("choices") not in (None, "") and pd.notna(row["choices"]) else None
        items.append({
            "instruction": row["instruction"],
            "answer": row["answer"],
            "kind": row["kind"],
            "choices": choices,
        })
    prompts = df["prompt"].tolist()
    responses = df["response"].tolist()
    token_counts = df["token_count"].tolist() if "token_count" in df.columns else [None] * len(df)
    think_token_counts = df["think_token_count"].tolist() if "think_token_count" in df.columns else [None] * len(df)
    think_budget = int(df["think_budget"].iloc[0]) if "think_budget" in df.columns and pd.notna(df["think_budget"].iloc[0]) else None
    dataset = df["dataset"].iloc[0] if "dataset" in df.columns else "aime"
    print(f"Loaded {len(df)} rows from {path} (think_budget={think_budget}, dataset={dataset})")
    return items, prompts, responses, token_counts, think_token_counts, think_budget, dataset


# ---------------------------------------------------------------------------
# Scoring. No external evaluator: extract the model's final answer from its
# own response text and exact-match it against the dataset's gold answer.
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_FINAL_ANSWER_RE = re.compile(r"final answer is[:\s]*\**\s*([A-Za-z0-9\-\.\,\/\s]+?)\**\s*(?:[\.\n]|$)", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([A-H])\b")
_INT_RE = re.compile(r"-?\d+")


def _extract_raw_answer(response: str) -> str | None:
    """Pull the candidate final-answer substring out of free-form model text,
    preferring an explicit \\boxed{...} then a 'final answer is ...' phrase."""
    matches = _BOXED_RE.findall(response)
    if matches:
        return matches[-1].strip()
    match = _FINAL_ANSWER_RE.search(response)
    if match:
        return match.group(1).strip()
    return None


def score_response(item: dict, response: str) -> dict:
    raw = _extract_raw_answer(response)
    parsed = None
    correct = False
    parse_error = raw is None

    if raw is not None:
        if item["kind"] == "numeric":
            m = _INT_RE.search(raw)
            if m:
                parsed = m.group(0)
                try:
                    correct = int(parsed) == int(item["answer"])
                except ValueError:
                    correct = False
            else:
                parse_error = True
        elif item["kind"] == "mcq":
            m = _LETTER_RE.search(raw.upper())
            if m:
                parsed = m.group(1)
                correct = parsed == item["answer"]
            else:
                parse_error = True

    return {
        "raw_extracted": raw,
        "parsed_answer": parsed,
        "gold_answer": item["answer"],
        "correct": correct,
        "parse_error": parse_error,
    }


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


def compute_metrics(items: list[dict], scores: list[dict]) -> dict:
    total = len(items)
    n_correct = sum(1 for s in scores if s["correct"])
    n_parse_error = sum(1 for s in scores if s["parse_error"])
    scorable = total - n_parse_error

    by_kind: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0, "parse_error": 0})
    for item, score in zip(items, scores):
        b = by_kind[item["kind"]]
        b["total"] += 1
        b["correct"] += int(score["correct"])
        b["parse_error"] += int(score["parse_error"])

    def pct(num, denom):
        return round(100 * num / denom, 2) if denom > 0 else None

    return {
        "total": total,
        "accuracy_%": pct(n_correct, scorable),
        "accuracy_%_incl_parse_errors": pct(n_correct, total),
        "correct": n_correct,
        "parse_errors": n_parse_error,
        "by_kind": {
            kind: {
                "total": b["total"],
                "accuracy_%": pct(b["correct"], b["total"] - b["parse_error"]),
                "parse_errors": b["parse_error"],
            }
            for kind, b in by_kind.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()

    # Operation modes:
    #   generate: run generation and save to parquet, skip scoring
    #   evaluate: run scoring on existing parquet, skip generation

    mode.add_argument("--generate", action="store_true",
                      help="Only run generation, save parquet, skip scoring")
    mode.add_argument("--evaluate", action="store_true",
                      help="Only score an existing parquet (skip generation)")

    parser.add_argument("--dataset", type=str, default="aime", choices=DATASETS,
                        help="Dataset to evaluate on: 'aime' (MathArena/aime_2025), "
                             "'gpqa' (Idavidrein/gpqa, gpqa_diamond), or "
                             "'lsat' (hails/agieval-lsat-lr). Ignored with --evaluate "
                             "(read from parquet).")
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
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="GPUs per model replica (vLLM tensor parallelism). Default: 4.")
    parser.add_argument("--num-dp-workers", type=int, default=1,
                        help="Number of data-parallel model replicas, each using "
                             "--tensor-parallel-size GPUs, sharding the dataset across "
                             "them. Requires num-dp-workers * tensor-parallel-size GPUs. "
                             "Default: 1 (no data parallelism).")
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
        if args.num_dp_workers > 1:
            prompts, responses, token_counts, think_token_counts = generate_responses_data_parallel(
                items,
                model_id=args.model_id,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                num_dp_workers=args.num_dp_workers,
                tensor_parallel_size=args.tensor_parallel_size,
                think_budget=think_budget,
            )
        else:
            prompts, responses, token_counts, think_token_counts = generate_responses(
                items,
                model_id=args.model_id,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                think_budget=think_budget,
                tensor_parallel_size=args.tensor_parallel_size,
            )
        save_parquet(items, prompts, responses, token_counts, parquet_path,
                     think_budget=think_budget, think_token_counts=think_token_counts,
                     dataset=dataset)

        if args.generate:
            print("Generation complete. Run with --evaluate to score against gold answers.")
            return

    # Stage 2: score against the dataset's own gold answers
    print("\nScoring responses against gold answers...")
    scores = [score_response(item, resp) for item, resp in zip(items, responses)]
    metrics = compute_metrics(items, scores)
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
