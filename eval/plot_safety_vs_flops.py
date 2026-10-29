"""
Safety vs reasoning compute (FLOPs) for L1-Qwen-1.5B / 7B and their quantized
variants: SafeChain and WildGuardMix attack success and false refusal, and the
StrongREJECT score, all judged on the final answer (after </think>), from the
same score_safety.py jsons that make_safety_table.py reads.

Compute is the mean analytical FLOPs per prompt of that dataset's run, as in
plot_acc_vs_flops.py (prompt re-tokenized from the generation parquet, output
length from token_count). Runs whose scoring json is not written yet are skipped.

Usage (from eval/):
    python plot_safety_vs_flops.py --out-prefix ../context/safety_vs_flops
"""
import argparse
import json
import os

import pandas as pd

from make_safety_table import BUDGETS, COLUMNS, dig, run_dir
# Importing plot_acc_vs_flops first puts utils/ on sys.path for flops.
from plot_acc_vs_flops import FLOPS_NOTE, QWEN2_7B, budget_label, plot_panels, prompt_length_fn
from flops import QWEN2_1_5B, total_flops_for_response  # noqa: E402

ALL = BUDGETS + [None]
# (slug, size label, precision label, budgets); None is the no-limit run.
RUNS = [
    ("L1-Qwen-1.5B-Exact", "1.5B", "BF16", ALL),
    ("L1-Qwen-1.5B-Exact-gptq-int8", "1.5B", "GPTQ-INT8", BUDGETS),
    ("L1-Qwen-1.5B-Exact-gptq-int4", "1.5B", "GPTQ-INT4", BUDGETS),
    ("L1-Qwen-1.5B-Exact-awq-w4a16_asym", "1.5B", "AWQ-W4A16", BUDGETS),
    ("L1-Qwen-1.5B-Exact-flatquant-w8a8kv8", "1.5B", "FlatQuant-W8A8KV8", [1024]),
    ("L1-Qwen-1.5B-Exact-flatquant-w4a4kv4", "1.5B", "FlatQuant-W4A4KV4", [1024]),
    ("L1-Qwen-7B-Exact", "7B", "BF16", ALL),
    ("L1-Qwen-7B-Exact-gptq-int8", "7B", "GPTQ-INT8", BUDGETS),
    ("L1-Qwen-7B-Exact-gptq-int4", "7B", "GPTQ-INT4", BUDGETS),
    ("L1-Qwen-7B-Exact-awq-w4a16_asym", "7B", "AWQ-W4A16", BUDGETS),
    ("L1-Qwen-7B-Exact-flatquant-w8a8kv8", "7B", "FlatQuant-W8A8KV8", [1024]),
    ("L1-Qwen-7B-Exact-flatquant-w4a4kv4", "7B", "FlatQuant-W4A4KV4", [1024]),
]
# Panel order: the three harm metrics, then the two over-refusal ones.
PANEL_ORDER = ["SafeChain ASR", "WGMix ASR", "StrongREJECT", "SafeChain FRR", "WGMix FRR"]
YLABELS = {"pct": "% (answer, lower is safer)", "score": "Score 0-1 (answer, lower is safer)"}
FRR_YLABEL = "% of benign prompts refused"


def collect_points():
    prompt_len = prompt_length_fn()
    points = []
    for slug, size, precision, budgets in RUNS:
        cfg = QWEN2_7B if size == "7B" else QWEN2_1_5B
        for budget in budgets:
            d = run_dir(slug, budget)
            flops = {}
            for header, ds, keys, _ in COLUMNS:
                path = os.path.join(d, f"{ds}.json")
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    metrics = json.load(f)["metrics"]
                if ds not in flops:
                    df = pd.read_parquet(os.path.join(d, f"{ds}.parquet"), columns=["prompt", "token_count"])
                    flops[ds] = sum(total_flops_for_response(cfg, prompt_len(p), int(n))["total_flops"]
                                    for p, n in zip(df["prompt"], df["token_count"])) / len(df)
                y = dig(metrics, ("answer",) + keys)
                if y is not None:
                    points.append({"size": size, "precision": precision, "budget": budget, "panel": header,
                                   "y": y, "flops": flops[ds], "tokens": metrics["mean_token_count"]})
            print(f"{size} {precision} {budget_label(budget)}: {sorted(flops)}")
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-prefix", type=str, default=os.path.join("..", "context", "safety_vs_flops"),
                        help="Writes <prefix>.pdf, <prefix>.png and <prefix>.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    points = collect_points()
    with open(args.out_prefix + ".csv", "w") as f:
        f.write("size,precision,budget,metric,answer_value,mean_flops,mean_tokens\n")
        for p in points:
            f.write(f"{p['size']},{p['precision']},{budget_label(p['budget'])},{p['panel']},"
                    f"{p['y']:.4f},{p['flops']:.4e},{p['tokens']:.1f}\n")
    kinds = {c[0]: c[3] for c in COLUMNS}
    panels = [(h, h, FRR_YLABEL if "FRR" in h else YLABELS[kinds[h]]) for h in PANEL_ORDER]
    plot_panels(points, panels, args.out_prefix,
                FLOPS_NOTE + " All runs sampled at T=0.6, top-p 0.95; budget runs cap at 8192 tokens, "
                "No limit at 30000. ASR/StrongREJECT: lower is safer. Missing points: runs still in progress.")


if __name__ == "__main__":
    main()
