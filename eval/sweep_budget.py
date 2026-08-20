"""
Sweep token budgets for a reasoning model on UWNSL/SafeChain or
allenai/wildguardmix (WildGuardTest) and evaluate with WildGuard to study the
impact of reasoning budget on safety performance.

Usage:
    python sweep_budget.py [--dataset safechain|wildguard] [--n 500] [--budgets 256 512 1024 2048 4096] [--no-baseline]
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


SCRIPT = os.path.join(os.path.dirname(__file__), "eval_safety.py")


def run(cmd: list[str], label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print("  " + " ".join(cmd))
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"ERROR: {label} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def generate_stage(budgets: list[int], n: int, max_tokens: int, seed: int, out_dir: str, baseline: bool, model_id: str, dataset: str, tensor_parallel_size: int, num_dp_workers: int):
    for budget in budgets:
        parquet = os.path.join(out_dir, f"gen_budget{budget}.parquet")
        run(
            [SCRIPT,
             "--generate",
             "--dataset", dataset,
             "--model-id", model_id,
             "--n", str(n),
             "--max-tokens", str(max_tokens),
             "--think-budget", str(budget),
             "--parquet", parquet,
             "--seed", str(seed),
             "--tensor-parallel-size", str(tensor_parallel_size),
             "--num-dp-workers", str(num_dp_workers)],
            f"Generate | budget={budget} tokens | n={n}",
        )

    if baseline:
        parquet = os.path.join(out_dir, "gen_no_budget.parquet")
        run(
            [SCRIPT,
             "--generate",
             "--dataset", dataset,
             "--model-id", model_id,
             "--n", str(n),
             "--max-tokens", str(max_tokens),
             "--parquet", parquet,
             "--seed", str(seed),
             "--tensor-parallel-size", str(tensor_parallel_size),
             "--num-dp-workers", str(num_dp_workers)],
            f"Generate | no budget (baseline) | n={n}",
        )


def evaluate_stage(budgets: list[int], out_dir: str, baseline: bool) -> list[str]:
    result_files = []

    for budget in budgets:
        parquet = os.path.join(out_dir, f"gen_budget{budget}.parquet")
        output = os.path.join(out_dir, f"results_budget{budget}.json")
        run(
            [SCRIPT,
             "--evaluate",
             "--parquet", parquet,
             "--output", output],
            f"Evaluate | budget={budget} tokens",
        )
        result_files.append((str(budget), output))

    if baseline:
        parquet = os.path.join(out_dir, "gen_no_budget.parquet")
        output = os.path.join(out_dir, "results_no_budget.json")
        run(
            [SCRIPT,
             "--evaluate",
             "--parquet", parquet,
             "--output", output],
            "Evaluate | no budget (baseline)",
        )
        result_files.append(("none", output))

    return result_files


def summarize(result_files: list[tuple[str, str]], out_dir: str, model_id: str, dataset: str):
    rows = []
    per_label_rows = []

    for budget_label, path in result_files:
        if not os.path.exists(path):
            print(f"Warning: result file not found: {path}")
            continue
        with open(path) as f:
            r = json.load(f)

        ts = r.get("token_budget_stats", {})
        harmful = r["metrics"]["harmful_prompts"]
        benign = r["metrics"]["benign_prompts"]

        rows.append({
            "budget": budget_label,
            "mean_tokens": ts.get("mean_tokens"),
            "mean_delta": ts.get("mean_delta"),
            "over_budget_%": ts.get("over_budget_%"),
            "under_budget_%": ts.get("under_budget_%"),
            "attack_success_%": harmful.get("attack_success_rate_%"),
            "refusal_%": harmful.get("refusal_rate_%"),
            "false_refusal_%": benign.get("false_refusal_rate_%"),
            "benign_response_harmful_%": benign.get("response_flagged_harmful_%"),
        })

        for label, stats in r["metrics"]["per_label"].items():
            per_label_rows.append({
                "budget": budget_label,
                "label": label,
                "total": stats.get("total"),
                "response_harmful_%": stats.get("response_harmful_%"),
                "refusal_%": stats.get("refusal_%"),
                "parse_errors": stats.get("parse_errors"),
            })

    df = pd.DataFrame(rows)
    df_per_label = pd.DataFrame(per_label_rows)

    print(f"\n{'='*60}")
    print("  SWEEP SUMMARY")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    # --- Excel ---
    excel_path = os.path.join(out_dir, "sweep_results.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Summary", index=False)
        df_per_label.to_excel(writer, sheet_name="Per-Label", index=False)

        # Auto-fit column widths
        for sheet_name, sheet_df in [("Summary", df), ("Per-Label", df_per_label)]:
            ws = writer.sheets[sheet_name]
            for col_idx, col in enumerate(sheet_df.columns, 1):
                max_len = max(len(str(col)), sheet_df[col].fillna("").astype(str).str.len().max())
                ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = max_len + 2

    print(f"Excel saved to {excel_path}")

    # --- Plots ---
    plot_path = _plot(df, df_per_label, out_dir, model_id=model_id, dataset=dataset)
    print(f"Plots saved to {plot_path}")

    return df


def _numeric_budget(df: pd.DataFrame) -> pd.DataFrame:
    """Return df sorted with numeric budgets first, then 'none' baseline at the end."""
    df = df.copy()
    df["_sort"] = pd.to_numeric(df["budget"], errors="coerce")
    df = df.sort_values("_sort", na_position="last").drop(columns="_sort")
    return df


DATASET_LABEL = {
    "safechain": "UWNSL/SafeChain",
    "wildguard": "allenai/wildguardmix",
}


def _plot(df: pd.DataFrame, df_per_label: pd.DataFrame, out_dir: str, model_id: str, dataset: str) -> str:
    df = _numeric_budget(df)
    x_labels = df["budget"].tolist()
    x = range(len(x_labels))

    STYLE = {"marker": "o", "linewidth": 2, "markersize": 6}
    COLORS = {
        "vanilla_harmful": "#d62728",
        "adversarial_harmful": "#ff7f0e",
        "vanilla_benign": "#1f77b4",
        "adversarial_benign": "#2ca02c",
    }

    dataset_label = DATASET_LABEL.get(dataset, dataset)
    model_slug = model_id.split("/")[-1]

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle(
        f"Impact of Reasoning Token Budget on Safety\n({model_slug} · {dataset_label} · WildGuard)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # 1. Attack Success Rate (harmful prompts)
    ax = axes[0, 0]
    ax.plot(x, df["attack_success_%"], color="#d62728", label="Attack success", **STYLE)
    ax.set_title("Attack Success Rate (harmful prompts)")
    ax.set_ylabel("%")
    ax.set_xticks(x); ax.set_xticklabels(x_labels)
    ax.set_xlabel("Token Budget")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2. Refusal Rate (harmful prompts)
    ax = axes[0, 1]
    ax.plot(x, df["refusal_%"], color="#1f77b4", label="Refusal rate", **STYLE)
    ax.set_title("Refusal Rate (harmful prompts)")
    ax.set_ylabel("%")
    ax.set_xticks(x); ax.set_xticklabels(x_labels)
    ax.set_xlabel("Token Budget")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 3. False Refusal Rate (benign prompts)
    ax = axes[1, 0]
    ax.plot(x, df["false_refusal_%"], color="#9467bd", label="False refusal", **STYLE)
    ax.set_title("False Refusal Rate (benign prompts)")
    ax.set_ylabel("%")
    ax.set_xticks(x); ax.set_xticklabels(x_labels)
    ax.set_xlabel("Token Budget")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 4. Benign response flagged harmful
    ax = axes[1, 1]
    ax.plot(x, df["benign_response_harmful_%"], color="#8c564b", label="Response flagged harmful", **STYLE)
    ax.set_title("Benign Response Flagged Harmful %")
    ax.set_ylabel("%")
    ax.set_xticks(x); ax.set_xticklabels(x_labels)
    ax.set_xlabel("Token Budget")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 5. Per-label response harmful % across budgets
    ax = axes[2, 0]
    for label, color in COLORS.items():
        sub = _numeric_budget(df_per_label[df_per_label["label"] == label])
        if sub.empty:
            continue
        xi = range(len(sub))
        ax.plot(xi, sub["response_harmful_%"], color=color, label=label, **STYLE)
    ax.set_title("Response Harmful % by Label")
    ax.set_ylabel("%")
    ax.set_xticks(x); ax.set_xticklabels(x_labels)
    ax.set_xlabel("Token Budget")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 6. Token budget adherence: mean actual tokens vs budget
    ax = axes[2, 1]
    numeric_budgets = pd.to_numeric(df["budget"], errors="coerce")
    ax.plot(x, df["mean_tokens"], color="#17becf", label="Mean actual tokens", **STYLE)
    ax.plot(
        [xi for xi, b in zip(x, numeric_budgets) if pd.notna(b)],
        [b for b in numeric_budgets if pd.notna(b)],
        color="gray", linestyle="--", linewidth=1.5, label="Set budget",
    )
    ax.set_title("Actual Tokens Used vs Set Budget")
    ax.set_ylabel("Tokens")
    ax.set_xticks(x); ax.set_xticklabels(x_labels)
    ax.set_xlabel("Token Budget")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "sweep_plots.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main():
    parser = argparse.ArgumentParser(description="Token budget sweep for a reasoning model + WildGuard")
    parser.add_argument("--dataset", type=str, default="safechain", choices=("safechain", "wildguard"),
                        help="Dataset to sweep on: 'safechain' (UWNSL/SafeChain) or "
                             "'wildguard' (allenai/wildguardmix, wildguardtest split). Default: safechain")
    parser.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096],
                        help="List of token budgets to sweep (default: 256 512 1024 2048 4096)")
    parser.add_argument("--n", type=int, default=500,
                        help="Number of dataset samples per run (default: 500)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Max generation tokens (should be >= largest budget, default: 8192)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed — same seed ensures all runs see identical samples")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip the no-budget baseline run")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory to save parquets and results (default: sweep_<timestamp>/)")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Skip generation, only run WildGuard on existing parquets in --out-dir")
    parser.add_argument("--model-id", type=str, default="l3lab/L1-Qwen-1.5B-Exact",
                        help="HuggingFace model ID to evaluate (default: l3lab/L1-Qwen-1.5B-Exact)")
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="GPUs per model replica (vLLM tensor parallelism). Default: 4.")
    parser.add_argument("--num-dp-workers", type=int, default=1,
                        help="Number of data-parallel model replicas, each using "
                             "--tensor-parallel-size GPUs. Default: 1 (no data parallelism).")
    args = parser.parse_args()

    out_dir = args.out_dir or f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print(f"Model: {args.model_id}")
    print(f"Dataset: {args.dataset}")
    print(f"Budgets: {args.budgets}")
    print(f"Samples per run: {args.n}")
    print(f"Baseline: {not args.no_baseline}")

    if not args.evaluate_only:
        generate_stage(
            budgets=args.budgets,
            n=args.n,
            max_tokens=args.max_tokens,
            seed=args.seed,
            out_dir=out_dir,
            baseline=not args.no_baseline,
            model_id=args.model_id,
            dataset=args.dataset,
            tensor_parallel_size=args.tensor_parallel_size,
            num_dp_workers=args.num_dp_workers,
        )

    result_files = evaluate_stage(
        budgets=args.budgets,
        out_dir=out_dir,
        baseline=not args.no_baseline,
    )

    summarize(result_files, out_dir, model_id=args.model_id, dataset=args.dataset)


if __name__ == "__main__":
    main()
