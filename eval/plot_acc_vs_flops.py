"""
Accuracy vs reasoning compute (FLOPs) for L1-Qwen and its quantized variants,
one panel per skill dataset plus the four-dataset average.

Each point is one (model, precision, budget) run from the same results that
make_budget_table.py reads. Compute is the mean analytical FLOPs per problem
(utils/flops.py: prefill + decode backbone, lm_head per sampled token, causal
attention), using each response's re-tokenized prompt and its token_count.
FLOPs are precision-independent (see utils/flops.py), so quantized variants
differ from BF16 here only through how many tokens they generate.

Usage (from eval/):
    python plot_acc_vs_flops.py --out-prefix ../context/acc_vs_flops
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from flops import QWEN2_1_5B, ModelFlopsConfig, total_flops_for_response  # noqa: E402
from make_budget_table import BUDGETS, DATASETS, cell_path  # noqa: E402

# Real config.json of l3lab/L1-Qwen-7B-Exact (Qwen2.5-7B backbone).
QWEN2_7B = ModelFlopsConfig(
    hidden_size=3584,
    num_hidden_layers=28,
    num_attention_heads=28,
    num_key_value_heads=4,
    intermediate_size=18944,
    vocab_size=152064,
    tie_word_embeddings=False,
)
# Both L1 sizes share the Qwen2.5 tokenizer; any local checkpoint has it.
TOKENIZER_DIR = os.path.join(os.path.dirname(__file__), "..", "quantize", "checkpoints",
                             "L1-Qwen-1.5B-Exact-gptq-int8")

# (slug, size label, precision label, budgets); None is the no-limit run.
RUNS = [
    ("L1-Qwen-1.5B-Exact", "1.5B", "BF16", BUDGETS + [None]),
    ("L1-Qwen-1.5B-Exact-gptq-int8", "1.5B", "GPTQ-INT8*", BUDGETS),
    ("L1-Qwen-1.5B-Exact-gptq-int4", "1.5B", "GPTQ-INT4*", BUDGETS),
    ("L1-Qwen-1.5B-Exact-awq-w4a16_asym", "1.5B", "AWQ-W4A16*", BUDGETS),
    ("L1-Qwen-1.5B-Exact-flatquant-w8a8kv8", "1.5B", "FlatQuant-W8A8KV8", [1024]),
    ("L1-Qwen-1.5B-Exact-flatquant-w4a4kv4", "1.5B", "FlatQuant-W4A4KV4", [1024]),
    ("L1-Qwen-7B-Exact-flatquant-w8a8kv8", "7B", "FlatQuant-W8A8KV8", [1024]),
    ("L1-Qwen-7B-Exact-flatquant-w4a4kv4", "7B", "FlatQuant-W4A4KV4", [1024]),
]
# Categorical slots 1-6 of the dataviz reference palette, in fixed order.
PRECISION_COLORS = {
    "BF16": "#2a78d6",
    "GPTQ-INT8": "#eb6834",
    "GPTQ-INT4": "#1baf7a",
    "AWQ-W4A16": "#eda100",
    "FlatQuant-W8A8KV8": "#e87ba4",
    "FlatQuant-W4A4KV4": "#008300",
}
BUDGET_MARKERS = {512: "o", 1024: "s", 2048: "^", 4096: "p", None: "*"}


def budget_label(budget):
    return "No limit" if budget is None else str(budget)


def prompt_length_fn():
    """Cached prompt -> token count with the shared L1 tokenizer."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
    prompt_lens = {}

    def prompt_len(prompt):
        if prompt not in prompt_lens:
            prompt_lens[prompt] = len(tok.encode(prompt, add_special_tokens=False))
        return prompt_lens[prompt]
    return prompt_len


def collect_points():
    prompt_len = prompt_length_fn()
    points = []
    for slug, size, precision, budgets in RUNS:
        cfg = QWEN2_7B if size == "7B" else QWEN2_1_5B
        for budget in budgets:
            per_ds = {}
            for ds, _ in DATASETS:
                path = cell_path(slug, budget, ds)
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    result = json.load(f)
                rows = result["data"]
                flops = sum(total_flops_for_response(cfg, prompt_len(r["prompt"]), int(r["token_count"]))["total_flops"]
                            for r in rows) / len(rows)
                acc = 100.0 * result["metrics"]["correct"] / result["metrics"]["total"]
                tokens = sum(int(r["token_count"]) for r in rows) / len(rows)
                per_ds[ds] = (acc, flops, tokens)
            if len(per_ds) == len(DATASETS):
                n = len(DATASETS)
                per_ds["avg"] = tuple(sum(v[i] for v in per_ds.values()) / n for i in range(3))
            for ds, (acc, flops, tokens) in per_ds.items():
                points.append({"size": size, "precision": precision, "budget": budget, "panel": ds,
                               "y": acc, "flops": flops, "tokens": tokens})
            print(f"{size} {precision} {budget_label(budget)}: {len(per_ds)} cells")
    return points


def plot_panels(points, panels, out_prefix, footnote):
    """panels: [(key, title, y label)]; each point carries "panel", "size",
    "precision", "budget", "flops" and "y". 1.5B runs are filled markers on
    solid/dashed lines, 7B runs hollow markers on dotted lines."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    ink, muted, grid = "#2b2b2b", "#6b6b6b", "#e6e6e6"
    series_keys = list(dict.fromkeys((p["size"], p["precision"]) for p in points))
    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        for size, precision in series_keys:
            pts = sorted((p for p in points if p["panel"] == key and p["size"] == size
                          and p["precision"] == precision), key=lambda p: p["flops"])
            if not pts:
                continue
            color = PRECISION_COLORS[precision.rstrip("*")]
            hollow = size == "7B"
            if len(pts) > 1:
                ax.plot([p["flops"] / 1e12 for p in pts], [p["y"] for p in pts], color=color, lw=2,
                        ls=line_style(size, precision), zorder=2)
            for p in pts:
                ax.scatter(p["flops"] / 1e12, p["y"], marker=BUDGET_MARKERS[p["budget"]],
                           s=110 if p["budget"] is None else 60, zorder=3, linewidths=1.6,
                           facecolors="white" if hollow else color, edgecolors=color if hollow else "white")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, color=ink, fontsize=12, fontweight="bold")
        ax.grid(True, which="major", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(muted)
        ax.tick_params(colors=muted, labelsize=9)
        ax.set_xlabel("Mean TFLOPs per problem (log)", color=muted, fontsize=10)
        ax.set_ylabel(ylabel, color=muted, fontsize=10)

    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    series = []
    for size, precision in series_keys:
        color = PRECISION_COLORS[precision.rstrip("*")]
        hollow = size == "7B"
        series.append(Line2D([], [], color=color, lw=2, ls=line_style(size, precision), marker="s",
                             markersize=7, markeredgewidth=1.6, markerfacecolor="white" if hollow else color,
                             markeredgecolor=color if hollow else "white", label=f"{size} {precision.replace('FlatQuant-', 'FQ-')}"))
    budgets = [Line2D([], [], ls="none", marker=m, markersize=12 if b is None else 8, color=muted,
                      label=budget_label(b)) for b, m in BUDGET_MARKERS.items()]
    leg1 = legend_ax.legend(handles=series, title="L1 model / precision", loc="upper left",
                            frameon=False, fontsize=8.5, title_fontsize=10, labelcolor=ink,
                            ncol=2 if len(series) > 8 else 1, columnspacing=0.8)
    legend_ax.add_artist(leg1)
    legend_ax.legend(handles=budgets, title="Reasoning budget", loc="lower left",
                     frameon=False, fontsize=9, title_fontsize=10, labelcolor=ink, ncol=3)
    fig.text(0.01, 0.005, footnote, fontsize=8, color=muted, wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=200)
    print(f"Wrote {out_prefix}.pdf and {out_prefix}.png")


def line_style(size, precision):
    if size == "7B":
        return ":"
    return "-" if precision == "BF16" else "--"


FLOPS_NOTE = ("FLOPs are analytical and precision-independent (quantization changes kernels, not "
              "multiply-adds), so quantized variants differ from BF16 only in tokens generated. FQ = FlatQuant.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-prefix", type=str, default=os.path.join("..", "context", "acc_vs_flops"),
                        help="Writes <prefix>.pdf, <prefix>.png and <prefix>.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    points = collect_points()
    with open(args.out_prefix + ".csv", "w") as f:
        f.write("size,precision,budget,dataset,acc,mean_flops,mean_tokens\n")
        for p in points:
            f.write(f"{p['size']},{p['precision']},{budget_label(p['budget'])},{p['panel']},"
                    f"{p['y']:.2f},{p['flops']:.4e},{p['tokens']:.1f}\n")
    panels = [(ds, title, "Accuracy (%)") for ds, title in DATASETS] + [("avg", "Average", "Accuracy (%)")]
    plot_panels(points, panels, args.out_prefix,
                FLOPS_NOTE + " * Greedy decoding; others sampled at T=0.6, top-p 0.95. "
                "L1-7B BF16/GPTQ/AWQ skill runs are not done yet.")


if __name__ == "__main__":
    main()
