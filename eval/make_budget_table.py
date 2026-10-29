"""
Build the model x reasoning-budget Acc/Len table (Markdown + LaTeX) from
completed full-dataset skill runs under results/full_eval/.

Any (model, budget, dataset) cell without a finished run falls back to the
reference table's value and is marked with a dagger, so rerunning this
script as jobs finish swaps reference cells for our own results.

Accuracy counts unparsed answers as incorrect (correct / total), and Len is
the mean total output tokens per response.

Usage (from eval/):
    python make_budget_table.py --out-prefix ../context/budget_results_table
"""
import argparse
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "full_eval")
# Unconstrained runs (no "Think for N tokens." prompt, 30000-token cap on
# every dataset) from scripts/jobs/run_skill_grid.sh ... 32k none.
NO_LIMIT_DIR = os.path.join(os.path.dirname(__file__), "results", "full_eval_32k")
# BF16 sections sampled at temperature 0.6 / top-p 0.95 (TEMPERATURE=0.6 TOP_P=0.95
# run_skill_grid.sh); greedy decoding made DeepSeek-R1 and DeepScaleR loop until
# max_tokens. The GPTQ/AWQ sections are still the greedy runs.
SAMPLED_MODELS = {"DeepSeek-R1-Distill-Qwen-1.5B", "DeepScaleR-1.5B-Preview", "L1-Qwen-1.5B-Exact"}
SAMPLED_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "full_eval_t06")
SAMPLED_NO_LIMIT_DIR = os.path.join(os.path.dirname(__file__), "results", "full_eval_32k_t06")
DATASETS = [("aime", "AIME"), ("math", "MATH"), ("gsm8k", "GSM8K"), ("gpqa", "GPQA")]
BUDGETS = [512, 1024, 2048, 4096]
# Sections that also get a "No limit" row (budget None).
NO_LIMIT_MODELS = {"DeepSeek-R1-Distill-Qwen-1.5B", "DeepScaleR-1.5B-Preview", "L1-Qwen-1.5B-Exact"}
MODELS = [
    ("DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-1.5B"),
    ("DeepScaleR-1.5B-Preview", "DeepScaleR-1.5B"),
    ("L1-Qwen-1.5B-Exact", "L1-Qwen-1.5B"),
    ("L1-Qwen-1.5B-Exact-gptq-int8", "L1-Qwen-1.5B (GPTQ-INT8)*"),
    ("L1-Qwen-1.5B-Exact-gptq-int4", "L1-Qwen-1.5B (GPTQ-INT4)*"),
    ("L1-Qwen-1.5B-Exact-awq-w4a16_asym", "L1-Qwen-1.5B (AWQ-W4A16)*"),
    ("flatquant-w8a8kv8", "DeepSeek-R1-1.5B (FlatQuant-W8A8KV8)§"),
    ("L1-Qwen-1.5B-Exact-flatquant-w8a8kv8", "L1-Qwen-1.5B (FlatQuant-W8A8KV8)‡"),
    ("L1-Qwen-1.5B-Exact-flatquant-w4a4kv4", "L1-Qwen-1.5B (FlatQuant-W4A4KV4)‡"),
    ("L1-Qwen-7B-Exact-flatquant-w8a8kv8", "L1-Qwen-7B (FlatQuant-W8A8KV8)‡"),
    ("L1-Qwen-7B-Exact-flatquant-w4a4kv4", "L1-Qwen-7B (FlatQuant-W4A4KV4)‡"),
]

# FlatQuant sections: budget 1024 only, sampled at temperature 0.6 / top-p 0.95
# (scripts/jobs/run_flatquant_grid.sh), so they read from full_eval_t06.
FLATQUANT_DIR = os.path.join(os.path.dirname(__file__), "results", "full_eval_t06")
FLATQUANT_BUDGETS = [1024]
# DeepSeek FlatQuant W8A8KV8 (run_skill_sweep.py): greedy, no budget prompt,
# caps 8192 (AIME, MATH, GPQA) / 4096 (GSM8K), in full_eval/flatquant-w8a8kv8.
DEEPSEEK_FLATQUANT = "flatquant-w8a8kv8"

# Reference table (results.png): {slug: {budget: [aime, math, gsm8k, gpqa, mmlu, avg]}},
# kept verbatim; the mmlu entry and the reference avg (which includes MMLU) are unused.
# each entry (acc %, len tokens).
REFERENCE = {
    "DeepSeek-R1-Distill-Qwen-1.5B": {
        512: [(25.5, 538), (81.6, 555), (83.5, 493), (30.3, 519), (41.9, 489), (52.6, 519)],
        1024: [(24.0, 1057), (87.8, 1007), (80.3, 1063), (30.2, 1097), (41.4, 974), (52.7, 1040)],
        2048: [(26.1, 2049), (90.3, 1974), (84.3, 1975), (30.0, 2206), (43.2, 2131), (54.8, 2067)],
        4096: [(26.1, 4230), (90.7, 4401), (86.8, 4271), (33.8, 4421), (39.8, 4246), (55.4, 4314)],
    },
    "DeepScaleR-1.5B-Preview": {
        512: [(48.1, 504), (89.2, 494), (89.7, 508), (49.7, 535), (36.3, 514), (62.6, 511)],
        1024: [(45.9, 1066), (91.5, 999), (90.2, 1085), (54.5, 998), (39.5, 1031), (64.3, 1036)],
        2048: [(50.1, 2184), (97.5, 2016), (94.2, 1955), (52.6, 2043), (40.6, 2028), (67.0, 2045)],
        4096: [(44.8, 4453), (100.0, 4453), (98.5, 4173), (49.9, 4054), (39.5, 4043), (66.5, 4171)],
    },
}


def cell_path(slug: str, budget: int | None, dataset: str) -> str:
    if slug == DEEPSEEK_FLATQUANT:
        path = os.path.join(RESULTS_DIR, slug, f"{dataset}.json")
    elif budget is None:
        no_limit_dir = SAMPLED_NO_LIMIT_DIR if slug in SAMPLED_MODELS else NO_LIMIT_DIR
        path = os.path.join(no_limit_dir, slug, f"{dataset}.json")
    elif "flatquant" in slug:
        path = os.path.join(FLATQUANT_DIR, f"{slug}-budget{budget}", f"{dataset}.json")
    else:
        results_dir = SAMPLED_RESULTS_DIR if slug in SAMPLED_MODELS else RESULTS_DIR
        path = os.path.join(results_dir, f"{slug}-budget{budget}", f"{dataset}.json")
    return path


def our_cell(slug: str, budget: int | None, dataset: str):
    path = cell_path(slug, budget, dataset)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        result = json.load(f)
    rows = result["data"]
    acc = 100.0 * result["metrics"]["correct"] / result["metrics"]["total"]
    length = sum(int(r["token_count"]) for r in rows) / len(rows)
    return acc, length


def build_rows():
    """Yield (model name, budget, cells, avg) where each cell / avg is
    (acc, len, is_reference) or None."""
    for slug, name in MODELS:
        ref = REFERENCE.get(slug, {})
        if slug == DEEPSEEK_FLATQUANT:
            budgets = [None]
        elif "flatquant" in slug:
            budgets = FLATQUANT_BUDGETS
        else:
            budgets = BUDGETS + [None] if slug in NO_LIMIT_MODELS else BUDGETS
        for budget in budgets:
            cells = []
            for i, (ds, _) in enumerate(DATASETS):
                ours = our_cell(slug, budget, ds)
                if ours:
                    cells.append((*ours, False))
                elif budget in ref:
                    cells.append((*ref[budget][i], True))
                else:
                    cells.append(None)

            # Average of the displayed cells, daggered if any came from the
            # reference (the reference's own avg includes MMLU, so it isn't reused).
            if all(cells):
                avg = (sum(c[0] for c in cells) / len(cells), sum(c[1] for c in cells) / len(cells),
                       any(c[2] for c in cells))
            else:
                avg = None
            yield name, budget, cells, avg


def fmt(cell, dagger: str) -> str:
    if cell is None:
        return "--"
    acc, length, is_ref = cell
    return f"{acc:.1f} / {length:.0f}" + (dagger if is_ref else "")


def budget_label(budget: int | None) -> str:
    return "No limit" if budget is None else str(budget)


def note(rows) -> str:
    text = ("Acc (%) / Len (tok). Accuracy counts unparsed answers as incorrect; Len is mean "
            "output tokens; Avg. is over the four datasets. No limit: no budget prompt, 30000-token "
            "cap; budget rows cap at 30000 (AIME), 8192 (MATH, GPQA), 4096 (GSM8K).")
    text += " Unmarked rows sample at temperature 0.6, top-p 0.95; rows marked * use greedy decoding."
    if any("‡" in name for name, _, _, _ in rows):
        text += " ‡ Budget 1024 only, sampled at temperature 0.6, top-p 0.95."
    if any("§" in name for name, _, _, _ in rows):
        text += (" § Greedy, no budget prompt, AIME capped at 8192 tokens (MATH, GPQA 8192; "
                 "GSM8K 4096) instead of 30000.")
    if any(c and c[2] for _, _, cells, _ in rows for c in cells):
        text += " Cells marked † are copied from the reference table; our run is not complete yet."
    return text


def to_markdown(rows) -> str:
    out = ["| Model | Budget | " + " | ".join(n for _, n in DATASETS) + " | Avg. |",
           "|---|---|" + "---|" * (len(DATASETS) + 1)]
    last = None
    for name, budget, cells, avg in rows:
        label = "**" + name.replace("*", r"\*") + "**" if name != last else ""
        last = name
        out.append(f"| {label} | {budget_label(budget)} | " + " | ".join(fmt(c, "†") for c in cells) + f" | {fmt(avg, '†')} |")
    return "\n".join(out) + f"\n\n{note(rows)}\n"


def latex_marks(text: str) -> str:
    return (text.replace("†", r"$^\dagger$").replace("‡", r"$^\ddagger$")
            .replace("§", r"$^\S$").replace("*", r"$^*$"))


def to_latex(rows) -> str:
    rows = list(rows)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{ll" + "c" * (len(DATASETS) + 1) + "}",
        r"\toprule",
        r" & Budget & " + " & ".join(rf"\textbf{{{n}}}" for _, n in DATASETS) + r" & \textbf{Avg.} \\",
        rf"\cmidrule(lr){{3-{2 + len(DATASETS)}}}\cmidrule(lr){{{3 + len(DATASETS)}-{3 + len(DATASETS)}}}",
        rf" & & \multicolumn{{{len(DATASETS)}}}{{c}}{{\textit{{Acc (\%) / Len (tok)}}}} & \textit{{Acc / Len}} \\",
        r"\midrule",
    ]
    for i, (name, budget, cells, avg) in enumerate(rows):
        first = i == 0 or rows[i - 1][0] != name
        last = i == len(rows) - 1 or rows[i + 1][0] != name
        n_rows = sum(r[0] == name for r in rows)
        head = rf"\multirow{{{n_rows}}}{{*}}{{{latex_marks(name)}}}" if first else ""
        lines.append(f"{head} & {budget_label(budget)} & " + " & ".join(fmt(c, r"$^\dagger$") for c in cells + [avg]) + r" \\")
        if last:
            lines.append(r"\midrule" if i < len(rows) - 1 else r"\bottomrule")
    caption = latex_marks(note(rows)).replace("%", r"\%")
    lines += [r"\end{tabular}", rf"\caption{{{caption}}}", r"\label{tab:budget}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-prefix", type=str, default=os.path.join("..", "context", "budget_results_table"),
                        help="Writes <prefix>.md and <prefix>.tex")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    rows = list(build_rows())
    md, tex = to_markdown(rows), to_latex(rows)
    with open(args.out_prefix + ".md", "w") as f:
        f.write(md)
    with open(args.out_prefix + ".tex", "w") as f:
        f.write(tex)
    # Compilable document around the same table, for viewing or sharing
    # without a paper to \input it into.
    # Natural size fits a landscape page; scaling to \textwidth made it too
    # tall for the caption once the "No limit" rows were added.
    standalone = tex
    with open(args.out_prefix + "_standalone.tex", "w") as f:
        f.write("\\documentclass[11pt]{article}\n"
                "\\usepackage[margin=0.4in,landscape]{geometry}\n"
                "\\usepackage{booktabs,multirow,graphicx}\n"
                "\\begin{document}\n"
                + standalone.replace(r"\begin{table}[t]", r"\begin{table}[h]") +
                "\\end{document}\n")
    print(md)
    print(f"Wrote {args.out_prefix}.md, {args.out_prefix}.tex and {args.out_prefix}_standalone.tex")


if __name__ == "__main__":
    main()
