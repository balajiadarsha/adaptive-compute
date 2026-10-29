"""
Collate safety results (score_safety.py jsons) into the safety tables:
SafeChain and WildGuardMix attack success / false refusal (WildGuard) and the
StrongREJECT score, each cell "answer / trace", plus mean output length and
the share of responses with no final answer.

Two tables: length control (BF16 models across budgets) and quantization.
Missing runs print as "--", so this also serves as the status table while
runs are in flight.

Usage (from eval/): python make_safety_table.py
Writes ../context/safety_results_table.md, .tex and _standalone.tex.
"""
import argparse
import json
import os

HERE = os.path.dirname(__file__)
# Budget rows: 8192-token cap. No-limit rows: 30000-token cap, no budget prompt.
BUDGET_DIR = os.path.join(HERE, "results", "safety_t06")
NO_LIMIT_DIR = os.path.join(HERE, "results", "safety_32k_t06")
BUDGETS = [512, 1024, 2048, 4096]
ALL = BUDGETS + [None]

# (slug, display name, budgets); a None budget is the "No limit" row.
LENGTH_CONTROL = [
    ("DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-1.5B", ALL),
    ("DeepScaleR-1.5B-Preview", "DeepScaleR-1.5B", ALL),
    ("L1-Qwen-1.5B-Exact", "L1-Qwen-1.5B", ALL),
    ("L1-Qwen-7B-Exact", "L1-Qwen-7B", ALL),
]
QUANTIZATION = [
    ("DeepSeek-R1-Distill-Qwen-1.5B-gptq-int8", "DeepSeek-R1-1.5B (GPTQ-INT8)", [None]),
    ("DeepSeek-R1-Distill-Qwen-1.5B-gptq-int4", "DeepSeek-R1-1.5B (GPTQ-INT4)", [None]),
    ("DeepSeek-R1-Distill-Qwen-1.5B-awq-w4a16_asym", "DeepSeek-R1-1.5B (AWQ-W4A16)", [None]),
    ("DeepSeek-R1-Distill-Qwen-1.5B-flatquant-w8a8kv8", "DeepSeek-R1-1.5B (FlatQuant-W8A8KV8)§", ["fq"]),
    ("L1-Qwen-1.5B-Exact-gptq-int8", "L1-Qwen-1.5B (GPTQ-INT8)", BUDGETS),
    ("L1-Qwen-1.5B-Exact-gptq-int4", "L1-Qwen-1.5B (GPTQ-INT4)", BUDGETS),
    ("L1-Qwen-1.5B-Exact-awq-w4a16_asym", "L1-Qwen-1.5B (AWQ-W4A16)", BUDGETS),
    ("L1-Qwen-1.5B-Exact-flatquant-w8a8kv8", "L1-Qwen-1.5B (FlatQuant-W8A8KV8)", [1024]),
    ("L1-Qwen-1.5B-Exact-flatquant-w4a4kv4", "L1-Qwen-1.5B (FlatQuant-W4A4KV4)", [1024]),
    ("L1-Qwen-7B-Exact-gptq-int8", "L1-Qwen-7B (GPTQ-INT8)", BUDGETS),
    ("L1-Qwen-7B-Exact-gptq-int4", "L1-Qwen-7B (GPTQ-INT4)", BUDGETS),
    ("L1-Qwen-7B-Exact-awq-w4a16_asym", "L1-Qwen-7B (AWQ-W4A16)", BUDGETS),
    ("L1-Qwen-7B-Exact-flatquant-w8a8kv8", "L1-Qwen-7B (FlatQuant-W8A8KV8)", [1024]),
    ("L1-Qwen-7B-Exact-flatquant-w4a4kv4", "L1-Qwen-7B (FlatQuant-W4A4KV4)", [1024]),
]

# (header, dataset, metric path within the answer/trace metrics, kind)
COLUMNS = [
    ("SafeChain ASR", "safechain", ("harmful_prompts", "attack_success_rate_%"), "pct"),
    ("SafeChain FRR", "safechain", ("benign_prompts", "false_refusal_rate_%"), "pct"),
    ("WGMix ASR", "wildguard", ("harmful_prompts", "attack_success_rate_%"), "pct"),
    ("WGMix FRR", "wildguard", ("benign_prompts", "false_refusal_rate_%"), "pct"),
    ("StrongREJECT", "strongreject", ("strongreject_score",), "score"),
]
DATASETS = ["safechain", "wildguard", "strongreject"]


def run_dir(slug: str, budget) -> str:
    if budget is None:
        return os.path.join(NO_LIMIT_DIR, slug)
    if budget == "fq":  # DeepSeek FlatQuant: no budget prompt, 8192 cap
        return os.path.join(BUDGET_DIR, slug)
    return os.path.join(BUDGET_DIR, f"{slug}-budget{budget}")


def load(slug: str, budget) -> dict:
    out = {}
    for ds in DATASETS:
        path = os.path.join(run_dir(slug, budget), f"{ds}.json")
        if os.path.exists(path):
            with open(path) as f:
                out[ds] = json.load(f)["metrics"]
    return out


def dig(d, keys):
    for k in keys:
        d = d.get(k) if d else None
    return d


def build_rows(models):
    """Yield (name, budget, cells): one "answer / trace" string per column,
    then mean length and no-answer %, or "--" where results are missing."""
    for slug, name, budgets in models:
        for budget in budgets:
            res = load(slug, budget)
            cells = []
            for _, ds, keys, kind in COLUMNS:
                m = res.get(ds)
                ans, tr = (dig(m, ("answer",) + keys), dig(m, ("trace",) + keys)) if m else (None, None)
                if ans is None or tr is None:
                    cells.append("--")
                elif kind == "pct":
                    cells.append(f"{ans:.1f} / {tr:.1f}")
                else:
                    cells.append(f"{ans:.2f} / {tr:.2f}")
            if len(res) == len(DATASETS):
                cells.append(f"{sum(m['mean_token_count'] for m in res.values()) / len(res):.0f}")
                cells.append(f"{sum(m['no_answer_%'] for m in res.values()) / len(res):.1f}")
            else:
                cells += ["--", "--"]
            yield name, budget, cells


def budget_label(budget) -> str:
    return "No limit" if budget in (None, "fq") else str(budget)


HEADERS = [c[0] for c in COLUMNS] + ["Len", "No ans."]

NOTE = ("Each cell is answer / trace: the final answer (after </think>) and the reasoning trace are judged "
        "separately. ASR: attack success rate (% of harmful prompts with a harmful response, WildGuard). "
        "FRR: false refusal rate (% of benign prompts refused). StrongREJECT: mean score of the fine-tuned "
        "StrongREJECT judge, 0 (refusal or useless) to 1 (fully helpful to the attacker). Lower is safer for "
        "all three; Len is mean output tokens and No ans. the % of responses that never close </think> "
        "(averaged over the three datasets); a missing answer counts as neither harmful nor a refusal. "
        "SafeChain: 1000 prompts disjoint from the GPTQ/AWQ calibration set; WildGuardMix: all 1699 "
        "WildGuardTest prompts; StrongREJECT: all 313 prompts. Sampling at temperature 0.6, top-p 0.95; "
        "budget rows cap at 8192 tokens, No limit rows at 30000 with no budget prompt.")
FQ_NOTE = " § No budget prompt, 8192-token cap."


def to_markdown(title, rows) -> str:
    out = [f"### {title}", "", "| Model | Budget | " + " | ".join(HEADERS) + " |",
           "|---|---|" + "---|" * len(HEADERS)]
    last = None
    for name, budget, cells in rows:
        label = f"**{name}**" if name != last else ""
        last = name
        out.append(f"| {label} | {budget_label(budget)} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def latex_escape(text: str) -> str:
    return (text.replace("%", r"\%").replace("§", r"$^\S$")
            .replace("</think>", r"\texttt{</think>}"))


def to_latex(caption, label, rows) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll" + "c" * len(HEADERS) + "}",
        r"\toprule",
        r" & Budget & " + " & ".join(rf"\textbf{{{h}}}$\downarrow$" if i < len(COLUMNS) else rf"\textbf{{{h}}}"
                                     for i, h in enumerate(HEADERS)) + r" \\",
        rf"\cmidrule(lr){{3-{2 + len(COLUMNS)}}}",
        rf" & & \multicolumn{{{len(COLUMNS)}}}{{c}}{{\textit{{Answer / Trace}}}} & \textit{{tok}} & \textit{{\%}} \\",
        r"\midrule",
    ]
    for i, (name, budget, cells) in enumerate(rows):
        first = i == 0 or rows[i - 1][0] != name
        last = i == len(rows) - 1 or rows[i + 1][0] != name
        n_rows = sum(r[0] == name for r in rows)
        head = rf"\multirow{{{n_rows}}}{{*}}{{{latex_escape(name)}}}" if first else ""
        lines.append(f"{head} & {budget_label(budget)} & " + " & ".join(cells) + r" \\")
        if last:
            lines.append(r"\midrule" if i < len(rows) - 1 else r"\bottomrule")
    lines += [r"\end{tabular}", rf"\caption{{{latex_escape(caption)}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-prefix", type=str, default=os.path.join("..", "context", "safety_results_table"),
                        help="Writes <prefix>.md, <prefix>.tex and <prefix>_standalone.tex")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    lc_rows = list(build_rows(LENGTH_CONTROL))
    q_rows = list(build_rows(QUANTIZATION))
    lc_caption = "Safety under length control (BF16). " + NOTE
    q_caption = ("Safety of quantized variants. Same columns and settings as Table~\\ref{tab:safety-length}; "
                 "FlatQuant L1 rows use budget 1024 only." + FQ_NOTE)

    md = (to_markdown("Safety under length control (BF16)", lc_rows) + "\n"
          + to_markdown("Safety of quantized variants", q_rows) + "\n" + NOTE + FQ_NOTE + "\n")
    tex = (to_latex(lc_caption, "tab:safety-length", lc_rows) + "\n"
           + to_latex(q_caption, "tab:safety-quant", q_rows))
    with open(args.out_prefix + ".md", "w") as f:
        f.write(md.replace("~\\ref{tab:safety-length}", " 1"))
    with open(args.out_prefix + ".tex", "w") as f:
        f.write(tex)
    with open(args.out_prefix + "_standalone.tex", "w") as f:
        f.write("\\documentclass[11pt]{article}\n"
                "\\usepackage[margin=0.4in,landscape]{geometry}\n"
                "\\usepackage{booktabs,multirow}\n"
                "\\begin{document}\n"
                + tex.replace(r"\begin{table}[t]", r"\begin{table}[p]") +
                "\\end{document}\n")
    print(md)
    print(f"Wrote {args.out_prefix}.md, {args.out_prefix}.tex and {args.out_prefix}_standalone.tex")


if __name__ == "__main__":
    main()
