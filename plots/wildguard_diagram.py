"""
Generate a diagram describing the WildGuard safety classifier.
Output: wildguard_diagram.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def rounded_box(ax, x, y, w, h, color, text, fontsize=10, text_color="white", bold=False):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.04",
        facecolor=color, edgecolor="white", linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x, y, text,
        ha="center", va="center", fontsize=fontsize,
        color=text_color, fontweight="bold" if bold else "normal",
        zorder=4, wrap=True,
        multialignment="center",
    )


def arrow(ax, x0, y0, x1, y1, color="#555555"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8),
        zorder=2,
    )


fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

# ── Title ──────────────────────────────────────────────────────────────────
ax.text(7, 8.5, "WildGuard", ha="center", va="center",
        fontsize=22, fontweight="bold", color="white")
ax.text(7, 8.1, "Open One-Stop Safety Classifier for LLM Interactions  ·  NeurIPS 2024",
        ha="center", va="center", fontsize=10, color="#aaaaaa")

# ── Input column ────────────────────────────────────────────────────────────
ax.text(1.5, 7.3, "INPUTS", ha="center", fontsize=9, color="#aaaaaa", fontweight="bold")
rounded_box(ax, 1.5, 6.6, 2.4, 0.7, "#1a6b8a", "User Prompt", fontsize=10, bold=True)
rounded_box(ax, 1.5, 5.7, 2.4, 0.7, "#1a6b8a", "AI Response\n(optional)", fontsize=9)

# bracket / brace connecting inputs
ax.annotate("", xy=(3.1, 6.15), xytext=(3.1, 6.6),
            arrowprops=dict(arrowstyle="-", color="#555", lw=1.5))
ax.annotate("", xy=(3.1, 6.15), xytext=(3.1, 5.7),
            arrowprops=dict(arrowstyle="-", color="#555", lw=1.5))
arrow(ax, 3.1, 6.15, 4.05, 6.15)

# ── Prompt template ─────────────────────────────────────────────────────────
rounded_box(ax, 5.0, 6.15, 1.9, 1.1, "#2d4059",
            "Instruction\nTemplate\n[INST]…[/INST]", fontsize=8.5)
arrow(ax, 5.95, 6.15, 6.8, 6.15)

# ── Model box ───────────────────────────────────────────────────────────────
model_box = FancyBboxPatch((6.8, 5.3), 2.8, 1.7,
    boxstyle="round,pad=0.06",
    facecolor="#7b2d8b", edgecolor="#cc55dd", linewidth=2, zorder=3)
ax.add_patch(model_box)
ax.text(8.2, 6.45, "allenai/wildguard", ha="center", va="center",
        fontsize=10, fontweight="bold", color="white", zorder=4)
ax.text(8.2, 6.1, "Mistral-7B fine-tune", ha="center", va="center",
        fontsize=8.5, color="#ddaaff", zorder=4)
ax.text(8.2, 5.78, "WildGuardMix training data", ha="center", va="center",
        fontsize=7.5, color="#ccaaee", zorder=4)
ax.text(8.2, 5.52, "110k prompt–response pairs", ha="center", va="center",
        fontsize=7.5, color="#ccaaee", zorder=4)
arrow(ax, 9.6, 6.15, 10.4, 6.15)

# ── Parser ──────────────────────────────────────────────────────────────────
rounded_box(ax, 11.1, 6.15, 1.3, 0.8, "#2d4059", "Output\nParser", fontsize=8.5)
arrow(ax, 11.75, 6.15, 12.3, 6.15)

# ── Outputs column ──────────────────────────────────────────────────────────
ax.text(12.9, 7.3, "OUTPUTS", ha="center", fontsize=9, color="#aaaaaa", fontweight="bold")

OUT_COLORS = ["#c0392b", "#8e44ad", "#16a085"]
OUT_LABELS = [
    "Prompt\nHarmfulness",
    "Response\nRefusal",
    "Response\nHarmfulness",
]
OUT_Y = [6.7, 6.15, 5.6]
for color, label, y in zip(OUT_COLORS, OUT_LABELS, OUT_Y):
    arrow(ax, 12.3, 6.15, 12.3, y)
    rounded_box(ax, 13.1, y, 1.55, 0.6, color, label, fontsize=8.5)

# ── Value pills under outputs ───────────────────────────────────────────────
pill_data = [
    (13.1, 6.25, "#e74c3c", "harmful"),
    (13.1, 5.95, "#27ae60", "unharmful"),
    (13.1, 5.5, "#8e44ad", "refusal"),
    (13.1, 5.2, "#2980b9", "compliance"),
    (13.1, 4.75, "#c0392b", "harmful"),
    (13.1, 4.45, "#27ae60", "unharmful"),
]
# Draw small value chips beneath each output box
pill_y_offsets = {
    6.7:  [(6.35, "#e74c3c", "harmful"), (6.05, "#27ae60", "unharmful")],
    6.15: [(5.8,  "#8e44ad", "refusal"), (5.5,  "#2980b9", "compliance"), (5.2, "#888888", "n/a")],
    5.6:  [(5.05, "#c0392b", "harmful"), (4.75, "#27ae60", "unharmful"), (4.45, "#888888", "n/a")],
}
for out_y, pills in pill_y_offsets.items():
    for py, pc, pt in pills:
        pill = FancyBboxPatch((12.32, py - 0.15), 1.55, 0.28,
            boxstyle="round,pad=0.03", facecolor=pc, edgecolor="none",
            alpha=0.85, zorder=3)
        ax.add_patch(pill)
        ax.text(13.1, py, pt, ha="center", va="center",
                fontsize=7.5, color="white", zorder=4)

# ── Backend section ─────────────────────────────────────────────────────────
ax.text(7, 4.1, "INFERENCE BACKENDS", ha="center", fontsize=9,
        color="#aaaaaa", fontweight="bold")

for bx, label, sublabel, color in [
    (4.5, "vLLM", "Recommended\nhigh throughput", "#1a5276"),
    (7.0, "HuggingFace\nTransformers", "GPU / CPU\nflexible device", "#145a32"),
    (9.5, "Subprocess\nvLLM", "Ephemeral model\nzero GPU leak", "#4a235a"),
]:
    rounded_box(ax, bx, 3.5, 2.2, 0.85, color, label, fontsize=9, bold=True)
    ax.text(bx, 3.0, sublabel, ha="center", va="center",
            fontsize=7.5, color="#aaaaaa")

# ── Key features section ────────────────────────────────────────────────────
ax.text(7, 2.35, "KEY FEATURES", ha="center", fontsize=9,
        color="#aaaaaa", fontweight="bold")

features = [
    (1.5,  "#17202a", "Prompt-only\nor Prompt+Response"),
    (4.0,  "#17202a", "Batched\nInference"),
    (6.5,  "#17202a", "Ephemeral\nGPU Model"),
    (9.0,  "#17202a", "Parse Error\nDetection"),
    (11.5, "#17202a", "WildGuardMix\nTraining Data"),
]
for fx, fc, fl in features:
    rounded_box(ax, fx, 1.75, 2.2, 0.75, fc, fl, fontsize=8.5,
                text_color="#dddddd")
    box_outline = FancyBboxPatch((fx - 1.1, 1.75 - 0.375), 2.2, 0.75,
        boxstyle="round,pad=0.04", fill=False,
        edgecolor="#444455", linewidth=1, zorder=3)
    ax.add_patch(box_outline)

# ── Accuracy callout ────────────────────────────────────────────────────────
callout = FancyBboxPatch((0.2, 0.2), 4.2, 0.85,
    boxstyle="round,pad=0.05", facecolor="#1c1c2e",
    edgecolor="#7b2d8b", linewidth=1.5, zorder=3)
ax.add_patch(callout)
ax.text(2.3, 0.625, "F1 ≥ 0.97 on WildGuardTest",
        ha="center", va="center", fontsize=9, color="white",
        fontweight="bold", zorder=4)

callout2 = FancyBboxPatch((5.0, 0.2), 4.0, 0.85,
    boxstyle="round,pad=0.05", facecolor="#1c1c2e",
    edgecolor="#1a6b8a", linewidth=1.5, zorder=3)
ax.add_patch(callout2)
ax.text(7.0, 0.625, "allenai/wildguard  ·  pip install wildguard",
        ha="center", va="center", fontsize=9, color="#88ccee", zorder=4)

callout3 = FancyBboxPatch((9.6, 0.2), 4.2, 0.85,
    boxstyle="round,pad=0.05", facecolor="#1c1c2e",
    edgecolor="#16a085", linewidth=1.5, zorder=3)
ax.add_patch(callout3)
ax.text(11.7, 0.625, "UWNSL/WildGuardMix  ·  110k samples",
        ha="center", va="center", fontsize=9, color="#88ddcc", zorder=4)

plt.tight_layout(pad=0.3)
out = "wildguard_diagram.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved {out}")
