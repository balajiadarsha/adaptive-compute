import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

columns = [
    "budget", "mean_tokens", "mean_delta", "over_budget_%", "under_budget_%",
    "attack_success_%", "refusal_%", "false_refusal_%", "benign_response_harmful_%"
]

col_labels = [
    "Budget", "Mean\nTokens", "Mean\nDelta", "Over\nBudget %", "Under\nBudget %",
    "Attack\nSuccess %", "Refusal\n%", "False\nRefusal %", "Benign\nHarmful %"
]

data = [
    ["256",  "5704.5", "5448.5", "89.0", "11.0", "6.52", "4.35", "0.00", "0.0"],
    ["1024", "6542.5", "5518.5", "79.0", "21.0", "0.00", "2.17", "0.00", "0.0"],
    ["4096", "6192.2", "2096.2", "74.0", "26.0", "8.70", "2.17", "0.00", "0.0"],
    ["none", "7641.4", "—",      "—",    "—",    "2.17", "2.17", "1.85", "0.0"],
]

# Color scale helpers
def green_red(val, lo, hi):
    """Green = good (low), Red = bad (high)."""
    if val == "—":
        return "#f5f5f5"
    try:
        v = float(val)
    except ValueError:
        return "#f5f5f5"
    t = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)
    r = int(255 * (0.9 * t + 0.95 * (1 - t)))
    g = int(255 * (0.95 * (1 - t) + 0.85 * t))
    b = int(255 * 0.92)
    return f"#{r:02x}{g:02x}{b:02x}"

def red_green(val, lo, hi):
    """Red = bad (high attack success), Green = good (low)."""
    if val == "—":
        return "#f5f5f5"
    try:
        v = float(val)
    except ValueError:
        return "#f5f5f5"
    t = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)
    r = int(255 * (0.95 * t + 0.95 * (1 - t)))
    g = int(255 * (0.85 * (1 - t) + 0.95 * t))
    b = int(255 * 0.92)
    return f"#{r:02x}{g:02x}{b:02x}"

# Cell colors: row x col
NCOLS = len(columns)
NROWS = len(data)

cell_colors = [["#ffffff"] * NCOLS for _ in range(NROWS)]

for r, row in enumerate(data):
    # mean_delta: lower = better (model respects budget more)
    cell_colors[r][2] = green_red(row[2], 0, 6000)
    # over_budget_%: lower = better
    cell_colors[r][3] = green_red(row[3], 0, 100)
    # under_budget_%: higher = better (neutral gray scale)
    cell_colors[r][4] = "#f5f5f5"
    # attack_success_%: lower = better
    cell_colors[r][5] = red_green(row[5], 0, 10)
    # refusal_%: higher = better
    cell_colors[r][6] = red_green(row[6], 0, 5)
    # false_refusal_%: lower = better
    cell_colors[r][7] = red_green(row[7], 0, 3)
    # benign_response_harmful_%: lower = better
    cell_colors[r][8] = red_green(row[8], 0, 3)

fig, ax = plt.subplots(figsize=(16, 3.2))
fig.patch.set_facecolor("white")
ax.axis("off")

table = ax.table(
    cellText=data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    cellColours=cell_colors,
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

# Style header
for col in range(NCOLS):
    cell = table[0, col]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold", fontsize=10)
    cell.set_edgecolor("#ffffff")

# Style data rows
for row in range(1, NROWS + 1):
    for col in range(NCOLS):
        cell = table[row, col]
        cell.set_edgecolor("#dddddd")
        cell.set_text_props(fontsize=11)
        if row % 2 == 0 and cell_colors[row - 1][col] == "#ffffff":
            cell.set_facecolor("#f9f9f9")

# Bold the budget column
for row in range(1, NROWS + 1):
    table[row, 0].set_text_props(fontweight="bold")
    table[row, 0].set_facecolor("#ecf0f1")

ax.set_title(
    "Token Budget Sweep — L1-Qwen-1.5B-Exact on UWNSL/SafeChain (WildGuard evaluation)",
    fontsize=13, fontweight="bold", pad=18, color="#2c3e50"
)

# Legend
legend_elements = [
    mpl.patches.Patch(facecolor="#f4c2c2", label="Higher = worse"),
    mpl.patches.Patch(facecolor="#c2f4c2", label="Lower = better"),
    mpl.patches.Patch(facecolor="#f5f5f5", label="Neutral / N/A"),
]
ax.legend(handles=legend_elements, loc="lower right",
          bbox_to_anchor=(1.0, -0.18), ncol=3, fontsize=9,
          framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout()
out = "sweep_table.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
