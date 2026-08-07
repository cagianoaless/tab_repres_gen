from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors

INK = "#1D252C"
MUTED = "#8A949A"
GRID = "#D9DEE1"
LOGISTIC = "#5EA7D1"
CATBOOST = "#E0914E"
TABPFN = "#53B39F"

ALPHA = 0.55
SAVE_DPI = 600

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.6,
        "axes.titlesize": 8.8,
        "axes.labelsize": 7.6,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.65,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)

DECODER_ORDER = ["Logistic", "CatBoost", "TabPFN"]
DECODER_COLORS = {"Logistic": LOGISTIC, "CatBoost": CATBOOST, "TabPFN": TABPFN}

panel_a = {
    "Logistic": [0.4295, 0.3995, 0.2661],
    "CatBoost": [0.2725, 0.2555, 0.1630],
    "TabPFN": [0.2894, 0.2914, 0.1871],
}
panel_b = {
    "Logistic": {"add_permuted": -0.030, "restore_alignment": -0.134},
    "CatBoost": {"add_permuted": -0.018, "restore_alignment": -0.092},
    "TabPFN": {"add_permuted":  0.003, "restore_alignment": -0.105},
}
panel_c = {
    "Logistic": {"length4": 0.489, "lengths5_10": -0.001},
    "CatBoost": {"length4": 0.372, "lengths5_10": 0.010},
    "TabPFN": {"length4": 0.758, "lengths5_10": 0.033},
}

def rgba(color, alpha):
    r, g, b, _ = mcolors.to_rgba(color)
    return (r, g, b, alpha)

def clean_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.6, length=2.5)
    if grid_axis == "y":
        ax.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
    elif grid_axis == "x":
        ax.grid(axis="x", color=GRID, linewidth=0.55, zorder=0)

def panel_tag(ax, letter):
    ax.text(-0.13, 1.06, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, fontweight="bold")

def draw_panel_a(ax):
    clean_axis(ax, "y")
    x = np.arange(3)
    width = 0.20
    offsets = [-0.24, 0.0, 0.24]
    for offset, decoder in zip(offsets, DECODER_ORDER):
        color = DECODER_COLORS[decoder]
        vals = panel_a[decoder]
        ax.bar(
            x + offset, vals, width=width,
            facecolor=rgba(color, 0.08), edgecolor=rgba(color, ALPHA),
            linewidth=1.0, hatch="///", zorder=3, label=decoder
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Multiset\nonly", "+ permuted\nabsolute", "+ aligned\nabsolute"])
    ax.set_ylabel("OOD accuracy")
    ax.set_ylim(0.10, 0.47)
    ax.set_xlim(-0.50, 2.50)
    ax.set_title("Same columns, different alignment", loc="left", pad=7)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=rgba(LOGISTIC, 0.08), edgecolor=rgba(LOGISTIC, ALPHA), hatch="///", linewidth=1.0),
        plt.Rectangle((0, 0), 1, 1, facecolor=rgba(CATBOOST, 0.08), edgecolor=rgba(CATBOOST, ALPHA), hatch="///", linewidth=1.0),
        plt.Rectangle((0, 0), 1, 1, facecolor=rgba(TABPFN, 0.08), edgecolor=rgba(TABPFN, ALPHA), hatch="///", linewidth=1.0),
    ]
    ax.legend(handles, ["Logistic", "CatBoost", "TabPFN"], frameon=False, loc="upper right", handlelength=1.0, handletextpad=0.35, borderpad=0, labelspacing=0.25)
    panel_tag(ax, "a")

def draw_panel_b(ax):
    clean_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    y_positions = {"Logistic": 2, "CatBoost": 1, "TabPFN": 0}
    for decoder in DECODER_ORDER:
        y = y_positions[decoder]
        color = DECODER_COLORS[decoder]
        add = panel_b[decoder]["add_permuted"]
        align = panel_b[decoder]["restore_alignment"]
        total = add + align
        ax.plot([0, add], [y, y], color=rgba(MUTED, ALPHA), linewidth=1.7, zorder=2)
        ax.plot([add, total], [y, y], color=rgba(color, ALPHA), linewidth=1.7, zorder=3)
        ax.scatter([0], [y], s=18, facecolor="white", edgecolor=INK, linewidth=0.8, zorder=4)
        ax.scatter([add], [y], s=20, marker="s", facecolor=rgba("#B1B8BD", ALPHA), edgecolor=rgba(MUTED, ALPHA), linewidth=0.8, zorder=4)
        ax.scatter([total], [y], s=22, marker="o", facecolor=rgba(color, ALPHA), edgecolor=rgba(color, ALPHA), linewidth=0.8, zorder=5)
        add_y = y + (0.18 if decoder == "Logistic" else 0.14)
        add_x = add / 2 - (0.006 if decoder == "Logistic" else 0.0)
        ax.text(add_x, add_y, f"{add:+.3f}", ha="center", va="bottom", fontsize=5.8, color=MUTED)
        ax.text((add + total) / 2, y - 0.15, f"{align:+.3f}", ha="center", va="top", fontsize=5.8, color=color)
    ax.axvline(0, color=INK, linewidth=0.7, zorder=1)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels(DECODER_ORDER)
    ax.set_xlim(-0.19, 0.025); ax.set_ylim(-0.55, 2.55)
    ax.set_xlabel("Accuracy change relative to multiset-only")
    ax.set_title("Block addition vs alignment", loc="left", pad=7)
    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=INK, markersize=3.6, label="multiset"),
        Line2D([], [], marker="s", linestyle="none", markerfacecolor=rgba("#B1B8BD", ALPHA), markeredgecolor=rgba(MUTED, ALPHA), markersize=3.6, label="+ permuted"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=rgba(MUTED, ALPHA), markeredgecolor=INK, markersize=3.6, label="aligned"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.003), ncol=3, handlelength=0.0, columnspacing=0.7, handletextpad=0.28, borderpad=0, fontsize=5.2, labelspacing=0.12)
    panel_tag(ax, "b")

def draw_panel_c(ax):
    clean_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    y_positions = {"Logistic": 2, "CatBoost": 1, "TabPFN": 0}
    for decoder in DECODER_ORDER:
        y = y_positions[decoder]
        color = DECODER_COLORS[decoder]
        v_short = panel_c[decoder]["length4"]
        v_long = panel_c[decoder]["lengths5_10"]
        ax.scatter([v_long], [y - 0.11], s=24, facecolor="white", edgecolor=rgba(color, ALPHA), linewidth=0.9, zorder=3)
        ax.scatter([v_short], [y + 0.11], s=26, facecolor=rgba(color, ALPHA), edgecolor=rgba(color, ALPHA), linewidth=0.9, zorder=3)
        if decoder == "Logistic":
            long_dx = 0.040; long_ha = "left"
        else:
            long_dx = 0.022 if v_long >= 0 else -0.022
            long_ha = "left" if v_long >= 0 else "right"
        ax.text(v_long + long_dx, y - 0.11, f"{v_long:+.3f}", ha=long_ha, va="center", fontsize=5.8, color=color)
        short_dx = 0.022 if v_short >= 0 else -0.022
        short_ha = "left" if v_short >= 0 else "right"
        ax.text(v_short + short_dx, y + 0.11, f"{v_short:+.3f}", ha=short_ha, va="center", fontsize=5.8, color=color)
    ax.axvline(0, color=INK, linewidth=0.7, zorder=1)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels(DECODER_ORDER)
    ax.set_xlim(-0.045, 0.89); ax.set_ylim(-0.55, 2.55)
    ax.set_xlabel("Accuracy change after adding\nlength-4 context")
    ax.set_title("Exposure is length-specific", loc="left", pad=7)
    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=rgba(INK, ALPHA), markeredgecolor=INK, markersize=3.6, label="length 4"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=INK, markersize=3.6, label="lengths 5-10"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.99), ncol=2, handletextpad=0.22, columnspacing=0.50, borderpad=0, fontsize=5.6)
    panel_tag(ax, "c")

def build_figure():
    fig = plt.figure(figsize=(7.1, 2.75))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.28, 1.0, 1.08], wspace=0.58, left=0.075, right=0.985, top=0.86, bottom=0.22)
    draw_panel_a(fig.add_subplot(grid[0]))
    draw_panel_b(fig.add_subplot(grid[1]))
    draw_panel_c(fig.add_subplot(grid[2]))
    return fig

def main():
    fig = build_figure()
    out_base = Path(__file__).resolve().parents[1] / "generated" / "figures" / "clutrr"
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base) + ".png", dpi=SAVE_DPI)
    fig.savefig(str(out_base) + ".pdf", dpi=SAVE_DPI)
    plt.close(fig)

if __name__ == "__main__":
    main()
