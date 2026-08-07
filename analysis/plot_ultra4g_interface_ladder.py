#!/usr/bin/env python3
"""Plot the Ultra57 interface ladder and native-readout boundary."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

INK = "#1D252C"
MUTED = "#66737B"
GRID = "#D9DEE1"
WIN = "#2A8F7B"
TIE = "#BCC5C9"
LOSS = "#C45B4A"

CONTRAST_LABELS = {
    "local_relational_vs_entity_tokens": "Local coordinates − entity tokens",
    "ultra_score_given_global_structure": "ULTRA score | graph structure",
    "global_structure_given_ultra": "Global descriptors | ULTRA score",
    "full_interface_vs_native_ultra": "Full table − native ULTRA",
}
CONTRAST_ORDER = list(CONTRAST_LABELS)
CONTRAST_COLORS = {
    "local_relational_vs_entity_tokens": "#0072B2",
    "ultra_score_given_global_structure": "#008C72",
    "global_structure_given_ultra": "#D88700",
    "full_interface_vs_native_ultra": "#59666E",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.titlesize": 8.8,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
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


def panel_tag(ax: Axes, letter: str, x: float = -0.08) -> None:
    ax.text(
        x,
        1.10,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )


def style_axis(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", width=0.6, length=2.5)
    ax.grid(axis="x", color=GRID, linewidth=0.55, zorder=0)


def load_data(repo: Path) -> pd.DataFrame:
    root = repo / "experiments" / "ultra4g" / "results"
    summary = pd.read_csv(root / "q64-encoder-ladder" / "contrast_summary.csv")
    summary = summary.loc[
        summary["context_per_class"].eq(200)
        & summary["direction"].eq("combined")
        & summary["contrast"].isin(CONTRAST_ORDER)
    ].copy()
    if len(summary) != 4:
        raise ValueError(f"expected four interface contrasts, found {len(summary)}")
    if not summary["dataset_count"].eq(57).all():
        raise ValueError("interface contrasts do not cover all 57 datasets")
    if not summary["family_count"].eq(16).all():
        raise ValueError("interface contrasts do not cover all 16 families")

    multiplicity = pd.read_csv(root / "multiplicity" / "multiplicity_summary.csv")
    multiplicity["contrast"] = multiplicity["contrast_id"].str.replace(
        "_c200_combined", "", regex=False
    )
    multiplicity = multiplicity.loc[
        multiplicity["contrast"].isin(CONTRAST_ORDER),
        ["contrast", "joint_max_t_adjusted_p"],
    ]
    if len(multiplicity) != 4:
        raise ValueError("joint max-t table is missing an interface contrast")

    leave_one_out = pd.read_csv(
        root / "multiplicity" / "leave_one_family_out_summary.csv"
    )
    leave_one_out["contrast"] = leave_one_out["contrast_id"].str.replace(
        "_c200_combined", "", regex=False
    )
    leave_one_out = leave_one_out.loc[
        leave_one_out["contrast"].isin(CONTRAST_ORDER),
        ["contrast", "most_influential_omitted_family"],
    ]
    if len(leave_one_out) != 4:
        raise ValueError("leave-one-family-out table is missing an interface contrast")

    frame = summary.merge(multiplicity, on="contrast", validate="one_to_one").merge(
        leave_one_out, on="contrast", validate="one_to_one"
    )
    frame["contrast"] = pd.Categorical(
        frame["contrast"], categories=CONTRAST_ORDER, ordered=True
    )
    return frame.sort_values("contrast").reset_index(drop=True)


def format_adjusted_p(value: float) -> str:
    if value < 0.001:
        return "<.001"
    return f"{value:.3f}".removeprefix("0")


def draw_forest(ax: Axes, data: pd.DataFrame) -> None:
    style_axis(ax)
    y_positions = {contrast: 3 - index for index, contrast in enumerate(CONTRAST_ORDER)}
    ax.axvline(0, color=INK, linewidth=0.8, zorder=1)

    for record in data.itertuples(index=False):
        contrast = str(record.contrast)
        y = y_positions[contrast]
        color = CONTRAST_COLORS[contrast]
        ax.plot(
            [record.fixed_panel_bootstrap_low, record.fixed_panel_bootstrap_high],
            [y, y],
            color=color,
            linewidth=1.2,
            alpha=0.35,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            [record.leave_one_family_out_min, record.leave_one_family_out_max],
            [y, y],
            color=color,
            linewidth=3.1,
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter(
            [record.equal_family_effect],
            [y],
            marker="D",
            s=32,
            facecolor=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        ax.text(
            record.equal_family_effect,
            y + 0.17,
            f"{record.equal_family_effect:+.3f}",
            ha="center",
            va="bottom",
            fontsize=6.1,
            color=color,
            fontweight="bold",
        )
        ax.text(
            0.505,
            y,
            (
                f"{record.positive_families:.0f}/16 positive  ·  "
                f"p$_{{maxT}}$ {format_adjusted_p(record.joint_max_t_adjusted_p)}"
            ),
            ha="right",
            va="center",
            fontsize=5.7,
            color=MUTED,
        )

    native = data.loc[data["contrast"].eq("full_interface_vs_native_ultra")].iloc[0]
    ax.text(
        native["leave_one_family_out_min"],
        y_positions["full_interface_vs_native_ultra"] - 0.20,
        (
            f"omit {native['most_influential_omitted_family']}: "
            f"{native['leave_one_family_out_min']:+.3f}"
        ),
        ha="center",
        va="top",
        fontsize=5.5,
        color=MUTED,
    )

    ax.set_xlim(-0.14, 0.53)
    ax.set_xticks([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_ylim(-0.48, 3.48)
    ax.set_yticks([y_positions[contrast] for contrast in CONTRAST_ORDER])
    ax.set_yticklabels([CONTRAST_LABELS[contrast] for contrast in CONTRAST_ORDER])
    ax.set_xlabel("Paired ΔMRR")
    ax.set_title("Equal-family effect and sensitivity", loc="left", pad=18, fontweight="bold")
    panel_tag(ax, "A", x=-0.42)


def draw_direction_counts(ax: Axes, data: pd.DataFrame) -> None:
    style_axis(ax)
    y_positions = {contrast: 3 - index for index, contrast in enumerate(CONTRAST_ORDER)}
    for record in data.itertuples(index=False):
        contrast = str(record.contrast)
        y = y_positions[contrast]
        left = 0.0
        segments = [
            (float(record.dataset_wins), WIN, "W"),
            (float(record.dataset_ties), TIE, "T"),
            (float(record.dataset_losses), LOSS, "L"),
        ]
        for width, color, label in segments:
            if width <= 0:
                continue
            ax.barh(
                y,
                width,
                left=left,
                height=0.50,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=2,
            )
            if width >= 6:
                ax.text(
                    left + width / 2,
                    y,
                    f"{label} {int(width)}",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color="white" if color != TIE else INK,
                    fontweight="bold",
                    zorder=3,
                )
            left += width
        ax.text(
            58.2,
            y,
            f"{record.dataset_wins:.0f}/{record.dataset_ties:.0f}/{record.dataset_losses:.0f}",
            ha="left",
            va="center",
            fontsize=5.8,
            color=MUTED,
        )

    ax.text(
        58.2,
        3.43,
        "W/T/L",
        ha="left",
        va="bottom",
        fontsize=5.6,
        color=MUTED,
        fontweight="bold",
    )

    ax.set_xlim(0, 68)
    ax.set_xticks([0, 20, 40, 57])
    ax.set_ylim(-0.48, 3.48)
    ax.set_yticks([y_positions[contrast] for contrast in CONTRAST_ORDER])
    ax.tick_params(axis="y", labelleft=False)
    ax.set_xlabel("Datasets")
    ax.set_title("Direction under ±0.005 tie band", loc="left", pad=18, fontweight="bold")
    panel_tag(ax, "B", x=-0.10)


def build_figure(data: pd.DataFrame) -> Figure:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.15),
        sharey=True,
        gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.22},
    )
    draw_forest(axes[0], data)
    draw_direction_counts(axes[1], data)

    sensitivity_handle = Line2D(
        [], [], color=INK, linewidth=1.2, alpha=0.35, label="family-bootstrap sensitivity"
    )
    lofo_handle = Line2D(
        [], [], color=INK, linewidth=3.1, label="leave-one-family-out range"
    )
    effect_handle = Line2D(
        [],
        [],
        marker="D",
        linestyle="none",
        markersize=4.8,
        markerfacecolor=INK,
        markeredgecolor="white",
        label="equal-family effect",
    )
    direction_handles = [
        Line2D([], [], color=WIN, linewidth=5, label="win"),
        Line2D([], [], color=TIE, linewidth=5, label="tie"),
        Line2D([], [], color=LOSS, linewidth=5, label="loss"),
    ]
    figure.legend(
        handles=[effect_handle, lofo_handle, sensitivity_handle, *direction_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=6,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.9,
        fontsize=5.8,
    )
    figure.suptitle(
        "What does a useful graph-to-table row need to expose?",
        x=0.045,
        y=1.025,
        ha="left",
        fontsize=10.2,
        fontweight="bold",
    )
    figure.text(
        0.045,
        0.955,
        "Frozen TabPFN · 200 examples per class · 57 KGs / 16 provenance families",
        ha="left",
        va="top",
        fontsize=6.7,
        color=MUTED,
    )
    figure.text(
        0.995,
        0.015,
        (
            "q64 query budget; all-entity filtered ranking. "
            "pmaxT adjusts eight exploratory contrasts jointly."
        ),
        ha="right",
        va="bottom",
        fontsize=5.5,
        color=MUTED,
    )
    figure.subplots_adjust(left=0.285, right=0.99, top=0.78, bottom=0.22)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to docs/figures.",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    output_dir = (args.output_dir or repo / "generated" / "figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    data = load_data(repo)
    figure = build_figure(data)
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"ultra4g_interface_ladder.{suffix}", dpi=400)
    plt.close(figure)


if __name__ == "__main__":
    main()
