#!/usr/bin/env python3
"""Render the cross-task representation contrast from committed unit-level data."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "generated" / "figures" / "composition"
DECODERS = ("logistic", "catboost", "tabpfn")
LABELS = {"logistic": "Logistic", "catboost": "CatBoost", "tabpfn": "TabPFN-3"}
COLORS = {"logistic": "#0072B2", "catboost": "#D98B00", "tabpfn": "#008C72"}
TIE_BAND = 0.0


def clutrr_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = pd.read_csv(
        ROOT / "experiments" / "clutrr" / "results" / "shortcut-diagnostic" / "cells.csv"
    )
    cells = cells.loc[
        (cells["context_per_relation"] == 80)
        & cells["decoder"].isin(DECODERS)
        & cells["encoder"].isin(("relation_multiset", "ordered_typed_path")),
        ["decoder", "seed", "encoder", "ood_equal_length_accuracy"],
    ]
    pivot = cells.pivot(
        index=["decoder", "seed"],
        columns="encoder",
        values="ood_equal_length_accuracy",
    ).reset_index()
    if len(pivot) != 9 or pivot.isna().any().any():
        raise ValueError("CLUTRR replication-unit matrix is incomplete")
    pivot["delta"] = pivot["relation_multiset"] - pivot["ordered_typed_path"]
    summary = (
        pivot.groupby("decoder", observed=True)["delta"]
        .agg(mean="mean", low="min", high="max")
        .reset_index()
    )
    primary = pd.read_csv(
        ROOT
        / "experiments"
        / "clutrr"
        / "results"
        / "metric-robustness"
        / "primary_metric_robustness.csv"
    )
    primary = primary.loc[
        primary["decoder"].isin(DECODERS) & primary["metric"].eq("accuracy"),
        ["decoder", "multiset_advantage"],
    ].rename(columns={"multiset_advantage": "primary_mean"})
    summary = summary.merge(primary, on="decoder", validate="one_to_one")
    summary["mean"] = summary.pop("primary_mean")
    return pivot, summary


def graphlog_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = ROOT / "experiments" / "graphlog" / "results" / "decoder-sensitivity"
    units = pd.read_csv(root / "primary_contrasts_by_world.csv")
    units = units.loc[
        units["decoder"].isin(DECODERS) & units["contrast"].eq("shared_multiset_minus_absolute"),
        ["decoder", "world", "seed_mean_delta"],
    ].rename(columns={"seed_mean_delta": "delta"})
    summary = pd.read_csv(root / "primary_contrasts.csv")
    summary = summary.loc[
        summary["decoder"].isin(DECODERS)
        & summary["contrast"].eq("shared_multiset_minus_absolute")
        & summary["context_per_relation"].eq(40),
        [
            "decoder",
            "equal_world_mean_delta",
            "fixed_panel_sensitivity_low",
            "fixed_panel_sensitivity_high",
            "world_wins",
            "world_ties",
            "world_losses",
        ],
    ].rename(
        columns={
            "equal_world_mean_delta": "mean",
            "fixed_panel_sensitivity_low": "low",
            "fixed_panel_sensitivity_high": "high",
        }
    )
    if len(units) != 171 or len(summary) != 3:
        raise ValueError("GraphLog 57-world matrix is incomplete")
    return units, summary


def dihedral_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for stream in ("cpu", "tabpfn"):
        frame = pd.read_csv(
            ROOT / "experiments" / "dihedral16" / "results" / stream / "primary_seed_contrasts.csv"
        )
        frames.append(frame)
    units = pd.concat(frames, ignore_index=True)
    units = units.loc[
        units["decoder"].isin(DECODERS) & units["contrast"].eq("shared_lossy_minus_absolute"),
        ["decoder", "seed", "difference"],
    ].rename(columns={"difference": "delta"})
    if len(units) != 30:
        raise ValueError("dihedral 10-population matrix is incomplete")
    summary = (
        units.groupby("decoder", observed=True)["delta"]
        .agg(mean="mean", low="min", high="max")
        .reset_index()
    )
    return units, summary


def panel(
    axis,
    units: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    title: str,
    subtitle: str,
    tag: str,
    show_wtl: bool = False,
) -> None:
    rng = np.random.default_rng(260807)
    positions = {"logistic": 2, "catboost": 1, "tabpfn": 0}
    summary = summary.set_index("decoder")
    for decoder in DECODERS:
        y = positions[decoder]
        values = units.loc[units["decoder"].eq(decoder), "delta"].to_numpy(float)
        jitter = rng.uniform(-0.19, 0.19, size=len(values))
        axis.scatter(
            values,
            y + jitter,
            s=17,
            color=COLORS[decoder],
            alpha=0.20,
            linewidth=0,
            zorder=2,
        )
        row = summary.loc[decoder]
        axis.plot(
            [float(row["low"]), float(row["high"])],
            [y, y],
            color=COLORS[decoder],
            linewidth=2.1,
            alpha=0.8,
            zorder=3,
        )
        mean = float(row["mean"])
        axis.scatter(
            mean,
            y,
            marker="D",
            s=62,
            color=COLORS[decoder],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        label = "0.0" if abs(mean) < 0.0005 else f"{mean:+.3f}"
        if show_wtl:
            label += (
                f"  ({int(row['world_wins'])}/{int(row['world_ties'])}/{int(row['world_losses'])})"
            )
        axis.text(
            mean,
            y + 0.29,
            label,
            color=COLORS[decoder],
            fontsize=7.3,
            ha="center",
            va="bottom",
        )

    axis.axvline(0, color="#697279", linewidth=0.8, zorder=1)
    axis.set_xlim(-0.62, 0.50)
    axis.set_ylim(-0.55, 2.55)
    axis.set_xticks((-0.4, -0.2, 0.0, 0.2, 0.4))
    axis.set_yticks((2, 1, 0), [LABELS[decoder] for decoder in DECODERS])
    axis.grid(axis="x", color="#D9DEE1", linewidth=0.55, zorder=0)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.set_title(title, loc="left", fontweight="bold", fontsize=10.5, pad=36)
    axis.text(
        0,
        1.095,
        subtitle,
        transform=axis.transAxes,
        color="#667178",
        fontsize=7.2,
        ha="left",
        va="bottom",
    )
    axis.text(
        0,
        1.015,
        "absolute better  \u2190   \u2192  shared better",
        transform=axis.transAxes,
        color="#667178",
        fontsize=7.0,
        ha="left",
        va="bottom",
    )
    axis.text(-0.15, 1.18, tag, transform=axis.transAxes, fontsize=10.5, fontweight="bold")


def main() -> None:
    clutrr = clutrr_data()
    graphlog = graphlog_data()
    dihedral = dihedral_data()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#1D252C",
            "text.color": "#1D252C",
            "xtick.color": "#1D252C",
            "ytick.color": "#1D252C",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.72, bottom=0.27, wspace=0.32)
    panel(
        axes[0],
        *clutrr,
        title="CLUTRR",
        subtitle="accuracy · 3 paired context seeds",
        tag="a",
    )
    panel(
        axes[1],
        *graphlog,
        title="GraphLog",
        subtitle="balanced accuracy · 57 worlds · labels show W/T/L",
        tag="b",
        show_wtl=True,
    )
    panel(
        axes[2],
        *dihedral,
        title="Dihedral composition",
        subtitle="accuracy · 10 generated populations",
        tag="c",
    )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color="#AEB6BB",
            markersize=4,
            label="replication unit",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="-",
            color="#1D252C",
            markersize=5,
            label="panel mean; line = range or sensitivity interval",
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.075)
    )
    fig.supxlabel("Shared multiset minus absolute path (native metric)", y=0.015, fontsize=9.5)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
