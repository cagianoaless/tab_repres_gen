#!/usr/bin/env python3
"""Render the cross-task representation contrast from committed summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "generated" / "figures" / "composition.png"
DECODERS = ("logistic", "catboost", "tabpfn")
LABELS = {"logistic": "Logistic", "catboost": "CatBoost", "tabpfn": "TabPFN-3"}
COLORS = {"logistic": "#0072B2", "catboost": "#D98B00", "tabpfn": "#008C72"}


def _endpoint_values() -> pd.DataFrame:
    clutrr = pd.DataFrame(
        {
            "task": "CLUTRR",
            "decoder": DECODERS,
            "delta": (0.4295 - 0.1565, 0.2725 - 0.1090, 0.2894 - 0.1031),
        }
    )

    graphlog = pd.read_csv(
        ROOT / "experiments" / "graphlog" / "results" / "decoder-sensitivity" / "primary_contrasts.csv"
    )
    graphlog = graphlog.loc[
        graphlog["decoder"].isin(DECODERS)
        & graphlog["contrast"].eq("shared_multiset_minus_absolute")
        & graphlog["context_per_relation"].eq(40),
        ["decoder", "equal_world_mean_delta"],
    ].rename(columns={"equal_world_mean_delta": "delta"})
    graphlog.insert(0, "task", "GraphLog")

    dihedral = pd.DataFrame(
        {
            "task": "Dihedral-16",
            "decoder": DECODERS,
            "delta": (-0.0004, 0.0126, -0.0011),
        }
    )
    return pd.concat([clutrr, graphlog, dihedral], ignore_index=True)


def main() -> None:
    data = _endpoint_values()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.7})
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.15), sharey=True, constrained_layout=True)
    for axis, task in zip(axes, ("CLUTRR", "GraphLog", "Dihedral-16"), strict=True):
        rows = data.loc[data["task"].eq(task)].set_index("decoder")
        for y, decoder in enumerate(reversed(DECODERS)):
            value = float(rows.loc[decoder, "delta"])
            axis.scatter(value, y, s=26, color=COLORS[decoder], alpha=0.7, zorder=3)
        axis.axvline(0, color="#777777", linewidth=0.8, linestyle=(0, (2, 2)))
        axis.set_title(task, loc="left", fontweight="semibold")
        axis.set_xlabel("Multiset minus absolute accuracy")
        axis.grid(axis="x", color="#E6E6E6", linewidth=0.55)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.set_yticks((2, 1, 0), [LABELS[d] for d in DECODERS])
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()

