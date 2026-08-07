from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "generated" / "figures"
OUTPUT_STEM = "audit"


# ---------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------

INK = "#1D252C"
MUTED = "#647078"
BORDER = "#C9D0D4"
LIGHT = "#F3F5F6"

TABLE = "#0072B2"
CONTEXT = "#D98B00"
DECODER = "#008C72"

FILL_ALPHA = 0.55


# ---------------------------------------------------------------------
# Graph controls
# ---------------------------------------------------------------------

# [left, bottom, width, height] in panel-a axes coordinates.
#
# This inset is intentionally tall enough to give the graph about
# one physical inch of drawing space. Because the axes are much wider
# than they are tall, width and height fractions should not be equal.
GRAPH_BOUNDS = [0.020, 0.405, 0.125, 0.4]


GRAPH_POSITIONS = {
    "a": (0.07, 0.70),
    "b": (0.43, 0.92),
    "c": (0.93, 0.68),
    "d": (0.74, 0.10),
    "e": (0.15, 0.13),
}

GRAPH_EDGES = [
    ("a", "b"),
    ("b", "c"),
    ("c", "d"),
    ("d", "e"),
    ("e", "a"),
    ("b", "d"),
]

GRAPH_NODE_RADIUS = 0.027
GRAPH_EDGE_WIDTH = 0.58
GRAPH_ARROW_SIZE = 4.9
GRAPH_EDGE_COLOR = MUTED


# ---------------------------------------------------------------------
# General plotting helpers
# ---------------------------------------------------------------------

def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def add_box(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    *,
    edge: str = BORDER,
    face: str = "white",
    title_color: str = INK,
    linewidth: float = 0.9,
    alpha: float = 1.0,
    fontsize: float = 8.2,
    fontweight: str = "bold",
) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            joinstyle="miter",
            alpha=alpha,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        title,
        ha="center",
        va="center",
        color=title_color,
        fontsize=fontsize,
        fontweight=fontweight,
    )


def add_arrow(
    ax: Axes,
    x_start: float,
    x_end: float,
    y: float,
    *,
    linewidth: float = 0.8,
    mutation_scale: float = 8.0,
    color: str = INK,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x_start, y),
            (x_end, y),
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


# ---------------------------------------------------------------------
# Directed graph
# ---------------------------------------------------------------------

def draw_graph_inset(parent_ax: Axes) -> None:
    """Draw the small directed graph without stretching it."""
    graph_ax = parent_ax.inset_axes(GRAPH_BOUNDS)

    graph_ax.set_xlim(0, 1)
    graph_ax.set_ylim(0, 1)

    # Essential: preserve the graph geometry rather than stretching it
    # to fill the rectangular inset.
    graph_ax.set_aspect("equal", adjustable="box")
    graph_ax.axis("off")

    for source, target in GRAPH_EDGES:
        graph_ax.add_patch(
            FancyArrowPatch(
                GRAPH_POSITIONS[source],
                GRAPH_POSITIONS[target],
                arrowstyle="-|>",
                mutation_scale=GRAPH_ARROW_SIZE,
                linewidth=GRAPH_EDGE_WIDTH,
                color=GRAPH_EDGE_COLOR,
                shrinkA=3.3,
                shrinkB=3.3,
                connectionstyle="arc3,rad=0",
                zorder=1,
            )
        )

    for x, y in GRAPH_POSITIONS.values():
        graph_ax.add_patch(
            Circle(
                (x, y),
                GRAPH_NODE_RADIUS,
                facecolor=INK,
                edgecolor=INK,
                linewidth=0.3,
                zorder=2,
            )
        )


# ---------------------------------------------------------------------
# Panel a
# ---------------------------------------------------------------------

def draw_pipeline(ax: Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        -0.005,
        1.02,
        "a",
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.025,
        1.02,
        "End-to-end performance obscures attribution",
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        transform=ax.transAxes,
    )

    box_y = 0.42
    box_height = 0.33

    draw_graph_inset(ax)

    boxes = [
        (
            0.215,
            0.145,
            r"Table $\phi$",
            TABLE,
            TABLE,
            "white",
            FILL_ALPHA,
        ),
        (
            0.410,
            0.145,
            r"Context $S$",
            CONTEXT,
            CONTEXT,
            "white",
            FILL_ALPHA,
        ),
        (
            0.605,
            0.145,
            r"Decoder $D$",
            DECODER,
            DECODER,
            "white",
            FILL_ALPHA,
        ),
        (
            0.805,
            0.170,
            "Evaluation",
            INK,
            LIGHT,
            INK,
            1.0,
        ),
    ]

    for x, width, title, edge, face, title_color, alpha in boxes:
        add_box(
            ax,
            x,
            box_y,
            width,
            box_height,
            title,
            edge=edge,
            face=face,
            title_color=title_color,
            linewidth=1.15 if edge not in (BORDER, INK) else 0.9,
            alpha=alpha,
        )

    arrow_y = box_y + box_height / 2

    # The first value is the right edge of the graph drawing area.
    add_arrow(ax, 0.174, 0.205, arrow_y)
    add_arrow(ax, 0.360, 0.400, arrow_y)
    add_arrow(ax, 0.555, 0.595, arrow_y)
    add_arrow(ax, 0.750, 0.795, arrow_y)

    bracket_y = 0.22
    ax.plot(
        [0.215, 0.750],
        [bracket_y, bracket_y],
        color=MUTED,
        linewidth=0.7,
    )
    ax.plot(
        [0.215, 0.215],
        [bracket_y, bracket_y + 0.05],
        color=MUTED,
        linewidth=0.7,
    )
    ax.plot(
        [0.750, 0.750],
        [bracket_y, bracket_y + 0.05],
        color=MUTED,
        linewidth=0.7,
    )

    ax.text(
        0.4825,
        0.11,
        (
            r"One number $M(\phi,S,D)$ cannot identify "
            r"which component produced the result."
        ),
        ha="center",
        va="center",
        fontsize=7.4,
        color=MUTED,
    )


# ---------------------------------------------------------------------
# Panel b
# ---------------------------------------------------------------------

def add_audit_component(
    ax: Axes,
    x: float,
    y: float,
    label: str,
    *,
    changed: bool,
    color: str,
) -> None:
    width = 0.060
    height = 0.135

    if changed:
        face = color
        edge = color
        text_color = "white"
        alpha = FILL_ALPHA
    else:
        face = LIGHT
        edge = BORDER
        text_color = INK
        alpha = 1.0

    add_box(
        ax,
        x,
        y,
        width,
        height,
        label,
        edge=edge,
        face=face,
        title_color=text_color,
        linewidth=0.85,
        alpha=alpha,
        fontsize=7.8,
    )


def draw_audit_pair(
    ax: Axes,
    x0: float,
    title: str,
    changed_index: int,
    color: str,
    delta_label: str,
) -> None:
    ax.text(
        x0 + 0.145,
        0.82,
        title,
        ha="center",
        va="center",
        fontsize=7.9,
        fontweight="bold",
        color=color,
    )

    component_labels = [r"$\phi$", r"$S$", r"$D$"]
    changed_labels = [
        (r"$\phi_a$", r"$\phi_b$"),
        (r"$S_a$", r"$S_b$"),
        (r"$D_a$", r"$D_b$"),
    ]

    x_positions = [
        x0 + 0.055,
        x0 + 0.140,
        x0 + 0.225,
    ]
    row_y_positions = (0.545, 0.315)
    row_names = ("Ref.", "Var.")

    for row_index, (y, row_name) in enumerate(
        zip(row_y_positions, row_names)
    ):
        ax.text(
            x0 + 0.006,
            y + 0.0675,
            row_name,
            ha="left",
            va="center",
            fontsize=6.15,
            color=MUTED,
        )

        for component_index, x in enumerate(x_positions):
            label = component_labels[component_index]

            if component_index == changed_index:
                label = changed_labels[component_index][row_index]

            add_audit_component(
                ax,
                x,
                y,
                label,
                changed=component_index == changed_index,
                color=color,
            )

            if component_index < 2:
                add_arrow(
                    ax,
                    x + 0.062,
                    x_positions[component_index + 1] - 0.004,
                    y + 0.0675,
                    linewidth=0.65,
                    mutation_scale=7,
                )

    ax.text(
        x0 + 0.145,
        0.155,
        delta_label,
        ha="center",
        va="center",
        fontsize=7.0,
        color=INK,
    )


def draw_paired_audit(ax: Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        -0.005,
        1.02,
        "b",
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.025,
        1.02,
        "A paired audit compares a reference with a one-component variant",
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.995,
        1.02,
        "",
        ha="right",
        va="bottom",
        fontsize=6.7,
        color=MUTED,
        transform=ax.transAxes,
    )

    draw_audit_pair(
        ax,
        0.005,
        "Table representation comparison",
        0,
        TABLE,
        r"$\Delta_{\phi}(S,D)$",
    )
    draw_audit_pair(
        ax,
        0.345,
        "Sampled context comparison",
        1,
        CONTEXT,
        r"$\Delta_{S}(\phi,D)$",
    )
    draw_audit_pair(
        ax,
        0.685,
        "Decoder comparison",
        2,
        DECODER,
        r"$\Delta_{D}(\phi,S)$",
    )

    for x in (0.332, 0.672):
        ax.plot(
            [x, x],
            [0.12, 0.86],
            color=BORDER,
            linewidth=0.65,
        )

    ax.text(
        0.5,
        -0.015,
        (
            ""
            ""
        ),
        ha="center",
        va="top",
        fontsize=7.0,
        color=MUTED,
        transform=ax.transAxes,
    )


# ---------------------------------------------------------------------
# Figure and export
# ---------------------------------------------------------------------

def build_figure() -> Figure:
    configure_matplotlib()

    # Slightly taller than the previous version so panel a gives the graph
    # more physical space.
    figure = plt.figure(figsize=(7.35, 4.30))

    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=[1.08, 1.10],
        hspace=0.30,
        left=0.025,
        right=0.985,
        top=0.94,
        bottom=0.10,
    )

    draw_pipeline(figure.add_subplot(grid[0]))
    draw_paired_audit(figure.add_subplot(grid[1]))

    return figure


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure = build_figure()

    pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"

    figure.savefig(pdf_path, dpi=400)
    figure.savefig(png_path, dpi=400)

    plt.close(figure)

    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
