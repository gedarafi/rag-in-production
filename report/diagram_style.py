"""Shared grayscale style and drawing helpers for the report diagrams.

Coordinates are in centimetres; figures are saved as PDF in report/figures.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

# same font as the slides (Palatino)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "TeX Gyre Pagella", "DejaVu Serif"],
    "font.monospace": ["Menlo", "DejaVu Sans Mono"],
    "font.size": 8,
    "pdf.fonttype": 42,
})

INK = "0.10"
EDGE = "0.35"
BOX_FILL = "0.97"
LLM_FILL = "0.84"
TOOL_FILL = "1.0"
GROUP_EDGE = "0.60"
GROUP_FILL = "0.94"
NOTE = "0.30"

TITLE_SIZE = 8.0
DETAIL_SIZE = 6.8
NOTE_SIZE = 6.8
LINE_WIDTH = 0.7
CENTIMETRES_PER_POINT = 2.54 / 72


def new_figure(width: float, height: float):
    """A blank figure of the given size in centimetres, with one axes in centimetre units."""
    figure = plt.figure(figsize=(width / 2.54, height / 2.54))
    axes = figure.add_axes([0, 0, 1, 1])
    axes.set_xlim(0, width)
    axes.set_ylim(0, height)
    axes.set_aspect("equal")
    axes.axis("off")
    return figure, axes


def save(figure, name: str) -> None:
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    @property
    def left(self):
        return (self.x - self.width / 2, self.y)

    @property
    def right(self):
        return (self.x + self.width / 2, self.y)

    @property
    def top(self):
        return (self.x, self.y + self.height / 2)

    @property
    def bottom(self):
        return (self.x, self.y - self.height / 2)


def draw_box(axes, x, y, width, height, title, detail="", kind="box", title_family="serif") -> Box:
    """A rounded box with a title and an optional smaller detail text below it.

    kind is "box" (light fill), "llm" (darker fill, bold title) or "tool" (white fill).
    """
    fill = {"box": BOX_FILL, "llm": LLM_FILL, "tool": TOOL_FILL}[kind]
    edge_width = LINE_WIDTH * (1.4 if kind == "llm" else 1.0)
    axes.add_patch(FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0,rounding_size=0.12",
        linewidth=edge_width, edgecolor=INK if kind == "llm" else EDGE, facecolor=fill, zorder=2,
    ))

    title_line = TITLE_SIZE * 1.2 * CENTIMETRES_PER_POINT
    detail_line = DETAIL_SIZE * 1.2 * CENTIMETRES_PER_POINT
    title_lines = title.count("\n") + 1
    detail_lines = detail.count("\n") + 1 if detail else 0
    gap = 0.06 if detail else 0
    block = title_lines * title_line + gap + detail_lines * detail_line
    top = y + block / 2

    axes.text(x, top, title, ha="center", va="top", fontsize=TITLE_SIZE, color=INK, family=title_family,
              weight="bold" if kind == "llm" else "normal", linespacing=1.2, zorder=3)
    if detail:
        axes.text(x, top - title_lines * title_line - gap, detail, ha="center", va="top",
                  fontsize=DETAIL_SIZE, color=NOTE, style="italic", linespacing=1.2, zorder=3)
    return Box(x, y, width, height)


def draw_group(axes, left, bottom, right, top, label, dashed=False, filled=True) -> None:
    """A rounded region behind boxes, labelled in its top-left corner."""
    axes.add_patch(FancyBboxPatch(
        (left, bottom), right - left, top - bottom,
        boxstyle="round,pad=0,rounding_size=0.2",
        linewidth=LINE_WIDTH * 0.8, edgecolor=GROUP_EDGE, facecolor=GROUP_FILL if filled else "none",
        linestyle=(0, (4, 2)) if dashed else "-", zorder=0 if filled else 1,
    ))
    axes.text(left + 0.15, top - 0.12, label, ha="left", va="top", fontsize=NOTE_SIZE, color=NOTE,
              style="italic", zorder=3)


def draw_arrow(axes, start, end, label="", label_position="above", dashed=False, curvature=0.0,
               label_offset=0.12) -> None:
    """A straight or curved arrow from start to end, with an optional label next to its midpoint."""
    axes.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>,head_length=3.2,head_width=1.6",
        connectionstyle=f"arc3,rad={curvature}", linewidth=LINE_WIDTH, color=EDGE,
        linestyle=(0, (3, 2)) if dashed else "-", shrinkA=0, shrinkB=0, zorder=1,
    ))
    if label:
        middle_x, middle_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        draw_label(axes, middle_x, middle_y, label, label_position, label_offset)


def draw_path(axes, points, label="", label_at=None, label_position="above", dashed=True) -> None:
    """A polyline through points with an arrow head on the last segment."""
    style = (0, (3, 2)) if dashed else "-"
    for start, end in zip(points[:-2], points[1:-1]):
        axes.add_line(Line2D([start[0], end[0]], [start[1], end[1]], linewidth=LINE_WIDTH, color=EDGE,
                             linestyle=style, zorder=1))
    draw_arrow(axes, points[-2], points[-1], dashed=dashed)
    if label:
        draw_label(axes, label_at[0], label_at[1], label, label_position, 0.0)


def draw_label(axes, x, y, text, position="above", offset=0.12) -> None:
    placement = {
        "above": dict(xy=(x, y + offset), ha="center", va="bottom"),
        "below": dict(xy=(x, y - offset), ha="center", va="top"),
        "left": dict(xy=(x - offset, y), ha="right", va="center"),
        "right": dict(xy=(x + offset, y), ha="left", va="center"),
    }[position]
    axes.text(*placement["xy"], text, ha=placement["ha"], va=placement["va"], fontsize=NOTE_SIZE,
              color=NOTE, style="italic", linespacing=1.15, zorder=3)
