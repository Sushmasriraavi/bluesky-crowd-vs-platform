"""One plotting style for every figure in the paper (matplotlib, static PNG + PDF).

Colors: validated categorical order (blue, orange, aqua) - never more than 3 series per chart;
text in ink colors, never series colors; thin marks, recessive grid, single y-axis.
Every figure is saved together with the CSV of the exact numbers it plots.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # slots 1-3 (validated all-pairs, light surface)
MUTED = "#8a8984"                              # reference lines, "other"
GAP = "#e9e8e4"                                # shaded data gaps
INK, INK2 = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.edgecolor": "#cfcdc6", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ebeae6", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 9,
})


def fig(w=7.0, h=3.6):
    f, ax = plt.subplots(figsize=(w, h))
    return f, ax


def save(f, out_dir: Path, name: str, note: str | None = None):
    """Save figure as PNG (slides) and PDF (paper). `note` = one-line caption under the plot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if note:
        f.text(0.01, 0.005, note, fontsize=8, color=INK2, ha="left", va="bottom")
    f.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    for ext in ("png", "pdf"):
        f.savefig(out_dir / f"{name}.{ext}", dpi=200)
    plt.close(f)


def table(df, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    return df


def shade_gaps(ax, gaps):
    """gaps: list of (start, end) datetimes -> grey bands labelled once."""
    for i, (a, b) in enumerate(gaps):
        ax.axvspan(a, b, color=GAP, lw=0, label="no firehose data" if i == 0 else None, zorder=0)


def ecdf(ax, values, color, label, logx=False):
    import numpy as np
    v = np.sort(np.asarray(values, dtype=float))
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return
    y = np.arange(1, len(v) + 1) / len(v)
    ax.step(v, y, where="post", color=color, label=f"{label} (n={len(v):,})")
    if logx:
        ax.set_xscale("log")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("share of accounts (cumulative)")


def dates(ax):
    """Compact date ticks (no overlapping labels)."""
    import matplotlib.dates as mdates
    loc = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
