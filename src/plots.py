"""Figures. Cluster-robust intervals, shared scales, no chart decoration."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.config import FIGURES, MODE_NAMES
from src.mnl import MNLResult

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GOLD = "#E69F00"
PURPLE = "#CC79A7"
BLACK = "#222222"
GRAY = "#8A8A8A"
LIGHT = "#F4F1EA"
SIZE_COLORS = {"1.5b": ORANGE, "3b": GOLD, "7b": PURPLE}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "text.color": "#222222",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

LABELS = {
    "ASC_TRAIN": "ASC train vs Swissmetro",
    "ASC_CAR": "ASC car vs Swissmetro",
    "B_TIME": "Time, per 100 minutes",
    "B_COST": "Cost, per 100 CHF",
    "ASC_TRAIN_BUSINESS": "Train ASC, business vs commute",
    "ASC_CAR_BUSINESS": "Car ASC, business vs commute",
    "B_GA_TRAIN": "GA holder, train",
    "B_GA_SM": "GA holder, Swissmetro",
    "B_HEADWAY": "Headway, per 100 minutes",
    "B_TIME_SENIOR": "Extra time coefficient, age 55+",
    "B_COST_HIGHINC": "Extra cost coefficient, income >100k",
}


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_coefficients(base: MNLResult, rich: MNLResult) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.6), constrained_layout=True)
    for ax, result, title in (
        (axes[0], base, "A. Base MNL"),
        (axes[1], rich, "B. Richer MNL"),
    ):
        names = list(result.names)
        coef = result.coef
        se = result.se_cluster
        y = np.arange(len(names))[::-1]
        ax.axvline(0, color=GRAY, lw=0.8)
        ax.errorbar(
            coef,
            y,
            xerr=1.96 * se,
            fmt="o",
            color=BLUE,
            ecolor=BLUE,
            elinewidth=1.2,
            capsize=3,
            markersize=5,
        )
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS.get(n, n) for n in names])
        ax.set_xlabel("Coefficient (cluster-robust 95% interval)")
        ax.set_title(title)
    fig.suptitle("Swissmetro multinomial logit, full estimation sample", fontsize=12)
    _save(fig, "coefficients")


def plot_confusion(panels: list[tuple[str, np.ndarray]], n: int) -> None:
    """panels: (title, 3x3 counts) in display order."""
    n_panels = len(panels)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(3.55 * n_panels, 4.15), constrained_layout=True
    )
    if n_panels == 1:
        axes = [axes]
    vmax = max(int(cm.max()) for _, cm in panels)
    vmax = max(vmax, 1)
    for ax, (title, cm) in zip(axes, panels):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(3), MODE_NAMES, rotation=25, ha="right")
        ax.set_yticks(range(3), MODE_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
        ax.set_title(title)
        for i in range(3):
            row_sum = cm[i].sum()
            for j in range(3):
                pct = 100.0 * cm[i, j] / row_sum if row_sum else 0.0
                color = "white" if cm[i, j] > 0.55 * vmax else BLACK
                ax.text(
                    j,
                    i,
                    f"{int(cm[i, j])}\n{pct:.0f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f"Choices on the same {n} tasks", fontsize=12)
    _save(fig, "confusion")


def plot_shares(rows: list[dict], n: int) -> None:
    """rows: list of {name, shares: [train, sm, car], color}."""
    fig, ax = plt.subplots(figsize=(9.4, 4.6), constrained_layout=True)
    x = np.arange(3)
    width = min(0.14, 0.82 / max(len(rows), 1))
    fallback = [BLACK, GRAY, BLUE, GREEN, ORANGE, GOLD, PURPLE]
    for i, row in enumerate(rows):
        offset = (i - (len(rows) - 1) / 2) * width
        ax.bar(
            x + offset,
            row["shares"],
            width=width,
            color=row.get("color", fallback[i % len(fallback)]),
            label=row["name"],
        )
    ax.set_xticks(x, MODE_NAMES)
    ax.set_ylabel("Share of tasks")
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.set_title(f"Same {n} tasks. Logit and LLM bars are mean probabilities.")
    _save(fig, "market_shares")


def plot_elasticities(curves: dict, styles: list[tuple[str, str, str, str]]) -> None:
    """curves[(mode, attribute)] = list of {model, pct, share} including pct=0.

    styles: (model key, color, matplotlib marker, legend label).
    """
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), constrained_layout=True, sharex=True)
    panels = [
        (0, 0, "train", "time", "A. Train time"),
        (0, 1, "train", "cost", "B. Train cost, fare-paying tasks"),
        (1, 0, "sm", "time", "C. Swissmetro time"),
        (1, 1, "sm", "cost", "D. Swissmetro cost, fare-paying tasks"),
    ]
    for r, c, mode, attr, title in panels:
        ax = axes[r][c]
        for model, color, marker, _label in styles:
            pts = [p for p in curves.get((mode, attr), []) if p["model"] == model]
            pts = sorted(pts, key=lambda p: p["pct"])
            if not pts:
                continue
            ax.plot(
                [p["pct"] for p in pts],
                [p["share"] for p in pts],
                marker=marker,
                ms=4,
                color=color,
            )
        ax.axvline(0, color=GRAY, lw=0.6)
        ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.set_ylabel("Mean predicted probability")
        ax.set_xlabel("Change in the attribute (%)")
    handles = [
        plt.Line2D([0], [0], color=color, marker=marker, label=label)
        for _model, color, marker, label in styles
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Own-mode probability when one attribute is rescaled", fontsize=12, y=1.08)
    _save(fig, "elasticities")


def plot_bias(
    gap_series: list[dict],
    equal_series: list[dict],
    n_ga: int,
    n_noga: int,
    n_eq: int,
) -> None:
    """gap_series: {name, color, gap: [train, sm, car]}.

    equal_series: {name, color, shares: [3]}.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), constrained_layout=True)
    ax = axes[0]
    x = np.arange(3)
    width = min(0.18, 0.8 / max(len(gap_series), 1))
    for i, series in enumerate(gap_series):
        ax.bar(
            x + (i - (len(gap_series) - 1) / 2) * width,
            series["gap"],
            width=width,
            color=series["color"],
            label=series["name"],
        )
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xticks(x, MODE_NAMES)
    ax.set_ylabel("Share with GA minus share without")
    ax.set_title(f"A. Annual-pass gap (n={n_ga} with, {n_noga} without)")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    width = min(0.18, 0.8 / max(len(equal_series), 1))
    for i, series in enumerate(equal_series):
        ax.bar(
            x + (i - (len(equal_series) - 1) / 2) * width,
            series["shares"],
            width=width,
            color=series["color"],
            label=series["name"],
        )
    ax.set_xticks(x, MODE_NAMES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean predicted probability")
    ax.set_title(f"B. Equal time and cost (n={n_eq})")
    ax.legend(loc="best", fontsize=8)
    fig.suptitle("Pass holders and the label prior, by model size", fontsize=12)
    _save(fig, "bias_checks")


def plot_model_size(acc_rows: list[dict], share_rows: list[dict], n: int) -> None:
    """acc_rows: name, accuracy, lo, hi, color. share_rows: name, shares, color."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.5), constrained_layout=True)
    ax = axes[0]
    y = np.arange(len(acc_rows))[::-1]
    for i, row in enumerate(acc_rows):
        lo = min(row["lo"], row["accuracy"])
        hi = max(row["hi"], row["accuracy"])
        ax.errorbar(
            row["accuracy"],
            y[i],
            xerr=[[row["accuracy"] - lo], [hi - row["accuracy"]]],
            fmt="o",
            color=row["color"],
            ecolor=row["color"],
            capsize=3,
            markersize=6,
        )
    ax.set_yticks(y, [row["name"] for row in acc_rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy on the shared tasks")
    ax.set_title("A. Accuracy, respondent bootstrap")

    ax = axes[1]
    x = np.arange(3)
    width = min(0.16, 0.8 / max(len(share_rows), 1))
    for i, row in enumerate(share_rows):
        ax.bar(
            x + (i - (len(share_rows) - 1) / 2) * width,
            row["shares"],
            width=width,
            color=row["color"],
            label=row["name"],
        )
    ax.set_xticks(x, MODE_NAMES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean predicted probability")
    ax.set_title("B. Where the probability goes")
    ax.legend(ncol=2, fontsize=8)
    fig.suptitle(f"Model size on the same {n} choice tasks", fontsize=12)
    _save(fig, "model_size")
