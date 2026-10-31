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
BLACK = "#222222"
GRAY = "#8A8A8A"
LIGHT = "#F4F1EA"

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


def plot_confusion(cm_mnl: np.ndarray, cm_llm: np.ndarray, n: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), constrained_layout=True)
    vmax = max(int(cm_mnl.max()), int(cm_llm.max()), 1)
    for ax, cm, title in (
        (axes[0], cm_mnl, "A. Base MNL"),
        (axes[1], cm_llm, "B. LLM"),
    ):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(3), MODE_NAMES, rotation=20, ha="right")
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
                    f"{cm[i, j]}\n{pct:.0f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f"Choices on the LLM subsample (n = {n} tasks)", fontsize=12)
    _save(fig, "confusion")


def plot_shares(rows: list[dict], n: int) -> None:
    """rows: list of {name, shares: [train, sm, car]} using mean probabilities."""
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    x = np.arange(3)
    width = 0.16
    colors = [BLACK, GRAY, BLUE, GREEN, ORANGE]
    for i, row in enumerate(rows):
        offset = (i - (len(rows) - 1) / 2) * width
        ax.bar(x + offset, row["shares"], width=width, color=colors[i % len(colors)], label=row["name"])
    ax.set_xticks(x, MODE_NAMES)
    ax.set_ylabel("Share of tasks")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.set_title(f"Same {n} tasks. MNL and LLM bars are mean probabilities.")
    _save(fig, "market_shares")


def plot_elasticities(curves: dict) -> None:
    """curves[(mode, attribute)] = list of {model, pct, share} including pct=0.

    `mode` is train or sm. `attribute` is time or cost.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), constrained_layout=True, sharex=True)
    panels = [
        (0, 0, "train", "time", "A. Train time"),
        (0, 1, "train", "cost", "B. Train cost, fare-paying tasks"),
        (1, 0, "sm", "time", "C. Swissmetro time"),
        (1, 1, "sm", "cost", "D. Swissmetro cost, fare-paying tasks"),
    ]
    style = {"mnl": (BLUE, "MNL"), "llm": (ORANGE, "LLM")}
    for r, c, mode, attr, title in panels:
        ax = axes[r][c]
        for model, (color, label) in style.items():
            pts = [p for p in curves.get((mode, attr), []) if p["model"] == model]
            pts = sorted(pts, key=lambda p: p["pct"])
            if not pts:
                continue
            ax.plot(
                [p["pct"] for p in pts],
                [p["share"] for p in pts],
                marker="o",
                ms=4,
                color=color,
                label=label,
            )
        ax.axvline(0, color=GRAY, lw=0.6)
        ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.set_ylabel("Mean predicted probability")
        ax.set_xlabel("Change in the attribute (%)")
    handles = [
        plt.Line2D([0], [0], color=BLUE, marker="o", label="MNL"),
        plt.Line2D([0], [0], color=ORANGE, marker="o", label="LLM"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Own-mode probability when one attribute is rescaled", fontsize=12, y=1.06)
    _save(fig, "elasticities")


def plot_bias(ga_rows: list[dict], equal_shares: dict, n_ga: int, n_noga: int, n_eq: int) -> None:
    """ga_rows: {status, model, shares}. equal_shares: {model: [3]}."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), constrained_layout=True)
    ax = axes[0]
    models = ["Observed", "MNL", "LLM"]
    colors = {"Observed": BLACK, "MNL": BLUE, "LLM": ORANGE}
    # Gap: share among GA holders minus share among everyone else.
    # A model that ignores the pass has a gap near zero.
    x = np.arange(3)
    width = 0.24
    for m_i, model in enumerate(models):
        has = next(r for r in ga_rows if r["status"] == "Has GA" and r["model"] == model)
        none = next(r for r in ga_rows if r["status"] == "No GA" and r["model"] == model)
        gap = [has["shares"][k] - none["shares"][k] for k in range(3)]
        ax.bar(x + (m_i - 1) * width, gap, width=width, color=colors[model], label=model)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xticks(x, MODE_NAMES)
    ax.set_ylabel("Share with GA minus share without")
    ax.set_title(f"A. Annual-pass gap (n={n_ga} with, {n_noga} without)")
    ax.legend(loc="best")

    ax = axes[1]
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width / 2, equal_shares["mnl"], width, color=BLUE, label="MNL")
    ax.bar(x + width / 2, equal_shares["llm"], width, color=ORANGE, label="LLM")
    ax.set_xticks(x, MODE_NAMES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean predicted probability")
    ax.set_title(f"B. Equal time and cost (n={n_eq})")
    ax.legend()
    fig.suptitle("Where the two predictors put travellers when the trade-off changes", fontsize=12)
    _save(fig, "bias_checks")
