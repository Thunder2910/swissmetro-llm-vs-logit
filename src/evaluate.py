"""Accuracy, shares, elasticities, and bootstrap intervals.

Elasticities use the base-referenced arc:

    E = ((S_new - S_base) / S_base) / delta

where S is the mean predicted probability of the shocked mode and delta is
the proportional change (0.20 for a +20% shock, -0.20 for a -20% shock).
A negative own elasticity means the share moves against the attribute.

Bootstrap intervals resample respondents, keeping all of a person's tasks
together. They describe uncertainty from the finite sample of people, not
a model of how the LLM was trained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.mnl import chosen_loglik, hard_predict


def availability_of(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            df["train_av"].to_numpy(dtype=bool),
            df["sm_av"].to_numpy(dtype=bool),
            df["car_av"].to_numpy(dtype=bool),
        ]
    )


def majority_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Most common training choice, skipping a mode that is unavailable."""
    order = train_df["y"].value_counts().index.to_numpy()
    avail = availability_of(test_df)
    pred = np.empty(len(test_df), dtype=int)
    for i in range(len(test_df)):
        picked = None
        for y in order:
            if avail[i, int(y)]:
                picked = int(y)
                break
        if picked is None:
            raise RuntimeError("No available alternative for a majority prediction.")
        pred[i] = picked
    return pred


def confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=int)
    for a, b in zip(y.tolist(), pred.tolist()):
        if b < 0:
            continue
        cm[int(a), int(b)] += 1
    return cm


def brier(proba: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(3)[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def classification_report(
    name: str,
    y: np.ndarray,
    pred: np.ndarray,
    proba: np.ndarray | None,
    ids: np.ndarray,
) -> dict:
    valid = pred >= 0
    y_v = y[valid]
    pred_v = pred[valid]
    acc = float((pred_v == y_v).mean()) if len(y_v) else float("nan")
    cm = confusion(y, pred)
    recalls = []
    precisions = []
    for k in range(3):
        row = cm[k].sum()
        col = cm[:, k].sum()
        recalls.append(float(cm[k, k] / row) if row else float("nan"))
        precisions.append(float(cm[k, k] / col) if col else float("nan"))
    out = {
        "model": name,
        "n": int(valid.sum()),
        "n_invalid": int((~valid).sum()),
        "n_people": int(pd.unique(ids[valid]).size) if valid.any() else 0,
        "accuracy": acc,
        "recall": recalls,
        "precision": precisions,
        "confusion": cm.tolist(),
        "hard_share": [float((pred_v == k).mean()) if len(pred_v) else float("nan") for k in range(3)],
        "observed_share": [float((y == k).mean()) for k in range(3)],
    }
    if proba is not None:
        out["loglik"] = chosen_loglik(proba[valid], y_v)
        out["mean_loglik"] = float(out["loglik"] / max(int(valid.sum()), 1))
        out["brier"] = brier(proba[valid], y_v)
        out["prob_share"] = [float(proba[valid, k].mean()) for k in range(3)]
        out["mean_p_chosen"] = float(proba[valid][np.arange(valid.sum()), y_v].mean())
    return out


def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test. Pairs are tasks, so the p-value is anti-conservative."""
    a_right = pred_a == y
    b_right = pred_b == y
    only_b = int((~a_right & b_right).sum())
    only_a = int((a_right & ~b_right).sum())
    n_disc = only_a + only_b
    if n_disc == 0:
        p = 1.0
    else:
        p = float(binomtest(only_b, n_disc, 0.5).pvalue)
    return {
        "a_only_correct": only_a,
        "b_only_correct": only_b,
        "n_discordant": n_disc,
        "p_value_exact": p,
        "note": "Treats tasks as independent. Prefer the respondent bootstrap interval.",
    }


def respondent_draws(ids: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    uniq, inverse = np.unique(ids, return_inverse=True)
    buckets = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    g = len(buckets)
    draws = []
    for _ in range(n_boot):
        take = rng.integers(0, g, size=g)
        draws.append(np.concatenate([buckets[i] for i in take]))
    return draws


def _pct(draws_stat: np.ndarray) -> list[float]:
    if np.all(np.isnan(draws_stat)):
        return [float("nan"), float("nan")]
    lo, hi = np.nanpercentile(draws_stat, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_mean(values: np.ndarray, draws: list[np.ndarray]) -> dict:
    stats = np.array([float(values[idx].mean()) for idx in draws])
    return {"estimate": float(values.mean()), "ci95": _pct(stats)}


def bootstrap_accuracy_diff(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    draws: list[np.ndarray],
) -> dict:
    """Mean accuracy of A minus mean accuracy of B, same resampled people."""
    point = float(correct_a.mean() - correct_b.mean())
    stats = np.array(
        [float(correct_a[idx].mean() - correct_b[idx].mean()) for idx in draws]
    )
    return {"estimate": point, "ci95": _pct(stats)}


def own_elasticity(base: np.ndarray, shock: np.ndarray, delta: float, draws: list[np.ndarray]) -> dict:
    def one(idx: np.ndarray) -> float:
        s0 = float(base[idx].mean())
        s1 = float(shock[idx].mean())
        if s0 <= 1e-8:
            return float("nan")
        return ((s1 - s0) / s0) / delta

    point = one(np.arange(len(base)))
    stats = np.array([one(idx) for idx in draws])
    return {
        "share_base": float(base.mean()),
        "share_shock": float(shock.mean()),
        "delta": float(delta),
        "elasticity": float(point),
        "ci95": _pct(stats),
        "n": int(len(base)),
    }


def central_vot(
    s_time_minus: float,
    s_time_plus: float,
    s_cost_minus: float,
    s_cost_plus: float,
) -> dict:
    """Implied CHF/hour from ±10 minute and ±10 CHF shifts of one mode's share.

    VOT = 60 * (dS/d minute) / (dS/d CHF). Both derivatives are negative when
    the mode is disliked as it gets slower or dearer, and the ratio is then
    positive. A flat or wrong-signed cost response is reported and not dressed
    up as a value of time.
    """
    d_s_dt = (s_time_plus - s_time_minus) / 20.0
    d_s_dc = (s_cost_plus - s_cost_minus) / 20.0
    cost_span = s_cost_plus - s_cost_minus
    time_span = s_time_plus - s_time_minus
    interpretable = bool(d_s_dt < 0 and d_s_dc < 0 and abs(cost_span) >= 0.005)
    vot = float(60.0 * d_s_dt / d_s_dc) if d_s_dc != 0 else float("nan")
    return {
        "d_share_per_minute": float(d_s_dt),
        "d_share_per_chf": float(d_s_dc),
        "share_change_time_pm10": float(time_span),
        "share_change_cost_pm10": float(cost_span),
        "vot_chf_per_hour": vot,
        "interpretable": interpretable,
        "share_time_minus10": float(s_time_minus),
        "share_time_plus10": float(s_time_plus),
        "share_cost_minus10": float(s_cost_minus),
        "share_cost_plus10": float(s_cost_plus),
    }


def bootstrap_vot(
    time_minus: np.ndarray,
    time_plus: np.ndarray,
    cost_minus: np.ndarray,
    cost_plus: np.ndarray,
    draws: list[np.ndarray],
) -> dict:
    point = central_vot(
        float(time_minus.mean()),
        float(time_plus.mean()),
        float(cost_minus.mean()),
        float(cost_plus.mean()),
    )

    def one(idx: np.ndarray) -> float:
        got = central_vot(
            float(time_minus[idx].mean()),
            float(time_plus[idx].mean()),
            float(cost_minus[idx].mean()),
            float(cost_plus[idx].mean()),
        )
        return got["vot_chf_per_hour"]

    stats = np.array([one(idx) for idx in draws])
    point["ci95"] = _pct(stats)
    point["n"] = int(len(time_minus))
    return point


def align_predictions(frame: pd.DataFrame, pred: pd.DataFrame, scenario: str) -> pd.DataFrame:
    sub = pred.loc[pred["scenario"] == scenario, :].copy()
    merged = frame[["obs_id"]].merge(sub, on="obs_id", how="left", validate="one_to_one")
    if merged["p_train"].isna().any():
        missing = merged.loc[merged["p_train"].isna(), "obs_id"].head(5).tolist()
        raise RuntimeError(f"Missing predictions for scenario {scenario}: {missing}")
    return merged


def proba_matrix(aligned: pd.DataFrame) -> np.ndarray:
    return aligned[["p_train", "p_sm", "p_car"]].to_numpy(dtype=float)


def hard_from_aligned(aligned: pd.DataFrame) -> np.ndarray:
    return aligned["pred"].to_numpy(dtype=int)
