"""Multinomial logit by Newton-Raphson, with classical and cluster-robust SEs.

Utility is linear in parameters. Swissmetro is the reference alternative
(its ASC is fixed at 0). Time and cost enter as minutes/100 and CHF/100,
which is the scaling in the Biogeme Swissmetro example, so a coefficient
is the utility change for a 100-minute or 100-franc change.

The log-likelihood of a linear-in-parameters logit is concave, so Newton
steps from the origin reach the same optimum Biogeme reports on this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

SPEC_BASE = "base"
SPEC_RICH = "rich"
SPEC_CONST = "constants"

BASE_NAMES = ["ASC_TRAIN", "ASC_CAR", "B_TIME", "B_COST"]
RICH_NAMES = BASE_NAMES + [
    "ASC_TRAIN_BUSINESS",
    "ASC_CAR_BUSINESS",
    "B_GA_TRAIN",
    "B_GA_SM",
    "B_HEADWAY",
    "B_TIME_SENIOR",
    "B_COST_HIGHINC",
]
CONST_NAMES = ["ASC_TRAIN", "ASC_CAR"]


def _as_float(df: pd.DataFrame, col: str) -> np.ndarray:
    return df[col].to_numpy(dtype=float)


def design_matrix(df: pd.DataFrame, spec: str) -> tuple[np.ndarray, list[str]]:
    """Return X with shape (n, 3, k) for train, Swissmetro, car."""
    n = len(df)
    tt = np.column_stack(
        [
            _as_float(df, "TRAIN_TT") / 100.0,
            _as_float(df, "SM_TT") / 100.0,
            _as_float(df, "CAR_TT") / 100.0,
        ]
    )
    tc = np.column_stack(
        [
            _as_float(df, "train_cost") / 100.0,
            _as_float(df, "sm_cost") / 100.0,
            _as_float(df, "car_cost") / 100.0,
        ]
    )
    if spec == SPEC_CONST:
        names = list(CONST_NAMES)
        x = np.zeros((n, 3, len(names)))
        x[:, 0, 0] = 1.0
        x[:, 2, 1] = 1.0
        return x, names

    names = list(BASE_NAMES)
    cols = [
        np.array([1.0, 0.0, 0.0]),  # ASC_TRAIN
        np.array([0.0, 0.0, 1.0]),  # ASC_CAR
    ]
    # Broadcast the ASC columns.
    blocks = [
        np.broadcast_to(cols[0], (n, 3)).copy(),
        np.broadcast_to(cols[1], (n, 3)).copy(),
        tt,
        tc,
    ]
    if spec == SPEC_RICH:
        names = list(RICH_NAMES)
        business = _as_float(df, "business")[:, None]
        ga = (df["GA"].to_numpy() == 1).astype(float)[:, None]
        senior = _as_float(df, "senior")[:, None]
        high = _as_float(df, "high_income")[:, None]
        head = np.column_stack(
            [
                _as_float(df, "TRAIN_HE") / 100.0,
                _as_float(df, "SM_HE") / 100.0,
                np.zeros(n),
            ]
        )
        blocks.extend(
            [
                business * np.array([1.0, 0.0, 0.0]),
                business * np.array([0.0, 0.0, 1.0]),
                ga * np.array([1.0, 0.0, 0.0]),
                ga * np.array([0.0, 1.0, 0.0]),
                head,
                senior * tt,
                high * tc,
            ]
        )
    elif spec != SPEC_BASE:
        raise ValueError(f"Unknown spec: {spec}")
    x = np.stack(blocks, axis=2)
    return x, names


def _softmax(v: np.ndarray, avail: np.ndarray) -> np.ndarray:
    masked = np.where(avail, v, -np.inf)
    finite = np.where(np.isfinite(masked), masked, -1.0e100)
    m = finite.max(axis=1, keepdims=True)
    exp_v = np.exp(np.where(avail, masked - m, -np.inf))
    exp_v = np.where(avail, exp_v, 0.0)
    denom = exp_v.sum(axis=1, keepdims=True)
    if np.any(denom <= 0):
        raise RuntimeError("An observation has no available alternative.")
    return exp_v / denom


def loglik_and_derivatives(
    theta: np.ndarray,
    x: np.ndarray,
    avail: np.ndarray,
    y: np.ndarray,
    with_hessian: bool = True,
) -> tuple[float, np.ndarray, np.ndarray | None, np.ndarray]:
    """Log-likelihood, its gradient, its Hessian, and per-row scores.

    Scores are gradients of the observation log-likelihood, shape (n, k).
    """
    v = np.einsum("njk,k->nj", x, theta)
    p = _softmax(v, avail)
    n = len(y)
    idx = np.arange(n)
    # log P_y = V_y - logsumexp over available alternatives.
    finite = np.where(avail, v, -1.0e100)
    m = finite.max(axis=1)
    shift = np.where(avail, v - m[:, None], -1.0e300)
    log_denom = m + np.log(np.exp(shift).sum(axis=1))
    ll_i = v[idx, y] - log_denom
    if not np.all(avail[idx, y]):
        raise RuntimeError("Chosen alternative is marked unavailable.")
    ll = float(ll_i.sum())

    x_bar = np.einsum("nj,njk->nk", p, x)
    x_chosen = x[idx, y, :]
    scores = x_chosen - x_bar
    grad = scores.sum(axis=0)

    hess = None
    if with_hessian:
        xx = np.einsum("nj,njk,njl->nkl", p, x, x)
        outer = x_bar[:, :, None] * x_bar[:, None, :]
        hess = -(xx - outer).sum(axis=0)
    return ll, grad, hess, scores


def null_loglik(avail: np.ndarray) -> float:
    """Equal probability among available alternatives (all coefficients zero)."""
    n_av = avail.sum(axis=1).astype(float)
    if np.any(n_av < 1):
        raise RuntimeError("Empty choice set.")
    return float((-np.log(n_av)).sum())


@dataclass
class MNLResult:
    spec: str
    names: list[str]
    coef: np.ndarray
    se_classical: np.ndarray
    se_robust: np.ndarray
    se_cluster: np.ndarray
    cov_classical: np.ndarray
    cov_robust: np.ndarray
    cov_cluster: np.ndarray
    loglik: float
    null_loglik: float
    n_obs: int
    n_ids: int
    grad_norm: float
    iterations: int
    converged: bool

    @property
    def k(self) -> int:
        return len(self.names)

    @property
    def rho2(self) -> float:
        return 1.0 - (self.loglik / self.null_loglik)

    @property
    def rho2bar(self) -> float:
        return 1.0 - ((self.loglik - self.k) / self.null_loglik)

    @property
    def aic(self) -> float:
        return -2.0 * self.loglik + 2.0 * self.k

    @property
    def bic(self) -> float:
        return -2.0 * self.loglik + self.k * np.log(self.n_obs)

    def t_stat(self, which: str = "cluster") -> np.ndarray:
        se = {"classical": self.se_classical, "robust": self.se_robust, "cluster": self.se_cluster}[
            which
        ]
        with np.errstate(divide="ignore"):
            return self.coef / se

    def p_value(self, which: str = "cluster") -> np.ndarray:
        t = self.t_stat(which)
        return 2.0 * norm.sf(np.abs(t))

    def as_dict(self) -> dict:
        def col(a: np.ndarray) -> list[float]:
            return [float(v) for v in a]

        return {
            "spec": self.spec,
            "names": list(self.names),
            "coef": col(self.coef),
            "se_classical": col(self.se_classical),
            "se_robust": col(self.se_robust),
            "se_cluster": col(self.se_cluster),
            "t_classical": col(self.t_stat("classical")),
            "t_robust": col(self.t_stat("robust")),
            "t_cluster": col(self.t_stat("cluster")),
            "p_cluster": col(self.p_value("cluster")),
            "loglik": self.loglik,
            "null_loglik": self.null_loglik,
            "rho2": self.rho2,
            "rho2bar": self.rho2bar,
            "aic": self.aic,
            "bic": self.bic,
            "n_obs": self.n_obs,
            "n_ids": self.n_ids,
            "k": self.k,
            "grad_norm": self.grad_norm,
            "iterations": self.iterations,
            "converged": self.converged,
        }


def _invert(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


def _se_from_cov(cov: np.ndarray) -> np.ndarray:
    diag = np.clip(np.diag(cov), 0.0, None)
    return np.sqrt(diag)


def fit_mnl(
    df: pd.DataFrame,
    spec: str,
    max_iter: int = 25,
    tol: float = 1e-10,
) -> MNLResult:
    x, names = design_matrix(df, spec)
    avail = np.column_stack(
        [
            df["train_av"].to_numpy(dtype=bool),
            df["sm_av"].to_numpy(dtype=bool),
            df["car_av"].to_numpy(dtype=bool),
        ]
    )
    y = df["y"].to_numpy(dtype=int)
    ids = df["ID"].to_numpy()
    theta = np.zeros(len(names))
    converged = False
    grad_norm = np.inf
    it_used = 0
    ll = grad = hess = scores = None
    for it in range(1, max_iter + 1):
        it_used = it
        ll, grad, hess, scores = loglik_and_derivatives(theta, x, avail, y, True)
        grad_norm = float(np.linalg.norm(grad))
        # Hessian of the log-likelihood is negative semidefinite.
        info = -hess
        try:
            step = np.linalg.solve(info, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(info, grad, rcond=None)[0]
        if grad_norm < 1e-8 and np.max(np.abs(step)) < tol:
            converged = True
            break
        accepted = False
        base_ll = ll
        for shrink in range(12):
            trial = theta + step / (2**shrink)
            trial_ll, _, _, _ = loglik_and_derivatives(trial, x, avail, y, False)
            if trial_ll >= base_ll - 1e-10:
                theta = trial
                accepted = True
                break
        if not accepted:
            break
        if np.max(np.abs(step)) < tol:
            ll, grad, hess, scores = loglik_and_derivatives(theta, x, avail, y, True)
            grad_norm = float(np.linalg.norm(grad))
            converged = grad_norm < 1e-6
            break
    if ll is None or hess is None or scores is None or grad is None:
        raise RuntimeError("MNL fit failed before the first iteration.")
    # Recompute at the returned theta so SEs match the stored coefficients.
    ll, grad, hess, scores = loglik_and_derivatives(theta, x, avail, y, True)
    grad_norm = float(np.linalg.norm(grad))
    info = -hess
    cov_classical = _invert(info)
    meat = scores.T @ scores
    cov_robust = cov_classical @ meat @ cov_classical
    # Cluster scores by respondent, then the same sandwich.
    uniq, inverse = np.unique(ids, return_inverse=True)
    clustered = np.zeros((len(uniq), len(names)))
    np.add.at(clustered, inverse, scores)
    meat_c = clustered.T @ clustered
    cov_cluster = cov_classical @ meat_c @ cov_classical
    if grad_norm < 1e-6:
        converged = True
    return MNLResult(
        spec=spec,
        names=names,
        coef=theta.copy(),
        se_classical=_se_from_cov(cov_classical),
        se_robust=_se_from_cov(cov_robust),
        se_cluster=_se_from_cov(cov_cluster),
        cov_classical=cov_classical,
        cov_robust=cov_robust,
        cov_cluster=cov_cluster,
        loglik=float(ll),
        null_loglik=null_loglik(avail),
        n_obs=int(len(df)),
        n_ids=int(len(uniq)),
        grad_norm=grad_norm,
        iterations=it_used,
        converged=converged,
    )


def predict_proba(df: pd.DataFrame, result: MNLResult) -> np.ndarray:
    x, names = design_matrix(df, result.spec)
    if names != result.names:
        raise RuntimeError("Design names do not match the fitted model.")
    avail = np.column_stack(
        [
            df["train_av"].to_numpy(dtype=bool),
            df["sm_av"].to_numpy(dtype=bool),
            df["car_av"].to_numpy(dtype=bool),
        ]
    )
    v = np.einsum("njk,k->nj", x, result.coef)
    return _softmax(v, avail)


def hard_predict(proba: np.ndarray, avail: np.ndarray) -> np.ndarray:
    """Argmax over available modes. Ties break toward the earlier mode."""
    masked = np.where(avail, proba, -np.inf)
    return np.argmax(masked, axis=1).astype(int)


def chosen_loglik(proba: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(proba[np.arange(len(y)), y], 1e-300, None)
    return float(np.log(p).sum())


def value_of_time(
    result: MNLResult,
    time_names: tuple[str, ...] = ("B_TIME",),
    cost_names: tuple[str, ...] = ("B_COST",),
    cov: str = "cluster",
) -> dict:
    """CHF per hour. Both attributes are scaled by 100, so the 100s cancel.

    VOT = 60 * (sum of named time coefficients) / (sum of named cost coefficients).
    The delta-method standard error uses the selected covariance matrix.
    """
    index = {n: i for i, n in enumerate(result.names)}
    bt = sum(result.coef[index[n]] for n in time_names)
    bc = sum(result.coef[index[n]] for n in cost_names)
    vot = float(60.0 * bt / bc)
    g = np.zeros(result.k)
    for n in time_names:
        g[index[n]] += 60.0 / bc
    for n in cost_names:
        g[index[n]] += -60.0 * bt / (bc**2)
    cov_m = {"classical": result.cov_classical, "robust": result.cov_robust, "cluster": result.cov_cluster}[
        cov
    ]
    var = float(g @ cov_m @ g)
    se = float(np.sqrt(max(var, 0.0)))
    return {
        "vot_chf_per_hour": vot,
        "se": se,
        "time_coef": float(bt),
        "cost_coef": float(bc),
        "defined": bool(bc < 0 and bt < 0),
        "time_names": list(time_names),
        "cost_names": list(cost_names),
    }


def money_metric(result: MNLResult, name: str, cov: str = "cluster") -> dict:
    """CHF equivalent of a coefficient, using the base cost coefficient.

    dU/dCHF = B_COST/100. A positive coefficient is worth
    beta * 100 / (-B_COST) francs.
    """
    index = {n: i for i, n in enumerate(result.names)}
    b = float(result.coef[index[name]])
    bc = float(result.coef[index["B_COST"]])
    wtp = b * 100.0 / (-bc)
    # delta method for b * 100 / (-bc)
    g = np.zeros(result.k)
    g[index[name]] = 100.0 / (-bc)
    g[index["B_COST"]] += b * 100.0 / (bc**2)
    cov_m = {"classical": result.cov_classical, "robust": result.cov_robust, "cluster": result.cov_cluster}[
        cov
    ]
    se = float(np.sqrt(max(float(g @ cov_m @ g), 0.0)))
    return {"name": name, "chf": float(wtp), "se": se}


def likelihood_ratio(restricted: MNLResult, unrestricted: MNLResult) -> dict:
    from scipy.stats import chi2

    df = unrestricted.k - restricted.k
    stat = 2.0 * (unrestricted.loglik - restricted.loglik)
    if df <= 0:
        raise ValueError("Unrestricted model is not larger.")
    return {
        "lr": float(stat),
        "df": int(df),
        "p_value": float(chi2.sf(stat, df)),
        "restricted_ll": restricted.loglik,
        "unrestricted_ll": unrestricted.loglik,
    }
