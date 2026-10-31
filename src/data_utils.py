"""Load the Swissmetro file and apply the Biogeme estimation filters.

The standard sample keeps trip purposes 1 (commute) and 3 (business),
drops unknown choices (CHOICE == 0), treats a mode as available only when
its availability flag is on (and, for train and car, SP != 0), and sets
the out-of-pocket train and Swissmetro cost to zero when the respondent
holds a General Abonnement.

That last step matters. In the raw file, TRAIN_CO for a GA holder is the
annual pass price, not the price of the trip (Bierlaire, Swissmetro data
description). The marginal trip cost is zero. SM_CO is a hypothetical
fare computed without the pass; the Biogeme specification also sets it
to zero for GA holders, which is the assumption used here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_SHA256, DATA_URL


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_data(path: Path) -> Path:
    """Use the committed file when the checksum matches, otherwise download."""
    if path.exists() and sha256_file(path) == DATA_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    urllib.request.urlretrieve(DATA_URL, path)
    digest = sha256_file(path)
    if digest != DATA_SHA256:
        raise RuntimeError(
            f"Downloaded Swissmetro file checksum {digest} does not match {DATA_SHA256}"
        )
    return path


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def estimation_sample(raw: pd.DataFrame) -> pd.DataFrame:
    """Biogeme swissmetro_data.py filter and variable definitions."""
    keep = raw["PURPOSE"].isin([1, 3]) & (raw["CHOICE"] != 0)
    df = raw.loc[keep].copy().reset_index(drop=True)
    df["obs_id"] = np.arange(len(df), dtype=int)

    # Non-zero availability, matching Biogeme's av = flag * (SP != 0).
    sp_ok = df["SP"].to_numpy() != 0
    df["train_av"] = ((df["TRAIN_AV"].to_numpy() != 0) & sp_ok).astype(np.int8)
    df["sm_av"] = (df["SM_AV"].to_numpy() != 0).astype(np.int8)
    df["car_av"] = ((df["CAR_AV"].to_numpy() != 0) & sp_ok).astype(np.int8)

    ga_pays = df["GA"].to_numpy() == 0
    df["train_cost"] = np.where(ga_pays, df["TRAIN_CO"].to_numpy(dtype=float), 0.0)
    df["sm_cost"] = np.where(ga_pays, df["SM_CO"].to_numpy(dtype=float), 0.0)
    df["car_cost"] = df["CAR_CO"].to_numpy(dtype=float)

    # Choice codes 1, 2, 3 become column indices 0, 1, 2.
    df["y"] = df["CHOICE"].to_numpy(dtype=int) - 1
    df["senior"] = df["AGE"].isin([4, 5]).astype(np.int8)
    df["high_income"] = (df["INCOME"].to_numpy() == 3).astype(np.int8)
    df["business"] = (df["PURPOSE"].to_numpy() == 3).astype(np.int8)
    return df


def availability_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            df["train_av"].to_numpy(dtype=bool),
            df["sm_av"].to_numpy(dtype=bool),
            df["car_av"].to_numpy(dtype=bool),
        ]
    )


def _within_id_constant(df: pd.DataFrame, col: str) -> bool:
    return bool((df.groupby("ID")[col].nunique() == 1).all())


def sample_description(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Facts used in the README. Computed, not typed by hand."""
    non_ga = df["GA"].to_numpy() == 0
    ga = ~non_ga
    ratio = (df.loc[non_ga, "SM_CO"] / df.loc[non_ga, "TRAIN_CO"]).replace(
        [np.inf, -np.inf], np.nan
    )
    tasks = df.groupby("ID").size()
    people = df.groupby("ID").agg(
        ga=("GA", "max"),
        purpose=("PURPOSE", "max"),
        male=("MALE", "max"),
        age=("AGE", "max"),
        income=("INCOME", "max"),
        car_av=("car_av", "max"),
    )
    chosen_ok = (
        ((df["y"] == 0) & (df["train_av"] == 1))
        | ((df["y"] == 1) & (df["sm_av"] == 1))
        | ((df["y"] == 2) & (df["car_av"] == 1))
    )
    desc = {
        "n_raw": int(len(raw)),
        "n_obs": int(len(df)),
        "n_excluded": int(len(raw) - len(df)),
        "n_people": int(df["ID"].nunique()),
        "tasks_per_person_min": int(tasks.min()),
        "tasks_per_person_max": int(tasks.max()),
        "choice_counts": {
            "train": int((df["CHOICE"] == 1).sum()),
            "swissmetro": int((df["CHOICE"] == 2).sum()),
            "car": int((df["CHOICE"] == 3).sum()),
        },
        "n_train_available": int(df["train_av"].sum()),
        "n_sm_available": int(df["sm_av"].sum()),
        "n_car_available": int(df["car_av"].sum()),
        "n_choice_unavailable": int((~chosen_ok).sum()),
        "n_sp_not_1": int((df["SP"] != 1).sum()),
        "n_business_obs": int((df["PURPOSE"] == 3).sum()),
        "n_commute_obs": int((df["PURPOSE"] == 1).sum()),
        "n_business_people": int((people["purpose"] == 3).sum()),
        "n_commute_people": int((people["purpose"] == 1).sum()),
        "n_ga_obs": int(ga.sum()),
        "n_ga_people": int((people["ga"] == 1).sum()),
        "n_male_people": int((people["male"] == 1).sum()),
        "n_female_people": int((people["male"] == 0).sum()),
        "n_senior_people": int(people["age"].isin([4, 5]).sum()),
        "n_high_income_people": int((people["income"] == 3).sum()),
        "n_people_without_car": int((people["car_av"] == 0).sum()),
        "covariate_constant_within_id": {
            col: _within_id_constant(df, col)
            for col in ("GA", "PURPOSE", "AGE", "INCOME", "MALE", "car_av", "WHO")
        },
        "median_train_co_if_ga": float(df.loc[ga, "TRAIN_CO"].median()),
        "median_train_co_if_no_ga": float(df.loc[non_ga, "TRAIN_CO"].median()),
        "median_sm_co_if_ga": float(df.loc[ga, "SM_CO"].median()),
        "median_sm_co_if_no_ga": float(df.loc[non_ga, "SM_CO"].median()),
        "median_sm_over_train_fare_if_no_ga": float(ratio.median()),
        "mean_train_tt": float(df["TRAIN_TT"].mean()),
        "mean_sm_tt": float(df["SM_TT"].mean()),
        "mean_car_tt_if_available": float(df.loc[df["car_av"] == 1, "CAR_TT"].mean()),
        "data_url": DATA_URL,
        "data_sha256": DATA_SHA256,
    }
    return desc


def split_by_respondent(
    df: pd.DataFrame, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out a fraction of respondent IDs. Every task of a person stays together."""
    rng = np.random.default_rng(seed)
    ids = np.sort(df["ID"].unique())
    rng.shuffle(ids)
    n_test = int(round(test_fraction * len(ids)))
    test_ids = set(ids[:n_test].tolist())
    is_test = df["ID"].isin(test_ids).to_numpy()
    return ~is_test, is_test


def stratified_task_sample(
    df: pd.DataFrame,
    n: int,
    seed: int,
    max_per_id: int,
    stratify_col: str = "CHOICE",
) -> pd.DataFrame:
    """Draw tasks without replacement, close to the choice mix, capped per person.

    Greedy acceptance after a shuffle. The achieved size can be slightly
    below `n` if the cap binds; the caller records the achieved size.
    """
    rng = np.random.default_rng(seed)
    counts = df[stratify_col].value_counts().to_dict()
    total = float(len(df))
    raw_quota = {k: n * (v / total) for k, v in counts.items()}
    quota = {k: int(np.floor(v)) for k, v in raw_quota.items()}
    leftover = n - sum(quota.values())
    order = sorted(raw_quota, key=lambda k: raw_quota[k] - quota[k], reverse=True)
    for k in order[:leftover]:
        quota[k] += 1

    taken_class = {k: 0 for k in quota}
    taken_id: dict[int, int] = {}
    chosen: list[int] = []
    order_idx = df.index.to_numpy().copy()
    rng.shuffle(order_idx)
    for idx in order_idx:
        row = df.loc[idx]
        cls = row[stratify_col]
        pid = int(row["ID"])
        if taken_class.get(cls, 0) >= quota.get(cls, 0):
            continue
        if taken_id.get(pid, 0) >= max_per_id:
            continue
        chosen.append(int(idx))
        taken_class[cls] = taken_class.get(cls, 0) + 1
        taken_id[pid] = taken_id.get(pid, 0) + 1
        if len(chosen) >= n:
            break
    out = df.loc[chosen].sort_values("obs_id")
    return out


def apply_attribute_shock(
    df: pd.DataFrame,
    mode: str,
    attribute: str,
    kind: str,
    level: float,
) -> pd.DataFrame:
    """Shock payable time or cost. Percentage shocks leave true zeros at zero.

    `level` is a percent for kind='pct' (20 means +20%) and an absolute
    number of minutes or CHF for kind='abs'.
    """
    out = df.copy()
    time_col = {"train": "TRAIN_TT", "sm": "SM_TT", "car": "CAR_TT"}[mode]
    cost_col = {"train": "train_cost", "sm": "sm_cost", "car": "car_cost"}[mode]
    col = time_col if attribute == "time" else cost_col
    values = out[col].to_numpy(dtype=float).copy()
    if kind == "pct":
        values = values * (1.0 + level / 100.0)
    elif kind == "abs":
        values = values + level
        values = np.clip(values, 0.0, None)
    else:
        raise ValueError(kind)
    # Shared scenario for the MNL and the prompt: one decimal place.
    out[col] = np.round(values, 1)
    return out


def equalise_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Controlled probe: same time, cost, and headway on every available mode."""
    out = df.copy()
    for col, value in (
        ("TRAIN_TT", 90.0),
        ("SM_TT", 90.0),
        ("CAR_TT", 90.0),
        ("train_cost", 50.0),
        ("sm_cost", 50.0),
        ("car_cost", 50.0),
        ("TRAIN_HE", 30.0),
        ("SM_HE", 30.0),
    ):
        out[col] = value
    out["SM_SEATS"] = 0
    return out
