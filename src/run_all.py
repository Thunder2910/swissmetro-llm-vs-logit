"""Run the Swissmetro comparison and write results/ and figures/.

MNL estimation is exact (Newton on a concave likelihood). LLM scores are
cached by a hash of the prompt and the scoring code, so a rerun continues
rather than starting over.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BIOGEME_B01A,
    DATA_PATH,
    ELAST_MAX_TASKS_PER_PERSON,
    ELAST_N,
    FIGURES,
    LLM_N,
    SEVEN_B_SHOCK_N,
    SEVEN_B_SHOCK_SEED,
    MAX_TASKS_PER_PERSON,
    MODE_KEYS,
    MODE_NAMES,
    N_BOOT,
    PCT_SHOCKS,
    PILOT_N,
    PROBE_N,
    RESULTS,
    SEED,
    TEST_FRACTION,
)
from src.data_utils import (
    apply_attribute_shock,
    ensure_data,
    equalise_attributes,
    estimation_sample,
    load_raw,
    sample_description,
    split_by_respondent,
    stratified_task_sample,
)
from src.evaluate import (
    align_predictions,
    availability_of,
    bootstrap_accuracy_diff,
    bootstrap_mean,
    bootstrap_vot,
    classification_report,
    hard_from_aligned,
    majority_predict,
    mcnemar,
    own_elasticity,
    proba_matrix,
    respondent_draws,
)
from src.mnl import hard_predict
from src.mnl import (
    SPEC_BASE,
    SPEC_CONST,
    SPEC_RICH,
    fit_mnl,
    likelihood_ratio,
    money_metric,
    predict_proba,
    value_of_time,
)
from src.model_specs import ModelSpec, cache_path, ensure_weights, parse_model_keys
from src.plots import (
    BLUE,
    GREEN,
    GRAY,
    SIZE_COLORS,
    plot_bias,
    plot_coefficients,
    plot_confusion,
    plot_elasticities,
    plot_model_size,
    plot_shares,
)
from src.prompts import chat_pieces, prompt_fingerprint, render_prompt


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(obj), indent=2), encoding="utf-8")


def coef_table(result, sample: str) -> list[dict]:
    rows = []
    d = result.as_dict()
    for i, name in enumerate(d["names"]):
        rows.append(
            {
                "sample": sample,
                "spec": result.spec,
                "name": name,
                "coef": d["coef"][i],
                "se_classical": d["se_classical"][i],
                "se_robust": d["se_robust"][i],
                "se_cluster": d["se_cluster"][i],
                "t_classical": d["t_classical"][i],
                "t_robust": d["t_robust"][i],
                "t_cluster": d["t_cluster"][i],
                "p_cluster": d["p_cluster"][i],
            }
        )
    return rows


FIELDS = [
    "obs_id",
    "ID",
    "scenario",
    "y",
    "p_train",
    "p_sm",
    "p_car",
    "p_raw_train",
    "p_raw_sm",
    "p_raw_car",
    "digit_mass",
    "pred",
    "top_token",
    "top_is_digit",
    "top_is_available",
    "n_tokens",
    "seconds",
]


def load_done(path: Path) -> set[tuple[int, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    prev = pd.read_csv(path)
    if prev.empty:
        return set()
    return set(zip(prev["obs_id"].astype(int), prev["scenario"].astype(str)))


def score_scenarios(
    jobs: list[tuple[str, pd.DataFrame, str]],
    cache_file: Path,
    spec: ModelSpec,
) -> pd.DataFrame:
    """jobs: (scenario name, frame, wording key)."""
    from src.llm_choice import ChoiceModel

    done = load_done(cache_file)
    pending = 0
    for scenario, frame, _wording in jobs:
        for obs_id in frame["obs_id"].astype(int):
            if (int(obs_id), scenario) not in done:
                pending += 1
    print(f"{spec.model_id}: {pending} prompts to score ({len(done)} cached keys)")
    if pending == 0:
        cached = pd.read_csv(cache_file)
        return cached.drop_duplicates(["obs_id", "scenario"], keep="last")
    ensure_weights(spec)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    new_file = not cache_file.exists() or cache_file.stat().st_size == 0
    model = None
    t0 = time.perf_counter()
    scored = 0
    with cache_file.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for scenario, frame, wording in jobs:
            todo = frame.loc[~frame["obs_id"].astype(int).map(lambda i: (int(i), scenario) in done)]
            if todo.empty:
                print(f"  {scenario}: cached ({len(frame)})")
                continue
            if model is None:
                print(f"Loading {spec.model_id} from {spec.path.name}...")
                model = ChoiceModel(model_path=str(spec.path))
                groups = {str(k): v for k, v in model.groups.items()}
                write_json(RESULTS / f"digit_token_ids_{spec.key}.json", groups)
                print("Digit token ids", groups)
            print(f"  {scenario}: scoring {len(todo)}")
            for _, row in todo.iterrows():
                prefix, suffix = chat_pieces(row, wording)
                avail = np.array([row["train_av"], row["sm_av"], row["car_av"]], dtype=bool)
                score = model.score(prefix, suffix, avail)
                token = score.top_token.replace("\n", "\\n").replace("\r", "")
                writer.writerow(
                    {
                        "obs_id": int(row["obs_id"]),
                        "ID": int(row["ID"]),
                        "scenario": scenario,
                        "y": int(row["y"]),
                        "p_train": score.p_cond[0],
                        "p_sm": score.p_cond[1],
                        "p_car": score.p_cond[2],
                        "p_raw_train": score.p_raw[0],
                        "p_raw_sm": score.p_raw[1],
                        "p_raw_car": score.p_raw[2],
                        "digit_mass": score.digit_mass,
                        "pred": int(score.pred),
                        "top_token": token,
                        "top_is_digit": int(score.top_is_digit),
                        "top_is_available": int(score.top_is_available),
                        "n_tokens": int(score.n_tokens),
                        "seconds": round(score.seconds, 4),
                    }
                )
                scored += 1
                if scored % 10 == 0:
                    fh.flush()
                    elapsed = time.perf_counter() - t0
                    rate = elapsed / scored
                    left = pending - scored
                    print(
                        f"    {scored}/{pending}  {rate:.2f}s each  "
                        f"ETA {left * rate / 60:.1f} min  last={scenario}"
                    )
        fh.flush()
    if model is not None:
        closer = getattr(model.llm, "close", None)
        if closer:
            closer()
        del model
        gc.collect()
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
    pred = pd.read_csv(cache_file)
    pred = pred.drop_duplicates(["obs_id", "scenario"], keep="last")
    return pred


def shock_jobs(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame, str]]:
    jobs = []
    for mode in ("train", "sm"):
        for attribute in ("time", "cost"):
            for pct in PCT_SHOCKS:
                name = f"{mode}_{attribute}_{pct:+d}pct"
                jobs.append((name, apply_attribute_shock(frame, mode, attribute, "pct", pct), "base"))
            for level in (-10, 10):
                unit = "min" if attribute == "time" else "chf"
                name = f"{mode}_{attribute}_{level:+d}{unit}"
                jobs.append((name, apply_attribute_shock(frame, mode, attribute, "abs", level), "base"))
    return jobs


def shares_from_proba(p: np.ndarray) -> list[float]:
    return [float(p[:, k].mean()) for k in range(3)]


def take(frame: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Positional subset. Avoids pandas' boolean `.loc` alignment rules."""
    return frame.iloc[np.flatnonzero(np.asarray(mask))].copy()


def analyse_llm(
    pred: pd.DataFrame,
    spec: ModelSpec,
    llm_df: pd.DataFrame,
    elast_df: pd.DataFrame,
    ga_df: pd.DataFrame,
    train_df: pd.DataFrame,
    train_base,
    oos_models: dict,
    pilot_summary: dict,
    fingerprint: str,
    shock_df: pd.DataFrame | None = None,
) -> tuple[dict, dict]:
    """Metrics for one instruct model on the shared tasks. Returns (block, curves).

    `elast_df` is the shared 84-task probe (equalised attributes). `shock_df`
    is the frame whose time and cost shocks were actually scored. For 1.5B
    and 3B those are the same 84 tasks. For 7B, shocks use a subset.
    """
    if shock_df is None:
        shock_df = elast_df
    key = spec.key
    y = llm_df["y"].to_numpy(dtype=int)
    ids = llm_df["ID"].to_numpy()
    avail = availability_of(llm_df)
    reports = {}
    probas = {}
    preds = {}
    for name, result in oos_models.items():
        proba = predict_proba(llm_df, result)
        pred_y = hard_predict(proba, avail)
        probas[name] = proba
        preds[name] = pred_y
        reports[name] = classification_report(name, y, pred_y, proba, ids)
    preds["majority"] = majority_predict(train_df, llm_df)
    reports["majority"] = classification_report("majority", y, preds["majority"], None, ids)

    base_aligned = align_predictions(llm_df, pred, "base")
    probas[key] = proba_matrix(base_aligned)
    preds[key] = hard_from_aligned(base_aligned)
    reports[key] = classification_report(key, y, preds[key], probas[key], ids)
    draws = respondent_draws(ids, N_BOOT, SEED)
    correct = {
        name: (preds[name] == y).astype(float) for name in ("mnl", "mnl_rich", "majority", key)
    }
    boot_acc = {name: bootstrap_mean(correct[name], draws) for name in correct}
    boot_diff = {
        "mnl_minus_llm": bootstrap_accuracy_diff(correct["mnl"], correct[key], draws),
        "mnl_minus_majority": bootstrap_accuracy_diff(correct["mnl"], correct["majority"], draws),
        "llm_minus_majority": bootstrap_accuracy_diff(correct[key], correct["majority"], draws),
        "rich_minus_mnl": bootstrap_accuracy_diff(correct["mnl_rich"], correct["mnl"], draws),
    }
    paired = {
        "mnl_vs_llm": mcnemar(y, preds["mnl"], preds[key]),
        "llm_vs_majority": mcnemar(y, preds["majority"], preds[key]),
    }
    corr = {}
    for k in range(3):
        corr[MODE_KEYS[k]] = float(np.corrcoef(probas["mnl"][:, k], probas[key][:, k])[0, 1])

    paying = shock_df["GA"].to_numpy() == 0

    def mnl_p(result, frame: pd.DataFrame) -> np.ndarray:
        return predict_proba(frame, result)

    def llm_p(scenario: str, frame: pd.DataFrame) -> np.ndarray:
        return proba_matrix(align_predictions(frame, pred, scenario))

    curves: dict[tuple[str, str], list[dict]] = {}
    elast_rows = []
    mode_index = {"train": 0, "sm": 1}
    base_mnl_e = mnl_p(train_base, shock_df)
    base_llm_e = llm_p("base", shock_df)
    for mode in ("train", "sm"):
        for attribute in ("time", "cost"):
            use_paying = attribute == "cost"
            mask = paying if use_paying else np.ones(len(shock_df), dtype=bool)
            if int(mask.sum()) < 5:
                print(f"Skipping {mode} {attribute}: only {int(mask.sum())} tasks")
                continue
            frame_m = take(shock_df, mask)
            if len(frame_m) != int(mask.sum()):
                raise RuntimeError("Shock subsample is misaligned.")
            draws_m = respondent_draws(frame_m["ID"].to_numpy(), N_BOOT, SEED + 1)
            b_mnl = base_mnl_e[mask, mode_index[mode]]
            b_llm = base_llm_e[mask, mode_index[mode]]
            if len(b_mnl) != len(frame_m) or list(frame_m["obs_id"]) != list(
                take(shock_df, mask)["obs_id"]
            ):
                raise RuntimeError("Probability rows do not match the shock frame.")
            series = {("mnl", 0): b_mnl, (key, 0): b_llm}
            for pct in PCT_SHOCKS:
                scenario = f"{mode}_{attribute}_{pct:+d}pct"
                shocked = take(apply_attribute_shock(shock_df, mode, attribute, "pct", pct), mask)
                if list(shocked["obs_id"]) != list(frame_m["obs_id"]):
                    raise RuntimeError(f"Shock frame for {scenario} is not aligned.")
                p_mnl = mnl_p(train_base, shocked)[:, mode_index[mode]]
                p_llm = llm_p(scenario, frame_m)[:, mode_index[mode]]
                delta = pct / 100.0
                for model_name, base_s, shock_s in (
                    ("mnl", b_mnl, p_mnl),
                    (key, b_llm, p_llm),
                ):
                    stat = own_elasticity(base_s, shock_s, delta, draws_m)
                    elast_rows.append(
                        {
                            "model": model_name,
                            "mode": mode,
                            "attribute": attribute,
                            "sample": "paying" if use_paying else "all",
                            "pct": pct,
                            "n": int(mask.sum()),
                            "n_people": int(pd.unique(shock_df.loc[mask, "ID"]).size),
                            **stat,
                        }
                    )
                    series[(model_name, pct)] = shock_s
            curve_key = (mode, attribute)
            curves[curve_key] = []
            for model_name in ("mnl", key):
                for pct in (0, *PCT_SHOCKS):
                    curves[curve_key].append(
                        {
                            "model": model_name,
                            "pct": pct,
                            "share": float(series[(model_name, pct)].mean()),
                        }
                    )

    vot_rows = []
    for mode, time_col, cost_col in (
        ("train", "TRAIN_TT", "train_cost"),
        ("sm", "SM_TT", "sm_cost"),
    ):
        mask = (shock_df[time_col].to_numpy() >= 20) & (shock_df[cost_col].to_numpy() >= 10)
        frame_m = take(shock_df, mask)
        if mask.sum() < 20:
            vot_rows.append(
                {"mode": mode, "model": key, "n": int(mask.sum()), "note": "too few unclipped tasks"}
            )
            continue
        draws_m = respondent_draws(frame_m["ID"].to_numpy(), N_BOOT, SEED + 5)
        packs = {}
        for attribute, suffix in (("time", "min"), ("cost", "chf")):
            for level in (-10, 10):
                scenario = f"{mode}_{attribute}_{level:+d}{suffix}"
                shocked = take(
                    apply_attribute_shock(shock_df, mode, attribute, "abs", float(level)),
                    mask,
                )
                packs[(attribute, level, "mnl")] = mnl_p(train_base, shocked)[:, mode_index[mode]]
                packs[(attribute, level, key)] = llm_p(scenario, frame_m)[:, mode_index[mode]]
        for model_name in ("mnl", key):
            got = bootstrap_vot(
                packs[("time", -10, model_name)],
                packs[("time", 10, model_name)],
                packs[("cost", -10, model_name)],
                packs[("cost", 10, model_name)],
                draws_m,
            )
            got["mode"] = mode
            got["model"] = model_name
            got["n_people"] = int(frame_m["ID"].nunique())
            vot_rows.append(got)

    ga_fare = []
    ga_elast = shock_df.loc[shock_df["GA"] == 1]
    if len(ga_elast) >= 8:
        for mode in ("train", "sm"):
            scenario = f"{mode}_cost_+10chf"
            base_g = llm_p("base", ga_elast)[:, mode_index[mode]]
            shock_g = llm_p(scenario, ga_elast)[:, mode_index[mode]]
            base_m = mnl_p(train_base, ga_elast)[:, mode_index[mode]]
            shock_m = mnl_p(
                train_base, apply_attribute_shock(ga_elast, mode, "cost", "abs", 10)
            )[:, mode_index[mode]]
            ga_fare.append(
                {
                    "mode": mode,
                    "n": int(len(ga_elast)),
                    "llm_share_base": float(base_g.mean()),
                    "llm_share_plus10": float(shock_g.mean()),
                    "mnl_share_base": float(base_m.mean()),
                    "mnl_share_plus10": float(shock_m.mean()),
                }
            )

    ga_wording = None
    if len(ga_df) and (pred["scenario"] == "ga_silent").any():
        base_g = llm_p("base", ga_df)
        silent = llm_p("ga_silent", ga_df)
        ga_wording = {
            "n": int(len(ga_df)),
            "n_people": int(ga_df["ID"].nunique()),
            "base_prob_share": shares_from_proba(base_g),
            "silent_prob_share": shares_from_proba(silent),
            "mean_abs_prob_change": float(np.abs(base_g - silent).mean()),
        }

    equal_frame = equalise_attributes(elast_df)
    equal_mnl = mnl_p(train_base, equal_frame)
    equal_llm = llm_p("equalised", elast_df)
    equalised = {
        "n": int(len(elast_df)),
        "mnl_prob_share": shares_from_proba(equal_mnl),
        "llm_prob_share": shares_from_proba(equal_llm),
        "llm_hard_share": [
            float((hard_predict(equal_llm, availability_of(elast_df)) == k).mean()) for k in range(3)
        ],
    }

    def group_shares(frame: pd.DataFrame) -> dict:
        if len(frame) == 0:
            return {
                "n": 0,
                "n_people": 0,
                "observed": [None, None, None],
                "mnl": [None, None, None],
                "llm": [None, None, None],
            }
        y_g = frame["y"].to_numpy(dtype=int)
        observed = [float((y_g == k).mean()) for k in range(3)]
        return {
            "n": int(len(frame)),
            "n_people": int(frame["ID"].nunique()),
            "observed": observed,
            "mnl": shares_from_proba(predict_proba(frame, train_base)),
            "llm": shares_from_proba(llm_p("base", frame)),
        }

    ga_split = {
        "ga": group_shares(llm_df.loc[llm_df["GA"] == 1]),
        "no_ga": group_shares(llm_df.loc[llm_df["GA"] == 0]),
    }
    fmt = {
        "mean_digit_mass": float(base_aligned["digit_mass"].mean()),
        "top_token_is_digit": float(base_aligned["top_is_digit"].mean()),
        "top_token_available": float(base_aligned["top_is_available"].mean()),
        "mean_seconds": float(base_aligned["seconds"].mean()),
        "mean_tokens": float(base_aligned["n_tokens"].mean()),
    }
    agreement = float((preds["mnl"] == preds[key]).mean())
    block = {
        "key": key,
        "label": spec.label,
        "model_id": spec.model_id,
        "quantization": spec.quant,
        "fingerprint": fingerprint,
        "format": fmt,
        "pilot": pilot_summary,
        "reports": reports,
        "bootstrap_accuracy": boot_acc,
        "bootstrap_accuracy_diff": boot_diff,
        "mcnemar": paired,
        "probability_correlation": corr,
        "hard_agreement_mnl_llm": agreement,
        "elasticities": elast_rows,
        "implied_vot": vot_rows,
        "ga_split": ga_split,
        "ga_plus_10chf": ga_fare,
        "ga_wording": ga_wording,
        "equalised": equalised,
        "n_paying": int((elast_df["GA"] == 0).sum()),
        "shock_sample": {
            "n": int(len(shock_df)),
            "n_people": int(shock_df["ID"].nunique()),
            "n_paying": int((shock_df["GA"] == 0).sum()),
            "parent_n": int(len(elast_df)),
            "seed": SEVEN_B_SHOCK_SEED if len(shock_df) != len(elast_df) else SEED + 11,
            "max_per_person": 1 if len(shock_df) != len(elast_df) else ELAST_MAX_TASKS_PER_PERSON,
        },
    }
    return block, curves


def publish_predictions(jobs, pred: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    needed = [
        (int(obs_id), scenario)
        for scenario, frame, _wording in jobs
        for obs_id in frame["obs_id"].tolist()
    ]
    need = pd.DataFrame(needed, columns=["obs_id", "scenario"]).drop_duplicates()
    out = need.merge(pred, on=["obs_id", "scenario"], how="left")
    if out["p_train"].isna().any():
        missing = out.loc[out["p_train"].isna(), ["obs_id", "scenario"]].head(5)
        raise RuntimeError(f"{spec.model_id} cache is missing rows:\n{missing}")
    path = RESULTS / f"llm_predictions_{spec.key}.csv"
    out.to_csv(path, index=False)
    if spec.key == "1.5b":
        out.to_csv(RESULTS / "llm_predictions.csv", index=False)
    return out


def pilot_block(pred: pd.DataFrame, spec: ModelSpec) -> dict:
    pilot_pred = pred.loc[pred["scenario"] == "pilot"]
    format_rate = float(pilot_pred["top_is_digit"].mean()) if len(pilot_pred) else 0.0
    summary = {
        "model": spec.model_id,
        "n": int(len(pilot_pred)),
        "format_rate": format_rate,
        "mean_digit_mass": float(pilot_pred["digit_mass"].mean()) if len(pilot_pred) else None,
        "mean_seconds": float(pilot_pred["seconds"].mean()) if len(pilot_pred) else None,
        "source": "training respondents only",
    }
    write_json(RESULTS / f"pilot_{spec.key}.json", summary)
    print(f"{spec.label} train-set pilot format rate: {format_rate:.3f}")
    if format_rate < 0.80:
        raise RuntimeError(
            f"{spec.model_id} did not follow the digit format on the training pilot "
            f"({format_rate:.3f}). The test set was not scored."
        )
    return summary


def render_model_figures(blocks: dict[str, dict], curves_by_key: dict, n: int) -> None:
    first = next(iter(blocks.values()))
    letters = "ABCDEFGH"
    panels = [("A. Base MNL", np.array(first["reports"]["mnl"]["confusion"]))]
    for i, block in enumerate(blocks.values(), start=1):
        panels.append(
            (f"{letters[i]}. {block['label']}", np.array(block["reports"][block["key"]]["confusion"]))
        )
    plot_confusion(panels, n=n)

    share_rows = [
        {"name": "Observed", "shares": first["reports"]["mnl"]["observed_share"], "color": "#222222"},
        {"name": "Majority", "shares": first["reports"]["majority"]["hard_share"], "color": GRAY},
        {"name": "MNL", "shares": first["reports"]["mnl"]["prob_share"], "color": BLUE},
        {"name": "Richer MNL", "shares": first["reports"]["mnl_rich"]["prob_share"], "color": GREEN},
    ]
    for key, block in blocks.items():
        share_rows.append(
            {
                "name": block["label"],
                "shares": block["reports"][key]["prob_share"],
                "color": SIZE_COLORS[key],
            }
        )
    plot_shares(share_rows, n=n)

    merged: dict[tuple[str, str], list] = {}
    styles = [("mnl", BLUE, "o", "MNL")]
    markers = {"1.5b": "s", "3b": "D", "7b": "^"}
    for key, _block in blocks.items():
        styles.append((key, SIZE_COLORS[key], markers[key], blocks[key]["label"]))
        for curve_key, pts in curves_by_key[key].items():
            bucket = merged.setdefault(curve_key, [])
            have_mnl = any(pt["model"] == "mnl" for pt in bucket)
            for pt in pts:
                if pt["model"] == "mnl" and have_mnl:
                    continue
                bucket.append(pt)
    plot_elasticities(merged, styles)

    gap_series = []
    # Observed and MNL gaps are the same for every model; take them from the first.
    ga = first["ga_split"]
    for name, field, color in (
        ("Observed", "observed", "#222222"),
        ("MNL", "mnl", BLUE),
    ):
        gap_series.append(
            {
                "name": name,
                "color": color,
                "gap": [ga["ga"][field][k] - ga["no_ga"][field][k] for k in range(3)],
            }
        )
    equal_series = [
        {"name": "MNL", "color": BLUE, "shares": first["equalised"]["mnl_prob_share"]}
    ]
    for key, block in blocks.items():
        g = block["ga_split"]
        gap_series.append(
            {
                "name": block["label"],
                "color": SIZE_COLORS[key],
                "gap": [g["ga"]["llm"][k] - g["no_ga"]["llm"][k] for k in range(3)],
            }
        )
        equal_series.append(
            {
                "name": block["label"],
                "color": SIZE_COLORS[key],
                "shares": block["equalised"]["llm_prob_share"],
            }
        )
    plot_bias(
        gap_series,
        equal_series,
        n_ga=ga["ga"]["n"],
        n_noga=ga["no_ga"]["n"],
        n_eq=first["equalised"]["n"],
    )

    acc_rows = []
    for name, report_name, color in (
        ("Majority", "majority", GRAY),
        ("MNL", "mnl", BLUE),
        ("Richer MNL", "mnl_rich", GREEN),
    ):
        boot = first["bootstrap_accuracy"][report_name]
        acc_rows.append(
            {
                "name": name,
                "accuracy": boot["estimate"],
                "lo": boot["ci95"][0],
                "hi": boot["ci95"][1],
                "color": color,
            }
        )
    prob_rows = [
        {"name": "MNL", "shares": first["reports"]["mnl"]["prob_share"], "color": BLUE},
    ]
    for key, block in blocks.items():
        boot = block["bootstrap_accuracy"][key]
        acc_rows.append(
            {
                "name": block["label"],
                "accuracy": boot["estimate"],
                "lo": boot["ci95"][0],
                "hi": boot["ci95"][1],
                "color": SIZE_COLORS[key],
            }
        )
        prob_rows.append(
            {
                "name": block["label"],
                "shares": block["reports"][key]["prob_share"],
                "color": SIZE_COLORS[key],
            }
        )
    plot_model_size(acc_rows, prob_rows, n=n)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Swissmetro MNL vs local LLMs")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--llm-n", type=int, default=LLM_N)
    parser.add_argument("--elast-n", type=int, default=ELAST_N)
    parser.add_argument(
        "--models",
        default="1.5b,3b,7b",
        help="Comma-separated keys: 1.5b, 3b, 7b. Same tasks and prompt for each.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    raw = load_raw(ensure_data(DATA_PATH))
    df = estimation_sample(raw)
    description = sample_description(raw, df)
    write_json(RESULTS / "sample_description.json", description)
    print(
        f"Sample: {description['n_obs']} tasks, {description['n_people']} people, "
        f"excluded {description['n_excluded']}"
    )

    print("Estimating full-sample models...")
    full_base = fit_mnl(df, SPEC_BASE)
    full_rich = fit_mnl(df, SPEC_RICH)
    if not full_base.converged or abs(full_base.loglik - BIOGEME_B01A["loglik"]) > 0.01:
        raise RuntimeError(
            f"Base MNL did not reproduce Biogeme: ll={full_base.loglik}, converged={full_base.converged}"
        )
    by_name = dict(zip(full_base.names, full_base.coef))
    for name, published in BIOGEME_B01A["coef"].items():
        if abs(by_name[name] - published) > 1e-3:
            raise RuntimeError(f"{name}={by_name[name]} differs from published {published}")
    print(f"Base MNL matches Biogeme. ll={full_base.loglik:.3f}")

    is_train, is_test = split_by_respondent(df, TEST_FRACTION, SEED)
    train_df = df.loc[is_train].copy()
    test_df = df.loc[is_test].copy()
    print(
        f"Split: train {train_df['ID'].nunique()} people / {len(train_df)} tasks; "
        f"test {test_df['ID'].nunique()} people / {len(test_df)} tasks"
    )
    train_base = fit_mnl(train_df, SPEC_BASE)
    train_rich = fit_mnl(train_df, SPEC_RICH)
    train_const = fit_mnl(train_df, SPEC_CONST)
    for result, label in (
        (full_base, "full base"),
        (full_rich, "full rich"),
        (train_base, "train base"),
        (train_rich, "train rich"),
        (train_const, "train constants"),
    ):
        if not result.converged:
            raise RuntimeError(f"{label} did not converge (grad {result.grad_norm})")

    coef_rows = []
    for result, sample in (
        (full_base, "full"),
        (full_rich, "full"),
        (train_base, "train"),
        (train_rich, "train"),
        (train_const, "train"),
    ):
        coef_rows.extend(coef_table(result, sample))
    pd.DataFrame(coef_rows).to_csv(RESULTS / "mnl_coefficients.csv", index=False)

    vot_full = value_of_time(full_base, cov="cluster")
    vot_full_robust = value_of_time(full_base, cov="robust")
    vot_rich = value_of_time(full_rich, cov="cluster")
    vot_rich_senior = value_of_time(full_rich, ("B_TIME", "B_TIME_SENIOR"), ("B_COST",), cov="cluster")
    vot_rich_high = value_of_time(full_rich, ("B_TIME",), ("B_COST", "B_COST_HIGHINC"), cov="cluster")
    head_vot = value_of_time(full_rich, ("B_HEADWAY",), ("B_COST",), cov="cluster")
    head_over_time = float(
        full_rich.coef[full_rich.names.index("B_HEADWAY")]
        / full_rich.coef[full_rich.names.index("B_TIME")]
    )
    lr_full = likelihood_ratio(full_base, full_rich)
    lr_train = likelihood_ratio(train_base, train_rich)

    # Timing-probe rows were scored while choosing the sample size. They are
    # removed from the LLM evaluation so those eight tasks are not also the
    # reported test. The prompt was not edited after the probe.
    probe = stratified_task_sample(df, PROBE_N, SEED, max_per_id=1)
    probe_ids = set(probe["obs_id"].astype(int).tolist())
    llm_frame = test_df.loc[~test_df["obs_id"].isin(probe_ids)].copy()
    llm_df = stratified_task_sample(llm_frame, args.llm_n, SEED + 7, MAX_TASKS_PER_PERSON)
    elast_df = stratified_task_sample(llm_df, args.elast_n, SEED + 11, ELAST_MAX_TASKS_PER_PERSON)
    seven_shock = stratified_task_sample(
        elast_df, min(SEVEN_B_SHOCK_N, len(elast_df)), SEVEN_B_SHOCK_SEED, max_per_id=1
    )
    pilot_df = stratified_task_sample(train_df, PILOT_N, SEED + 3, max_per_id=1)
    ga_df = llm_df.loc[llm_df["GA"] == 1].copy()
    print(
        f"LLM sample {len(llm_df)} tasks / {llm_df['ID'].nunique()} people; "
        f"elasticity {len(elast_df)} / {elast_df['ID'].nunique()}; "
        f"7B shocks {len(seven_shock)} / {seven_shock['ID'].nunique()}; "
        f"GA holders in LLM sample {len(ga_df)}; pilot {len(pilot_df)}"
    )
    sample_index = pd.concat(
        [
            probe.assign(role="timing_probe"),
            pilot_df.assign(role="pilot_train"),
            llm_df.assign(role="llm_eval"),
            elast_df.assign(role="elasticity"),
            seven_shock.assign(role="shock_7b"),
        ],
        ignore_index=True,
    )
    sample_index[["role", "obs_id", "ID", "CHOICE", "GA", "PURPOSE"]].to_csv(
        RESULTS / "sample_index.csv", index=False
    )
    example = llm_df.iloc[0]
    (RESULTS / "example_prompt.txt").write_text(render_prompt(example, "base"), encoding="utf-8")

    def pack_model(result, vot=None):
        d = result.as_dict()
        if vot is not None:
            d["vot"] = vot
        return d

    mnl_bundle = {
        "full_base": pack_model(full_base, vot_full),
        "full_rich": pack_model(full_rich, vot_rich),
        "train_base": pack_model(train_base, value_of_time(train_base)),
        "train_rich": pack_model(train_rich, value_of_time(train_rich)),
        "train_constants": pack_model(train_const),
        "vot_full_cluster": vot_full,
        "vot_full_robust_se": vot_full_robust,
        "vot_rich_base_person": vot_rich,
        "vot_rich_senior": vot_rich_senior,
        "vot_rich_high_income": vot_rich_high,
        "headway_chf_per_hour": head_vot,
        "headway_over_invehicle_time": head_over_time,
        "lr_full": lr_full,
        "lr_train": lr_train,
        "swissmetro_advantage_over_train_chf": money_metric(full_base, "ASC_TRAIN"),
        "swissmetro_advantage_over_car_chf": money_metric(full_base, "ASC_CAR"),
        "ga_train_chf": money_metric(full_rich, "B_GA_TRAIN"),
        "benchmark": {
            "loglik_diff": full_base.loglik - BIOGEME_B01A["loglik"],
            "coef_diff": {k: by_name[k] - v for k, v in BIOGEME_B01A["coef"].items()},
            "published_loglik": BIOGEME_B01A["loglik"],
        },
    }
    write_json(RESULTS / "mnl_summary.json", mnl_bundle)
    plot_coefficients(full_base, full_rich)

    # Out-of-sample MNL on the full held-out people, before spending LLM time.
    oos_models = {
        "mnl": train_base,
        "mnl_rich": train_rich,
        "constants": train_const,
    }
    full_test_reports = {}
    p_test = {}
    y_test = test_df["y"].to_numpy(dtype=int)
    ids_test = test_df["ID"].to_numpy()
    avail_test = availability_of(test_df)
    for name, result in oos_models.items():
        proba = predict_proba(test_df, result)
        pred = hard_predict(proba, avail_test)
        p_test[name] = proba
        full_test_reports[name] = classification_report(name, y_test, pred, proba, ids_test)
    maj_test = majority_predict(train_df, test_df)
    full_test_reports["majority"] = classification_report(
        "majority", y_test, maj_test, None, ids_test
    )
    write_json(RESULTS / "oos_full_test.json", full_test_reports)
    print(
        "Full test accuracy: "
        + ", ".join(f"{k}={v['accuracy']:.3f}" for k, v in full_test_reports.items())
    )

    if args.skip_llm:
        print("Skipping LLM (--skip-llm).")
        return

    specs = parse_model_keys(args.models)
    fingerprint = prompt_fingerprint()
    pilot_jobs: list[tuple[str, pd.DataFrame, str]] = [("pilot", pilot_df, "base")]

    def jobs_for(spec: ModelSpec) -> list[tuple[str, pd.DataFrame, str]]:
        # 7B keeps the 420 base tasks, the 84-task equalised probe, and the
        # pass-wording check. Only the time/cost grid is reduced.
        shock_df = seven_shock if spec.key == "7b" else elast_df
        rest: list[tuple[str, pd.DataFrame, str]] = [("base", llm_df, "base")]
        rest.extend(shock_jobs(shock_df))
        rest.append(("equalised", equalise_attributes(elast_df), "equalised"))
        if len(ga_df):
            rest.append(("ga_silent", ga_df, "ga_silent"))
        return pilot_jobs + rest

    blocks: dict[str, dict] = {}
    curves_by_key: dict[str, dict] = {}
    for spec in specs:
        print(f"=== {spec.model_id} ({spec.quant}) ===")
        cache_file = cache_path(spec)
        t_model = time.perf_counter()
        model_jobs = jobs_for(spec)
        # Pilot first, so a model that ignores the digit format does not
        # spend the shock grid before we notice.
        score_scenarios(pilot_jobs, cache_file, spec)
        pilot_pred = publish_predictions(pilot_jobs, pd.read_csv(cache_file), spec)
        pilot_summary = pilot_block(pilot_pred, spec)
        if args.pilot_only:
            continue
        raw_pred = score_scenarios(model_jobs, cache_file, spec)
        pred = publish_predictions(model_jobs, raw_pred, spec)
        shock_df = seven_shock if spec.key == "7b" else elast_df
        block, curves = analyse_llm(
            pred,
            spec,
            llm_df,
            elast_df,
            ga_df,
            train_df,
            train_base,
            oos_models,
            pilot_summary,
            fingerprint,
            shock_df=shock_df,
        )
        block["seconds_model"] = time.perf_counter() - t_model
        blocks[spec.key] = block
        curves_by_key[spec.key] = curves
        write_json(RESULTS / f"llm_{spec.key}.json", block)
        acc = block["reports"][spec.key]["accuracy"]
        print(f"{spec.label} accuracy on {len(llm_df)} tasks: {acc:.3f}")

    if args.pilot_only:
        print("Pilot only; stopping.")
        return
    if not blocks:
        raise RuntimeError("No model was scored.")

    elast_rows = []
    vot_rows = []
    mnl_elast_kept = False
    mnl_vot_kept = False
    for key, block in blocks.items():
        for row in block["elasticities"]:
            tagged = dict(row)
            tagged["llm_key"] = key
            if tagged["model"] == "mnl":
                if mnl_elast_kept:
                    continue
                tagged["llm_key"] = "mnl"
            elast_rows.append(tagged)
        if any(row["model"] == "mnl" for row in block["elasticities"]):
            mnl_elast_kept = True
        for row in block["implied_vot"]:
            tagged = dict(row)
            tagged["llm_key"] = key
            if tagged.get("model") == "mnl":
                if mnl_vot_kept:
                    continue
                tagged["llm_key"] = "mnl"
            vot_rows.append(tagged)
        if any(row.get("model") == "mnl" for row in block["implied_vot"]):
            mnl_vot_kept = True
    pd.DataFrame(elast_rows).to_csv(RESULTS / "elasticities.csv", index=False)
    pd.DataFrame(vot_rows).to_csv(RESULTS / "implied_vot.csv", index=False)

    comparison = []
    for key, block in blocks.items():
        report = block["reports"][key]
        diff = block["bootstrap_accuracy_diff"]["mnl_minus_llm"]
        vot = {row["mode"]: row for row in block["implied_vot"] if row.get("model") == key}
        comparison.append(
            {
                "key": key,
                "label": block["label"],
                "model_id": block["model_id"],
                "quantization": block["quantization"],
                "accuracy": report["accuracy"],
                "accuracy_ci95": block["bootstrap_accuracy"][key]["ci95"],
                "loglik": report["loglik"],
                "mean_loglik": report["mean_loglik"],
                "brier": report["brier"],
                "prob_share": report["prob_share"],
                "hard_share": report["hard_share"],
                "recall": report["recall"],
                "mean_p_chosen": report["mean_p_chosen"],
                "mnl_minus_llm": diff["estimate"],
                "mnl_minus_llm_ci95": diff["ci95"],
                "equalised_prob_share": block["equalised"]["llm_prob_share"],
                "equalised_hard_share": block["equalised"]["llm_hard_share"],
                "ga_train_probability": block["ga_split"]["ga"]["llm"][0],
                "noga_train_probability": block["ga_split"]["no_ga"]["llm"][0],
                "implied_vot": vot,
                "shock_n": block["shock_sample"]["n"],
                "format": block["format"],
            }
        )
    pd.DataFrame(
        [
            {
                "key": row["key"],
                "model_id": row["model_id"],
                "accuracy": row["accuracy"],
                "loglik": row["loglik"],
                "mean_loglik": row["mean_loglik"],
                "p_train": row["prob_share"][0],
                "p_sm": row["prob_share"][1],
                "p_car": row["prob_share"][2],
                "hard_train": row["hard_share"][0],
                "hard_sm": row["hard_share"][1],
                "hard_car": row["hard_share"][2],
                "mnl_minus_llm": row["mnl_minus_llm"],
                "shock_n": row["shock_n"],
                "ga_p_train": row["ga_train_probability"],
                "equalised_p_car": row["equalised_prob_share"][2],
                "equalised_p_sm": row["equalised_prob_share"][1],
            }
            for row in comparison
        ]
    ).to_csv(RESULTS / "model_size.csv", index=False)

    # Same 36 tasks for every model's arc value of time, so the size
    # comparison is not a mix of the 84-task grid and the 7B subset.
    common_vot = []
    mnl_vot_kept = False
    for spec in specs:
        if spec.key == "7b":
            src_vot = blocks["7b"]["implied_vot"]
        else:
            cached = pd.read_csv(cache_path(spec)).drop_duplicates(
                ["obs_id", "scenario"], keep="last"
            )
            src, _ = analyse_llm(
                cached,
                spec,
                llm_df,
                elast_df,
                ga_df,
                train_df,
                train_base,
                oos_models,
                blocks[spec.key]["pilot"],
                fingerprint,
                shock_df=seven_shock,
            )
            src_vot = src["implied_vot"]
        for row in src_vot:
            tagged = dict(row)
            if tagged.get("model") == "mnl":
                if mnl_vot_kept:
                    continue
                mnl_vot_kept = True
                tagged["llm_key"] = "mnl"
            else:
                tagged["llm_key"] = spec.key
            tagged["shock_n"] = int(len(seven_shock))
            tagged["shock_seed"] = SEVEN_B_SHOCK_SEED
            common_vot.append(tagged)
    pd.DataFrame(common_vot).to_csv(RESULTS / "implied_vot_common.csv", index=False)

    first = next(iter(blocks.values()))
    summary = {
        "description": description,
        "split": {
            "seed": SEED,
            "test_fraction": TEST_FRACTION,
            "n_train_people": int(train_df["ID"].nunique()),
            "n_train_tasks": int(len(train_df)),
            "n_test_people": int(test_df["ID"].nunique()),
            "n_test_tasks": int(len(test_df)),
            "probe_n": int(len(probe)),
            "probe_in_test": int(probe["obs_id"].isin(set(test_df["obs_id"])).sum()),
        },
        "mnl": mnl_bundle,
        "oos_full_test": full_test_reports,
        "llm_sample": {
            "n": int(len(llm_df)),
            "n_people": int(llm_df["ID"].nunique()),
            "n_elast": int(len(elast_df)),
            "n_elast_people": int(elast_df["ID"].nunique()),
            "n_elast_paying": int(first["n_paying"]),
            "choice_counts": {
                MODE_NAMES[k]: int((llm_df["CHOICE"] == k + 1).sum()) for k in range(3)
            },
            "fingerprint": fingerprint,
            "models": [spec.model_id for spec in specs],
            "seven_b_shock": {
                "n": int(len(seven_shock)),
                "n_people": int(seven_shock["ID"].nunique()),
                "seed": SEVEN_B_SHOCK_SEED,
                "max_per_person": 1,
                "parent_n": int(len(elast_df)),
                "parent_seed": SEED + 11,
                "reason": (
                    "Qwen2.5-7B-Instruct Q4_K_M takes about 4.4 seconds per prompt "
                    "on this machine. The full shock grid on 84 tasks would add "
                    "about three hours after the 420 base tasks, so time and cost "
                    "shocks for 7B use this subset of the 84. Accuracy, market "
                    "shares, the pass split, and the equalised-attribute probe "
                    "still use the same tasks as the 1.5B and 3B models."
                ),
            },
        },
        "llms": blocks,
        "comparison": comparison,
        "shock_common_vot": common_vot,
        "seconds_total": time.perf_counter() - started,
    }
    write_json(RESULTS / "summary.json", summary)
    render_model_figures(blocks, curves_by_key, n=len(llm_df))
    print("Wrote results/summary.json and figures/.")
    print(
        "Accuracy: "
        + ", ".join(f"{block['label']}={block['reports'][key]['accuracy']:.3f}" for key, block in blocks.items())
    )
    print(f"Total minutes: {(time.perf_counter() - started) / 60:.1f}")


if __name__ == "__main__":
    main()
