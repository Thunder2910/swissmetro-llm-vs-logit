"""Run the Swissmetro comparison and write results/ and figures/.

MNL estimation is exact (Newton on a concave likelihood). LLM scores are
cached by a hash of the prompt and the scoring code, so a rerun continues
rather than starting over.
"""

from __future__ import annotations

import argparse
import csv
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
from src.plots import plot_bias, plot_coefficients, plot_confusion, plot_elasticities, plot_shares
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


def score_scenarios(jobs: list[tuple[str, pd.DataFrame, str]], cache_path: Path) -> pd.DataFrame:
    """jobs: (scenario name, frame, wording key)."""
    from src.llm_choice import ChoiceModel, ensure_model

    ensure_model()
    done = load_done(cache_path)
    pending = 0
    for scenario, frame, _wording in jobs:
        for obs_id in frame["obs_id"].astype(int):
            if (obs_id, scenario) not in done:
                pending += 1
    print(f"LLM tasks to score: {pending} (cached keys: {len(done)})")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not cache_path.exists() or cache_path.stat().st_size == 0
    model = None
    t0 = time.perf_counter()
    scored = 0
    with cache_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for scenario, frame, wording in jobs:
            todo = frame.loc[~frame["obs_id"].astype(int).map(lambda i: (i, scenario) in done)]
            if todo.empty:
                print(f"  {scenario}: cached ({len(frame)})")
                continue
            if model is None:
                print("Loading model...")
                model = ChoiceModel()
                groups = {k: v for k, v in model.groups.items()}
                write_json(RESULTS / "digit_token_ids.json", groups)
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
    pred = pd.read_csv(cache_path)
    pred = pred.drop_duplicates(["obs_id", "scenario"], keep="last")
    final = RESULTS / "llm_predictions.csv"
    pred.to_csv(final, index=False)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Swissmetro MNL vs local LLM")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--llm-n", type=int, default=LLM_N)
    parser.add_argument("--elast-n", type=int, default=ELAST_N)
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
    pilot_df = stratified_task_sample(train_df, PILOT_N, SEED + 3, max_per_id=1)
    ga_df = llm_df.loc[llm_df["GA"] == 1].copy()
    print(
        f"LLM sample {len(llm_df)} tasks / {llm_df['ID'].nunique()} people; "
        f"elasticity {len(elast_df)} / {elast_df['ID'].nunique()}; "
        f"GA holders in LLM sample {len(ga_df)}; pilot {len(pilot_df)}"
    )
    sample_index = pd.concat(
        [
            probe.assign(role="timing_probe"),
            pilot_df.assign(role="pilot_train"),
            llm_df.assign(role="llm_eval"),
            elast_df.assign(role="elasticity"),
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

    fingerprint = prompt_fingerprint()
    cache_path = RESULTS / "cache" / f"predictions_{fingerprint}.csv"
    jobs: list[tuple[str, pd.DataFrame, str]] = [("pilot", pilot_df, "base")]
    if not args.pilot_only:
        jobs.append(("base", llm_df, "base"))
        jobs.extend(shock_jobs(elast_df))
        jobs.append(("equalised", equalise_attributes(elast_df), "equalised"))
        if len(ga_df):
            jobs.append(("ga_silent", ga_df, "ga_silent"))
    pred = score_scenarios(jobs, cache_path)
    # The cache may contain timing-probe rows from earlier runs. Publish only
    # the scenarios this run defined.
    needed = [
        (int(obs_id), scenario)
        for scenario, frame, _wording in jobs
        for obs_id in frame["obs_id"].tolist()
    ]
    need = pd.DataFrame(needed, columns=["obs_id", "scenario"]).drop_duplicates()
    pred = need.merge(pred, on=["obs_id", "scenario"], how="left")
    if pred["p_train"].isna().any():
        missing = pred.loc[pred["p_train"].isna(), ["obs_id", "scenario"]].head(5)
        raise RuntimeError(f"LLM cache is missing rows:\n{missing}")
    pred.to_csv(RESULTS / "llm_predictions.csv", index=False)

    pilot_pred = pred.loc[pred["scenario"] == "pilot"]
    format_rate = float(pilot_pred["top_is_digit"].mean()) if len(pilot_pred) else 0.0
    pilot_summary = {
        "n": int(len(pilot_pred)),
        "format_rate": format_rate,
        "mean_digit_mass": float(pilot_pred["digit_mass"].mean()) if len(pilot_pred) else None,
        "mean_seconds": float(pilot_pred["seconds"].mean()) if len(pilot_pred) else None,
        "source": "training respondents only",
    }
    write_json(RESULTS / "pilot.json", pilot_summary)
    print(f"Train-set pilot format rate: {format_rate:.3f}")
    if format_rate < 0.80:
        raise RuntimeError(
            "The model did not follow the digit format on the training pilot. "
            "The test set was not scored. Inspect prompts/ before changing anything."
        )
    if args.pilot_only:
        print("Pilot only; stopping.")
        return

    # Comparison on the LLM subsample.
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
    probas["llm"] = proba_matrix(base_aligned)
    preds["llm"] = hard_from_aligned(base_aligned)
    reports["llm"] = classification_report("llm", y, preds["llm"], probas["llm"], ids)
    draws = respondent_draws(ids, N_BOOT, SEED)
    correct = {name: (preds[name] == y).astype(float) for name in ("mnl", "mnl_rich", "majority", "llm")}
    boot_acc = {name: bootstrap_mean(correct[name], draws) for name in correct}
    boot_diff = {
        "mnl_minus_llm": bootstrap_accuracy_diff(correct["mnl"], correct["llm"], draws),
        "mnl_minus_majority": bootstrap_accuracy_diff(correct["mnl"], correct["majority"], draws),
        "llm_minus_majority": bootstrap_accuracy_diff(correct["llm"], correct["majority"], draws),
        "rich_minus_mnl": bootstrap_accuracy_diff(correct["mnl_rich"], correct["mnl"], draws),
    }
    paired = {
        "mnl_vs_llm": mcnemar(y, preds["mnl"], preds["llm"]),
        "llm_vs_majority": mcnemar(y, preds["majority"], preds["llm"]),
    }
    # McNemar above is (a=mnl, b=llm): b_only means LLM correct and MNL wrong.
    corr = {}
    for k in range(3):
        corr[MODE_KEYS[k]] = float(np.corrcoef(probas["mnl"][:, k], probas["llm"][:, k])[0, 1])

    # Elasticities on the paired subsample.
    paying = elast_df["GA"].to_numpy() == 0

    def mnl_p(result, frame: pd.DataFrame) -> np.ndarray:
        return predict_proba(frame, result)

    def llm_p(scenario: str, frame: pd.DataFrame) -> np.ndarray:
        return proba_matrix(align_predictions(frame, pred, scenario))

    curves: dict[tuple[str, str], list[dict]] = {}
    elast_rows = []
    mode_index = {"train": 0, "sm": 1}

    base_mnl_e = mnl_p(train_base, elast_df)
    base_llm_e = llm_p("base", elast_df)
    for mode in ("train", "sm"):
        for attribute in ("time", "cost"):
            use_paying = attribute == "cost"
            mask = paying if use_paying else np.ones(len(elast_df), dtype=bool)
            if int(mask.sum()) < 5:
                print(f"Skipping {mode} {attribute}: only {int(mask.sum())} tasks")
                continue
            frame_m = take(elast_df, mask)
            if len(frame_m) != int(mask.sum()):
                raise RuntimeError("Shock subsample is misaligned.")
            draws_m = respondent_draws(frame_m["ID"].to_numpy(), N_BOOT, SEED + 1)
            b_mnl = base_mnl_e[mask, mode_index[mode]]
            b_llm = base_llm_e[mask, mode_index[mode]]
            if len(b_mnl) != len(frame_m) or list(frame_m["obs_id"]) != list(
                take(elast_df, mask)["obs_id"]
            ):
                raise RuntimeError("Probability rows do not match the shock frame.")
            series = {("mnl", 0): b_mnl, ("llm", 0): b_llm}
            for pct in PCT_SHOCKS:
                scenario = f"{mode}_{attribute}_{pct:+d}pct"
                shocked = take(apply_attribute_shock(elast_df, mode, attribute, "pct", pct), mask)
                if list(shocked["obs_id"]) != list(frame_m["obs_id"]):
                    raise RuntimeError(f"Shock frame for {scenario} is not aligned.")
                p_mnl = mnl_p(train_base, shocked)[:, mode_index[mode]]
                p_llm = llm_p(scenario, frame_m)[:, mode_index[mode]]
                delta = pct / 100.0
                for model_name, base_s, shock_s in (
                    ("mnl", b_mnl, p_mnl),
                    ("llm", b_llm, p_llm),
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
                            "n_people": int(pd.unique(elast_df.loc[mask, "ID"]).size),
                            **stat,
                        }
                    )
                    series[(model_name, pct)] = shock_s
            curve_key = (mode, attribute)
            curves[curve_key] = []
            for model_name in ("mnl", "llm"):
                for pct in (0, *PCT_SHOCKS):
                    curves[curve_key].append(
                        {"model": model_name, "pct": pct, "share": float(series[(model_name, pct)].mean())}
                    )

    # Implied VOT from ±10 minutes and ±10 CHF, on tasks where neither shift clips.
    vot_rows = []
    for mode, time_col, cost_col in (
        ("train", "TRAIN_TT", "train_cost"),
        ("sm", "SM_TT", "sm_cost"),
    ):
        mask = (elast_df[time_col].to_numpy() >= 20) & (elast_df[cost_col].to_numpy() >= 10)
        frame_m = take(elast_df, mask)
        if mask.sum() < 20:
            vot_rows.append({"mode": mode, "n": int(mask.sum()), "note": "too few unclipped tasks"})
            continue
        draws_m = respondent_draws(frame_m["ID"].to_numpy(), N_BOOT, SEED + 5)
        packs = {}
        for attribute, suffix in (("time", "min"), ("cost", "chf")):
            for level in (-10, 10):
                scenario = f"{mode}_{attribute}_{level:+d}{suffix}"
                shocked = take(
                    apply_attribute_shock(elast_df, mode, attribute, "abs", float(level)),
                    mask,
                )
                packs[(attribute, level, "mnl")] = mnl_p(train_base, shocked)[:, mode_index[mode]]
                packs[(attribute, level, "llm")] = llm_p(scenario, frame_m)[:, mode_index[mode]]
        for model_name in ("mnl", "llm"):
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

    # GA holders: introducing a 10 CHF fare on a mode that was free.
    ga_fare = []
    ga_elast = elast_df.loc[elast_df["GA"] == 1]
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

    # Wording check: same zeros, pass not mentioned.
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
        "llm_hard_share": [float((hard_predict(equal_llm, availability_of(elast_df)) == k).mean()) for k in range(3)],
    }

    # GA vs not on the main LLM sample.
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
        mnl_s = shares_from_proba(predict_proba(frame, train_base))
        llm_s = shares_from_proba(llm_p("base", frame))
        return {
            "n": int(len(frame)),
            "n_people": int(frame["ID"].nunique()),
            "observed": observed,
            "mnl": mnl_s,
            "llm": llm_s,
        }

    ga_split = {
        "ga": group_shares(llm_df.loc[llm_df["GA"] == 1]),
        "no_ga": group_shares(llm_df.loc[llm_df["GA"] == 0]),
    }

    # Format diagnostics on the evaluation sample, not the pilot.
    fmt = {
        "mean_digit_mass": float(base_aligned["digit_mass"].mean()),
        "top_token_is_digit": float(base_aligned["top_is_digit"].mean()),
        "top_token_available": float(base_aligned["top_is_available"].mean()),
        "mean_seconds": float(base_aligned["seconds"].mean()),
        "mean_tokens": float(base_aligned["n_tokens"].mean()),
    }

    # Agreement of hard predictions.
    agreement = float((preds["mnl"] == preds["llm"]).mean())

    elast_df_out = pd.DataFrame(elast_rows)
    elast_df_out.to_csv(RESULTS / "elasticities.csv", index=False)
    pd.DataFrame(vot_rows).to_csv(RESULTS / "implied_vot.csv", index=False)

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
            "n_elast_paying": int(paying.sum()),
            "choice_counts": {
                MODE_NAMES[k]: int((llm_df["CHOICE"] == k + 1).sum()) for k in range(3)
            },
            "fingerprint": fingerprint,
            "model_id": "Qwen2.5-1.5B-Instruct",
            "quantization": "Q4_K_M",
            "format": fmt,
            "pilot": pilot_summary,
        },
        "reports_llm_sample": reports,
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
        "seconds_total": time.perf_counter() - started,
    }
    write_json(RESULTS / "summary.json", summary)

    # Figures that need both predictors.
    plot_confusion(
        np.array(reports["mnl"]["confusion"]),
        np.array(reports["llm"]["confusion"]),
        n=len(llm_df),
    )
    plot_shares(
        [
            {"name": "Observed", "shares": reports["llm"]["observed_share"]},
            {"name": "Majority", "shares": reports["majority"]["hard_share"]},
            {"name": "MNL", "shares": reports["mnl"]["prob_share"]},
            {"name": "Richer MNL", "shares": reports["mnl_rich"]["prob_share"]},
            {"name": "LLM", "shares": reports["llm"]["prob_share"]},
        ],
        n=len(llm_df),
    )
    plot_elasticities(curves)
    if ga_split["ga"]["n"] > 0 and ga_split["no_ga"]["n"] > 0:
        ga_rows = []
        for status, key in (("No GA", "no_ga"), ("Has GA", "ga")):
            block = ga_split[key]
            for model, field in (("Observed", "observed"), ("MNL", "mnl"), ("LLM", "llm")):
                ga_rows.append({"status": status, "model": model, "shares": block[field]})
        plot_bias(
            ga_rows,
            {"mnl": equalised["mnl_prob_share"], "llm": equalised["llm_prob_share"]},
            n_ga=ga_split["ga"]["n"],
            n_noga=ga_split["no_ga"]["n"],
            n_eq=equalised["n"],
        )
    else:
        print("Skipping bias figure; the LLM sample is missing a GA group.")
    print("Wrote results/summary.json and figures/.")
    print(
        "LLM-sample accuracy: "
        + ", ".join(f"{k}={reports[k]['accuracy']:.3f}" for k in ("majority", "mnl", "mnl_rich", "llm"))
    )
    print(f"Total minutes: {(time.perf_counter() - started) / 60:.1f}")


if __name__ == "__main__":
    main()
