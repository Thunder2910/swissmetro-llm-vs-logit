"""Persona and scenario prompts.

Labels follow the EPFL Swissmetro codebook
(transp-or.epfl.ch/documents/technicalReports/CS_SwissmetroDescription.pdf).
The prompt never includes the respondent's choice.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.config import PROMPTS

# Bump if rendering logic changes in a way the template files do not show.
PROMPT_LOGIC_ID = "v1"

PURPOSE = {1: "commuting", 3: "business"}
AGE = {
    1: "24 or younger",
    2: "25 to 39",
    3: "40 to 54",
    4: "55 to 65",
    5: "over 65",
    6: "not stated",
}
INCOME = {
    0: "under 50,000 CHF per year",
    1: "under 50,000 CHF per year",
    2: "50,000 to 100,000 CHF per year",
    3: "over 100,000 CHF per year",
    4: "not stated",
}
PAYER = {
    0: "not stated",
    1: "the traveller",
    2: "the employer",
    3: "shared half and half",
}
LUGGAGE = {0: "none", 1: "one piece", 3: "several pieces"}
GENDER = {0: "female", 1: "male"}


def load_text(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8").strip() + "\n"


def prompt_fingerprint() -> str:
    """Changes when the prompt text or the scoring code changes, so old scores are not reused."""
    h = hashlib.sha256()
    h.update(PROMPT_LOGIC_ID.encode())
    here = Path(__file__).resolve().parent
    for path in (
        PROMPTS / "system.txt",
        PROMPTS / "user_template.txt",
        Path(__file__).resolve(),
        here / "llm_choice.py",
    ):
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _fmt(value: float) -> str:
    x = float(value)
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.1f}"


def _label(mapping: dict, code: int, what: str) -> str:
    if int(code) not in mapping:
        return f"not stated ({what} code {int(code)})"
    return mapping[int(code)]


def _mode_text(available: bool, time: float, cost: float, headway: float | None, seats: str | None) -> str:
    if not available:
        return "not available"
    parts = [f"{_fmt(time)} min", f"{_fmt(cost)} CHF"]
    if headway is not None:
        parts.append(f"headway {_fmt(headway)} min")
    if seats is not None:
        parts.append(seats)
    return ", ".join(parts)


def render_user(row: pd.Series, scenario: str) -> str:
    """Fill the user template. `scenario` changes wording, not the choice."""
    ga = int(row["GA"]) == 1
    rail_free = float(row["train_cost"]) == 0.0 and float(row["sm_cost"]) == 0.0
    if scenario == "equalised":
        ga_text = "yes" if ga else "no"
        extra = (
            "Controlled comparison: every available mode takes 90 min and costs 50 CHF. "
            "Train and Swissmetro both run every 30 min with train-style seating.\n"
        )
    elif scenario == "ga_silent":
        ga_text = "not stated in this question"
        extra = ""
    elif ga and rail_free:
        # Only claim a zero fare when the numbers below really are zero.
        # A cost shock on a pass holder must not describe a fare the options contradict.
        ga_text = "yes. The train and Swissmetro fares below are 0 CHF because of this pass"
        extra = ""
    elif ga:
        ga_text = "yes. Use the fares printed on the options below"
        extra = ""
    else:
        ga_text = "no"
        extra = ""
    seats = "airline-style seats" if int(row["SM_SEATS"]) == 1 else "train-style seating"
    # In the equalised probe the dataframe already has seats set to train-style.
    user = load_text("user_template.txt").format(
        purpose=_label(PURPOSE, int(row["PURPOSE"]), "purpose"),
        age=_label(AGE, int(row["AGE"]), "age"),
        gender=_label(GENDER, int(row["MALE"]), "gender"),
        income=_label(INCOME, int(row["INCOME"]), "income"),
        payer=_label(PAYER, int(row["WHO"]), "who pays"),
        luggage=_label(LUGGAGE, int(row["LUGGAGE"]), "luggage"),
        ga=ga_text,
        extra=extra,
        train=_mode_text(bool(row["train_av"]), row["TRAIN_TT"], row["train_cost"], row["TRAIN_HE"], None),
        sm=_mode_text(bool(row["sm_av"]), row["SM_TT"], row["sm_cost"], row["SM_HE"], seats),
        car=_mode_text(bool(row["car_av"]), row["CAR_TT"], row["car_cost"], None, None),
    )
    return user


def render_prompt(row: pd.Series, scenario: str) -> str:
    system = load_text("system.txt").strip()
    user = render_user(row, scenario).strip()
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def chat_pieces(row: pd.Series, scenario: str) -> tuple[str, str]:
    """Prefix (cached) through the start of the user turn, and the suffix."""
    system = load_text("system.txt").strip()
    user = render_user(row, scenario).strip()
    prefix = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"
    suffix = f"{user}<|im_end|>\n<|im_start|>assistant\n"
    return prefix, suffix
