"""The prompt must not leak the choice or the annual pass price."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PATH
from src.data_utils import apply_attribute_shock, ensure_data, estimation_sample, load_raw
from src.prompts import render_prompt, render_user


class TestPrompts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        raw = load_raw(ensure_data(DATA_PATH))
        cls.df = estimation_sample(raw)

    def test_ga_prompt_shows_zero_and_hides_annual_price(self) -> None:
        row = self.df.loc[self.df["GA"] == 1].iloc[0]
        text = render_user(row, "base")
        self.assertIn("0 CHF", text)
        self.assertNotIn(str(int(row["TRAIN_CO"])), text)
        self.assertNotIn("chosen", text.lower())
        # The raw TRAIN_CO for a pass holder is an annual price, not a trip fare.
        self.assertGreater(float(row["TRAIN_CO"]), 500)
        full = render_prompt(row, "base")
        self.assertIn("<|im_start|>assistant", full)
        self.assertNotIn("CHOICE", full)

    def test_cost_shock_does_not_claim_a_zero_fare(self) -> None:
        row = self.df.loc[self.df["GA"] == 1].iloc[0]
        shocked = apply_attribute_shock(self.df.loc[[row.name]], "train", "cost", "abs", 10).iloc[0]
        text = render_user(shocked, "base")
        self.assertIn("10 CHF", text)
        self.assertNotIn("fares below are 0 CHF", text)

    def test_unavailable_car_is_labelled(self) -> None:
        row = self.df.loc[self.df["car_av"] == 0].iloc[0]
        text = render_user(row, "base")
        self.assertIn("3. Car: not available", text)
        self.assertNotIn("0 min, 0 CHF", text)


if __name__ == "__main__":
    unittest.main()
