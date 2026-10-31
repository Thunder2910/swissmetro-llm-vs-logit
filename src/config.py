"""Paths, seeds, and sample sizes.

LLM sample sizes were set after a local timing probe on this machine
(see results/run_config.json). They are large enough for a comparison
and small enough to finish on CPU.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "swissmetro.dat"
MODEL_PATH = ROOT / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
PROMPTS = ROOT / "prompts"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DATA_URL = "https://transp-or.epfl.ch/data/swissmetro.dat"
DATA_SHA256 = "27432693cf052985d79a950b4b888be3efca798fc89b0d3ffefe40608ede00f2"

MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/"
    "qwen2.5-1.5b-instruct-q4_k_m.gguf"
)
MODEL_BYTES = 1_117_320_736
MODEL_ID = "Qwen2.5-1.5B-Instruct"
MODEL_QUANT = "Q4_K_M"
MODEL_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Published Biogeme example b01a (full filtered sample).
# Coefficients are the unrounded 3.3.x estimates; fit statistics are the
# values reported by Biogeme (three decimals).
BIOGEME_B01A = {
    "n_obs": 6768,
    "n_excluded": 3960,
    "null_loglik": -6964.663,
    "loglik": -5331.252,
    "coef": {
        "ASC_TRAIN": -0.701187,
        "ASC_CAR": -0.154633,
        "B_TIME": -1.277859,
        "B_COST": -1.083790,
    },
}

SEED = 42
TEST_FRACTION = 0.20

# LLM experiment size. Overridden by CLI flags.
# About 0.85 s per prompt on 4 CPU threads (Q4_K_M). 420 base tasks plus
# a 84-task shock subsample is roughly 45 minutes.
LLM_N = 420
ELAST_N = 84
PILOT_N = 24
PROBE_N = 8
MAX_TASKS_PER_PERSON = 4
ELAST_MAX_TASKS_PER_PERSON = 2

PCT_SHOCKS = (-50, -20, -10, 10, 20, 50)
ABS_LEVELS = (-10, 10)  # minutes, or CHF, central difference

N_BOOT = 1000
N_THREADS = 4
N_CTX = 1024
N_BATCH = 512

MODE_NAMES = ("Train", "Swissmetro", "Car")
MODE_KEYS = ("train", "sm", "car")
