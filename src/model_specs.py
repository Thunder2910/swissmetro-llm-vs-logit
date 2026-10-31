"""Local instruct models scored with the same prompt.

The 1.5B cache file name is the historical one (prompt fingerprint only).
Larger models use a separate file so the caches cannot mix. Weights are
gitignored; sizes are the official Qwen GGUF byte counts.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.config import RESULTS, ROOT
from src.prompts import prompt_fingerprint

HF = "https://huggingface.co"


@dataclass(frozen=True)
class WeightFile:
    filename: str
    url: str
    nbytes: int


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    quant: str
    label: str
    files: tuple[WeightFile, ...]

    @property
    def path(self) -> Path:
        """File passed to llama.cpp. Split GGUFs load from the first shard."""
        return ROOT / "models" / self.files[0].filename


def _spec_3b() -> ModelSpec:
    name = "qwen2.5-3b-instruct-q4_k_m.gguf"
    return ModelSpec(
        key="3b",
        model_id="Qwen2.5-3B-Instruct",
        quant="Q4_K_M",
        label="3B",
        files=(
            WeightFile(
                filename=name,
                url=f"{HF}/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/{name}",
                nbytes=2_104_932_768,
            ),
        ),
    )


def _spec_7b() -> ModelSpec:
    # Official Q4_K_M is published as two shards. llama.cpp loads the pair
    # when the first shard is opened and the second sits beside it.
    first = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    second = "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"
    repo = f"{HF}/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main"
    return ModelSpec(
        key="7b",
        model_id="Qwen2.5-7B-Instruct",
        quant="Q4_K_M",
        label="7B",
        files=(
            WeightFile(filename=first, url=f"{repo}/{first}", nbytes=3_993_201_344),
            WeightFile(filename=second, url=f"{repo}/{second}", nbytes=689_872_288),
        ),
    )


def _spec_15() -> ModelSpec:
    name = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    return ModelSpec(
        key="1.5b",
        model_id="Qwen2.5-1.5B-Instruct",
        quant="Q4_K_M",
        label="1.5B",
        files=(
            WeightFile(
                filename=name,
                url=f"{HF}/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/{name}",
                nbytes=1_117_320_736,
            ),
        ),
    )


MODELS: tuple[ModelSpec, ...] = (_spec_15(), _spec_3b(), _spec_7b())
BY_KEY = {spec.key: spec for spec in MODELS}


def parse_model_keys(text: str) -> list[ModelSpec]:
    keys = [part.strip() for part in text.split(",") if part.strip()]
    if not keys:
        raise SystemExit("Pass at least one model key, for example 1.5b,3b,7b.")
    missing = [key for key in keys if key not in BY_KEY]
    if missing:
        known = ", ".join(BY_KEY)
        raise SystemExit(f"Unknown model key(s): {', '.join(missing)}. Known keys: {known}.")
    return [BY_KEY[key] for key in keys]


def cache_path(spec: ModelSpec) -> Path:
    """1.5B keeps the original cache name so the finished run is reused."""
    fingerprint = prompt_fingerprint()
    if spec.key == "1.5b":
        return RESULTS / "cache" / f"predictions_{fingerprint}.csv"
    return RESULTS / "cache" / f"predictions_{spec.key}_{fingerprint}.csv"


def ensure_weights(spec: ModelSpec) -> Path:
    for item in spec.files:
        path = ROOT / "models" / item.filename
        if path.exists() and path.stat().st_size == item.nbytes:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {item.url}")
        subprocess.check_call(
            ["curl", "-L", "--retry", "5", "--retry-delay", "2", "-C", "-", "-o", str(path), item.url]
        )
        if not path.exists() or path.stat().st_size != item.nbytes:
            got = path.stat().st_size if path.exists() else 0
            raise RuntimeError(f"{item.filename} is {got} bytes; expected {item.nbytes}.")
    return spec.path
