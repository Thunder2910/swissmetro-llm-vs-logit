"""Score Swissmetro scenarios with a local instruct model.

Choices are not sampled. The next-token logits of the digit tokens
"1", "2", and "3" are turned into a distribution over available modes.
Temperature is irrelevant because nothing is drawn. The Llama seed is
still fixed so the setup is explicit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.config import MODEL_PATH, N_BATCH, N_CTX, N_THREADS, SEED


@dataclass
class Score:
    p_cond: np.ndarray  # length 3, renormalised over available digit groups
    p_raw: np.ndarray  # length 3, full-vocabulary probability of each digit group
    digit_mass: float  # sum of the three groups before availability masking
    pred: int  # 0, 1, 2
    top_token: str
    top_is_digit: bool
    top_is_available: bool
    n_tokens: int
    seconds: float


class ChoiceModel:
    def __init__(self, model_path: str | None = None) -> None:
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=str(model_path or MODEL_PATH),
            n_ctx=N_CTX,
            n_batch=N_BATCH,
            n_threads=N_THREADS,
            n_threads_batch=N_THREADS,
            seed=SEED,
            logits_all=False,
            verbose=False,
            use_mmap=True,
        )
        self.groups = self._digit_groups()
        self._cache_ok: bool | None = None
        self._prefix_tokens: list[int] | None = None
        self._prefix_key: str | None = None

    def n_vocab(self) -> int:
        return int(self.llm.n_vocab())

    def _digit_groups(self) -> dict[int, list[int]]:
        """Token ids whose text is exactly 1, 2, or 3, aside from whitespace."""
        groups: dict[int, list[int]] = {1: [], 2: [], 3: []}
        for i in range(self.n_vocab()):
            raw = self.llm.detokenize([i])
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if len(text) > 3:
                continue
            stripped = text.strip()
            if stripped in {"1", "2", "3"} and set(text) <= set(" \t\r\n123"):
                groups[int(stripped)].append(i)
        for digit, ids in groups.items():
            if not ids:
                raise RuntimeError(f"No token decodes to digit {digit}")
        return groups

    def _tokenize(self, text: str) -> list[int]:
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))

    def _logits(self) -> np.ndarray:
        n_vocab = self.n_vocab()
        ptr = self.llm._ctx.get_logits()
        if not ptr:
            raise RuntimeError("Model returned no logits. Check the llama.cpp build.")
        arr = np.ctypeslib.as_array(ptr, shape=(n_vocab,))
        return np.array(arr, dtype=np.float64, copy=True)

    def _prepare_prefix(self, prefix: str) -> None:
        if self._prefix_key == prefix and self._prefix_tokens is not None:
            return
        tokens = self._tokenize(prefix)
        self.llm.reset()
        self.llm.eval(tokens)
        self._prefix_tokens = tokens
        self._prefix_key = prefix

    def _eval_prompt(self, prefix: str, suffix: str) -> int:
        """Evaluate the chat prompt. Reuse the system-prefix cache when BPE allows it."""
        full_text = prefix + suffix
        full = self._tokenize(full_text)
        if len(full) >= N_CTX:
            raise RuntimeError(f"Prompt is {len(full)} tokens, context is {N_CTX}.")
        if self._cache_ok is None:
            self._prepare_prefix(prefix)
            suffix_ids = self._tokenize(suffix)
            self._cache_ok = self._prefix_tokens + suffix_ids == full
        if self._cache_ok:
            self._prepare_prefix(prefix)
            assert self._prefix_tokens is not None
            self.llm.n_tokens = len(self._prefix_tokens)
            suffix_ids = full[len(self._prefix_tokens) :]
            self.llm.eval(suffix_ids)
        else:
            self.llm.reset()
            self.llm.eval(full)
        return len(full)

    def score(self, prefix: str, suffix: str, available: np.ndarray) -> Score:
        t0 = time.perf_counter()
        n_tokens = self._eval_prompt(prefix, suffix)
        logits = self._logits()
        # Full-vocabulary log-softmax, then sum the token ids in each digit group.
        m = float(logits.max())
        exp_shift = np.exp(logits - m)
        z = float(exp_shift.sum())
        prob = exp_shift / z
        p_raw = np.zeros(3)
        for digit, ids in self.groups.items():
            p_raw[digit - 1] = float(prob[np.array(ids, dtype=int)].sum())
        avail = np.asarray(available, dtype=bool)
        masked = np.where(avail, p_raw, 0.0)
        mass_avail = float(masked.sum())
        if mass_avail <= 0:
            p_cond = np.full(3, np.nan)
            pred = -1
        else:
            p_cond = masked / mass_avail
            pred = int(np.argmax(np.where(avail, p_cond, -np.inf)))
        top_id = int(np.argmax(logits))
        top_text = self.llm.detokenize([top_id]).decode("utf-8", errors="replace")
        top_digit = top_text.strip() if top_text.strip() in {"1", "2", "3"} else ""
        top_available = bool(top_digit) and bool(avail[int(top_digit) - 1])
        return Score(
            p_cond=p_cond,
            p_raw=p_raw,
            digit_mass=float(p_raw.sum()),
            pred=pred,
            top_token=top_text,
            top_is_digit=bool(top_digit),
            top_is_available=top_available,
            n_tokens=n_tokens,
            seconds=time.perf_counter() - t0,
        )


def ensure_model(path=MODEL_PATH):
    """Download the GGUF if the committed copy is absent. Weights are not committed."""
    from src.config import MODEL_BYTES, MODEL_URL

    if path.exists() and path.stat().st_size == MODEL_BYTES:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    print(f"Downloading {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, path)
    if path.stat().st_size != MODEL_BYTES:
        raise RuntimeError(
            f"Model size {path.stat().st_size} does not match expected {MODEL_BYTES}"
        )
    return path
