from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .arikan import apply_arikan_transform
from .gf2 import dense_mod2, enumerate_binary_vectors


def _validate_priors(priors: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(priors, dtype=np.float64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError("priors must be rank 1")
    if np.any((arr <= 0.0) | (arr >= 0.5)):
        raise ValueError("all priors must lie strictly between 0 and 1/2")
    length = int(arr.shape[0])
    if length <= 0 or length & (length - 1):
        raise ValueError("priors length must be a positive power of two")
    return arr


@dataclass(frozen=True)
class ReliabilityEstimate:
    success_probability: np.ndarray
    conditional_entropy_bits: np.ndarray
    samples: int
    seed: int


class ExactSCPosterior:
    def __init__(self, priors: np.ndarray | Sequence[float]):
        self.priors = _validate_priors(priors)
        self.length = int(self.priors.shape[0])

    def posterior(self, prefix: np.ndarray | Sequence[int], index: int) -> np.ndarray:
        prefix_arr = dense_mod2(prefix).reshape(-1)
        if int(index) < 0 or int(index) >= self.length:
            raise IndexError("index out of range")
        if int(prefix_arr.shape[0]) != int(index):
            raise ValueError("prefix length must match the queried index")
        return self._posterior_recursive(self.priors, prefix_arr, int(index))

    def _posterior_recursive(self, priors: np.ndarray, prefix: np.ndarray, index: int) -> np.ndarray:
        length = int(priors.shape[0])
        if length == 1:
            return np.array([1.0 - priors[0], priors[0]], dtype=np.float64)
        half = length // 2
        if index < half:
            return self._posterior_recursive(priors[:half], prefix, index)
        local_index = index - half
        a_prefix = prefix[:half]
        if int(a_prefix.shape[0]) != half:
            raise ValueError("second-half synthetic bits require the full first-half prefix")
        b_prefix = prefix[half:]
        c_prefix = np.bitwise_xor(a_prefix[:local_index], b_prefix)
        c_post = self._posterior_recursive(priors[half:], c_prefix.astype(np.uint8, copy=False), local_index)
        a_bit = int(a_prefix[local_index])
        return np.array([c_post[a_bit], c_post[a_bit ^ 1]], dtype=np.float64)


def exhaustive_posterior_bit(
    priors: np.ndarray | Sequence[float],
    prefix: np.ndarray | Sequence[int],
    index: int,
) -> np.ndarray:
    prior_arr = _validate_priors(priors)
    prefix_arr = dense_mod2(prefix).reshape(-1)
    if int(prefix_arr.shape[0]) != int(index):
        raise ValueError("prefix length must match the queried index")
    suffix_width = int(prior_arr.shape[0]) - int(index) - 1
    suffixes = enumerate_binary_vectors(suffix_width)
    probs = np.zeros(2, dtype=np.float64)
    for bit in (0, 1):
        for suffix in suffixes:
            u = np.concatenate([prefix_arr, np.array([bit], dtype=np.uint8), suffix.astype(np.uint8, copy=False)])
            e = apply_arikan_transform(u)
            prob = np.prod(np.where(e, prior_arr, 1.0 - prior_arr), dtype=np.float64)
            probs[bit] += float(prob)
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("degenerate posterior")
    return probs / total


def estimate_reliability_monte_carlo(
    priors: np.ndarray | Sequence[float],
    *,
    samples: int,
    seed: int,
    progress_every_samples: int = 0,
) -> ReliabilityEstimate:
    prior_arr = _validate_priors(priors)
    evaluator = ExactSCPosterior(prior_arr)
    rng = np.random.default_rng(int(seed))
    n = int(prior_arr.shape[0])
    success = np.zeros(n, dtype=np.float64)
    entropy = np.zeros(n, dtype=np.float64)
    for sample_index in range(int(samples)):
        e = (rng.random(n) < prior_arr).astype(np.uint8)
        u = apply_arikan_transform(e)
        for index in range(n):
            post = evaluator.posterior(u[:index], index)
            guess = int(post[1] > post[0])
            success[index] += float(guess == int(u[index]))
            entropy[index] += float(-np.sum(np.where(post > 0.0, post * np.log2(post), 0.0)))
        if progress_every_samples and (sample_index + 1) % int(progress_every_samples) == 0:
            print(
                f"[polar-dem reliability] samples={sample_index + 1}/{samples} "
                f"mean_success={success.mean() / max(sample_index + 1, 1):.4f}"
            )
    return ReliabilityEstimate(
        success_probability=success / float(samples),
        conditional_entropy_bits=entropy / float(samples),
        samples=int(samples),
        seed=int(seed),
    )
