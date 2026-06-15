from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import logsumexp

from .gf2 import bits_to_int, dense_mod2, enumerate_binary_vectors, matvec_mod2


def log_prior_probability(error: np.ndarray | Sequence[int], priors: np.ndarray | Sequence[float]) -> float:
    err = dense_mod2(error).reshape(-1)
    prior_arr = np.asarray(priors, dtype=np.float64).reshape(-1)
    if int(err.shape[0]) != int(prior_arr.shape[0]):
        raise ValueError("error/prior length mismatch")
    return float(np.sum(np.where(err == 1, np.log(prior_arr), np.log1p(-prior_arr)), dtype=np.float64))


@dataclass(frozen=True)
class ExactMapResult:
    best_error: np.ndarray
    best_log_probability: float
    best_logical_bits: np.ndarray | None
    best_logical_key: int | None
    feasible_count: int
    logical_log_masses: dict[int, float]
    all_feasible_errors: tuple[np.ndarray, ...]
    all_feasible_log_probs: tuple[float, ...]


def exact_map_for_syndrome(
    m_det: np.ndarray | Sequence[Sequence[int]],
    priors: np.ndarray | Sequence[float],
    syndrome: np.ndarray | Sequence[int],
    *,
    observables: np.ndarray | Sequence[Sequence[int]] | None = None,
) -> ExactMapResult:
    matrix = dense_mod2(m_det)
    syndrome_arr = dense_mod2(syndrome).reshape(-1)
    obs = None if observables is None else dense_mod2(observables)
    n = int(matrix.shape[1])
    errors = enumerate_binary_vectors(n)
    feasible_errors: list[np.ndarray] = []
    feasible_logs: list[float] = []
    logical_mass_terms: dict[int, list[float]] = {}
    best_index = -1
    best_log = float("-inf")
    best_logical_bits = None
    best_logical_key = None
    for error in errors:
        if not np.array_equal(matvec_mod2(matrix, error), syndrome_arr):
            continue
        feasible_errors.append(error.copy())
        log_prob = log_prior_probability(error, priors)
        feasible_logs.append(log_prob)
        logical_bits = None
        logical_key = None
        if obs is not None:
            logical_bits = matvec_mod2(obs, error)
            logical_key = bits_to_int(logical_bits.tolist())
            logical_mass_terms.setdefault(logical_key, []).append(log_prob)
        if log_prob > best_log:
            best_log = log_prob
            best_index = len(feasible_errors) - 1
            best_logical_bits = None if logical_bits is None else logical_bits.copy()
            best_logical_key = logical_key
    if best_index < 0:
        raise ValueError("no feasible error matches the supplied syndrome")
    logical_log_masses = {
        int(key): float(logsumexp(np.asarray(values, dtype=np.float64)))
        for key, values in logical_mass_terms.items()
    }
    return ExactMapResult(
        best_error=feasible_errors[best_index].copy(),
        best_log_probability=float(best_log),
        best_logical_bits=best_logical_bits,
        best_logical_key=best_logical_key,
        feasible_count=len(feasible_errors),
        logical_log_masses=logical_log_masses,
        all_feasible_errors=tuple(entry.copy() for entry in feasible_errors),
        all_feasible_log_probs=tuple(float(value) for value in feasible_logs),
    )
