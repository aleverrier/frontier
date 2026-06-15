from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import logsumexp

from .arikan import apply_arikan_transform, arikan_matrix
from .dynamic_frozen import DynamicFrozenSystem, derive_dynamic_frozen_system
from .exact_map import log_prior_probability
from .gf2 import bits_to_int, dense_mod2, matvec_mod2, matmul_mod2
from .sc_posterior import ExactSCPosterior


@dataclass(frozen=True)
class SCLCandidate:
    u: np.ndarray
    e: np.ndarray
    path_metric: float
    log_probability: float
    logical_bits: np.ndarray | None
    logical_key: int | None


@dataclass(frozen=True)
class SCLDecodeResult:
    dynamic_frozen: DynamicFrozenSystem
    q_matrix: np.ndarray
    best_candidate: SCLCandidate
    candidates: tuple[SCLCandidate, ...]
    logical_log_masses: dict[int, float]


def decode_scl(
    m_det: np.ndarray | Sequence[Sequence[int]],
    priors: np.ndarray | Sequence[float],
    syndrome: np.ndarray | Sequence[int],
    *,
    observables: np.ndarray | Sequence[Sequence[int]] | None = None,
    list_size: int,
) -> SCLDecodeResult:
    matrix = dense_mod2(m_det)
    syndrome_arr = dense_mod2(syndrome).reshape(-1)
    obs = None if observables is None else dense_mod2(observables)
    prior_arr = np.asarray(priors, dtype=np.float64).reshape(-1)
    if int(matrix.shape[1]) != int(prior_arr.shape[0]):
        raise ValueError("M_det/prior length mismatch")
    q = matmul_mod2(matrix, arikan_matrix(int(matrix.shape[1])))
    dynamic = derive_dynamic_frozen_system(q)
    rhs = dynamic.transformed_syndrome(syndrome_arr)
    for row in dynamic.consistency_rows:
        if np.any(dynamic.right_pivot_matrix[row]) or int(rhs[row]) == 0:
            continue
        raise ValueError("inconsistent syndrome for the supplied dynamic frozen system")
    evaluator = ExactSCPosterior(prior_arr)
    frozen_lookup = {rule.index: rule for rule in dynamic.rules}
    paths: list[tuple[np.ndarray, float]] = [(np.zeros(0, dtype=np.uint8), 0.0)]
    n = int(matrix.shape[1])
    for index in range(n):
        next_paths: list[tuple[np.ndarray, float]] = []
        for prefix, metric in paths:
            post = evaluator.posterior(prefix, index)
            if index in frozen_lookup:
                forced = dynamic.forced_bit(index, prefix, rhs)
                prob = float(post[forced])
                if prob <= 0.0:
                    continue
                next_paths.append((np.concatenate([prefix, np.array([forced], dtype=np.uint8)]), metric + float(np.log(prob))))
                continue
            for bit in (0, 1):
                prob = float(post[bit])
                if prob <= 0.0:
                    continue
                next_paths.append((np.concatenate([prefix, np.array([bit], dtype=np.uint8)]), metric + float(np.log(prob))))
        next_paths.sort(key=lambda item: item[1], reverse=True)
        paths = next_paths[: int(list_size)]
        if not paths:
            raise ValueError("decoder pruned all paths; check the dynamic frozen constraints")
    candidates: list[SCLCandidate] = []
    logical_logs: dict[int, list[float]] = {}
    for u, path_metric in paths:
        e = apply_arikan_transform(u)
        if not np.array_equal(matvec_mod2(matrix, e), syndrome_arr):
            continue
        logical_bits = None
        logical_key = None
        if obs is not None:
            logical_bits = matvec_mod2(obs, e)
            logical_key = bits_to_int(logical_bits.tolist())
        log_prob = log_prior_probability(e, prior_arr)
        candidates.append(
            SCLCandidate(
                u=u.copy(),
                e=e.copy(),
                path_metric=float(path_metric),
                log_probability=float(log_prob),
                logical_bits=None if logical_bits is None else logical_bits.copy(),
                logical_key=logical_key,
            )
        )
        if logical_key is not None:
            logical_logs.setdefault(int(logical_key), []).append(log_prob)
    if not candidates:
        raise ValueError("no surviving candidate satisfies the syndrome")
    candidates.sort(key=lambda item: item.log_probability, reverse=True)
    logical_log_masses = {
        int(key): float(logsumexp(np.asarray(values, dtype=np.float64)))
        for key, values in logical_logs.items()
    }
    return SCLDecodeResult(
        dynamic_frozen=dynamic,
        q_matrix=q,
        best_candidate=candidates[0],
        candidates=tuple(candidates),
        logical_log_masses=logical_log_masses,
    )
