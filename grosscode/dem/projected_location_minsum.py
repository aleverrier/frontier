from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from grosscode.core import DecoderConfig
from grosscode.dem.projected_location import PROJECTED_STATE_LABELS, ProjectedLocationProblem

def _small_logsumexp(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("-inf")
    vmax = float(np.max(arr))
    if not np.isfinite(vmax):
        return vmax
    return float(vmax + math.log(float(np.sum(np.exp(arr - vmax)))))


def _compress_scores_to_llr_logsumexp(
    scores: np.ndarray, zeros: np.ndarray, ones: np.ndarray, llr_clip: float
) -> float:
    llr = _small_logsumexp(np.asarray(scores, dtype=np.float64)[zeros]) - _small_logsumexp(
        np.asarray(scores, dtype=np.float64)[ones]
    )
    return float(np.clip(llr, -float(llr_clip), float(llr_clip)))


def _compress_scores_to_llr_max(scores: np.ndarray, zeros: np.ndarray, ones: np.ndarray, llr_clip: float) -> float:
    arr = np.asarray(scores, dtype=np.float64)
    llr = float(np.max(arr[zeros]) - np.max(arr[ones]))
    return float(np.clip(llr, -float(llr_clip), float(llr_clip)))


def _check_update_minsum_binary(
    incoming_v2c: np.ndarray,
    syndrome_bit: int,
    old_c2v: np.ndarray,
    *,
    normalization: float,
    offset: float,
    damping: float,
    llr_clip: float,
) -> np.ndarray:
    degree = int(incoming_v2c.size)
    if degree == 0:
        return np.asarray(old_c2v, dtype=np.float64).copy()

    incoming = np.asarray(incoming_v2c, dtype=np.float64).reshape(-1)
    signs = np.where(incoming >= 0.0, 1.0, -1.0)
    abs_vals = np.abs(incoming)
    parity_sign = -1.0 if (int(syndrome_bit) & 1) else 1.0
    prod_sign = parity_sign * float(np.prod(signs))

    if degree == 1:
        forced = float(llr_clip) if (int(syndrome_bit) & 1) == 0 else -float(llr_clip)
        raw = np.asarray([forced], dtype=np.float64)
    else:
        idx_min = int(np.argmin(abs_vals))
        min1 = float(abs_vals[idx_min])
        masked = abs_vals.copy()
        masked[idx_min] = np.inf
        min2 = float(np.min(masked))
        mags = np.full(degree, float(normalization) * max(min1 - float(offset), 0.0), dtype=np.float64)
        mags[idx_min] = float(normalization) * max(min2 - float(offset), 0.0)
        raw = prod_sign * signs * mags

    raw = np.clip(raw, -float(llr_clip), float(llr_clip))
    if float(damping) > 0.0:
        return np.asarray((1.0 - float(damping)) * raw + float(damping) * np.asarray(old_c2v, dtype=np.float64))
    return raw


def _binary_contribution(sign_row: np.ndarray, llr: float) -> np.ndarray:
    return 0.5 * float(llr) * np.asarray(sign_row, dtype=np.float64)


@dataclass(frozen=True)
class ProjectedLocationDecodeResult:
    estimate_symbols: np.ndarray
    posterior_log_scores: np.ndarray
    converged: bool
    iterations: int
    edge_updates: int
    unsatisfied_checks: int
    unsatisfied_vector: np.ndarray
    logical_action: np.ndarray


@dataclass
class ProjectedLocationMinSumDecoder:
    problem: ProjectedLocationProblem
    config: DecoderConfig = DecoderConfig()
    compression_mode: str = "max"

    def __post_init__(self) -> None:
        self.config.validate("minsum")
        if bool(self.config.self_corrected):
            raise NotImplementedError("self-corrected min-sum is not implemented for projected-location decoding")
        mode = str(self.compression_mode).strip().lower()
        if mode not in {"max", "logsumexp"}:
            raise ValueError("compression_mode must be one of: max, logsumexp")
        self.compression_mode = mode

    def _compress_scores_to_llr(self, scores: np.ndarray, mask: int) -> float:
        zeros, ones = self.problem.mask_state_split[int(mask)]
        if str(self.compression_mode) == "max":
            return _compress_scores_to_llr_max(scores, zeros, ones, float(self.config.llr_clip))
        return _compress_scores_to_llr_logsumexp(scores, zeros, ones, float(self.config.llr_clip))

    def _decode_flooding(self, target: np.ndarray) -> ProjectedLocationDecodeResult:
        belief = self.problem.initial_log_scores()
        c2v = np.zeros(self.problem.n_edges, dtype=np.float64)
        converged = False
        hard = np.zeros(self.problem.n, dtype=np.uint8)
        residual = target.copy()
        iterations = 0
        for it in range(1, int(self.config.max_iter) + 1):
            iterations = int(it)
            v2c = np.zeros(self.problem.n_edges, dtype=np.float64)
            for edge_index in range(self.problem.n_edges):
                var = int(self.problem.detector_edge_col[edge_index])
                mask = int(self.problem.detector_edge_mask[edge_index])
                scores_excluding = belief[var] - _binary_contribution(self.problem.mask_sign_table[mask], c2v[edge_index])
                v2c[edge_index] = self._compress_scores_to_llr(scores_excluding, mask)

            new_c2v = c2v.copy()
            for check_index, edges in enumerate(self.problem.detector_check_to_edges):
                if edges.size == 0:
                    continue
                new_c2v[edges] = _check_update_minsum_binary(
                    v2c[edges],
                    int(target[check_index]),
                    c2v[edges],
                    normalization=float(self.config.normalization),
                    offset=float(self.config.offset),
                    damping=float(self.config.damping),
                    llr_clip=float(self.config.llr_clip),
                )

            belief = self.problem.initial_log_scores()
            for edge_index in range(self.problem.n_edges):
                var = int(self.problem.detector_edge_col[edge_index])
                mask = int(self.problem.detector_edge_mask[edge_index])
                belief[var] += _binary_contribution(self.problem.mask_sign_table[mask], new_c2v[edge_index])
            belief -= np.max(belief, axis=1, keepdims=True)
            c2v = new_c2v
            hard = np.asarray(np.argmax(belief, axis=1), dtype=np.uint8)
            residual = self.problem.syndrome_from_symbols(hard) ^ target
            if int(np.count_nonzero(residual)) == 0:
                converged = True
                break
        return ProjectedLocationDecodeResult(
            estimate_symbols=hard,
            posterior_log_scores=belief,
            converged=bool(converged),
            iterations=int(iterations),
            edge_updates=int(iterations * self.problem.n_edges),
            unsatisfied_checks=int(np.count_nonzero(residual)),
            unsatisfied_vector=np.asarray(residual, dtype=np.uint8),
            logical_action=self.problem.logical_action_from_symbols(hard),
        )

    def _decode_layered(self, target: np.ndarray) -> ProjectedLocationDecodeResult:
        belief = self.problem.initial_log_scores()
        c2v = np.zeros(self.problem.n_edges, dtype=np.float64)
        converged = False
        hard = np.zeros(self.problem.n, dtype=np.uint8)
        residual = target.copy()
        iterations = 0
        for it in range(1, int(self.config.max_iter) + 1):
            iterations = int(it)
            for check_index, edges in enumerate(self.problem.detector_check_to_edges):
                if edges.size == 0:
                    continue
                incoming = np.zeros(edges.size, dtype=np.float64)
                for local_edge_index, edge_index in enumerate(edges.tolist()):
                    var = int(self.problem.detector_edge_col[edge_index])
                    mask = int(self.problem.detector_edge_mask[edge_index])
                    scores_excluding = belief[var] - _binary_contribution(self.problem.mask_sign_table[mask], c2v[edge_index])
                    incoming[local_edge_index] = self._compress_scores_to_llr(scores_excluding, mask)
                new_c2v = _check_update_minsum_binary(
                    incoming,
                    int(target[check_index]),
                    c2v[edges],
                    normalization=float(self.config.normalization),
                    offset=float(self.config.offset),
                    damping=float(self.config.damping),
                    llr_clip=float(self.config.llr_clip),
                )
                delta = np.asarray(new_c2v - c2v[edges], dtype=np.float64)
                c2v[edges] = new_c2v
                for local_edge_index, edge_index in enumerate(edges.tolist()):
                    if delta[local_edge_index] == 0.0:
                        continue
                    var = int(self.problem.detector_edge_col[edge_index])
                    mask = int(self.problem.detector_edge_mask[edge_index])
                    belief[var] += _binary_contribution(self.problem.mask_sign_table[mask], delta[local_edge_index])
            belief -= np.max(belief, axis=1, keepdims=True)
            hard = np.asarray(np.argmax(belief, axis=1), dtype=np.uint8)
            residual = self.problem.syndrome_from_symbols(hard) ^ target
            if int(np.count_nonzero(residual)) == 0:
                converged = True
                break
        return ProjectedLocationDecodeResult(
            estimate_symbols=hard,
            posterior_log_scores=belief,
            converged=bool(converged),
            iterations=int(iterations),
            edge_updates=int(iterations * self.problem.n_edges),
            unsatisfied_checks=int(np.count_nonzero(residual)),
            unsatisfied_vector=np.asarray(residual, dtype=np.uint8),
            logical_action=self.problem.logical_action_from_symbols(hard),
        )

    def decode(self, syndrome: np.ndarray) -> ProjectedLocationDecodeResult:
        target = np.asarray(syndrome, dtype=np.uint8).reshape(-1) & 1
        if int(target.size) != self.problem.m:
            raise ValueError(f"syndrome length mismatch: got {target.size}, expected {self.problem.m}")
        schedule = self.config.normalized_schedule()
        if schedule == "flooding":
            return self._decode_flooding(target)
        if schedule == "layered":
            return self._decode_layered(target)
        raise ValueError(f"unsupported schedule for projected-location min-sum: {self.config.schedule}")


__all__ = [
    "PROJECTED_STATE_LABELS",
    "ProjectedLocationDecodeResult",
    "ProjectedLocationMinSumDecoder",
]
