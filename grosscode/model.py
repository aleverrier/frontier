from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
import scipy.io as spio
import scipy.sparse as sp


QTANNER_ROOT = Path("/Users/anthony/research/qtanner-ssf")
if str(QTANNER_ROOT) not in sys.path:
    sys.path.insert(0, str(QTANNER_ROOT))

from qtanner_decoder import build_row_space, compute_syndrome, in_row_space, read_mtx, row_sets_to_masks  # type: ignore  # noqa: E402


DEFAULT_HX = QTANNER_ROOT / "gross_code" / "HX_Gross_144_12_12.mtx"
DEFAULT_HZ = QTANNER_ROOT / "gross_code" / "HZ_Gross_144_12_12.mtx"

Side = Literal["x", "z"]


def clip_prob(p: float, eps: float = 1e-12) -> float:
    return float(min(max(float(p), eps), 1.0 - eps))


def llr_from_prob(p: float) -> float:
    p_eff = clip_prob(p)
    return float(math.log((1.0 - p_eff) / p_eff))


def bits_to_mask(bits: Sequence[int]) -> int:
    mask = 0
    for idx, bit in enumerate(bits):
        if int(bit) & 1:
            mask ^= 1 << int(idx)
    return int(mask)


def logistic_cost(bits: np.ndarray, llr: np.ndarray) -> float:
    bits_arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    llr_arr = np.asarray(llr, dtype=np.float64).reshape(-1)
    if bits_arr.size != llr_arr.size:
        raise ValueError("bits and llr must have the same length")
    cost0 = np.empty_like(llr_arr, dtype=np.float64)
    cost1 = np.empty_like(llr_arr, dtype=np.float64)
    finite = np.isfinite(llr_arr)
    cost0[finite] = np.logaddexp(0.0, -llr_arr[finite])
    cost1[finite] = np.logaddexp(0.0, llr_arr[finite])
    pos_inf = np.isposinf(llr_arr)
    neg_inf = np.isneginf(llr_arr)
    cost0[pos_inf] = 0.0
    cost1[pos_inf] = float("inf")
    cost0[neg_inf] = float("inf")
    cost1[neg_inf] = 0.0
    return float(np.sum(np.where(bits_arr == 0, cost0, cost1)))


def logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    arr = np.asarray(list(values), dtype=np.float64)
    vmax = float(np.max(arr))
    if not np.isfinite(vmax):
        return vmax
    return float(vmax + math.log(float(np.sum(np.exp(arr - vmax)))))


@dataclass(frozen=True)
class RepeatedSyndromeShot:
    side: Side
    p: float
    p_meas: float
    rounds: int
    seed: int
    data_faults: np.ndarray
    measurement_faults: np.ndarray
    detector_slices: np.ndarray
    total_data: np.ndarray

    @property
    def total_data_mask(self) -> int:
        return bits_to_mask(self.total_data)


@lru_cache(maxsize=8)
def _read_mod2_sparse(path: str) -> sp.csr_matrix:
    mat = spio.mmread(path)
    if not sp.issparse(mat):
        mat = sp.csr_matrix(mat)
    mat = mat.tocsr().astype(np.uint8)
    if mat.nnz:
        mat.data %= 2
        mat.eliminate_zeros()
    return mat


def _load_side_paths(side: Side, hx_path: Path, hz_path: Path) -> Tuple[Path, Path]:
    if side == "x":
        return hz_path, hx_path
    if side == "z":
        return hx_path, hz_path
    raise ValueError(f"unsupported side '{side}'")


@dataclass
class GrossSideModel:
    side: Side
    h_path: Path
    stabilizer_path: Path
    h_sparse: sp.csr_matrix
    h_dense: np.ndarray
    h_row_masks: Tuple[int, ...]
    stabilizer_basis: Dict[int, int]
    stabilizer_pivots: Tuple[int, ...]
    n_data: int
    n_checks: int
    _local_round_pcm: sp.csr_matrix | None = field(default=None, init=False, repr=False)
    _full_pcm_cache: Dict[int, sp.csr_matrix] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def load(
        cls,
        *,
        side: Side,
        hx_path: Path = DEFAULT_HX,
        hz_path: Path = DEFAULT_HZ,
    ) -> "GrossSideModel":
        h_path, stabilizer_path = _load_side_paths(side, Path(hx_path), Path(hz_path))
        h_sets, n_h = read_mtx(str(h_path))
        stabilizer_sets, n_stab = read_mtx(str(stabilizer_path))
        if int(n_h) != int(n_stab):
            raise ValueError(f"matrix size mismatch: {h_path} has n={n_h}, {stabilizer_path} has n={n_stab}")
        h_row_masks = tuple(int(x) for x in row_sets_to_masks(h_sets, int(n_h)))
        stabilizer_masks = row_sets_to_masks(stabilizer_sets, int(n_h))
        stabilizer_basis, stabilizer_pivots = build_row_space(stabilizer_masks)
        h_sparse = _read_mod2_sparse(str(h_path))
        h_dense = np.asarray(h_sparse.toarray() % 2, dtype=np.uint8)
        return cls(
            side=side,
            h_path=Path(h_path),
            stabilizer_path=Path(stabilizer_path),
            h_sparse=h_sparse,
            h_dense=h_dense,
            h_row_masks=h_row_masks,
            stabilizer_basis=stabilizer_basis,
            stabilizer_pivots=tuple(int(x) for x in stabilizer_pivots),
            n_data=int(n_h),
            n_checks=int(h_sparse.shape[0]),
        )

    def sample_shot(
        self,
        *,
        p: float,
        rounds: int,
        seed: int,
        p_meas: float | None = None,
    ) -> RepeatedSyndromeShot:
        rounds_int = int(rounds)
        if rounds_int <= 0:
            raise ValueError("rounds must be positive")
        p_meas_eff = float(p if p_meas is None else p_meas)
        rng = np.random.default_rng(int(seed))
        data_faults = (rng.random((rounds_int, self.n_data)) < clip_prob(p)).astype(np.uint8)
        measurement_faults = (rng.random((rounds_int, self.n_checks)) < clip_prob(p_meas_eff)).astype(np.uint8)
        syndromes = ((data_faults @ self.h_dense.T) & 1).astype(np.uint8)
        detector_slices = syndromes.copy()
        detector_slices[0] ^= measurement_faults[0]
        if rounds_int > 1:
            detector_slices[1:] ^= measurement_faults[1:]
            detector_slices[1:] ^= measurement_faults[:-1]
        total_data = np.bitwise_xor.reduce(data_faults, axis=0).astype(np.uint8)
        return RepeatedSyndromeShot(
            side=self.side,
            p=float(p),
            p_meas=float(p_meas_eff),
            rounds=rounds_int,
            seed=int(seed),
            data_faults=data_faults,
            measurement_faults=measurement_faults,
            detector_slices=detector_slices,
            total_data=total_data,
        )

    def local_round_pcm(self) -> sp.csr_matrix:
        if self._local_round_pcm is not None:
            return self._local_round_pcm
        rows: List[int] = []
        cols: List[int] = []
        data: List[int] = []
        left_offset = self.n_data
        right_offset = self.n_data + self.n_checks
        coo = self.h_sparse.tocoo()
        rows.extend(int(x) for x in coo.row.tolist())
        cols.extend(int(x) for x in coo.col.tolist())
        data.extend(1 for _ in range(int(coo.nnz)))
        for check_idx in range(self.n_checks):
            rows.append(int(check_idx))
            cols.append(int(left_offset + check_idx))
            data.append(1)
            rows.append(int(check_idx))
            cols.append(int(right_offset + check_idx))
            data.append(1)
        self._local_round_pcm = sp.coo_matrix(
            (np.asarray(data, dtype=np.uint8), (np.asarray(rows), np.asarray(cols))),
            shape=(self.n_checks, self.n_data + 2 * self.n_checks),
            dtype=np.uint8,
        ).tocsr()
        return self._local_round_pcm

    def full_block_pcm(self, rounds: int) -> sp.csr_matrix:
        rounds_int = int(rounds)
        if rounds_int <= 0:
            raise ValueError("rounds must be positive")
        cached = self._full_pcm_cache.get(rounds_int)
        if cached is not None:
            return cached
        n_q = rounds_int * self.n_data
        n_m = rounds_int * self.n_checks
        total_vars = n_q + n_m
        rows: List[int] = []
        cols: List[int] = []
        data: List[int] = []
        coo = self.h_sparse.tocoo()
        for round_idx in range(rounds_int):
            row_offset = round_idx * self.n_checks
            q_offset = round_idx * self.n_data
            rows.extend((row_offset + int(r)) for r in coo.row.tolist())
            cols.extend((q_offset + int(c)) for c in coo.col.tolist())
            data.extend(1 for _ in range(int(coo.nnz)))
            right_m_offset = n_q + round_idx * self.n_checks
            for check_idx in range(self.n_checks):
                rows.append(int(row_offset + check_idx))
                cols.append(int(right_m_offset + check_idx))
                data.append(1)
                if round_idx > 0:
                    left_m_offset = n_q + (round_idx - 1) * self.n_checks
                    rows.append(int(row_offset + check_idx))
                    cols.append(int(left_m_offset + check_idx))
                    data.append(1)
        pcm = sp.coo_matrix(
            (np.asarray(data, dtype=np.uint8), (np.asarray(rows), np.asarray(cols))),
            shape=(rounds_int * self.n_checks, total_vars),
            dtype=np.uint8,
        ).tocsr()
        self._full_pcm_cache[rounds_int] = pcm
        return pcm

    def full_block_prior_llr(self, *, rounds: int, p_data: float, p_meas: float) -> np.ndarray:
        data_llr = np.full(int(rounds) * self.n_data, llr_from_prob(p_data), dtype=np.float64)
        meas_llr = np.full(int(rounds) * self.n_checks, llr_from_prob(p_meas), dtype=np.float64)
        return np.concatenate([data_llr, meas_llr])

    def unpack_full_block_estimate(self, estimate: Sequence[int], rounds: int) -> Tuple[np.ndarray, np.ndarray]:
        rounds_int = int(rounds)
        arr = np.asarray(estimate, dtype=np.uint8).reshape(-1)
        n_q = rounds_int * self.n_data
        expected = n_q + rounds_int * self.n_checks
        if arr.size != expected:
            raise ValueError(f"estimate length mismatch: got {arr.size}, expected {expected}")
        q_hat = arr[:n_q].reshape(rounds_int, self.n_data).copy()
        m_hat = arr[n_q:].reshape(rounds_int, self.n_checks).copy()
        return q_hat, m_hat

    def predicted_detector_slices(self, data_faults: np.ndarray, measurement_faults: np.ndarray) -> np.ndarray:
        q_arr = np.asarray(data_faults, dtype=np.uint8)
        m_arr = np.asarray(measurement_faults, dtype=np.uint8)
        if q_arr.ndim != 2 or q_arr.shape[1] != self.n_data:
            raise ValueError("data_faults must have shape (rounds, n_data)")
        if m_arr.shape != (q_arr.shape[0], self.n_checks):
            raise ValueError("measurement_faults shape mismatch")
        syndromes = ((q_arr @ self.h_dense.T) & 1).astype(np.uint8)
        detector_slices = syndromes.copy()
        detector_slices[0] ^= m_arr[0]
        if q_arr.shape[0] > 1:
            detector_slices[1:] ^= m_arr[1:]
            detector_slices[1:] ^= m_arr[:-1]
        return detector_slices

    def detector_slices_match(self, data_faults: np.ndarray, measurement_faults: np.ndarray, observed: np.ndarray) -> bool:
        predicted = self.predicted_detector_slices(data_faults, measurement_faults)
        return bool(np.array_equal(predicted.astype(np.uint8), np.asarray(observed, dtype=np.uint8)))

    def logical_status(self, true_total: np.ndarray, correction: np.ndarray) -> str:
        residual = np.asarray(true_total, dtype=np.uint8).reshape(-1) ^ np.asarray(correction, dtype=np.uint8).reshape(-1)
        if residual.size != self.n_data:
            raise ValueError("correction length mismatch")
        residual_mask = bits_to_mask(residual)
        if int(compute_syndrome(int(residual_mask), list(self.h_row_masks))) != 0:
            return "syndrome_fail"
        if residual_mask != 0 and not bool(in_row_space(int(residual_mask), self.stabilizer_basis, list(self.stabilizer_pivots))):
            return "logical_fail"
        return "success"
