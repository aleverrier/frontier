from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import scipy.sparse as sp


def binary_csr_mod2(matrix: sp.spmatrix) -> sp.csr_matrix:
    if not sp.issparse(matrix):
        raise TypeError("matrix must be a scipy sparse matrix")
    coo = matrix.tocoo(copy=True)
    coo.sum_duplicates()
    data = np.mod(coo.data.astype(np.int64), 2).astype(np.uint8)
    keep = data != 0
    out = sp.coo_matrix((data[keep], (coo.row[keep], coo.col[keep])), shape=coo.shape, dtype=np.uint8)
    out = out.tocsr()
    out.sort_indices()
    return out


def clip_probabilities(priors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.clip(np.asarray(priors, dtype=np.float64), float(eps), 1.0 - float(eps))


def llr_from_priors(priors: np.ndarray) -> np.ndarray:
    clipped = clip_probabilities(priors)
    return np.log((1.0 - clipped) / clipped)


@dataclass(frozen=True)
class DecoderConfig:
    max_iter: int = 60
    schedule: str = "layered"
    damping: float = 0.0
    normalization: float = 1.0
    offset: float = 0.0
    llr_clip: float = 30.0
    self_corrected: bool = False

    def normalized_schedule(self) -> str:
        schedule = str(self.schedule).strip().lower()
        if schedule == "serial":
            return "layered"
        return schedule

    def validate(self, algorithm: str) -> None:
        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be > 0")
        if self.normalized_schedule() not in {"layered", "flooding"}:
            raise ValueError("schedule must be one of layered, serial, flooding")
        if not (0.0 <= float(self.damping) < 1.0):
            raise ValueError("damping must lie in [0, 1)")
        if float(self.llr_clip) <= 0.0:
            raise ValueError("llr_clip must be > 0")
        if bool(self.self_corrected) and str(algorithm) != "minsum":
            raise ValueError("self_corrected is only supported for min-sum")
        if str(algorithm) == "minsum":
            if float(self.normalization) <= 0.0:
                raise ValueError("normalization must be > 0 for min-sum")
            if float(self.offset) < 0.0:
                raise ValueError("offset must be >= 0 for min-sum")


@dataclass(frozen=True)
class WindowConfig:
    window_size: int
    overlap_size: int = 0
    commit_size: Optional[int] = None

    def effective_commit_size(self) -> int:
        commit = int(self.window_size) - int(self.overlap_size) if self.commit_size is None else int(self.commit_size)
        return int(commit)

    def validate(self) -> None:
        if int(self.window_size) <= 0:
            raise ValueError("window_size must be > 0")
        if int(self.overlap_size) < 0:
            raise ValueError("overlap_size must be >= 0")
        commit = self.effective_commit_size()
        if commit <= 0:
            raise ValueError("commit_size must be > 0")
        if commit + int(self.overlap_size) != int(self.window_size):
            raise ValueError("commit_size + overlap_size must equal window_size")


@dataclass
class WindowStepResult:
    window_index: int
    column_start: int
    column_end: int
    commit_end_requested: int
    commit_end_actual: int
    overlap_start: int
    overlap_end: int
    active_row_count: int
    iterations: int
    converged: bool
    unsatisfied_checks: int


@dataclass
class SideDecodeResult:
    estimate: np.ndarray
    posterior_llr: np.ndarray
    converged: bool
    iterations: int
    unsatisfied_checks: int
    unsatisfied_vector: np.ndarray
    logical_action: np.ndarray
    window_steps: list[WindowStepResult] = field(default_factory=list)


@dataclass
class FrameDecodeResult:
    x: SideDecodeResult
    z: SideDecodeResult
    converged: bool
    logical_frame_action: dict[str, np.ndarray]
    unsatisfied_checks: dict[str, int]
    iterations: dict[str, int]


@dataclass(frozen=True)
class MessagePassingResult:
    estimate: np.ndarray
    posterior_llr: np.ndarray
    converged: bool
    iterations: int
    residual: np.ndarray
    erased_edge_count_by_iter: tuple[int, ...] = ()
    erased_edge_total: int = 0


@dataclass(frozen=True)
class TannerGraph:
    H: sp.csr_matrix
    H_csc: sp.csc_matrix
    edge_var: np.ndarray
    edge_check: np.ndarray
    check_to_edges: tuple[np.ndarray, ...]
    var_to_edges: tuple[np.ndarray, ...]

    @classmethod
    def from_csr(cls, matrix: sp.spmatrix) -> "TannerGraph":
        H = binary_csr_mod2(matrix).tocsr()
        H.sort_indices()
        H_csc = H.tocsc()
        edge_var = H.indices.astype(np.int32, copy=True)
        edge_check = np.empty(edge_var.size, dtype=np.int32)
        check_to_edges: list[np.ndarray] = []
        for check in range(int(H.shape[0])):
            start = int(H.indptr[check])
            stop = int(H.indptr[check + 1])
            edges = np.arange(start, stop, dtype=np.int32)
            check_to_edges.append(edges)
            edge_check[start:stop] = check
        var_edges: list[list[int]] = [[] for _ in range(int(H.shape[1]))]
        for edge_index, var_index in enumerate(edge_var.tolist()):
            var_edges[int(var_index)].append(int(edge_index))
        return cls(
            H=H,
            H_csc=H_csc,
            edge_var=edge_var,
            edge_check=edge_check,
            check_to_edges=tuple(np.asarray(edges, dtype=np.int32) for edges in check_to_edges),
            var_to_edges=tuple(np.asarray(edges, dtype=np.int32) for edges in var_edges),
        )

    @property
    def m(self) -> int:
        return int(self.H.shape[0])

    @property
    def n(self) -> int:
        return int(self.H.shape[1])

    @property
    def n_edges(self) -> int:
        return int(self.edge_var.size)

    def syndrome_from_bits(self, bits: np.ndarray) -> np.ndarray:
        data = np.asarray(bits, dtype=np.uint8).reshape(-1)
        if int(data.size) != self.n:
            raise ValueError(f"bit vector length mismatch: got {data.size}, expected {self.n}")
        out = np.zeros(self.m, dtype=np.uint8)
        np.bitwise_xor.at(out, self.edge_check, data[self.edge_var] & 1)
        return out


@dataclass(frozen=True)
class SideContext:
    name: str
    graph: TannerGraph
    observables: sp.csr_matrix
    priors: np.ndarray
    prior_llr: np.ndarray
    row_min_col: np.ndarray
    row_max_col: np.ndarray
    col_forward_reach: np.ndarray
    col_to_checks: tuple[np.ndarray, ...]

    @classmethod
    def from_matrices(
        cls,
        *,
        name: str,
        check_matrix: sp.spmatrix,
        observables: Optional[sp.spmatrix],
        priors: np.ndarray,
    ) -> "SideContext":
        graph = TannerGraph.from_csr(check_matrix)
        obs = sp.csr_matrix((0, graph.n), dtype=np.uint8) if observables is None else binary_csr_mod2(observables)
        obs = obs.tocsr()
        clipped_priors = clip_probabilities(np.asarray(priors, dtype=np.float64).reshape(-1))
        if int(clipped_priors.size) != graph.n:
            raise ValueError(f"prior vector length mismatch: got {clipped_priors.size}, expected {graph.n}")

        row_min = np.full(graph.m, graph.n, dtype=np.int32)
        row_max = np.full(graph.m, -1, dtype=np.int32)
        for row in range(graph.m):
            start = int(graph.H.indptr[row])
            stop = int(graph.H.indptr[row + 1])
            cols = graph.H.indices[start:stop]
            if cols.size:
                row_min[row] = int(cols[0])
                row_max[row] = int(cols[-1])

        col_to_checks: list[np.ndarray] = []
        col_forward = np.zeros(graph.n, dtype=np.int32)
        for col in range(graph.n):
            start = int(graph.H_csc.indptr[col])
            stop = int(graph.H_csc.indptr[col + 1])
            checks = graph.H_csc.indices[start:stop].astype(np.int32, copy=True)
            col_to_checks.append(checks)
            if checks.size:
                col_forward[col] = int(np.max(row_max[checks]))
            else:
                col_forward[col] = int(col)

        return cls(
            name=str(name),
            graph=graph,
            observables=obs,
            priors=clipped_priors,
            prior_llr=llr_from_priors(clipped_priors),
            row_min_col=row_min,
            row_max_col=row_max,
            col_forward_reach=col_forward,
            col_to_checks=tuple(col_to_checks),
        )

    @property
    def H(self) -> sp.csr_matrix:
        return self.graph.H

    @property
    def n(self) -> int:
        return self.graph.n

    @property
    def m(self) -> int:
        return self.graph.m

    def resolve_prior_llr(self, prior_llr: Optional[np.ndarray]) -> np.ndarray:
        if prior_llr is None:
            return np.asarray(self.prior_llr, dtype=np.float64).copy()
        out = np.asarray(prior_llr, dtype=np.float64).reshape(-1).copy()
        if int(out.size) != self.n:
            raise ValueError(f"prior_llr length mismatch: got {out.size}, expected {self.n}")
        return out

    def syndrome(self, bits: np.ndarray) -> np.ndarray:
        return self.graph.syndrome_from_bits(bits)

    def logical_action_for(self, bits: np.ndarray) -> np.ndarray:
        if int(self.observables.shape[0]) == 0:
            return np.zeros(0, dtype=np.uint8)
        vec = np.asarray(bits, dtype=np.uint8).reshape(-1)
        return np.asarray(self.observables.dot(vec) % 2, dtype=np.uint8).reshape(-1)

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        error = (rng.random(self.n) < self.priors).astype(np.uint8)
        return error, self.syndrome(error), self.logical_action_for(error)

    def fold_columns_into_syndrome(
        self,
        syndrome: np.ndarray,
        *,
        columns: Sequence[int],
        values: Sequence[int],
    ) -> np.ndarray:
        out = np.asarray(syndrome, dtype=np.uint8).reshape(-1).copy()
        for column, value in zip(columns, values):
            if int(value) & 1:
                checks = self.col_to_checks[int(column)]
                if checks.size:
                    out[checks] ^= 1
        return out


class SplitFrameDecoder:
    def __init__(self, *, x_decoder: object, z_decoder: object) -> None:
        self.x_decoder = x_decoder
        self.z_decoder = z_decoder

    def decode(
        self,
        *,
        x_syndrome: np.ndarray,
        z_syndrome: np.ndarray,
        x_prior_llr: Optional[np.ndarray] = None,
        z_prior_llr: Optional[np.ndarray] = None,
    ) -> FrameDecodeResult:
        x_result = self.x_decoder.decode(np.asarray(x_syndrome, dtype=np.uint8), prior_llr=x_prior_llr)
        z_result = self.z_decoder.decode(np.asarray(z_syndrome, dtype=np.uint8), prior_llr=z_prior_llr)
        return FrameDecodeResult(
            x=x_result,
            z=z_result,
            converged=bool(x_result.converged and z_result.converged),
            logical_frame_action={
                "x": np.asarray(x_result.logical_action, dtype=np.uint8),
                "z": np.asarray(z_result.logical_action, dtype=np.uint8),
            },
            unsatisfied_checks={
                "x": int(x_result.unsatisfied_checks),
                "z": int(z_result.unsatisfied_checks),
            },
            iterations={
                "x": int(x_result.iterations),
                "z": int(z_result.iterations),
            },
        )


def _phi(values: np.ndarray, llr_clip: float) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-12, float(llr_clip))
    tanh_vals = np.tanh(clipped / 2.0)
    tanh_vals = np.clip(tanh_vals, 1e-12, 1.0 - 1e-12)
    return -np.log(tanh_vals)


def _phi_inverse(values: np.ndarray, llr_clip: float) -> np.ndarray:
    expo = np.exp(-np.clip(np.asarray(values, dtype=np.float64), 0.0, float(llr_clip)))
    expo = np.clip(expo, 1e-12, 1.0 - 1e-12)
    out = 2.0 * np.arctanh(expo)
    return np.clip(out, 0.0, float(llr_clip))


def _check_update_bp(
    incoming_v2c: np.ndarray,
    syndrome_bit: int,
    old_c2v: np.ndarray,
    *,
    damping: float,
    llr_clip: float,
) -> np.ndarray:
    degree = int(incoming_v2c.size)
    if degree == 0:
        return old_c2v.copy()

    parity_sign = -1.0 if (int(syndrome_bit) & 1) else 1.0
    if degree == 1:
        raw = np.asarray([parity_sign * float(llr_clip)], dtype=np.float64)
        if float(damping) > 0.0:
            return (1.0 - float(damping)) * raw + float(damping) * old_c2v
        return raw

    signs = np.where(np.asarray(incoming_v2c, dtype=np.float64) >= 0.0, 1.0, -1.0)
    abs_vals = np.abs(np.asarray(incoming_v2c, dtype=np.float64))
    phi_vals = _phi(abs_vals, llr_clip)
    total_phi = float(np.sum(phi_vals))
    mags = _phi_inverse(total_phi - phi_vals, llr_clip)
    prod_sign = parity_sign * float(np.prod(signs))
    raw = prod_sign * signs * mags
    if float(damping) > 0.0:
        return (1.0 - float(damping)) * raw + float(damping) * old_c2v
    return raw


def _check_update_minsum(
    incoming_v2c: np.ndarray,
    syndrome_bit: int,
    old_c2v: np.ndarray,
    *,
    normalization: float,
    offset: float,
    damping: float,
) -> np.ndarray:
    degree = int(incoming_v2c.size)
    if degree == 0:
        return old_c2v.copy()

    signs = np.where(np.asarray(incoming_v2c, dtype=np.float64) >= 0.0, 1.0, -1.0)
    abs_vals = np.abs(np.asarray(incoming_v2c, dtype=np.float64))
    parity_sign = -1.0 if (int(syndrome_bit) & 1) else 1.0
    prod_sign = parity_sign * float(np.prod(signs))

    if degree == 1:
        mag = float(normalization) * max(float(abs_vals[0]) - float(offset), 0.0)
        raw = np.asarray([prod_sign * signs[0] * mag], dtype=np.float64)
        if float(damping) > 0.0:
            return (1.0 - float(damping)) * raw + float(damping) * old_c2v
        return raw

    idx_min = int(np.argmin(abs_vals))
    min1 = float(abs_vals[idx_min])
    masked = abs_vals.copy()
    masked[idx_min] = np.inf
    min2 = float(np.min(masked))
    mags = np.full(degree, float(normalization) * max(min1 - float(offset), 0.0), dtype=np.float64)
    mags[idx_min] = float(normalization) * max(min2 - float(offset), 0.0)
    raw = prod_sign * signs * mags
    if float(damping) > 0.0:
        return (1.0 - float(damping)) * raw + float(damping) * old_c2v
    return raw


def _same_sign_or_zero_mask(candidate: np.ndarray, previous: np.ndarray) -> np.ndarray:
    cand = np.asarray(candidate, dtype=np.float64)
    prev = np.asarray(previous, dtype=np.float64)
    return (
        (cand == 0.0)
        | (prev == 0.0)
        | ((cand > 0.0) & (prev > 0.0))
        | ((cand < 0.0) & (prev < 0.0))
    )


def _apply_scms_erasure(candidate: np.ndarray, previous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = _same_sign_or_zero_mask(candidate, previous)
    updated = np.where(keep, np.asarray(candidate, dtype=np.float64), 0.0)
    return updated, np.logical_not(keep)


def run_message_passing_result(
    *,
    graph: TannerGraph,
    syndrome_bits: np.ndarray,
    prior_llr: np.ndarray,
    config: DecoderConfig,
    algorithm: str,
) -> MessagePassingResult:
    config.validate(algorithm)
    target = np.asarray(syndrome_bits, dtype=np.uint8).reshape(-1) & 1
    if int(target.size) != graph.m:
        raise ValueError(f"syndrome length mismatch: got {target.size}, expected {graph.m}")

    prior = np.asarray(prior_llr, dtype=np.float64).reshape(-1).copy()
    if int(prior.size) != graph.n:
        raise ValueError(f"prior_llr length mismatch: got {prior.size}, expected {graph.n}")

    if graph.n_edges == 0:
        hard = (prior < 0.0).astype(np.uint8)
        residual = (graph.syndrome_from_bits(hard) ^ target).astype(np.uint8)
        return MessagePassingResult(
            estimate=hard,
            posterior_llr=prior,
            converged=bool(np.count_nonzero(residual) == 0),
            iterations=0,
            residual=residual,
        )

    if str(algorithm) == "bp":
        update = lambda incoming, syndrome_bit, old: _check_update_bp(  # noqa: E731
            incoming,
            syndrome_bit,
            old,
            damping=float(config.damping),
            llr_clip=float(config.llr_clip),
        )
    elif str(algorithm) == "minsum":
        update = lambda incoming, syndrome_bit, old: _check_update_minsum(  # noqa: E731
            incoming,
            syndrome_bit,
            old,
            normalization=float(config.normalization),
            offset=float(config.offset),
            damping=float(config.damping),
        )
    else:
        raise ValueError(f"unsupported algorithm '{algorithm}'")

    m_cv = np.zeros(graph.n_edges, dtype=np.float64)
    schedule = config.normalized_schedule()
    converged = False
    iterations = 0
    erased_edge_count_by_iter: list[int] = []
    use_scms = bool(config.self_corrected) and str(algorithm) == "minsum"

    if schedule == "flooding":
        m_vc = prior[graph.edge_var].copy()
        llr = prior.copy()
        residual = np.ones(graph.m, dtype=np.uint8)
        for it in range(1, int(config.max_iter) + 1):
            iterations = int(it)
            new_cv = m_cv.copy()
            for check, edges in enumerate(graph.check_to_edges):
                if edges.size == 0:
                    continue
                new_cv[edges] = update(m_vc[edges], int(target[check]), m_cv[edges])
            m_cv = new_cv
            llr = prior.copy()
            np.add.at(llr, graph.edge_var, m_cv)
            hard = (llr < 0.0).astype(np.uint8)
            residual = (graph.syndrome_from_bits(hard) ^ target).astype(np.uint8)
            candidate_vc = llr[graph.edge_var] - m_cv
            if use_scms:
                m_vc, erased_mask = _apply_scms_erasure(candidate_vc, m_vc)
                erased_edge_count_by_iter.append(int(np.count_nonzero(erased_mask)))
            else:
                m_vc = candidate_vc
            if int(np.count_nonzero(residual)) == 0:
                converged = True
                break
        return MessagePassingResult(
            estimate=hard,
            posterior_llr=llr,
            converged=converged,
            iterations=iterations,
            residual=residual,
            erased_edge_count_by_iter=tuple(erased_edge_count_by_iter),
            erased_edge_total=int(sum(erased_edge_count_by_iter)),
        )

    if use_scms:
        m_vc = prior[graph.edge_var].copy()
        llr = prior.copy()
        residual = np.ones(graph.m, dtype=np.uint8)
        for it in range(1, int(config.max_iter) + 1):
            iterations = int(it)
            v2c_snapshot = m_vc.copy()
            erased_mask_iter = np.zeros(graph.n_edges, dtype=bool)
            for check, edges in enumerate(graph.check_to_edges):
                if edges.size == 0:
                    continue
                vars_for_check = graph.edge_var[edges]
                new = update(m_vc[edges], int(target[check]), m_cv[edges])
                delta = new - m_cv[edges]
                m_cv[edges] = new
                np.add.at(llr, vars_for_check, delta)
                for var in np.unique(vars_for_check):
                    var_index = int(var)
                    var_edges = graph.var_to_edges[var_index]
                    candidate = llr[var_index] - m_cv[var_edges]
                    updated, erased_mask = _apply_scms_erasure(candidate, v2c_snapshot[var_edges])
                    m_vc[var_edges] = updated
                    erased_mask_iter[var_edges] = erased_mask
            hard = (llr < 0.0).astype(np.uint8)
            residual = (graph.syndrome_from_bits(hard) ^ target).astype(np.uint8)
            erased_edge_count_by_iter.append(int(np.count_nonzero(erased_mask_iter)))
            if int(np.count_nonzero(residual)) == 0:
                converged = True
                break
        return MessagePassingResult(
            estimate=hard,
            posterior_llr=llr,
            converged=converged,
            iterations=iterations,
            residual=residual,
            erased_edge_count_by_iter=tuple(erased_edge_count_by_iter),
            erased_edge_total=int(sum(erased_edge_count_by_iter)),
        )

    llr = prior.copy()
    residual = np.ones(graph.m, dtype=np.uint8)
    for it in range(1, int(config.max_iter) + 1):
        iterations = int(it)
        for check, edges in enumerate(graph.check_to_edges):
            if edges.size == 0:
                continue
            vars_for_check = graph.edge_var[edges]
            incoming = llr[vars_for_check] - m_cv[edges]
            new = update(incoming, int(target[check]), m_cv[edges])
            delta = new - m_cv[edges]
            m_cv[edges] = new
            np.add.at(llr, vars_for_check, delta)
        hard = (llr < 0.0).astype(np.uint8)
        residual = (graph.syndrome_from_bits(hard) ^ target).astype(np.uint8)
        if int(np.count_nonzero(residual)) == 0:
            converged = True
            break
    return MessagePassingResult(
        estimate=hard,
        posterior_llr=llr,
        converged=converged,
        iterations=iterations,
        residual=residual,
        erased_edge_count_by_iter=tuple(erased_edge_count_by_iter),
        erased_edge_total=int(sum(erased_edge_count_by_iter)),
    )


def run_message_passing(
    *,
    graph: TannerGraph,
    syndrome_bits: np.ndarray,
    prior_llr: np.ndarray,
    config: DecoderConfig,
    algorithm: str,
) -> tuple[np.ndarray, np.ndarray, bool, int, np.ndarray]:
    result = run_message_passing_result(
        graph=graph,
        syndrome_bits=syndrome_bits,
        prior_llr=prior_llr,
        config=config,
        algorithm=algorithm,
    )
    return result.estimate, result.posterior_llr, result.converged, result.iterations, result.residual


def load_dem_side_from_stim(
    *,
    name: str,
    stim_path: Path,
    expected_shape: Optional[tuple[int, int]] = None,
) -> SideContext:
    stim = importlib.import_module("stim")
    dem_mod = importlib.import_module("ldpc.ckt_noise.dem_matrices")
    detector_error_model_to_check_matrices = getattr(dem_mod, "detector_error_model_to_check_matrices")

    circuit = stim.Circuit.from_file(str(Path(stim_path)))
    dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    mats = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
    check_matrix = binary_csr_mod2(mats.check_matrix.tocsr().astype(np.uint8))
    observables = binary_csr_mod2(mats.observables_matrix.tocsr().astype(np.uint8))
    if expected_shape is not None and tuple(map(int, check_matrix.shape)) != tuple(map(int, expected_shape)):
        raise ValueError(
            f"{name}: expected check matrix shape {tuple(map(int, expected_shape))}, "
            f"got {tuple(map(int, check_matrix.shape))}"
        )
    return SideContext.from_matrices(
        name=str(name),
        check_matrix=check_matrix,
        observables=observables,
        priors=np.asarray(mats.priors, dtype=np.float64).reshape(-1),
    )
