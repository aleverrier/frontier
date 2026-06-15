from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp


def _as_int32_vector(values: np.ndarray | list[int] | tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int32).reshape(-1)
    return np.ascontiguousarray(arr)


def _as_uint8_vector(values: np.ndarray | list[int] | tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint8).reshape(-1)
    return np.ascontiguousarray(arr)


def _as_float64_vector(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.ascontiguousarray(arr)


def build_hybrid_orig_orders(
    *,
    n_rows: int,
    hybrid_aux_row_count: int,
    max_iter: int,
    schedule_seed: int,
) -> np.ndarray:
    row_count = int(n_rows)
    aux_rows = int(hybrid_aux_row_count)
    iters = int(max_iter)
    if row_count < 0 or aux_rows < 0 or aux_rows > row_count:
        raise ValueError("invalid row counts for hybrid order generation")
    if iters <= 0:
        raise ValueError("max_iter must be positive")
    orig_rows = np.arange(aux_rows, row_count, dtype=np.int32)
    if int(orig_rows.size) == 0:
        return np.empty(0, dtype=np.int32)
    if int(orig_rows.size) == 1:
        return np.ascontiguousarray(np.tile(orig_rows, int(iters)))
    rng = np.random.default_rng(int(schedule_seed))
    orders = np.empty((int(iters), int(orig_rows.size)), dtype=np.int32)
    for idx in range(int(iters)):
        orders[idx] = np.asarray(rng.permutation(orig_rows), dtype=np.int32)
    return np.ascontiguousarray(orders.reshape(-1))


def build_hybrid_orig_orders_batch(
    *,
    n_rows: int,
    hybrid_aux_row_count: int,
    max_iter: int,
    shot_seed: int,
    view_seed_offset: int,
    ensemble_size: int,
) -> np.ndarray:
    row_count = int(n_rows)
    aux_rows = int(hybrid_aux_row_count)
    iters = int(max_iter)
    branch_count = int(ensemble_size)
    if row_count < 0 or aux_rows < 0 or aux_rows > row_count:
        raise ValueError("invalid row counts for hybrid batch order generation")
    if iters <= 0:
        raise ValueError("max_iter must be positive")
    if branch_count <= 0:
        raise ValueError("ensemble_size must be positive")
    orig_rows = np.arange(aux_rows, row_count, dtype=np.int32)
    n_orig = int(orig_rows.size)
    if n_orig == 0:
        return np.empty(0, dtype=np.int32)
    if n_orig == 1:
        return np.ascontiguousarray(np.tile(orig_rows, int(branch_count * iters)))

    out = np.empty((int(branch_count), int(iters), int(n_orig)), dtype=np.int32)
    for branch_idx in range(int(branch_count)):
        branch_seed = int(shot_seed) * 1_000_003 + int(view_seed_offset) * 1009 + int(branch_idx) * 104729
        rng = np.random.default_rng(int(branch_seed))
        for iter_idx in range(int(iters)):
            out[branch_idx, iter_idx] = np.asarray(rng.permutation(orig_rows), dtype=np.int32)
    return np.ascontiguousarray(out.reshape(-1))


def paper_hybrid_aux_layer_bounds_from_meta(meta: Mapping[str, object]) -> tuple[int, ...]:
    if "aux_u_row_slice" not in meta or "aux_v_row_slice" not in meta:
        return ()
    bounds: list[int] = []
    u_slice = [int(x) for x in meta["aux_u_row_slice"]]
    v_slice = [int(x) for x in meta["aux_v_row_slice"]]
    if len(u_slice) == 2 and int(u_slice[0]) == 0 and int(u_slice[1]) > 0:
        bounds.append(int(u_slice[1]))
    if len(v_slice) == 2 and int(v_slice[1]) > int(v_slice[0]):
        bounds.append(int(v_slice[1]))
    return tuple(bounds)


@dataclass(frozen=True)
class NativeCsrMatrix:
    n_rows: int
    n_cols: int
    indptr: np.ndarray
    indices: np.ndarray

    @classmethod
    def from_scipy(cls, matrix: sp.spmatrix) -> "NativeCsrMatrix":
        csr = matrix.tocsr()
        return cls(
            n_rows=int(csr.shape[0]),
            n_cols=int(csr.shape[1]),
            indptr=_as_int32_vector(csr.indptr),
            indices=_as_int32_vector(csr.indices),
        )

    @property
    def nnz(self) -> int:
        return int(self.indices.size)

    def validate(self) -> None:
        if int(self.n_rows) < 0 or int(self.n_cols) < 0:
            raise ValueError("matrix dimensions must be non-negative")
        if self.indptr.dtype != np.int32 or self.indices.dtype != np.int32:
            raise ValueError("native CSR arrays must use int32")
        if self.indptr.ndim != 1 or self.indices.ndim != 1:
            raise ValueError("native CSR arrays must be 1-D")
        if int(self.indptr.size) != int(self.n_rows) + 1:
            raise ValueError("indptr length must be n_rows + 1")
        if int(self.indptr[0]) != 0:
            raise ValueError("indptr must start at 0")
        if int(self.indptr[-1]) != int(self.indices.size):
            raise ValueError("indptr must end at nnz")
        if np.any(self.indptr[1:] < self.indptr[:-1]):
            raise ValueError("indptr must be nondecreasing")
        if np.any(self.indices < 0) or np.any(self.indices >= int(self.n_cols)):
            raise ValueError("CSR column indices out of range")


@dataclass(frozen=True)
class NativePaperMinSumConfig:
    alpha: float = 0.96875
    beta: float = 0.0
    damp: float = 0.0
    max_iter: int = 400
    schedule: str = "hybrid_serial_layered"

    def validate(self) -> None:
        if not (0.0 < float(self.alpha) <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        if abs(float(self.beta)) > 0.0:
            raise ValueError("native paper backend v1 only supports beta=0")
        if abs(float(self.damp)) > 0.0:
            raise ValueError("native paper backend v1 only supports damp=0")
        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be positive")
        if str(self.schedule) != "hybrid_serial_layered":
            raise ValueError("native paper backend v1 only supports hybrid_serial_layered")


@dataclass(frozen=True)
class NativePaperGariProblem:
    label: str
    check_matrix: NativeCsrMatrix
    observables_top: NativeCsrMatrix
    observables_bottom: NativeCsrMatrix
    prior_llr: np.ndarray
    selection_prior_llr: np.ndarray
    top_rows: np.ndarray
    bottom_rows: np.ndarray
    hybrid_aux_row_count: int
    hybrid_aux_layer_bounds: tuple[int, ...]

    def validate(self) -> None:
        self.check_matrix.validate()
        self.observables_top.validate()
        self.observables_bottom.validate()
        if self.prior_llr.dtype != np.float64 or self.selection_prior_llr.dtype != np.float64:
            raise ValueError("prior LLR arrays must use float64")
        if self.top_rows.dtype != np.int32 or self.bottom_rows.dtype != np.int32:
            raise ValueError("row index arrays must use int32")
        if int(self.prior_llr.size) != int(self.check_matrix.n_cols):
            raise ValueError("prior_llr size must equal number of columns")
        if int(self.selection_prior_llr.size) != int(self.check_matrix.n_cols):
            raise ValueError("selection_prior_llr size must equal number of columns")
        if int(self.hybrid_aux_row_count) < 0 or int(self.hybrid_aux_row_count) > int(self.check_matrix.n_rows):
            raise ValueError("hybrid_aux_row_count out of range")
        if self.hybrid_aux_layer_bounds:
            if int(self.hybrid_aux_layer_bounds[-1]) != int(self.hybrid_aux_row_count):
                raise ValueError("hybrid_aux_layer_bounds must end at hybrid_aux_row_count")
            if any(int(a) >= int(b) for a, b in zip(self.hybrid_aux_layer_bounds, self.hybrid_aux_layer_bounds[1:])):
                raise ValueError("hybrid_aux_layer_bounds must be strictly increasing")


@dataclass(frozen=True)
class NativeDecodeRequest:
    syndrome: np.ndarray
    success_rows: np.ndarray
    p_scalar: float
    max_iter: int
    schedule_seed: int

    def validate(self, *, n_rows: int) -> None:
        if self.syndrome.dtype != np.uint8:
            raise ValueError("syndrome must use uint8")
        if self.success_rows.dtype != np.int32:
            raise ValueError("success_rows must use int32")
        if int(self.syndrome.size) != int(n_rows):
            raise ValueError("syndrome length must equal check row count")
        if np.any(self.success_rows < 0) or np.any(self.success_rows >= int(n_rows)):
            raise ValueError("success_rows out of range")
        if not (0.0 < float(self.p_scalar) < 1.0):
            raise ValueError("p_scalar must be in (0, 1)")
        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be positive")


@dataclass(frozen=True)
class NativeEnsembleRequest:
    syndrome: np.ndarray
    success_rows: np.ndarray
    p_scalar: float
    max_iter: int
    shot_seed: int
    view_seed_offset: int
    ensemble_size: int

    def validate(self, *, n_rows: int) -> None:
        if self.syndrome.dtype != np.uint8:
            raise ValueError("syndrome must use uint8")
        if self.success_rows.dtype != np.int32:
            raise ValueError("success_rows must use int32")
        if int(self.syndrome.size) != int(n_rows):
            raise ValueError("syndrome length must equal check row count")
        if np.any(self.success_rows < 0) or np.any(self.success_rows >= int(n_rows)):
            raise ValueError("success_rows out of range")
        if not (0.0 < float(self.p_scalar) < 1.0):
            raise ValueError("p_scalar must be in (0, 1)")
        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be positive")
        if int(self.ensemble_size) <= 0:
            raise ValueError("ensemble_size must be positive")


@dataclass(frozen=True)
class NativeBranchResult:
    e_hat: np.ndarray
    converged: bool
    iterations: int
    active_edge_count: int


@dataclass(frozen=True)
class NativeEnsembleResult:
    chosen_e_hat: np.ndarray | None
    converged: bool
    latency_iters: int
    total_iters: int
    total_edge_work: int
    branches_converged: int
    tie_candidates: int
    exception_branches: int


@runtime_checkable
class NativePaperMinSumBackend(Protocol):
    def supports_config(self, config: NativePaperMinSumConfig) -> bool: ...

    def decode(self, problem: NativePaperGariProblem, request: NativeDecodeRequest) -> NativeBranchResult: ...

    def decode_ensemble(self, problem: NativePaperGariProblem, request: NativeEnsembleRequest) -> NativeEnsembleResult: ...


def load_native_backend(module_name: str = "grosscode_native") -> NativePaperMinSumBackend | None:
    try:
        module = importlib.import_module(str(module_name))
    except ImportError:
        return None
    backend = getattr(module, "PaperGariMinSumBackend", None)
    if backend is None:
        raise RuntimeError(f"{module_name!r} does not expose PaperGariMinSumBackend")
    try:
        return backend()
    except RuntimeError:
        return None


def build_native_problem(
    *,
    label: str,
    matrix: sp.spmatrix,
    obs_top: sp.spmatrix,
    obs_bottom: sp.spmatrix,
    prior_llr: np.ndarray,
    selection_prior_llr: np.ndarray,
    meta: Mapping[str, object],
    top_rows: np.ndarray,
    bottom_rows: np.ndarray,
) -> NativePaperGariProblem:
    problem = NativePaperGariProblem(
        label=str(label),
        check_matrix=NativeCsrMatrix.from_scipy(matrix),
        observables_top=NativeCsrMatrix.from_scipy(obs_top),
        observables_bottom=NativeCsrMatrix.from_scipy(obs_bottom),
        prior_llr=_as_float64_vector(prior_llr),
        selection_prior_llr=_as_float64_vector(selection_prior_llr),
        top_rows=_as_int32_vector(top_rows),
        bottom_rows=_as_int32_vector(bottom_rows),
        hybrid_aux_row_count=int(meta.get("hybrid_aux_row_count", 0)),
        hybrid_aux_layer_bounds=paper_hybrid_aux_layer_bounds_from_meta(meta),
    )
    problem.validate()
    return problem


def build_decode_request(
    *,
    syndrome: np.ndarray,
    success_rows: np.ndarray,
    p_scalar: float,
    max_iter: int,
    schedule_seed: int,
    n_rows: int,
) -> NativeDecodeRequest:
    request = NativeDecodeRequest(
        syndrome=_as_uint8_vector(syndrome),
        success_rows=_as_int32_vector(success_rows),
        p_scalar=float(p_scalar),
        max_iter=int(max_iter),
        schedule_seed=int(schedule_seed),
    )
    request.validate(n_rows=int(n_rows))
    return request


def build_ensemble_request(
    *,
    syndrome: np.ndarray,
    success_rows: np.ndarray,
    p_scalar: float,
    max_iter: int,
    shot_seed: int,
    view_seed_offset: int,
    ensemble_size: int,
    n_rows: int,
) -> NativeEnsembleRequest:
    request = NativeEnsembleRequest(
        syndrome=_as_uint8_vector(syndrome),
        success_rows=_as_int32_vector(success_rows),
        p_scalar=float(p_scalar),
        max_iter=int(max_iter),
        shot_seed=int(shot_seed),
        view_seed_offset=int(view_seed_offset),
        ensemble_size=int(ensemble_size),
    )
    request.validate(n_rows=int(n_rows))
    return request
