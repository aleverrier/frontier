from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Literal

import networkx as nx
import numpy as np
import scipy.sparse as sp
from ldpc import BpDecoder

from grosscode.core import DecoderConfig, TannerGraph, _apply_scms_erasure, _check_update_minsum, llr_from_priors
from grosscode.decoders.structure_aware import (
    LayeredScheduleMode,
    SectorStructureModel,
    TriangleFactorizationMode,
    actual_interval_from_virtual,
    apply_sector_gauge_descent,
    build_reduced_local_model,
    build_sector_structure_model,
    resolve_schedule_direction,
    solve_reduced_local_model,
    virtual_column_span_arrays,
    virtual_commit_mask,
    virtual_separator_local_columns,
)
from grosscode.dem.builder import SplitSectorMetadata, SplitSectorProblem
from grosscode.utils.gf2 import csr_matvec_mod2


RelayFlipMode = Literal["erase", "blend", "damp"]
RelayCandidateScore = Literal["final_absllr", "mean_absllr", "temporal_instability", "instability_plus_residual"]
DecoderName = Literal[
    "bp",
    "minsum",
    "self_corrected_minsum",
    "local_round",
    "separator_wavefront",
    "round_wavefront",
    "relay_minsum",
    "separator_wavefront_sa",
    "round_wavefront_sa",
]


@dataclass(frozen=True)
class SplitSectorSyndrome:
    X: np.ndarray
    Z: np.ndarray


@dataclass(frozen=True)
class SplitSectorPriors:
    X: np.ndarray
    Z: np.ndarray


@dataclass(frozen=True)
class DecoderWindow:
    round_radius: int = 1
    max_passes: int = 3
    max_iter: int = 40
    separator_window_rounds: int = 3
    separator_overlap_rounds: int = 1
    separator_topk: int = 2
    separator_max_branches: int = 4
    separator_max_window_expansions: int = 2
    separator_mean_tail: int = 8
    separator_reliable_shell_hops: int = 1
    separator_reliable_topk: int = 6
    separator_reliable_abs_mean_threshold: float = 6.0
    relay_enable: bool = False
    relay_trigger_residual_stall_rounds: int = 4
    relay_trigger_rebound: int = 1
    relay_gamma_stable: float = 0.15
    relay_gamma_bulk: float = 0.0
    relay_gamma_frontier: float = -0.15
    relay_clip_B: float = 12.0
    relay_frontier_shell_radius: int = 1
    relay_flip_mode: RelayFlipMode = "erase"
    relay_flip_kappa: float = 0.25
    relay_flip_eta: float = 0.5
    relay_candidate_score: RelayCandidateScore = "instability_plus_residual"
    relay_num_legs: int = 1
    relay_leg_iters: int = 8
    triangle_factorization: TriangleFactorizationMode = "off"
    layered_schedule: LayeredScheduleMode = "off"
    gauge_descent: bool = False
    gauge_descent_max_iterations: int = 128


@dataclass(frozen=True)
class _RelayOptions:
    enabled: bool
    trigger_residual_stall_rounds: int
    trigger_rebound: int
    gamma_stable: float
    gamma_bulk: float
    gamma_frontier: float
    clip_B: float
    frontier_shell_radius: int
    flip_mode: RelayFlipMode
    flip_kappa: float
    flip_eta: float
    candidate_score: RelayCandidateScore
    num_legs: int
    leg_iters: int

    @classmethod
    def from_window(cls, window: DecoderWindow, *, force_enable: bool = False) -> "_RelayOptions":
        return cls(
            enabled=bool(force_enable or window.relay_enable),
            trigger_residual_stall_rounds=max(1, int(window.relay_trigger_residual_stall_rounds)),
            trigger_rebound=max(0, int(window.relay_trigger_rebound)),
            gamma_stable=float(window.relay_gamma_stable),
            gamma_bulk=float(window.relay_gamma_bulk),
            gamma_frontier=float(window.relay_gamma_frontier),
            clip_B=max(1.0, float(window.relay_clip_B)),
            frontier_shell_radius=max(0, int(window.relay_frontier_shell_radius)),
            flip_mode=str(window.relay_flip_mode),
            flip_kappa=float(window.relay_flip_kappa),
            flip_eta=float(window.relay_flip_eta),
            candidate_score=str(window.relay_candidate_score),
            num_legs=max(1, int(window.relay_num_legs)),
            leg_iters=max(1, int(window.relay_leg_iters)),
        )


@dataclass(frozen=True)
class SplitSectorDecodeResult:
    correction_X: np.ndarray
    correction_Z: np.ndarray
    logical_frame_action_X: np.ndarray
    logical_frame_action_Z: np.ndarray
    diagnostics: dict[str, object]


class SplitXZDecoder:
    def __init__(
        self,
        problem: SplitSectorProblem,
        *,
        bp_max_iter: int = 25,
        minsum_max_iter: int = 40,
        minsum_scale: float = 0.75,
        seed: int = 123,
        scms: bool = False,
    ) -> None:
        self.problem = problem
        self.bp_max_iter = int(bp_max_iter)
        self.minsum_max_iter = int(minsum_max_iter)
        self.minsum_scale = float(minsum_scale)
        self.seed = int(seed)
        self.scms = bool(scms)
        self._structure_models: dict[str, SectorStructureModel] = {}

    def _get_structure_model(
        self,
        *,
        matrix: sp.csr_matrix,
        observables: sp.csr_matrix,
        metadata: SplitSectorMetadata,
    ) -> SectorStructureModel:
        sector = str(metadata.sector)
        cached = self._structure_models.get(str(sector))
        if cached is not None:
            return cached
        model = build_sector_structure_model(
            matrix=matrix,
            observables=observables,
            metadata=metadata,
            sector=str(sector),
        )
        self._structure_models[str(sector)] = model
        return model

    def decode_split_xz(
        self,
        syndrome: SplitSectorSyndrome,
        priors: SplitSectorPriors,
        *,
        window: DecoderWindow | None = None,
        decoder: DecoderName = "bp",
    ) -> SplitSectorDecodeResult:
        correction_x, logical_x, diag_x = self._decode_sector(
            matrix=self.problem.D_X,
            observables=self.problem.O_X,
            metadata=self.problem.metadata_X,
            syndrome=np.asarray(syndrome.X, dtype=np.uint8).reshape(-1),
            priors=np.asarray(priors.X, dtype=np.float64).reshape(-1),
            decoder=decoder,
            window=window,
        )
        correction_z, logical_z, diag_z = self._decode_sector(
            matrix=self.problem.D_Z,
            observables=self.problem.O_Z,
            metadata=self.problem.metadata_Z,
            syndrome=np.asarray(syndrome.Z, dtype=np.uint8).reshape(-1),
            priors=np.asarray(priors.Z, dtype=np.float64).reshape(-1),
            decoder=decoder,
            window=window,
        )
        return SplitSectorDecodeResult(
            correction_X=correction_x,
            correction_Z=correction_z,
            logical_frame_action_X=logical_x,
            logical_frame_action_Z=logical_z,
            diagnostics={
                "decoder": decoder,
                "scms": bool(self.scms or decoder == "self_corrected_minsum"),
                "window": None if window is None else {
                    "round_radius": int(window.round_radius),
                    "max_passes": int(window.max_passes),
                    "max_iter": int(window.max_iter),
                    "separator_window_rounds": int(window.separator_window_rounds),
                    "separator_overlap_rounds": int(window.separator_overlap_rounds),
                    "separator_topk": int(window.separator_topk),
                    "separator_max_branches": int(window.separator_max_branches),
                    "separator_max_window_expansions": int(window.separator_max_window_expansions),
                    "separator_mean_tail": int(window.separator_mean_tail),
                    "separator_reliable_shell_hops": int(window.separator_reliable_shell_hops),
                    "separator_reliable_topk": int(window.separator_reliable_topk),
                    "separator_reliable_abs_mean_threshold": float(window.separator_reliable_abs_mean_threshold),
                    "relay_enable": bool(window.relay_enable),
                    "relay_trigger_residual_stall_rounds": int(window.relay_trigger_residual_stall_rounds),
                    "relay_trigger_rebound": int(window.relay_trigger_rebound),
                    "relay_gamma_stable": float(window.relay_gamma_stable),
                    "relay_gamma_bulk": float(window.relay_gamma_bulk),
                    "relay_gamma_frontier": float(window.relay_gamma_frontier),
                    "relay_clip_B": float(window.relay_clip_B),
                    "relay_frontier_shell_radius": int(window.relay_frontier_shell_radius),
                    "relay_flip_mode": str(window.relay_flip_mode),
                    "relay_flip_kappa": float(window.relay_flip_kappa),
                    "relay_flip_eta": float(window.relay_flip_eta),
                    "relay_candidate_score": str(window.relay_candidate_score),
                    "relay_num_legs": int(window.relay_num_legs),
                    "relay_leg_iters": int(window.relay_leg_iters),
                    "triangle_factorization": str(window.triangle_factorization),
                    "layered_schedule": str(window.layered_schedule),
                    "gauge_descent": bool(window.gauge_descent),
                    "gauge_descent_max_iterations": int(window.gauge_descent_max_iterations),
                },
                "sector_X": diag_x,
                "sector_Z": diag_z,
            },
        )

    def _decode_sector(
        self,
        *,
        matrix: sp.csr_matrix,
        observables: sp.csr_matrix,
        metadata: SplitSectorMetadata,
        syndrome: np.ndarray,
        priors: np.ndarray,
        decoder: DecoderName,
        window: DecoderWindow | None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        if syndrome.shape[0] != matrix.shape[0]:
            raise ValueError(f"syndrome length mismatch for sector {metadata.sector}")
        if priors.shape[0] != matrix.shape[1]:
            raise ValueError(f"priors length mismatch for sector {metadata.sector}")
        effective_window = window
        base_decoder = str(decoder)
        if str(decoder) in {"round_wavefront_sa", "separator_wavefront_sa"}:
            seed_window = window or DecoderWindow(max_iter=self.minsum_max_iter)
            effective_window = replace(
                seed_window,
                triangle_factorization=(
                    "nonoverlap"
                    if str(seed_window.triangle_factorization) == "off"
                    else str(seed_window.triangle_factorization)
                ),
                layered_schedule=(
                    "auto" if str(seed_window.layered_schedule) == "off" else str(seed_window.layered_schedule)
                ),
                gauge_descent=True if not bool(seed_window.gauge_descent) else bool(seed_window.gauge_descent),
            )
            base_decoder = "round_wavefront" if str(decoder) == "round_wavefront_sa" else "separator_wavefront"

        structure_model: SectorStructureModel | None = None
        if effective_window is not None and (
            str(effective_window.triangle_factorization) != "off"
            or bool(effective_window.gauge_descent)
            or str(effective_window.layered_schedule) != "off"
        ):
            structure_model = self._get_structure_model(
                matrix=matrix,
                observables=observables,
                metadata=metadata,
            )

        if base_decoder == "local_round":
            return self._decode_sector_local_round(
                matrix=matrix,
                observables=observables,
                metadata=metadata,
                syndrome=syndrome,
                priors=priors,
                window=effective_window or DecoderWindow(max_iter=self.minsum_max_iter),
                scms=self.scms,
            )
        if base_decoder == "round_wavefront":
            correction, logical, diagnostics = self._decode_sector_round_wavefront(
                matrix=matrix,
                observables=observables,
                metadata=metadata,
                syndrome=syndrome,
                priors=priors,
                window=effective_window or DecoderWindow(max_iter=self.minsum_max_iter),
                structure_model=structure_model,
            )
            diagnostics["structure_mode"] = str(decoder)
        elif base_decoder == "separator_wavefront":
            base_window = effective_window or DecoderWindow(max_iter=self.minsum_max_iter)
            base_correction, base_logical, base_diag = self._decode_sector_windowed(
                matrix=matrix,
                observables=observables,
                metadata=metadata,
                syndrome=syndrome,
                priors=priors,
                method="minimum_sum",
                max_iter=int(base_window.max_iter),
                window=base_window,
                scms=self.scms,
                structure_model=structure_model,
            )
            base_ok = bool(np.array_equal(csr_matvec_mod2(matrix, base_correction), syndrome))
            if base_ok:
                diag = {
                    "mode": "separator_wavefront_rescue",
                    "rescue_used": False,
                    "predecode": dict(base_diag),
                    "iterations": int(base_diag.get("iterations", 0)),
                    "converged": bool(base_diag.get("converged", False)),
                    "syndrome_ok": True,
                    "active_columns": int(base_diag.get("active_columns", 0)),
                    "residual_weight_final": int(base_diag.get("residual_weight_final", 0)),
                    "scms": bool(self.scms),
                }
                correction = base_correction
                logical = base_logical
                diagnostics = diag
            else:
                rescue_correction, rescue_logical, rescue_diag = self._decode_sector_separator_wavefront(
                    matrix=matrix,
                    observables=observables,
                    metadata=metadata,
                    syndrome=syndrome,
                    priors=priors,
                    window=base_window,
                    structure_model=structure_model,
                )
                diag = dict(rescue_diag)
                diag["mode"] = "separator_wavefront_rescue"
                diag["rescue_used"] = True
                diag["predecode"] = dict(base_diag)
                diag["iterations"] = int(base_diag.get("iterations", 0)) + int(rescue_diag.get("iterations", 0))
                diag["scms"] = bool(self.scms)
                correction = rescue_correction
                logical = rescue_logical
                diagnostics = diag

        elif base_decoder == "relay_minsum":
            base_window = replace(window or DecoderWindow(max_iter=self.minsum_max_iter), relay_enable=True)
            base_correction, base_logical, base_diag = self._decode_sector_windowed(
                matrix=matrix,
                observables=observables,
                metadata=metadata,
                syndrome=syndrome,
                priors=priors,
                method="minimum_sum",
                max_iter=int(base_window.max_iter),
                window=base_window,
                scms=self.scms,
                structure_model=structure_model,
            )
            base_ok = bool(np.array_equal(csr_matvec_mod2(matrix, base_correction), syndrome))
            if base_ok:
                diag = {
                    "mode": "relay_minsum_rescue",
                    "rescue_used": False,
                    "predecode": dict(base_diag),
                    "iterations": int(base_diag.get("iterations", 0)),
                    "converged": bool(base_diag.get("converged", False)),
                    "syndrome_ok": True,
                    "active_columns": int(base_diag.get("active_columns", 0)),
                    "residual_weight_final": int(base_diag.get("residual_weight_final", 0)),
                    "relay_enabled": True,
                    "scms": bool(self.scms),
                }
                correction = base_correction
                logical = base_logical
                diagnostics = diag
            else:
                rescue_correction, rescue_logical, rescue_diag = self._decode_sector_separator_wavefront(
                    matrix=matrix,
                    observables=observables,
                    metadata=metadata,
                    syndrome=syndrome,
                    priors=priors,
                    window=base_window,
                    relay_options=_RelayOptions.from_window(base_window, force_enable=True),
                    structure_model=structure_model,
                )
                diag = dict(rescue_diag)
                diag["mode"] = "relay_minsum_rescue"
                diag["rescue_used"] = True
                diag["predecode"] = dict(base_diag)
                diag["iterations"] = int(base_diag.get("iterations", 0)) + int(rescue_diag.get("iterations", 0))
                diag["relay_enabled"] = True
                diag["scms"] = bool(self.scms)
                correction = rescue_correction
                logical = rescue_logical
                diagnostics = diag
        elif base_decoder == "bp":
            method = "product_sum"
            max_iter = self.bp_max_iter
            scms_active = False
        elif base_decoder == "minsum":
            method = "minimum_sum"
            max_iter = self.minsum_max_iter
            scms_active = bool(self.scms)
        elif base_decoder == "self_corrected_minsum":
            method = "minimum_sum"
            max_iter = self.minsum_max_iter
            scms_active = True
        else:
            raise ValueError(f"unsupported decoder: {decoder}")

        if base_decoder in {"bp", "minsum", "self_corrected_minsum"} and effective_window is not None:
            correction, logical, diagnostics = self._decode_sector_windowed(
                matrix=matrix,
                observables=observables,
                metadata=metadata,
                syndrome=syndrome,
                priors=priors,
                method=method,
                max_iter=int(effective_window.max_iter),
                window=effective_window,
                scms=scms_active,
                structure_model=structure_model,
            )
        elif base_decoder in {"bp", "minsum", "self_corrected_minsum"}:
            correction, diagnostics = self._run_bp_like(
                matrix=matrix,
                syndrome=syndrome,
                priors=priors,
                method=method,
                max_iter=max_iter,
                scms=scms_active,
            )
            logical = csr_matvec_mod2(observables, correction)
        scms_active = bool(self.scms) if base_decoder == "minsum" else bool(base_decoder == "self_corrected_minsum")
        if structure_model is not None:
            diagnostics["triangle_catalog_size"] = int(structure_model.catalog_size)
            diagnostics["triangle_counts_by_kind"] = dict(structure_model.counts_by_kind)
            diagnostics["triangle_selected_count"] = int(len(structure_model.selection.selected_relations))
            diagnostics["triangle_selected_counts_by_kind"] = dict(structure_model.selection.selected_counts_by_kind)
            diagnostics["triangle_overlap_count"] = int(len(structure_model.selection.residual_relations))
            diagnostics["triangle_overlap_counts_by_kind"] = dict(structure_model.selection.overlapping_counts_by_kind)
        if effective_window is not None and bool(effective_window.gauge_descent) and structure_model is not None:
            gauge = apply_sector_gauge_descent(
                estimate=correction,
                priors=priors,
                matrix=matrix,
                observables=observables,
                selection=structure_model.selection,
                max_iterations=int(effective_window.gauge_descent_max_iterations),
            )
            correction = np.asarray(gauge.estimate, dtype=np.uint8).reshape(-1) & 1
            logical = csr_matvec_mod2(observables, correction)
            diagnostics["gauge_descent_enabled"] = True
            diagnostics["gauge_descent_iterations"] = int(gauge.iterations)
            diagnostics["gauge_descent_accepted_moves"] = int(len(gauge.accepted_steps))
            diagnostics["gauge_descent_total_delta"] = float(sum(float(step.delta_cost) for step in gauge.accepted_steps))
            diagnostics["gauge_descent_converged"] = bool(gauge.converged)
        else:
            diagnostics["gauge_descent_enabled"] = False
        diagnostics["syndrome_ok"] = bool(np.array_equal(csr_matvec_mod2(matrix, correction), syndrome))
        diagnostics["logical_weight"] = int(logical.sum())
        diagnostics["scms"] = bool(scms_active)
        diagnostics["decoder_mode"] = str(decoder)
        return correction, logical, diagnostics

    def _decode_sector_round_wavefront(
        self,
        *,
        matrix: sp.csr_matrix,
        observables: sp.csr_matrix,
        metadata: SplitSectorMetadata,
        syndrome: np.ndarray,
        priors: np.ndarray,
        window: DecoderWindow,
        structure_model: SectorStructureModel | None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        round_window = DecoderWindow(
            round_radius=int(window.round_radius),
            max_passes=int(window.max_passes),
            max_iter=int(window.max_iter),
            separator_window_rounds=int(window.separator_window_rounds),
            separator_overlap_rounds=int(window.separator_overlap_rounds),
            separator_topk=0,
            separator_max_branches=1,
            separator_max_window_expansions=0,
            separator_mean_tail=int(window.separator_mean_tail),
            separator_reliable_shell_hops=0,
            separator_reliable_topk=0,
            separator_reliable_abs_mean_threshold=float(window.separator_reliable_abs_mean_threshold),
            triangle_factorization=str(window.triangle_factorization),
            layered_schedule=str(window.layered_schedule),
            gauge_descent=bool(window.gauge_descent),
            gauge_descent_max_iterations=int(window.gauge_descent_max_iterations),
        )
        direction = resolve_schedule_direction(
            matrix=matrix,
            metadata=metadata,
            requested=str(round_window.layered_schedule),
        )
        correction, logical, diagnostics = self._decode_sector_separator_wavefront(
            matrix=matrix,
            observables=observables,
            metadata=metadata,
            syndrome=syndrome,
            priors=priors,
            window=round_window,
            structure_model=structure_model,
        )
        diagnostics = dict(diagnostics)
        diagnostics["mode"] = "round_wavefront"
        diagnostics["branching_enabled"] = False
        diagnostics["window_expansions_enabled"] = False
        diagnostics["reliable_shell_clamps_enabled"] = False
        diagnostics["schedule_note"] = (
            "Pure time-directional round-wavefront min-sum: fixed round windows, overlap commits, "
            f"carried LLRs on uncommitted columns, and resolved direction `{direction.resolved}`."
        )
        diagnostics["schedule_direction_requested"] = str(direction.requested)
        diagnostics["schedule_direction_resolved"] = str(direction.resolved)
        diagnostics["boundary_row_weight_first"] = float(direction.boundary_row_weight_first)
        diagnostics["boundary_row_weight_last"] = float(direction.boundary_row_weight_last)
        return correction, logical, diagnostics

    def _decode_sector_windowed(
        self,
        *,
        matrix: sp.csr_matrix,
        observables: sp.csr_matrix,
        metadata: SplitSectorMetadata,
        syndrome: np.ndarray,
        priors: np.ndarray,
        method: str,
        max_iter: int,
        window: DecoderWindow,
        scms: bool,
        structure_model: SectorStructureModel | None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        column_index, active_rounds = _select_window_columns(metadata, syndrome, int(window.round_radius))
        correction = np.zeros(matrix.shape[1], dtype=np.uint8)
        if column_index.size:
            if (
                str(method) == "minimum_sum"
                and not bool(scms)
                and structure_model is not None
                and str(window.triangle_factorization) == "nonoverlap"
            ):
                local_prior_llr = llr_from_priors(np.asarray(priors[column_index], dtype=np.float64))
                reduced_model = build_reduced_local_model(
                    local_sub_matrix=matrix[:, column_index].tocsr(),
                    local_cols=column_index,
                    local_prior_llr=local_prior_llr,
                    selection=structure_model.selection,
                )
                reduced_result = solve_reduced_local_model(
                    model=reduced_model,
                    syndrome=syndrome,
                    max_iter=int(max_iter),
                )
                correction[column_index] = np.asarray(reduced_result.bits_local, dtype=np.uint8)
                diagnostics = {
                    "iterations": int(reduced_result.iterations),
                    "converged": bool(reduced_result.converged),
                    "method": "minimum_sum_structure_aware",
                    "erased_edge_count_by_iter": [],
                    "erased_edge_total": 0,
                    "residual_weight_final": int(np.count_nonzero(reduced_result.residual)),
                    "triangle_factorization": str(window.triangle_factorization),
                    "reduced_triangle_count": int(reduced_result.reduced_triangle_count),
                    "residual_triangle_count": int(reduced_result.residual_triangle_count),
                    "reduced_variable_count": int(reduced_result.reduced_variable_count),
                }
            else:
                delta, diagnostics = self._run_bp_like(
                    matrix=matrix[:, column_index],
                    syndrome=syndrome,
                    priors=priors[column_index],
                    method=method,
                    max_iter=max_iter,
                    scms=scms,
                )
                correction[column_index] = delta
        else:
            diagnostics = {
                "iterations": 0,
                "converged": True,
                "erased_edge_count_by_iter": [],
                "erased_edge_total": 0,
                "residual_weight_final": 0,
                "reduced_triangle_count": 0,
                "residual_triangle_count": 0,
                "reduced_variable_count": 0,
            }
        logical = csr_matvec_mod2(observables, correction)
        diagnostics.update(
            {
                "mode": "windowed",
                "active_rounds": active_rounds,
                "active_columns": int(column_index.size),
            }
        )
        return correction, logical, diagnostics

    def _decode_sector_local_round(
        self,
        *,
        matrix: sp.csr_matrix,
        observables: sp.csr_matrix,
        metadata: SplitSectorMetadata,
        syndrome: np.ndarray,
        priors: np.ndarray,
        window: DecoderWindow,
        scms: bool,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        correction = np.zeros(matrix.shape[1], dtype=np.uint8)
        residual = syndrome.copy()
        passes: list[dict[str, object]] = []
        previous_weight = int(residual.sum())
        for pass_index in range(int(window.max_passes)):
            if not np.any(residual):
                break
            components = _active_round_components(metadata, residual)
            if not components:
                break
            any_progress = False
            for component in components:
                expanded = _expand_round_component(component, metadata.total_rounds, int(window.round_radius))
                column_index = _columns_for_rounds(metadata, expanded)
                if column_index.size == 0:
                    passes.append(
                        {
                            "pass": pass_index,
                            "component": component,
                            "expanded_rounds": expanded,
                            "active_columns": 0,
                            "residual_weight_before": int(residual.sum()),
                            "residual_weight_after": int(residual.sum()),
                        }
                    )
                    continue
                delta_sub, diagnostics = self._run_bp_like(
                    matrix=matrix[:, column_index],
                    syndrome=residual,
                    priors=priors[column_index],
                    method="minimum_sum",
                    max_iter=int(window.max_iter),
                    scms=scms,
                )
                if np.any(delta_sub):
                    any_progress = True
                delta = np.zeros(matrix.shape[1], dtype=np.uint8)
                delta[column_index] = delta_sub
                correction ^= delta
                residual = syndrome ^ csr_matvec_mod2(matrix, correction)
                passes.append(
                    {
                        "pass": pass_index,
                        "component": component,
                        "expanded_rounds": expanded,
                        "active_columns": int(column_index.size),
                        "iterations": int(diagnostics["iterations"]),
                        "converged": bool(diagnostics["converged"]),
                        "residual_weight_before": previous_weight,
                        "residual_weight_after": int(residual.sum()),
                    }
                )
                previous_weight = int(residual.sum())
            if not any_progress:
                break
        logical = csr_matvec_mod2(observables, correction)
        iterations_total = int(sum(int(item.get("iterations", 0)) for item in passes))
        converged = bool(not np.any(residual))
        return correction, logical, {
            "mode": "local_round",
            "passes": passes,
            "iterations": iterations_total,
            "converged": converged,
            "residual_weight_final": int(residual.sum()),
            "syndrome_ok": converged,
            "scms": bool(scms),
        }

    def _run_separator_subproblem(
        self,
        *,
        matrix: sp.csr_matrix,
        rows: np.ndarray,
        columns: np.ndarray,
        syndrome: np.ndarray,
        prior_llr: np.ndarray,
        max_iter: int,
        clamp_assignments: dict[int, int] | None = None,
        mean_tail: int = 8,
        sub_matrix: sp.csr_matrix | None = None,
        relay_options: _RelayOptions | None = None,
        relay_separator_local: np.ndarray | None = None,
        relay_carry_local: np.ndarray | None = None,
        relay_modified_local: np.ndarray | None = None,
        structure_selection: object | None = None,
        triangle_factorization: TriangleFactorizationMode = "off",
    ) -> dict[str, object]:
        local_rows = np.asarray(rows, dtype=np.int32)
        local_cols = np.asarray(columns, dtype=np.int32)
        local_syndrome = np.asarray(syndrome, dtype=np.uint8).reshape(-1).copy()
        local_prior = np.asarray(prior_llr, dtype=np.float64).reshape(-1).copy()

        if int(local_rows.size) != int(local_syndrome.size):
            raise ValueError("local syndrome length mismatch")
        if int(local_cols.size) != int(local_prior.size):
            raise ValueError("local prior length mismatch")
        if local_cols.size == 0:
            residual = local_syndrome.copy()
            empty_stats = _empty_temporal_stats(0)
            return {
                "bits": np.zeros(0, dtype=np.uint8),
                "llr": np.zeros(0, dtype=np.float64),
                "mean_llr": np.zeros(0, dtype=np.float64),
                "converged": bool(np.count_nonzero(residual) == 0),
                "iterations": 0,
                "residual": residual,
                "score": 0.0,
                "temporal": empty_stats,
                "relay": {
                    "activated": False,
                    "activation_iter": 0,
                    "legs_used": 0,
                    "stable_count": 0,
                    "bulk_count": 0,
                    "frontier_count": 0,
                    "message_flip_total": 0,
                    "llr_flip_total": 0,
                },
                "reduced_triangle_count": 0,
                "residual_triangle_count": 0,
                "reduced_variable_count": 0,
            }

        local_sub_matrix = matrix[local_rows][:, local_cols].tocsr() if sub_matrix is None else sub_matrix.tocsr()
        clamped = np.zeros(local_cols.size, dtype=bool)
        clamp_vals = np.zeros(local_cols.size, dtype=np.uint8)
        if clamp_assignments:
            sub_csc = local_sub_matrix.tocsc()
            for local_idx, raw_val in clamp_assignments.items():
                idx = int(local_idx)
                if idx < 0 or idx >= int(local_cols.size):
                    raise ValueError(f"local clamp index out of range: {idx}")
                bit = int(raw_val) & 1
                clamped[idx] = True
                clamp_vals[idx] = np.uint8(bit)
                if bit:
                    begin = int(sub_csc.indptr[idx])
                    end = int(sub_csc.indptr[idx + 1])
                    if begin < end:
                        local_syndrome[sub_csc.indices[begin:end]] ^= 1

        free_local = np.flatnonzero(~clamped).astype(np.int32, copy=False)
        local_bits = np.zeros(local_cols.size, dtype=np.uint8)
        local_llr = local_prior.copy()
        local_mean_llr = local_prior.copy()
        local_bits[clamped] = clamp_vals[clamped]
        local_llr[clamped] = np.where(clamp_vals[clamped] > 0, -64.0, 64.0)
        local_mean_llr[clamped] = local_llr[clamped]

        if free_local.size:
            reduced_matrix = local_sub_matrix[:, free_local]
            structure_active = (
                str(triangle_factorization) == "nonoverlap"
                and structure_selection is not None
                and not bool(self.scms)
                and not bool(relay_options is not None and relay_options.enabled)
            )
            if structure_active:
                reduced_model = build_reduced_local_model(
                    local_sub_matrix=reduced_matrix,
                    local_cols=local_cols[free_local],
                    local_prior_llr=local_prior[free_local],
                    selection=structure_selection,
                )
                reduced_result = solve_reduced_local_model(
                    model=reduced_model,
                    syndrome=local_syndrome,
                    max_iter=int(max_iter),
                )
                local_bits[free_local] = np.asarray(reduced_result.bits_local, dtype=np.uint8)
                local_llr[free_local] = np.asarray(reduced_result.llr_local, dtype=np.float64)
                local_mean_llr[free_local] = np.asarray(reduced_result.mean_llr_local, dtype=np.float64)
                iters_used = int(reduced_result.iterations)
                temporal_stats = _empty_temporal_stats(int(free_local.size))
                relay_diag = {
                    "activated": False,
                    "activation_iter": 0,
                    "legs_used": 0,
                    "stable_count": 0,
                    "bulk_count": 0,
                    "frontier_count": 0,
                    "message_flip_total": 0,
                    "llr_flip_total": 0,
                    "reduced_triangle_count": int(reduced_result.reduced_triangle_count),
                    "residual_triangle_count": int(reduced_result.residual_triangle_count),
                    "reduced_variable_count": int(reduced_result.reduced_variable_count),
                }
            else:
                cfg = DecoderConfig(
                    max_iter=int(max_iter),
                    schedule="layered",
                    damping=0.0,
                    normalization=float(self.minsum_scale),
                    offset=0.0,
                    llr_clip=30.0,
                    self_corrected=bool(self.scms),
                )
                dec_bits, dec_llr, dec_mean_llr, _converged, iters_used, _reduced_residual, temporal_stats, relay_diag = _run_minsum_with_trace_mean(
                    graph=TannerGraph.from_csr(reduced_matrix),
                    syndrome_bits=local_syndrome,
                    prior_llr=local_prior[free_local],
                    config=cfg,
                    mean_tail=int(mean_tail),
                    relay_options=relay_options,
                    relay_separator_local=_remap_local_indices(relay_separator_local, free_local),
                    relay_carry_local=_remap_local_indices(relay_carry_local, free_local),
                    relay_modified_local=_remap_local_indices(
                        np.asarray(sorted(clamp_assignments), dtype=np.int32) if clamp_assignments else relay_modified_local,
                        free_local,
                    ),
                )
                local_bits[free_local] = np.asarray(dec_bits, dtype=np.uint8)
                local_llr[free_local] = np.asarray(dec_llr, dtype=np.float64)
                local_mean_llr[free_local] = np.asarray(dec_mean_llr, dtype=np.float64)
        else:
            iters_used = 0
            temporal_stats = _empty_temporal_stats(0)
            relay_diag = {
                "activated": False,
                "activation_iter": 0,
                "legs_used": 0,
                "stable_count": 0,
                "bulk_count": 0,
                "frontier_count": 0,
                "message_flip_total": 0,
                "llr_flip_total": 0,
                "reduced_triangle_count": 0,
                "residual_triangle_count": 0,
                "reduced_variable_count": 0,
            }

        full_temporal = _empty_temporal_stats(int(local_cols.size))
        if free_local.size:
            _scatter_temporal_stats(full_temporal, free_local, temporal_stats)
        full_residual = (csr_matvec_mod2(local_sub_matrix, local_bits) ^ np.asarray(syndrome, dtype=np.uint8)).astype(np.uint8)
        weighted_prior = np.clip(local_prior, 0.0, None)
        score = float(np.dot(weighted_prior, local_bits.astype(np.float64)))
        relay_summary = dict(relay_diag)
        relay_summary["message_flip_total"] = int(np.sum(full_temporal["message_flip_count"], dtype=np.int64))
        relay_summary["llr_flip_total"] = int(np.sum(full_temporal["llr_flip_count"], dtype=np.int64))
        return {
            "bits": local_bits,
            "llr": local_llr,
            "mean_llr": local_mean_llr,
            "converged": bool(np.count_nonzero(full_residual) == 0),
            "iterations": int(iters_used),
            "residual": full_residual,
            "score": score,
            "temporal": full_temporal,
            "relay": relay_summary,
            "reduced_triangle_count": int(relay_summary.get("reduced_triangle_count", 0)),
            "residual_triangle_count": int(relay_summary.get("residual_triangle_count", 0)),
            "reduced_variable_count": int(relay_summary.get("reduced_variable_count", 0)),
        }

    def _decode_sector_separator_wavefront(
        self,
        *,
        matrix: sp.csr_matrix,
        observables: sp.csr_matrix,
        metadata: SplitSectorMetadata,
        syndrome: np.ndarray,
        priors: np.ndarray,
        window: DecoderWindow,
        relay_options: _RelayOptions | None = None,
        structure_model: SectorStructureModel | None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        total_rounds = int(metadata.total_rounds)
        if total_rounds <= 0:
            raise ValueError("metadata.total_rounds must be positive")

        base_window_rounds = max(1, int(window.separator_window_rounds))
        overlap_rounds = max(0, min(int(window.separator_overlap_rounds), int(base_window_rounds) - 1))
        separator_topk = max(0, int(window.separator_topk))
        separator_max_branches = max(1, int(window.separator_max_branches))
        max_expansions = max(0, int(window.separator_max_window_expansions))
        separator_mean_tail = max(1, int(window.separator_mean_tail))
        reliable_shell_hops = max(0, int(window.separator_reliable_shell_hops))
        reliable_topk = max(0, int(window.separator_reliable_topk))
        reliable_abs_mean_threshold = max(0.0, float(window.separator_reliable_abs_mean_threshold))

        correction = np.zeros(matrix.shape[1], dtype=np.uint8)
        residual = np.asarray(syndrome, dtype=np.uint8).reshape(-1).copy()
        carry_llr = llr_from_priors(np.asarray(priors, dtype=np.float64))
        steps: list[dict[str, object]] = []
        total_iterations = 0
        start_round = 0
        stalled = False
        active_columns_last = 0
        relay_mode = bool(relay_options is not None and relay_options.enabled)
        direction = resolve_schedule_direction(
            matrix=matrix,
            metadata=metadata,
            requested=str(window.layered_schedule),
        )
        span_start_v, _span_stop_v = virtual_column_span_arrays(metadata, direction=str(direction.resolved))

        while start_round < total_rounds:
            window_end = min(total_rounds - 1, int(start_round + base_window_rounds - 1))
            best_step: dict[str, object] | None = None
            expansions_used = 0

            while True:
                commit_end = total_rounds - 1 if window_end == total_rounds - 1 else max(start_round, window_end - overlap_rounds)
                active_actual_start, active_actual_end = actual_interval_from_virtual(
                    total_rounds=int(total_rounds),
                    start_round=int(start_round),
                    end_round=int(window_end),
                    direction=str(direction.resolved),
                )
                commit_actual_start, commit_actual_end = actual_interval_from_virtual(
                    total_rounds=int(total_rounds),
                    start_round=int(start_round),
                    end_round=int(commit_end),
                    direction=str(direction.resolved),
                )
                active_cols = _columns_for_round_interval(metadata, int(active_actual_start), int(active_actual_end))
                active_columns_last = int(active_cols.size)
                solve_rows = _rows_for_round_interval(metadata, int(active_actual_start), int(active_actual_end))
                success_rows = _rows_for_round_interval(metadata, int(commit_actual_start), int(commit_actual_end))
                local_prior = np.asarray(carry_llr[active_cols], dtype=np.float64)
                local_syndrome = np.asarray(residual[solve_rows], dtype=np.uint8)
                local_submatrix = matrix[solve_rows][:, active_cols].tocsr()
                separator_local = virtual_separator_local_columns(
                    metadata,
                    active_cols,
                    commit_end_round=int(commit_end),
                    direction=str(direction.resolved),
                )
                carry_local = np.flatnonzero(span_start_v[active_cols] < int(start_round)).astype(
                    np.int32,
                    copy=False,
                )

                base_result = self._run_separator_subproblem(
                    matrix=matrix,
                    rows=solve_rows,
                    columns=active_cols,
                    syndrome=local_syndrome,
                    prior_llr=local_prior,
                    max_iter=int(window.max_iter),
                    mean_tail=int(separator_mean_tail),
                    sub_matrix=local_submatrix,
                    relay_options=relay_options,
                    relay_separator_local=separator_local,
                    relay_carry_local=carry_local,
                    structure_selection=None if structure_model is None else structure_model.selection,
                    triangle_factorization=str(window.triangle_factorization),
                )
                total_iterations += int(base_result["iterations"])
                success_local = np.flatnonzero(np.isin(solve_rows, success_rows)).astype(np.int32, copy=False)
                branch_count = 1
                best = {
                    **base_result,
                    "success_weight": int(np.count_nonzero(np.asarray(base_result["residual"], dtype=np.uint8)[success_local])),
                    "total_weight": int(np.count_nonzero(np.asarray(base_result["residual"], dtype=np.uint8))),
                    "separator_columns": 0,
                    "reliable_clamps": 0,
                    "uncertain_columns": 0,
                    "chosen_local": np.zeros(0, dtype=np.int32),
                }

                if separator_local.size > 0 and separator_topk > 0:
                    chosen_local = _select_branch_candidates(
                        sub_matrix=local_submatrix,
                        separator_local=separator_local,
                        residual=np.asarray(base_result["residual"], dtype=np.uint8),
                        llr=np.asarray(base_result["llr"], dtype=np.float64),
                        mean_llr=np.asarray(base_result["mean_llr"], dtype=np.float64),
                        temporal=_as_temporal_stats(base_result.get("temporal")),
                        topk=min(int(separator_topk), int(separator_local.size)),
                        score_mode=relay_options.candidate_score if relay_mode else "mean_absllr",
                        shell_radius=relay_options.frontier_shell_radius if relay_mode else 1,
                    )
                    reliable_clamp_map = _select_reliable_shell_clamps(
                        sub_matrix=local_submatrix,
                        chosen_local=chosen_local,
                        separator_local=separator_local,
                        mean_llr=np.asarray(base_result["mean_llr"], dtype=np.float64),
                        shell_hops=int(reliable_shell_hops),
                        reliable_topk=int(reliable_topk),
                        reliable_abs_mean_threshold=float(reliable_abs_mean_threshold),
                    )
                    for assignment in _iter_assignments_low_weight(
                        int(chosen_local.size),
                        limit=max(1, int(separator_max_branches)),
                    ):
                        clamp_map = {int(local_idx): int(bit) for local_idx, bit in zip(chosen_local.tolist(), assignment)}
                        clamp_map.update(reliable_clamp_map)
                        cand_result = self._run_separator_subproblem(
                            matrix=matrix,
                            rows=solve_rows,
                            columns=active_cols,
                            syndrome=local_syndrome,
                            prior_llr=local_prior,
                            max_iter=int(window.max_iter),
                            clamp_assignments=clamp_map,
                            mean_tail=int(separator_mean_tail),
                            sub_matrix=local_submatrix,
                            relay_options=relay_options,
                            relay_separator_local=separator_local,
                            relay_carry_local=carry_local,
                            relay_modified_local=np.asarray(sorted(clamp_map), dtype=np.int32),
                            structure_selection=None if structure_model is None else structure_model.selection,
                            triangle_factorization=str(window.triangle_factorization),
                        )
                        total_iterations += int(cand_result["iterations"])
                        branch_count += 1
                        candidate = {
                            **cand_result,
                            "success_weight": int(np.count_nonzero(np.asarray(cand_result["residual"], dtype=np.uint8)[success_local])),
                            "total_weight": int(np.count_nonzero(np.asarray(cand_result["residual"], dtype=np.uint8))),
                            "separator_columns": int(chosen_local.size),
                            "reliable_clamps": int(len(reliable_clamp_map)),
                            "uncertain_columns": int(chosen_local.size),
                            "chosen_local": chosen_local,
                        }
                        if _candidate_key(candidate) < _candidate_key(best):
                            best = candidate
                        if int(best["success_weight"]) == 0 and int(best["total_weight"]) == 0:
                            break

                best_step = {
                    "window_start_round": int(start_round),
                    "window_end_round": int(window_end),
                    "commit_end_round": int(commit_end),
                    "active_columns": int(active_cols.size),
                    "active_rows": int(solve_rows.size),
                    "separator_columns": int(best["separator_columns"]),
                    "uncertain_columns": int(best["uncertain_columns"]),
                    "reliable_clamps": int(best["reliable_clamps"]),
                    "branches_tried": int(branch_count),
                    "iterations": int(best["iterations"]),
                    "success_residual_weight": int(best["success_weight"]),
                    "total_residual_weight": int(best["total_weight"]),
                    "expanded_rounds": int(expansions_used),
                    "relay_activated": bool(dict(best.get("relay", {})).get("activated", False)),
                    "relay_activation_iter": int(dict(best.get("relay", {})).get("activation_iter", 0)),
                    "relay_legs_used": int(dict(best.get("relay", {})).get("legs_used", 0)),
                    "relay_frontier_count": int(dict(best.get("relay", {})).get("frontier_count", 0)),
                    "relay_message_flip_total": int(dict(best.get("relay", {})).get("message_flip_total", 0)),
                    "relay_llr_flip_total": int(dict(best.get("relay", {})).get("llr_flip_total", 0)),
                    "relay_candidate_score": str(relay_options.candidate_score) if relay_mode else "mean_absllr",
                    "reduced_triangle_count": int(best.get("reduced_triangle_count", 0)),
                    "residual_triangle_count": int(best.get("residual_triangle_count", 0)),
                    "reduced_variable_count": int(best.get("reduced_variable_count", 0)),
                    "schedule_direction": str(direction.resolved),
                }
                if int(best["success_weight"]) == 0 or int(window_end) == int(total_rounds - 1) or int(expansions_used) >= int(max_expansions):
                    best_step.update(best)
                    break
                window_end = min(total_rounds - 1, int(window_end + 1))
                expansions_used += 1

            assert best_step is not None
            active_actual_start, active_actual_end = actual_interval_from_virtual(
                total_rounds=int(total_rounds),
                start_round=int(start_round),
                end_round=int(best_step["window_end_round"]),
                direction=str(direction.resolved),
            )
            active_cols = _columns_for_round_interval(metadata, int(active_actual_start), int(active_actual_end))
            commit_mask = virtual_commit_mask(
                metadata,
                active_cols,
                commit_end_round=int(best_step["commit_end_round"]),
                direction=str(direction.resolved),
            )
            commit_cols = active_cols[commit_mask]

            if int(best_step["success_residual_weight"]) > 0 and int(best_step["commit_end_round"]) < int(total_rounds - 1):
                stalled = True
                if active_cols.size:
                    correction[active_cols] = np.asarray(best_step["bits"], dtype=np.uint8)
                steps.append(
                    {
                        "stalled": True,
                        **{
                            k: v
                            for k, v in best_step.items()
                            if k not in {"bits", "llr", "mean_llr", "residual", "temporal", "relay", "chosen_local"}
                        },
                    }
                )
                break

            if commit_cols.size == 0:
                stalled = True
                steps.append(
                    {
                        "stalled": True,
                        **{
                            k: v
                            for k, v in best_step.items()
                            if k not in {"bits", "llr", "mean_llr", "residual", "temporal", "relay", "chosen_local"}
                        },
                    }
                )
                break

            correction[commit_cols] = np.asarray(best_step["bits"], dtype=np.uint8)[commit_mask]
            keep_mask = ~commit_mask
            if np.any(keep_mask):
                carry_llr[active_cols[keep_mask]] = np.asarray(best_step["llr"], dtype=np.float64)[keep_mask]
            residual = (csr_matvec_mod2(matrix, correction) ^ np.asarray(syndrome, dtype=np.uint8)).astype(np.uint8)
            steps.append(
                {
                    "stalled": False,
                    **{
                        k: v
                        for k, v in best_step.items()
                        if k not in {"bits", "llr", "mean_llr", "residual", "temporal", "relay", "chosen_local"}
                    },
                }
            )
            start_round = int(best_step["commit_end_round"]) + 1

        logical = csr_matvec_mod2(observables, correction)
        final_residual = (csr_matvec_mod2(matrix, correction) ^ np.asarray(syndrome, dtype=np.uint8)).astype(np.uint8)
        converged = bool(np.count_nonzero(final_residual) == 0)
        return correction, logical, {
            "mode": "separator_wavefront",
            "steps": steps,
            "iterations": int(total_iterations),
            "converged": converged,
            "syndrome_ok": converged,
            "active_columns": int(active_columns_last),
            "residual_weight_final": int(np.count_nonzero(final_residual)),
            "stalled": bool(stalled),
            "relay_enabled": bool(relay_mode),
            "scms": bool(self.scms),
            "schedule_direction_requested": str(direction.requested),
            "schedule_direction_resolved": str(direction.resolved),
            "boundary_row_weight_first": float(direction.boundary_row_weight_first),
            "boundary_row_weight_last": float(direction.boundary_row_weight_last),
            "triangle_factorization": str(window.triangle_factorization),
            "triangle_selected_count": 0 if structure_model is None else int(len(structure_model.selection.selected_relations)),
        }

    def _run_bp_like(
        self,
        *,
        matrix: sp.csr_matrix,
        syndrome: np.ndarray,
        priors: np.ndarray,
        method: str,
        max_iter: int,
        scms: bool,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if bool(scms) and str(method) == "minimum_sum":
            cfg = DecoderConfig(
                max_iter=int(max_iter),
                schedule="serial",
                damping=0.0,
                normalization=float(self.minsum_scale),
                offset=0.0,
                llr_clip=30.0,
                self_corrected=True,
            )
            bits, _llr, _mean_llr, converged, iterations, _residual, _temporal, relay_diag = _run_minsum_with_trace_mean(
                graph=TannerGraph.from_csr(matrix),
                syndrome_bits=np.asarray(syndrome, dtype=np.uint8),
                prior_llr=llr_from_priors(np.asarray(priors, dtype=np.float64)),
                config=cfg,
                mean_tail=1,
            )
            diagnostics = {
                "iterations": int(iterations),
                "converged": bool(converged),
                "method": method,
                "erased_edge_count_by_iter": [int(value) for value in relay_diag.get("erased_edge_count_by_iter", [])],
                "erased_edge_total": int(relay_diag.get("erased_edge_total", 0)),
            }
            return np.asarray(bits, dtype=np.uint8).reshape(-1) & 1, diagnostics

        decoder = BpDecoder(
            matrix,
            error_channel=np.asarray(priors, dtype=np.float64),
            max_iter=int(max_iter),
            bp_method=str(method),
            ms_scaling_factor=float(self.minsum_scale),
            schedule="serial",
            omp_thread_count=1,
            random_schedule_seed=self.seed,
            input_vector_type="syndrome",
        )
        correction = np.asarray(decoder.decode(np.asarray(syndrome, dtype=np.uint8)), dtype=np.uint8).reshape(-1) & 1
        diagnostics = {
            "iterations": int(getattr(decoder, "iter", int(max_iter))),
            "converged": bool(getattr(decoder, "converge", False)),
            "method": method,
            "erased_edge_count_by_iter": [],
            "erased_edge_total": 0,
        }
        return correction, diagnostics


def _select_window_columns(metadata: SplitSectorMetadata, syndrome: np.ndarray, round_radius: int) -> tuple[np.ndarray, list[int]]:
    active_rounds = sorted({int(metadata.detector_round_index[row]) for row in np.flatnonzero(syndrome)})
    if not active_rounds:
        return np.zeros(0, dtype=np.int32), []
    expanded = _expand_round_component(active_rounds, metadata.total_rounds, round_radius)
    return _columns_for_rounds(metadata, expanded), expanded


def _columns_for_rounds(metadata: SplitSectorMetadata, rounds: list[int]) -> np.ndarray:
    if not rounds:
        return np.zeros(0, dtype=np.int32)
    rounds_arr = np.asarray(rounds, dtype=np.int16)
    keep = np.isin(metadata.column_round_start, rounds_arr) | np.isin(metadata.column_round_stop, rounds_arr)
    return np.flatnonzero(keep).astype(np.int32)


def _columns_for_round_interval(metadata: SplitSectorMetadata, start_round: int, end_round: int) -> np.ndarray:
    keep = (metadata.column_round_start <= int(end_round)) & (metadata.column_round_stop >= int(start_round))
    return np.flatnonzero(keep).astype(np.int32, copy=False)


def _rows_for_round_interval(metadata: SplitSectorMetadata, start_round: int, end_round: int) -> np.ndarray:
    if int(start_round) > int(end_round):
        return np.zeros(0, dtype=np.int32)
    start_row = int(metadata.detector_round_slices[int(start_round)][1])
    stop_row = int(metadata.detector_round_slices[int(end_round)][2])
    return np.arange(start_row, stop_row, dtype=np.int32)


def _local_rows_for_round_interval(metadata: SplitSectorMetadata, start_round: int, end_round: int) -> np.ndarray:
    global_rows = _rows_for_round_interval(metadata, start_round, end_round)
    if global_rows.size == 0:
        return np.zeros(0, dtype=np.int32)
    return np.arange(global_rows.size, dtype=np.int32)


def _separator_local_columns(
    metadata: SplitSectorMetadata,
    active_cols: np.ndarray,
    commit_end_round: int,
) -> np.ndarray:
    active = np.asarray(active_cols, dtype=np.int32)
    crossing = np.flatnonzero(
        (metadata.column_round_start[active] <= int(commit_end_round))
        & (metadata.column_round_stop[active] > int(commit_end_round))
    ).astype(np.int32, copy=False)
    if crossing.size:
        return crossing
    future_touching = np.flatnonzero(metadata.column_round_stop[active] > int(commit_end_round)).astype(np.int32, copy=False)
    if future_touching.size:
        return future_touching
    return np.zeros(0, dtype=np.int32)


def _empty_temporal_stats(num_vars: int) -> dict[str, np.ndarray]:
    size = max(0, int(num_vars))
    return {
        "message_flip_count": np.zeros(size, dtype=np.int32),
        "llr_flip_count": np.zeros(size, dtype=np.int32),
        "llr_abs_delta_sum": np.zeros(size, dtype=np.float64),
        "residual_touch_count": np.zeros(size, dtype=np.int32),
        "instability_score": np.zeros(size, dtype=np.float64),
    }


def _as_temporal_stats(payload: object) -> dict[str, np.ndarray]:
    if isinstance(payload, dict):
        return {
            "message_flip_count": np.asarray(payload.get("message_flip_count", np.zeros(0, dtype=np.int32)), dtype=np.int32),
            "llr_flip_count": np.asarray(payload.get("llr_flip_count", np.zeros(0, dtype=np.int32)), dtype=np.int32),
            "llr_abs_delta_sum": np.asarray(payload.get("llr_abs_delta_sum", np.zeros(0, dtype=np.float64)), dtype=np.float64),
            "residual_touch_count": np.asarray(payload.get("residual_touch_count", np.zeros(0, dtype=np.int32)), dtype=np.int32),
            "instability_score": np.asarray(payload.get("instability_score", np.zeros(0, dtype=np.float64)), dtype=np.float64),
        }
    return _empty_temporal_stats(0)


def _scatter_temporal_stats(dst: dict[str, np.ndarray], indices: np.ndarray, src: dict[str, np.ndarray]) -> None:
    local = np.asarray(indices, dtype=np.int32)
    if local.size == 0:
        return
    for key in dst:
        dst[key][local] = np.asarray(src[key], dtype=dst[key].dtype)


def _remap_local_indices(local_indices: np.ndarray | None, free_local: np.ndarray) -> np.ndarray:
    if local_indices is None:
        return np.zeros(0, dtype=np.int32)
    source = np.asarray(local_indices, dtype=np.int32).reshape(-1)
    if source.size == 0:
        return np.zeros(0, dtype=np.int32)
    remap = {int(local_idx): int(reduced_idx) for reduced_idx, local_idx in enumerate(np.asarray(free_local, dtype=np.int32).tolist())}
    out = [remap[int(idx)] for idx in source.tolist() if int(idx) in remap]
    if not out:
        return np.zeros(0, dtype=np.int32)
    return np.asarray(out, dtype=np.int32)


def _llr_sign_flip_mask(previous: np.ndarray, current: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    prev = np.asarray(previous, dtype=np.float64)
    curr = np.asarray(current, dtype=np.float64)
    return (
        (np.signbit(prev) != np.signbit(curr))
        & (np.abs(prev) > float(eps))
        & (np.abs(curr) > float(eps))
    )


def _apply_flip_mode(
    current: np.ndarray,
    previous: np.ndarray,
    flip_mask: np.ndarray,
    relay_options: _RelayOptions,
    *,
    llr_clip: float,
) -> np.ndarray:
    out = np.asarray(current, dtype=np.float64).copy()
    mask = np.asarray(flip_mask, dtype=bool)
    if not np.any(mask):
        return out
    if relay_options.flip_mode == "erase":
        out[mask] = 0.0
    elif relay_options.flip_mode == "blend":
        out[mask] = out[mask] + float(relay_options.flip_kappa) * np.asarray(previous, dtype=np.float64)[mask]
    else:
        out[mask] = float(relay_options.flip_eta) * out[mask]
    return np.clip(out, -float(llr_clip), float(llr_clip))


def _residual_touch_count(graph: TannerGraph, residual_rows: np.ndarray) -> np.ndarray:
    touch = np.zeros(graph.n, dtype=np.int32)
    for check in np.asarray(residual_rows, dtype=np.int32).tolist():
        edges = graph.check_to_edges[int(check)]
        if edges.size:
            np.add.at(touch, graph.edge_var[edges], 1)
    return touch


def _variables_near_checks(graph: TannerGraph, residual_rows: np.ndarray, *, radius: int) -> np.ndarray:
    rows = np.asarray(residual_rows, dtype=np.int32)
    if rows.size == 0 or int(radius) <= 0:
        return np.zeros(0, dtype=np.int32)
    seen_checks = {int(row) for row in rows.tolist()}
    frontier_checks = set(seen_checks)
    seen_vars: set[int] = set()
    for _ in range(int(radius)):
        if not frontier_checks:
            break
        frontier_vars: set[int] = set()
        for check in frontier_checks:
            frontier_vars.update(int(v) for v in graph.edge_var[graph.check_to_edges[int(check)]].tolist())
        new_vars = frontier_vars - seen_vars
        seen_vars.update(new_vars)
        next_checks: set[int] = set()
        for var in new_vars:
            next_checks.update(int(c) for c in graph.edge_check[graph.var_to_edges[int(var)]].tolist())
        frontier_checks = next_checks - seen_checks
        seen_checks.update(frontier_checks)
    if not seen_vars:
        return np.zeros(0, dtype=np.int32)
    return np.asarray(sorted(seen_vars), dtype=np.int32)


def _tail_mean_llr(llr_tail: list[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    if not llr_tail:
        return np.asarray(fallback, dtype=np.float64).copy()
    return np.mean(np.stack(llr_tail, axis=0), axis=0)


def _relay_instability_score(
    *,
    message_flip_count: np.ndarray,
    llr_flip_count: np.ndarray,
    llr_abs_delta_sum: np.ndarray,
    mean_llr: np.ndarray,
) -> np.ndarray:
    denom = np.maximum(1.0, np.abs(np.asarray(mean_llr, dtype=np.float64)))
    return (
        np.asarray(message_flip_count, dtype=np.float64)
        + 0.5 * np.asarray(llr_flip_count, dtype=np.float64)
        + np.asarray(llr_abs_delta_sum, dtype=np.float64) / denom
    )


def _scale_leg_gamma(gamma: float, leg_index: int) -> float:
    scaled = float(gamma) * (1.0 + 0.25 * max(0, int(leg_index) - 1))
    return float(np.clip(scaled, -0.95, 0.95))


def _build_relay_gamma_profile(
    *,
    graph: TannerGraph,
    residual: np.ndarray,
    mean_llr: np.ndarray,
    carry_local: np.ndarray,
    separator_local: np.ndarray,
    modified_local: np.ndarray,
    temporal_stats: dict[str, np.ndarray],
    relay_options: _RelayOptions,
    leg_index: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    gamma = np.full(graph.n, float(relay_options.gamma_bulk), dtype=np.float64)
    residual_rows = np.flatnonzero(np.asarray(residual, dtype=np.uint8))
    frontier_mask = np.zeros(graph.n, dtype=bool)
    if residual_rows.size and int(relay_options.frontier_shell_radius) > 0:
        frontier_mask[_variables_near_checks(graph, residual_rows, radius=int(relay_options.frontier_shell_radius))] = True
    frontier_mask[np.asarray(separator_local, dtype=np.int32)] = True
    frontier_mask[np.asarray(modified_local, dtype=np.int32)] = True

    instability = np.asarray(temporal_stats["message_flip_count"], dtype=np.int32) + np.asarray(
        temporal_stats["llr_flip_count"],
        dtype=np.int32,
    )
    low_polarization = np.abs(np.asarray(mean_llr, dtype=np.float64)) < max(2.0, 0.25 * float(relay_options.clip_B))
    frontier_mask |= (instability >= 2) & low_polarization
    gamma[frontier_mask] = _scale_leg_gamma(float(relay_options.gamma_frontier), int(leg_index))

    carry_mask = np.zeros(graph.n, dtype=bool)
    carry_mask[np.asarray(carry_local, dtype=np.int32)] = True
    stable_threshold = max(4.0, 0.5 * float(relay_options.clip_B))
    stable_mask = (
        ~frontier_mask
        & (carry_mask | (np.abs(np.asarray(mean_llr, dtype=np.float64)) >= stable_threshold))
        & (instability <= 1)
    )
    gamma[stable_mask] = float(relay_options.gamma_stable)
    bulk_mask = ~(frontier_mask | stable_mask)
    return gamma, frontier_mask, {
        "stable_count": int(np.count_nonzero(stable_mask)),
        "bulk_count": int(np.count_nonzero(bulk_mask)),
        "frontier_count": int(np.count_nonzero(frontier_mask)),
        "leg": int(leg_index),
    }


def _relay_bias(prior: np.ndarray, previous_llr: np.ndarray, gamma: np.ndarray, *, clip_B: float, llr_clip: float) -> np.ndarray:
    clipped_prev = np.clip(np.asarray(previous_llr, dtype=np.float64), -float(clip_B), float(clip_B))
    bias = (1.0 - np.asarray(gamma, dtype=np.float64)) * np.asarray(prior, dtype=np.float64) + np.asarray(gamma, dtype=np.float64) * clipped_prev
    return np.clip(bias, -float(llr_clip), float(llr_clip))


def _run_minsum_with_trace_mean(
    *,
    graph: TannerGraph,
    syndrome_bits: np.ndarray,
    prior_llr: np.ndarray,
    config: DecoderConfig,
    mean_tail: int,
    relay_options: _RelayOptions | None = None,
    relay_separator_local: np.ndarray | None = None,
    relay_carry_local: np.ndarray | None = None,
    relay_modified_local: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, int, np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    config.validate("minsum")
    target = np.asarray(syndrome_bits, dtype=np.uint8).reshape(-1) & 1
    if int(target.size) != graph.m:
        raise ValueError(f"syndrome length mismatch: got {target.size}, expected {graph.m}")
    prior = np.asarray(prior_llr, dtype=np.float64).reshape(-1).copy()
    if int(prior.size) != graph.n:
        raise ValueError(f"prior_llr length mismatch: got {prior.size}, expected {graph.n}")

    if graph.n_edges == 0:
        hard = (prior < 0.0).astype(np.uint8)
        residual = (graph.syndrome_from_bits(hard) ^ target).astype(np.uint8)
        temporal = _empty_temporal_stats(graph.n)
        return hard, prior, prior.copy(), bool(np.count_nonzero(residual) == 0), 0, residual, temporal, {
            "activated": False,
            "activation_iter": 0,
            "legs_used": 0,
            "stable_count": 0,
            "bulk_count": int(graph.n),
            "frontier_count": 0,
            "erased_edge_count_by_iter": [],
            "erased_edge_total": 0,
        }

    tail_window = max(1, min(int(config.max_iter), int(mean_tail)))
    llr_tail: list[np.ndarray] = []
    schedule = config.normalized_schedule()
    m_cv = np.zeros(graph.n_edges, dtype=np.float64)
    converged = False
    iterations = 0
    erased_edge_count_by_iter: list[int] = []
    use_scms = bool(config.self_corrected)
    temporal = _empty_temporal_stats(graph.n)
    previous_llr = prior.copy()
    relay_enabled = bool(relay_options is not None and relay_options.enabled)
    relay_active = False
    relay_activation_iter = 0
    relay_elapsed_iters = 0
    relay_legs_used = 0
    relay_counts = {"stable_count": 0, "bulk_count": int(graph.n), "frontier_count": 0}
    frontier_mask = np.zeros(graph.n, dtype=bool)
    bias = prior.copy()
    best_residual_weight = graph.m + 1
    stall_rounds = 0
    separator_local = np.asarray(relay_separator_local if relay_separator_local is not None else np.zeros(0, dtype=np.int32), dtype=np.int32)
    carry_local = np.asarray(relay_carry_local if relay_carry_local is not None else np.zeros(0, dtype=np.int32), dtype=np.int32)
    modified_local = np.asarray(relay_modified_local if relay_modified_local is not None else np.zeros(0, dtype=np.int32), dtype=np.int32)

    def record_llr(llr_vec: np.ndarray) -> None:
        nonlocal previous_llr
        current = np.asarray(llr_vec, dtype=np.float64).copy()
        temporal["llr_flip_count"] += _llr_sign_flip_mask(previous_llr, current).astype(np.int32)
        temporal["llr_abs_delta_sum"] += np.abs(current - previous_llr)
        previous_llr = current
        llr_tail.append(current)
        if len(llr_tail) > int(tail_window):
            llr_tail.pop(0)

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
                new_cv[edges] = _check_update_minsum(
                    m_vc[edges],
                    int(target[check]),
                    m_cv[edges],
                    normalization=float(config.normalization),
                    offset=float(config.offset),
                    damping=float(config.damping),
                )
            m_cv = new_cv
            llr = prior.copy()
            np.add.at(llr, graph.edge_var, m_cv)
            record_llr(llr)
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
    else:
        llr = prior.copy()
        residual = np.ones(graph.m, dtype=np.uint8)
        prev_vc = prior[graph.edge_var].copy()
        m_vc = prev_vc.copy()
        for it in range(1, int(config.max_iter) + 1):
            iterations = int(it)
            if use_scms:
                v2c_snapshot = m_vc.copy()
                erased_mask_iter = np.zeros(graph.n_edges, dtype=bool)
            if relay_enabled and relay_active:
                current_leg = min(int(relay_options.num_legs), 1 + int(relay_elapsed_iters) // int(relay_options.leg_iters))
                relay_legs_used = max(int(relay_legs_used), int(current_leg))
                mean_hint = _tail_mean_llr(llr_tail, llr)
                gamma, frontier_mask, profile_counts = _build_relay_gamma_profile(
                    graph=graph,
                    residual=residual,
                    mean_llr=mean_hint,
                    carry_local=carry_local,
                    separator_local=separator_local,
                    modified_local=modified_local,
                    temporal_stats=temporal,
                    relay_options=relay_options,
                    leg_index=int(current_leg),
                )
                bias_next = _relay_bias(
                    prior,
                    llr,
                    gamma,
                    clip_B=float(relay_options.clip_B),
                    llr_clip=float(config.llr_clip),
                )
                llr += bias_next - bias
                bias = bias_next
                relay_counts = {
                    "stable_count": int(profile_counts["stable_count"]),
                    "bulk_count": int(profile_counts["bulk_count"]),
                    "frontier_count": int(profile_counts["frontier_count"]),
                }
                if use_scms:
                    for var_index in range(graph.n):
                        var_edges = graph.var_to_edges[var_index]
                        if var_edges.size == 0:
                            continue
                        candidate = llr[var_index] - m_cv[var_edges]
                        updated, erased_mask = _apply_scms_erasure(candidate, v2c_snapshot[var_edges])
                        m_vc[var_edges] = updated
                        erased_mask_iter[var_edges] = erased_mask
                        prev_vc[var_edges] = updated
            for check, edges in enumerate(graph.check_to_edges):
                if edges.size == 0:
                    continue
                vars_for_check = graph.edge_var[edges]
                incoming = m_vc[edges] if use_scms else llr[vars_for_check] - m_cv[edges]
                if relay_enabled:
                    flip_mask = _llr_sign_flip_mask(prev_vc[edges], incoming)
                    if relay_active and np.any(flip_mask):
                        flip_mask &= frontier_mask[vars_for_check]
                        if np.any(flip_mask):
                            incoming = _apply_flip_mode(
                                incoming,
                                prev_vc[edges],
                                flip_mask,
                                relay_options,
                                llr_clip=float(config.llr_clip),
                            )
                            np.add.at(temporal["message_flip_count"], vars_for_check[flip_mask], 1)
                    prev_vc[edges] = incoming
                new = _check_update_minsum(
                    incoming,
                    int(target[check]),
                    m_cv[edges],
                    normalization=float(config.normalization),
                    offset=float(config.offset),
                    damping=float(config.damping),
                )
                delta = new - m_cv[edges]
                m_cv[edges] = new
                np.add.at(llr, vars_for_check, delta)
                if use_scms:
                    for var in np.unique(vars_for_check):
                        var_index = int(var)
                        var_edges = graph.var_to_edges[var_index]
                        candidate = llr[var_index] - m_cv[var_edges]
                        updated, erased_mask = _apply_scms_erasure(candidate, v2c_snapshot[var_edges])
                        m_vc[var_edges] = updated
                        erased_mask_iter[var_edges] = erased_mask
                        if relay_enabled:
                            prev_vc[var_edges] = updated
            record_llr(llr)
            hard = (llr < 0.0).astype(np.uint8)
            residual = (graph.syndrome_from_bits(hard) ^ target).astype(np.uint8)
            if use_scms:
                erased_edge_count_by_iter.append(int(np.count_nonzero(erased_mask_iter)))
            if int(np.count_nonzero(residual)) == 0:
                converged = True
                break
            residual_weight = int(np.count_nonzero(residual))
            if relay_enabled and not relay_active:
                if int(residual_weight) < int(best_residual_weight):
                    best_residual_weight = int(residual_weight)
                    stall_rounds = 0
                else:
                    stall_rounds += 1
                rebound = int(residual_weight) - int(best_residual_weight)
                if int(residual_weight) > 0 and (
                    int(stall_rounds) >= int(relay_options.trigger_residual_stall_rounds)
                    or (int(relay_options.trigger_rebound) > 0 and int(rebound) >= int(relay_options.trigger_rebound))
                ):
                    relay_active = True
                    relay_activation_iter = int(it)
                    relay_elapsed_iters = 0
                    relay_legs_used = max(int(relay_legs_used), 1)
            elif relay_enabled and relay_active:
                relay_elapsed_iters += 1

    mean_llr = _tail_mean_llr(llr_tail, prior)
    temporal["residual_touch_count"] = _residual_touch_count(graph, np.flatnonzero(residual))
    temporal["instability_score"] = _relay_instability_score(
        message_flip_count=temporal["message_flip_count"],
        llr_flip_count=temporal["llr_flip_count"],
        llr_abs_delta_sum=temporal["llr_abs_delta_sum"],
        mean_llr=mean_llr,
    )
    return hard, llr, np.asarray(mean_llr, dtype=np.float64), converged, iterations, residual, temporal, {
        "activated": bool(relay_active),
        "activation_iter": int(relay_activation_iter),
        "legs_used": int(relay_legs_used),
        "stable_count": int(relay_counts["stable_count"]),
        "bulk_count": int(relay_counts["bulk_count"]),
        "frontier_count": int(relay_counts["frontier_count"]),
        "erased_edge_count_by_iter": [int(value) for value in erased_edge_count_by_iter],
        "erased_edge_total": int(sum(erased_edge_count_by_iter)),
    }


def _local_variable_shell(sub_matrix: sp.csr_matrix, seeds: np.ndarray, *, hops: int) -> np.ndarray:
    seed_arr = np.asarray(seeds, dtype=np.int32)
    if seed_arr.size == 0 or int(hops) <= 0:
        return np.zeros(0, dtype=np.int32)
    csr = sub_matrix.tocsr()
    csc = sub_matrix.tocsc()
    seen = {int(v) for v in seed_arr.tolist()}
    frontier = set(seen)
    seed_set = set(seen)
    for _ in range(int(hops)):
        if not frontier:
            break
        checks: set[int] = set()
        for var in frontier:
            begin = int(csc.indptr[int(var)])
            end = int(csc.indptr[int(var) + 1])
            checks.update(int(row) for row in csc.indices[begin:end].tolist())
        next_frontier: set[int] = set()
        for check in checks:
            begin = int(csr.indptr[int(check)])
            end = int(csr.indptr[int(check) + 1])
            next_frontier.update(int(col) for col in csr.indices[begin:end].tolist())
        next_frontier -= seen
        seen.update(next_frontier)
        frontier = next_frontier
    shell = sorted(seen - seed_set)
    if not shell:
        return np.zeros(0, dtype=np.int32)
    return np.asarray(shell, dtype=np.int32)


def _variables_near_residual_rows(sub_matrix: sp.csr_matrix, residual_rows: np.ndarray, *, radius: int) -> np.ndarray:
    rows = np.asarray(residual_rows, dtype=np.int32)
    if rows.size == 0 or int(radius) <= 0:
        return np.zeros(0, dtype=np.int32)
    csr = sub_matrix.tocsr()
    csc = sub_matrix.tocsc()
    seen_checks = {int(row) for row in rows.tolist()}
    frontier_checks = set(seen_checks)
    seen_vars: set[int] = set()
    for _ in range(int(radius)):
        if not frontier_checks:
            break
        frontier_vars: set[int] = set()
        for check in frontier_checks:
            begin = int(csr.indptr[int(check)])
            end = int(csr.indptr[int(check) + 1])
            frontier_vars.update(int(col) for col in csr.indices[begin:end].tolist())
        new_vars = frontier_vars - seen_vars
        seen_vars.update(new_vars)
        next_checks: set[int] = set()
        for var in new_vars:
            begin = int(csc.indptr[int(var)])
            end = int(csc.indptr[int(var) + 1])
            next_checks.update(int(row) for row in csc.indices[begin:end].tolist())
        frontier_checks = next_checks - seen_checks
        seen_checks.update(frontier_checks)
    if not seen_vars:
        return np.zeros(0, dtype=np.int32)
    return np.asarray(sorted(seen_vars), dtype=np.int32)


def _variable_residual_touch_count(sub_matrix: sp.csr_matrix, residual_rows: np.ndarray) -> np.ndarray:
    touch = np.zeros(sub_matrix.shape[1], dtype=np.int32)
    rows = np.asarray(residual_rows, dtype=np.int32)
    if rows.size == 0:
        return touch
    csr = sub_matrix.tocsr()
    for check in rows.tolist():
        begin = int(csr.indptr[int(check)])
        end = int(csr.indptr[int(check) + 1])
        if begin < end:
            np.add.at(touch, csr.indices[begin:end], 1)
    return touch


def _select_reliable_shell_clamps(
    *,
    sub_matrix: sp.csr_matrix,
    chosen_local: np.ndarray,
    separator_local: np.ndarray,
    mean_llr: np.ndarray,
    shell_hops: int,
    reliable_topk: int,
    reliable_abs_mean_threshold: float,
) -> dict[int, int]:
    if int(shell_hops) <= 0 or int(reliable_topk) <= 0:
        return {}
    shell_local = _local_variable_shell(sub_matrix, np.asarray(chosen_local, dtype=np.int32), hops=int(shell_hops))
    if shell_local.size == 0:
        return {}
    exclude = set(int(v) for v in np.asarray(separator_local, dtype=np.int32).tolist())
    candidates = [
        int(v)
        for v in shell_local.tolist()
        if int(v) not in exclude and np.isfinite(float(mean_llr[int(v)])) and abs(float(mean_llr[int(v)])) >= float(reliable_abs_mean_threshold)
    ]
    if not candidates:
        return {}
    ordered = sorted(candidates, key=lambda idx: (-abs(float(mean_llr[int(idx)])), int(idx)))
    chosen = ordered[: int(reliable_topk)]
    return {int(idx): int(float(mean_llr[int(idx)]) < 0.0) for idx in chosen}


def _select_branch_candidates(
    *,
    sub_matrix: sp.csr_matrix,
    separator_local: np.ndarray,
    residual: np.ndarray,
    llr: np.ndarray,
    mean_llr: np.ndarray,
    temporal: dict[str, np.ndarray],
    topk: int,
    score_mode: RelayCandidateScore,
    shell_radius: int,
) -> np.ndarray:
    if int(topk) <= 0:
        return np.zeros(0, dtype=np.int32)
    separator = np.asarray(separator_local, dtype=np.int32)
    if separator.size:
        candidates = separator
    else:
        residual_rows = np.flatnonzero(np.asarray(residual, dtype=np.uint8))
        if residual_rows.size == 0:
            return np.zeros(0, dtype=np.int32)
        candidates = _variables_near_residual_rows(sub_matrix, residual_rows, radius=max(1, int(shell_radius)))
    if candidates.size == 0:
        return np.zeros(0, dtype=np.int32)

    residual_touch = _variable_residual_touch_count(sub_matrix, np.flatnonzero(np.asarray(residual, dtype=np.uint8)))
    instability = np.asarray(temporal.get("instability_score", np.zeros(sub_matrix.shape[1], dtype=np.float64)), dtype=np.float64)
    mean_abs = np.abs(np.asarray(mean_llr, dtype=np.float64))
    final_abs = np.abs(np.asarray(llr, dtype=np.float64))
    separator_set = {int(idx) for idx in separator.tolist()}

    if str(score_mode) == "final_absllr":
        ordered = sorted(candidates.tolist(), key=lambda idx: (float(final_abs[int(idx)]), -int(residual_touch[int(idx)]), int(idx)))
    elif str(score_mode) == "mean_absllr":
        ordered = sorted(candidates.tolist(), key=lambda idx: (float(mean_abs[int(idx)]), -int(residual_touch[int(idx)]), int(idx)))
    elif str(score_mode) == "temporal_instability":
        ordered = sorted(
            candidates.tolist(),
            key=lambda idx: (-float(instability[int(idx)]), float(mean_abs[int(idx)]), -int(residual_touch[int(idx)]), int(idx)),
        )
    else:
        ordered = sorted(
            candidates.tolist(),
            key=lambda idx: (
                -(
                    4.0 * float(int(idx) in separator_set)
                    + 2.0 * float(residual_touch[int(idx)])
                    + float(instability[int(idx)])
                    - 0.25 * float(mean_abs[int(idx)])
                ),
                float(mean_abs[int(idx)]),
                int(idx),
            ),
        )
    chosen = ordered[: int(topk)]
    return np.asarray(chosen, dtype=np.int32)


def _candidate_key(candidate: dict[str, object]) -> tuple[int, int, float, int, int]:
    return (
        int(candidate.get("success_weight", 0)),
        int(candidate.get("total_weight", 0)),
        float(candidate.get("score", 0.0)),
        int(candidate.get("separator_columns", 0)),
        -int(candidate.get("reliable_clamps", 0)),
    )


def _iter_assignments_low_weight(width: int, *, limit: int) -> list[tuple[int, ...]]:
    if int(width) <= 0 or int(limit) <= 0:
        return []
    out: list[tuple[int, ...]] = [tuple(0 for _ in range(int(width)))]
    if len(out) >= int(limit):
        return out
    for weight in range(1, int(width) + 1):
        for ones in itertools.combinations(range(int(width)), weight):
            bits = [0] * int(width)
            for idx in ones:
                bits[int(idx)] = 1
            out.append(tuple(int(bit) for bit in bits))
            if len(out) >= int(limit):
                return out
    return out


def _active_round_components(metadata: SplitSectorMetadata, residual: np.ndarray) -> list[list[int]]:
    active_rounds = sorted({int(metadata.detector_round_index[row]) for row in np.flatnonzero(residual)})
    if not active_rounds:
        return []
    graph = nx.Graph()
    graph.add_nodes_from(active_rounds)
    for round_index in active_rounds:
        if round_index + 1 in active_rounds:
            graph.add_edge(round_index, round_index + 1)
    return [sorted(int(node) for node in component) for component in nx.connected_components(graph)]


def _expand_round_component(component: list[int], total_rounds: int, round_radius: int) -> list[int]:
    expanded: set[int] = set()
    for round_index in component:
        lo = max(0, int(round_index) - int(round_radius))
        hi = min(int(total_rounds) - 1, int(round_index) + int(round_radius))
        expanded.update(range(lo, hi + 1))
    return sorted(expanded)


def decode_split_xz(
    problem: SplitSectorProblem,
    syndrome: SplitSectorSyndrome,
    priors: SplitSectorPriors,
    *,
    window: DecoderWindow | None = None,
    decoder: DecoderName = "bp",
    scms: bool = False,
) -> SplitSectorDecodeResult:
    return SplitXZDecoder(problem, scms=bool(scms)).decode_split_xz(syndrome, priors, window=window, decoder=decoder)
