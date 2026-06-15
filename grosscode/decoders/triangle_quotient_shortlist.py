from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import scipy.sparse as sp

from grosscode.decoders.triangle_quotient_minsum import (
    TriangleQuotientDecodeResult,
    TriangleQuotientMinSumConfig,
    TriangleQuotientMinSumDecoder,
)
from grosscode.dem.reference_recovery import AugmentedReferenceRecoverySolver
from grosscode.dem.triangle_basis import TriangleBasisArtifact
from grosscode.utils.gf2 import binary_csr_mod2, csr_matvec_mod2, dense_mod2


def _logical_word_from_bits(bits: np.ndarray) -> int:
    word = 0
    for idx, bit in enumerate(dense_mod2(bits).reshape(-1).tolist()):
        if int(bit) != 0:
            word |= 1 << int(idx)
    return int(word)


def _logical_bits_from_word(word: int, width: int) -> np.ndarray:
    return np.asarray([(int(word) >> idx) & 1 for idx in range(int(width))], dtype=np.uint8)


def _objective_cost_from_error(cost_zero: np.ndarray, cost_one: np.ndarray, error_bits: np.ndarray) -> float:
    bits = dense_mod2(error_bits).reshape(-1)
    return float(
        np.sum(
            np.where(bits > 0, np.asarray(cost_one, dtype=np.float64), np.asarray(cost_zero, dtype=np.float64)),
            dtype=np.float64,
        )
    )


@dataclass(frozen=True)
class LogicalSectorShortlistConfig:
    top_singletons: int = 4
    pair_source_size: int = 4
    top_pairs: int = 2
    max_candidates: int = 6
    always_include_base: bool = True


@dataclass(frozen=True)
class LogicalSectorProposal:
    logical_bits: np.ndarray
    logical_word: int
    relative_mask: int
    relative_weight: int
    score: float
    toggle_weight: int


@dataclass(frozen=True)
class TriangleQuotientShortlistConfig:
    shortlist: LogicalSectorShortlistConfig = field(default_factory=LogicalSectorShortlistConfig)
    screen_config: TriangleQuotientMinSumConfig = field(
        default_factory=lambda: TriangleQuotientMinSumConfig(
            max_iter=8,
            damping=0.10,
            stable_rounds_required=1,
        )
    )
    refine_config: TriangleQuotientMinSumConfig = field(
        default_factory=lambda: TriangleQuotientMinSumConfig(
            max_iter=20,
            damping=0.15,
            stable_rounds_required=2,
        )
    )
    refine_top_k: int = 2


@dataclass(frozen=True)
class TriangleQuotientShortlistCandidateResult:
    proposal: LogicalSectorProposal
    proposer_rank: int
    screen_result: TriangleQuotientDecodeResult
    final_result: TriangleQuotientDecodeResult
    refined: bool


@dataclass(frozen=True)
class TriangleQuotientShortlistDecodeResult:
    final_decode: TriangleQuotientDecodeResult
    selected_proposal: LogicalSectorProposal
    selected_candidate_rank: int
    seed_logical_bits: np.ndarray
    seed_logical_word: int
    seed_syndrome_valid: bool
    candidate_results: tuple[TriangleQuotientShortlistCandidateResult, ...]
    screened_count: int
    refined_count: int
    screen_edge_update_total: int
    refine_edge_update_total: int
    selected_from_refine: bool


class LogicalSectorShortlistProposer:
    def __init__(
        self,
        *,
        logical_matrix: sp.csr_matrix,
        priors: np.ndarray,
        reference_solver: AugmentedReferenceRecoverySolver,
        config: LogicalSectorShortlistConfig | None = None,
    ) -> None:
        cfg = LogicalSectorShortlistConfig() if config is None else config
        if int(cfg.max_candidates) <= 0:
            raise ValueError("max_candidates must be positive")
        if int(cfg.top_singletons) < 0:
            raise ValueError("top_singletons must be non-negative")
        if int(cfg.pair_source_size) < 0:
            raise ValueError("pair_source_size must be non-negative")
        if int(cfg.top_pairs) < 0:
            raise ValueError("top_pairs must be non-negative")
        self.config = cfg
        self.logical_matrix = binary_csr_mod2(logical_matrix).tocsr()
        self.reference_solver = reference_solver
        self.n = int(self.logical_matrix.shape[1])
        self.logical_dim = int(self.logical_matrix.shape[0])
        clipped = np.clip(np.asarray(priors, dtype=np.float64).reshape(-1), 1e-15, 1.0 - 1e-15)
        self.cost_zero = -np.log(1.0 - clipped)
        self.cost_one = -np.log(clipped)
        zero_syndrome = np.zeros(int(self.reference_solver.detector_rows), dtype=np.uint8)
        self.logical_toggle_vectors = tuple(
            self.reference_solver.solve_reference(
                zero_syndrome,
                _logical_bits_from_word(1 << int(bit_idx), int(self.logical_dim)),
            )
            for bit_idx in range(int(self.logical_dim))
        )
        self._toggle_cache: dict[int, np.ndarray] = {0: np.zeros(int(self.n), dtype=np.uint8)}

    def _toggle_vector(self, relative_mask: int) -> np.ndarray:
        key = int(relative_mask)
        cached = self._toggle_cache.get(key)
        if cached is not None:
            return np.asarray(cached, dtype=np.uint8)
        out = np.zeros(int(self.n), dtype=np.uint8)
        changed = int(key)
        while int(changed) > 0:
            bit_idx = int(changed.bit_length() - 1)
            out ^= np.asarray(self.logical_toggle_vectors[int(bit_idx)], dtype=np.uint8)
            changed ^= 1 << int(bit_idx)
        self._toggle_cache[key] = out
        return np.asarray(out, dtype=np.uint8)

    def _score_relative_mask(self, *, center_estimate: np.ndarray, relative_mask: int) -> tuple[float, int]:
        toggle = self._toggle_vector(int(relative_mask))
        candidate = dense_mod2(center_estimate).reshape(-1) ^ toggle
        return (
            _objective_cost_from_error(self.cost_zero, self.cost_one, candidate),
            int(np.count_nonzero(toggle)),
        )

    def propose(
        self,
        *,
        center_estimate: np.ndarray,
        base_logical: np.ndarray,
    ) -> tuple[LogicalSectorProposal, ...]:
        center = dense_mod2(center_estimate).reshape(-1)
        base_bits = dense_mod2(base_logical).reshape(-1)
        if int(center.size) != int(self.n):
            raise ValueError("center_estimate size mismatch")
        if int(base_bits.size) != int(self.logical_dim):
            raise ValueError("base_logical size mismatch")

        base_word = _logical_word_from_bits(base_bits)
        candidate_info: dict[int, tuple[float, int, int]] = {}

        base_score, base_toggle_weight = self._score_relative_mask(center_estimate=center, relative_mask=0)
        candidate_info[0] = (float(base_score), 0, int(base_toggle_weight))

        singleton_rows: list[tuple[float, int, int]] = []
        for bit_idx in range(int(self.logical_dim)):
            mask = 1 << int(bit_idx)
            score, toggle_weight = self._score_relative_mask(center_estimate=center, relative_mask=mask)
            singleton_rows.append((float(score), int(bit_idx), int(toggle_weight)))
            candidate_info[int(mask)] = (float(score), 1, int(toggle_weight))
        singleton_rows.sort(key=lambda item: (float(item[0]), int(item[1])))

        source_bits = [int(item[1]) for item in singleton_rows[: int(self.config.pair_source_size)]]
        pair_rows: list[tuple[float, int, int]] = []
        for left_bit, right_bit in combinations(source_bits, 2):
            mask = (1 << int(left_bit)) | (1 << int(right_bit))
            score, toggle_weight = self._score_relative_mask(center_estimate=center, relative_mask=mask)
            pair_rows.append((float(score), int(mask), int(toggle_weight)))
        pair_rows.sort(key=lambda item: (float(item[0]), int(item[1])))
        for score, mask, toggle_weight in pair_rows[: int(self.config.top_pairs)]:
            candidate_info[int(mask)] = (float(score), 2, int(toggle_weight))

        ordered_masks: list[int] = []
        if bool(self.config.always_include_base):
            ordered_masks.append(0)
        remaining_masks = sorted(
            [int(mask) for mask in candidate_info.keys() if int(mask) != 0],
            key=lambda mask: (
                float(candidate_info[int(mask)][0]),
                int(candidate_info[int(mask)][1]),
                int(base_word ^ int(mask)),
            ),
        )
        for mask in remaining_masks:
            if int(mask) == 0 and bool(self.config.always_include_base):
                continue
            ordered_masks.append(int(mask))
            if int(len(ordered_masks)) >= int(self.config.max_candidates):
                break
        if not ordered_masks:
            ordered_masks = [0]

        proposals: list[LogicalSectorProposal] = []
        for relative_mask in ordered_masks:
            score, relative_weight, toggle_weight = candidate_info[int(relative_mask)]
            logical_word = int(base_word ^ int(relative_mask))
            proposals.append(
                LogicalSectorProposal(
                    logical_bits=_logical_bits_from_word(logical_word, int(self.logical_dim)),
                    logical_word=int(logical_word),
                    relative_mask=int(relative_mask),
                    relative_weight=int(relative_weight),
                    score=float(score),
                    toggle_weight=int(toggle_weight),
                )
            )
        return tuple(proposals)


class TriangleQuotientShortlistDecoder:
    def __init__(
        self,
        *,
        detector_matrix: sp.csr_matrix,
        logical_matrix: sp.csr_matrix,
        priors: np.ndarray,
        basis_artifact: TriangleBasisArtifact,
        reference_solver: AugmentedReferenceRecoverySolver | None = None,
        proposer: LogicalSectorShortlistProposer | None = None,
        config: TriangleQuotientShortlistConfig | None = None,
    ) -> None:
        cfg = TriangleQuotientShortlistConfig() if config is None else config
        if int(cfg.refine_top_k) < 0:
            raise ValueError("refine_top_k must be non-negative")
        self.config = cfg
        self.detector_matrix = binary_csr_mod2(detector_matrix).tocsr()
        self.logical_matrix = binary_csr_mod2(logical_matrix).tocsr()
        self.reference_solver = (
            AugmentedReferenceRecoverySolver.build(detector_matrix=self.detector_matrix, logical_matrix=self.logical_matrix)
            if reference_solver is None
            else reference_solver
        )
        self.proposer = (
            LogicalSectorShortlistProposer(
                logical_matrix=self.logical_matrix,
                priors=np.asarray(priors, dtype=np.float64),
                reference_solver=self.reference_solver,
                config=cfg.shortlist,
            )
            if proposer is None
            else proposer
        )
        self.screen_decoder = TriangleQuotientMinSumDecoder(
            detector_matrix=self.detector_matrix,
            logical_matrix=self.logical_matrix,
            priors=np.asarray(priors, dtype=np.float64),
            basis_artifact=basis_artifact,
            reference_solver=self.reference_solver,
            config=cfg.screen_config,
        )
        self.refine_decoder = TriangleQuotientMinSumDecoder(
            detector_matrix=self.detector_matrix,
            logical_matrix=self.logical_matrix,
            priors=np.asarray(priors, dtype=np.float64),
            basis_artifact=basis_artifact,
            reference_solver=self.reference_solver,
            config=cfg.refine_config,
        )

    def decode_from_seed(
        self,
        *,
        syndrome: np.ndarray,
        seed_estimate: np.ndarray,
        seed_logical: np.ndarray | None = None,
    ) -> TriangleQuotientShortlistDecodeResult:
        syndrome_bits = dense_mod2(syndrome).reshape(-1)
        seed_bits = dense_mod2(seed_estimate).reshape(-1)
        if int(syndrome_bits.size) != int(self.detector_matrix.shape[0]):
            raise ValueError("syndrome length mismatch")
        if int(seed_bits.size) != int(self.detector_matrix.shape[1]):
            raise ValueError("seed_estimate length mismatch")
        seed_logical_bits = (
            csr_matvec_mod2(self.logical_matrix, seed_bits)
            if seed_logical is None
            else dense_mod2(seed_logical).reshape(-1)
        )
        if int(seed_logical_bits.size) != int(self.logical_matrix.shape[0]):
            raise ValueError("seed_logical length mismatch")

        proposals = self.proposer.propose(center_estimate=seed_bits, base_logical=seed_logical_bits)
        candidate_results: list[TriangleQuotientShortlistCandidateResult] = []
        screen_edge_update_total = 0
        for rank, proposal in enumerate(proposals):
            screen_result = self.screen_decoder.decode(syndrome=syndrome_bits, target_logical=proposal.logical_bits)
            screen_edge_update_total += int(screen_result.edge_update_proxy)
            candidate_results.append(
                TriangleQuotientShortlistCandidateResult(
                    proposal=proposal,
                    proposer_rank=int(rank),
                    screen_result=screen_result,
                    final_result=screen_result,
                    refined=False,
                )
            )

        refine_count = min(int(self.config.refine_top_k), int(len(candidate_results)))
        refine_edge_update_total = 0
        if int(refine_count) > 0:
            refine_indices = sorted(
                range(len(candidate_results)),
                key=lambda idx: (
                    float(candidate_results[int(idx)].screen_result.objective_cost),
                    0 if bool(candidate_results[int(idx)].screen_result.converged) else 1,
                    int(candidate_results[int(idx)].proposal.relative_weight),
                    int(candidate_results[int(idx)].proposal.logical_word),
                ),
            )[: int(refine_count)]
            for idx in refine_indices:
                proposal = candidate_results[int(idx)].proposal
                refine_result = self.refine_decoder.decode(syndrome=syndrome_bits, target_logical=proposal.logical_bits)
                refine_edge_update_total += int(refine_result.edge_update_proxy)
                candidate_results[int(idx)] = TriangleQuotientShortlistCandidateResult(
                    proposal=proposal,
                    proposer_rank=int(candidate_results[int(idx)].proposer_rank),
                    screen_result=candidate_results[int(idx)].screen_result,
                    final_result=refine_result,
                    refined=True,
                )

        best_index = min(
            range(len(candidate_results)),
            key=lambda idx: (
                float(candidate_results[int(idx)].final_result.objective_cost),
                0 if bool(candidate_results[int(idx)].final_result.converged) else 1,
                int(candidate_results[int(idx)].proposal.relative_weight),
                int(candidate_results[int(idx)].proposal.logical_word),
            ),
        )
        best = candidate_results[int(best_index)]
        return TriangleQuotientShortlistDecodeResult(
            final_decode=best.final_result,
            selected_proposal=best.proposal,
            selected_candidate_rank=int(best.proposer_rank),
            seed_logical_bits=np.asarray(seed_logical_bits, dtype=np.uint8),
            seed_logical_word=int(_logical_word_from_bits(seed_logical_bits)),
            seed_syndrome_valid=bool(np.array_equal(csr_matvec_mod2(self.detector_matrix, seed_bits), syndrome_bits)),
            candidate_results=tuple(candidate_results),
            screened_count=int(len(candidate_results)),
            refined_count=int(sum(1 for item in candidate_results if bool(item.refined))),
            screen_edge_update_total=int(screen_edge_update_total),
            refine_edge_update_total=int(refine_edge_update_total),
            selected_from_refine=bool(best.refined),
        )
