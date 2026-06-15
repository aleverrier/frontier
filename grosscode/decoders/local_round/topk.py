from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from clamped_decoder import MinSumSyndromeDecoder
from grosscode.model import GrossSideModel, logsumexp, logistic_cost

from .base import LocalRoundFactor, LocalRoundInput, LocalRoundOutput


def _mask_to_index_pairs(mask: np.ndarray | None, values: np.ndarray | None, offset: int) -> Tuple[List[int], List[int]]:
    if mask is None or values is None:
        return [], []
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    val_arr = np.asarray(values, dtype=np.uint8).reshape(-1)
    if mask_arr.size != val_arr.size:
        raise ValueError("clamp mask/value length mismatch")
    indices = [int(offset + idx) for idx in np.flatnonzero(mask_arr)]
    vals = [int(val_arr[idx] & 1) for idx in np.flatnonzero(mask_arr)]
    return indices, vals


def _candidate_llr(bit_values: Sequence[np.ndarray], costs: Sequence[float]) -> np.ndarray:
    if not bit_values:
        raise ValueError("expected at least one candidate")
    n_bits = np.asarray(bit_values[0], dtype=np.uint8).reshape(-1).size
    out = np.zeros(n_bits, dtype=np.float64)
    neg_costs = [-float(x) for x in costs]
    for idx in range(n_bits):
        zero_terms: List[float] = []
        one_terms: List[float] = []
        for bits, nc in zip(bit_values, neg_costs):
            if int(np.asarray(bits, dtype=np.uint8).reshape(-1)[idx]) == 0:
                zero_terms.append(float(nc))
            else:
                one_terms.append(float(nc))
        out[idx] = float(logsumexp(zero_terms) - logsumexp(one_terms))
    return out


@dataclass
class TopKLocalRoundFactor(LocalRoundFactor):
    model: GrossSideModel
    max_iter: int = 40
    alpha: float = 0.75
    beta: float = 0.0
    damp: float = 0.0
    schedule: str = "layered"
    topk_interface_bits: int = 0
    max_enumerated_candidates: int = 64
    _decoder: MinSumSyndromeDecoder | None = field(default=None, init=False, repr=False)

    @property
    def n_data(self) -> int:
        return int(self.model.n_data)

    @property
    def n_checks(self) -> int:
        return int(self.model.n_checks)

    @property
    def left_offset(self) -> int:
        return int(self.n_data)

    @property
    def right_offset(self) -> int:
        return int(self.n_data + self.n_checks)

    def _get_decoder(self) -> MinSumSyndromeDecoder:
        if self._decoder is None:
            self._decoder = MinSumSyndromeDecoder(
                self.model.local_round_pcm(),
                alpha=float(self.alpha),
                beta=float(self.beta),
                damp=float(self.damp),
                schedule=str(self.schedule),
                clamp_strength=float("inf"),
            )
        return self._decoder

    def _build_prior_llr(self, round_input: LocalRoundInput) -> np.ndarray:
        data_prior = np.asarray(round_input.data_prior_llr, dtype=np.float64).reshape(self.n_data)
        left_prior = np.asarray(round_input.left_interface_llr, dtype=np.float64).reshape(self.n_checks)
        right_prior = np.asarray(round_input.right_interface_llr, dtype=np.float64).reshape(self.n_checks)
        return np.concatenate([data_prior, left_prior, right_prior])

    def _decode_once(
        self,
        *,
        decoder: MinSumSyndromeDecoder,
        detector_slice: np.ndarray,
        prior_llr: np.ndarray,
        clamp_vars: Sequence[int],
        clamp_vals: Sequence[int],
    ) -> Tuple[np.ndarray, bool, np.ndarray, int, int]:
        estimate, success, post_llr = decoder.decode(
            np.asarray(detector_slice, dtype=np.uint8).reshape(self.n_checks),
            p=0.05,
            max_iter=int(self.max_iter),
            prior_llr=np.asarray(prior_llr, dtype=np.float64),
            clamp_vars=list(clamp_vars) if clamp_vars else None,
            clamp_vals=list(clamp_vals) if clamp_vals else None,
        )
        return (
            np.asarray(estimate, dtype=np.uint8).reshape(-1),
            bool(success),
            np.asarray(post_llr, dtype=np.float64).reshape(-1),
            int(decoder.last_decode_iterations),
            int(decoder.last_decode_edge_work_pess),
        )

    def infer(self, round_input: LocalRoundInput) -> LocalRoundOutput:
        decoder = self._get_decoder()
        prior_llr = self._build_prior_llr(round_input)
        clamp_vars: List[int] = []
        clamp_vals: List[int] = []
        left_vars, left_vals = _mask_to_index_pairs(
            round_input.left_clamp_mask,
            round_input.left_clamp_values,
            self.left_offset,
        )
        right_vars, right_vals = _mask_to_index_pairs(
            round_input.right_clamp_mask,
            round_input.right_clamp_values,
            self.right_offset,
        )
        clamp_vars.extend(left_vars)
        clamp_vals.extend(left_vals)
        clamp_vars.extend(right_vars)
        clamp_vals.extend(right_vals)

        base_estimate, base_success, base_post, base_iters, base_edges = self._decode_once(
            decoder=decoder,
            detector_slice=round_input.detector_slice,
            prior_llr=prior_llr,
            clamp_vars=clamp_vars,
            clamp_vals=clamp_vals,
        )
        total_iters = int(base_iters)
        total_edges = int(base_edges)
        candidate_estimates = [base_estimate]
        candidate_costs = [logistic_cost(base_estimate, prior_llr)]
        successful_candidates = 1 if base_success else 0

        if int(self.topk_interface_bits) > 0:
            right_post = base_post[self.right_offset : self.right_offset + self.n_checks]
            order = np.argsort(np.abs(right_post))
            branch_count = min(
                int(self.topk_interface_bits),
                int(self.max_enumerated_candidates).bit_length() - 1,
                int(order.size),
            )
            branch_bits = [int(order[idx]) for idx in range(branch_count)]
            candidate_count = min(int(self.max_enumerated_candidates), 1 << int(branch_count))
            base_clamped = set(int(v) for v in clamp_vars)
            for assign_idx in range(candidate_count):
                extra_vars: List[int] = []
                extra_vals: List[int] = []
                for bit_pos, local_bit in enumerate(branch_bits):
                    full_idx = int(self.right_offset + local_bit)
                    if full_idx in base_clamped:
                        continue
                    extra_vars.append(full_idx)
                    extra_vals.append((int(assign_idx) >> bit_pos) & 1)
                if not extra_vars:
                    continue
                est, success, post, iters, edges = self._decode_once(
                    decoder=decoder,
                    detector_slice=round_input.detector_slice,
                    prior_llr=prior_llr,
                    clamp_vars=[*clamp_vars, *extra_vars],
                    clamp_vals=[*clamp_vals, *extra_vals],
                )
                total_iters += int(iters)
                total_edges += int(edges)
                if not success:
                    continue
                successful_candidates += 1
                candidate_estimates.append(est)
                candidate_costs.append(logistic_cost(est, prior_llr))
                base_post = post
                base_estimate = est
                base_success = success

        if successful_candidates > 1:
            data_bits = [cand[: self.n_data] for cand in candidate_estimates]
            left_bits = [cand[self.left_offset : self.left_offset + self.n_checks] for cand in candidate_estimates]
            right_bits = [cand[self.right_offset : self.right_offset + self.n_checks] for cand in candidate_estimates]
            data_llr = _candidate_llr(data_bits, candidate_costs)
            left_llr = _candidate_llr(left_bits, candidate_costs)
            right_llr = _candidate_llr(right_bits, candidate_costs)
        else:
            data_llr = base_post[: self.n_data]
            left_llr = base_post[self.left_offset : self.left_offset + self.n_checks]
            right_llr = base_post[self.right_offset : self.right_offset + self.n_checks]

        data_estimate = (np.asarray(data_llr) < 0.0).astype(np.uint8)
        left_estimate = (np.asarray(left_llr) < 0.0).astype(np.uint8)
        right_estimate = (np.asarray(right_llr) < 0.0).astype(np.uint8)
        return LocalRoundOutput(
            data_llr=np.asarray(data_llr, dtype=np.float64),
            left_interface_llr=np.asarray(left_llr, dtype=np.float64),
            right_interface_llr=np.asarray(right_llr, dtype=np.float64),
            data_estimate=data_estimate,
            left_interface_estimate=left_estimate,
            right_interface_estimate=right_estimate,
            success=bool(base_success),
            work_iterations=int(total_iters),
            edge_updates=int(total_edges),
            metadata={
                "topk_interface_bits": int(self.topk_interface_bits),
                "successful_candidates": int(successful_candidates),
            },
        )
