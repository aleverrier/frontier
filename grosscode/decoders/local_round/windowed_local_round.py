from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from grosscode.model import GrossSideModel, RepeatedSyndromeShot, llr_from_prob

from ..common import DecoderRunResult
from .base import LocalRoundInput
from .topk import TopKLocalRoundFactor


def _safe_extrinsic(posterior: np.ndarray, prior: np.ndarray) -> np.ndarray:
    post = np.asarray(posterior, dtype=np.float64)
    prev = np.asarray(prior, dtype=np.float64)
    diff = post - prev
    same_infinite = np.isinf(post) & np.isinf(prev) & (np.signbit(post) == np.signbit(prev))
    diff[same_infinite] = 0.0
    return np.nan_to_num(diff, nan=0.0, posinf=64.0, neginf=-64.0)


@dataclass
class WindowedLocalRoundDecoder:
    model: GrossSideModel
    factor: TopKLocalRoundFactor
    window_size: int = 3
    sweeps: int = 3
    belief_damping: float = 0.5
    label: str = "windowed_local_round"

    def decode(self, shot: RepeatedSyndromeShot) -> DecoderRunResult:
        rounds = int(shot.rounds)
        n = int(self.model.n_data)
        m = int(self.model.n_checks)
        meas_prior = np.full(m, llr_from_prob(shot.p_meas), dtype=np.float64)
        data_prior = np.full(n, llr_from_prob(shot.p), dtype=np.float64)
        strong_zero = np.full(m, 64.0, dtype=np.float64)
        committed_q = np.zeros((rounds, n), dtype=np.uint8)
        committed_m = np.zeros((rounds, m), dtype=np.uint8)
        left_boundary_llr = strong_zero.copy()
        total_work = 0
        total_edges = 0
        factor_calls = 0
        t0 = time.perf_counter()
        try:
            for start in range(rounds):
                end = min(rounds, start + max(1, int(self.window_size)))
                win_len = int(end - start)
                beliefs = [meas_prior.copy() for _ in range(win_len)]
                last_outputs = []
                for _sweep in range(max(1, int(self.sweeps))):
                    left_msgs = [np.zeros(m, dtype=np.float64) for _ in range(win_len)]
                    right_msgs = [np.zeros(m, dtype=np.float64) for _ in range(win_len)]
                    local_outputs = []
                    for offset in range(win_len):
                        global_round = start + offset
                        left_prior = left_boundary_llr if offset == 0 else beliefs[offset - 1]
                        right_prior = beliefs[offset]
                        round_input = LocalRoundInput(
                            detector_slice=np.asarray(shot.detector_slices[global_round], dtype=np.uint8),
                            data_prior_llr=data_prior,
                            left_interface_llr=left_prior,
                            right_interface_llr=right_prior,
                            left_clamp_mask=np.ones(m, dtype=bool) if global_round == 0 else None,
                            left_clamp_values=np.zeros(m, dtype=np.uint8) if global_round == 0 else None,
                            metadata={"round_index": int(global_round), "window_start": int(start)},
                        )
                        out = self.factor.infer(round_input)
                        factor_calls += 1
                        total_work += int(out.work_iterations)
                        total_edges += int(out.edge_updates)
                        left_msgs[offset] = _safe_extrinsic(out.left_interface_llr, left_prior)
                        right_msgs[offset] = _safe_extrinsic(out.right_interface_llr, right_prior)
                        local_outputs.append(out)
                    for offset in range(win_len):
                        updated = meas_prior.copy()
                        updated += right_msgs[offset]
                        if offset + 1 < win_len:
                            updated += left_msgs[offset + 1]
                        beliefs[offset] = (
                            float(self.belief_damping) * beliefs[offset]
                            + (1.0 - float(self.belief_damping)) * updated
                        )
                    last_outputs = local_outputs
                first_output = last_outputs[0]
                committed_q[start] = np.asarray(first_output.data_estimate, dtype=np.uint8)
                committed_m[start] = (np.asarray(beliefs[0], dtype=np.float64) < 0.0).astype(np.uint8)
                left_boundary_llr = beliefs[0].copy()
            correction = np.bitwise_xor.reduce(committed_q, axis=0).astype(np.uint8)
            if not self.model.detector_slices_match(committed_q, committed_m, shot.detector_slices):
                status = "syndrome_fail"
            else:
                status = str(self.model.logical_status(shot.total_data, correction))
        except Exception as exc:
            committed_q = np.zeros((rounds, n), dtype=np.uint8)
            committed_m = np.zeros((rounds, m), dtype=np.uint8)
            correction = np.zeros(n, dtype=np.uint8)
            status = "exception_fail"
            return DecoderRunResult(
                decoder=self.label,
                status=status,
                correction=correction,
                data_fault_estimate=committed_q,
                measurement_fault_estimate=committed_m,
                runtime_ms=1000.0 * float(time.perf_counter() - t0),
                work_iterations=int(total_work),
                edge_updates=int(total_edges),
                metadata={"exception": repr(exc), "factor_calls": int(factor_calls)},
            )
        return DecoderRunResult(
            decoder=self.label,
            status=status,
            correction=correction,
            data_fault_estimate=committed_q,
            measurement_fault_estimate=committed_m,
            runtime_ms=1000.0 * float(time.perf_counter() - t0),
            work_iterations=int(total_work),
            edge_updates=int(total_edges),
            metadata={
                "factor_calls": int(factor_calls),
                "window_size": int(self.window_size),
                "sweeps": int(self.sweeps),
                "topk_interface_bits": int(self.factor.topk_interface_bits),
            },
        )
