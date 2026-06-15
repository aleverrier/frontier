from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Tuple

import ldpc  # type: ignore
import numpy as np

from clamped_decoder import MinSumSyndromeDecoder
from grosscode.model import GrossSideModel, RepeatedSyndromeShot

from .common import DecoderRunResult


def _hard_bits_from_llr(llr: np.ndarray) -> np.ndarray:
    return (np.asarray(llr, dtype=np.float64) < 0.0).astype(np.uint8)


def _status_from_estimate(
    *,
    model: GrossSideModel,
    shot: RepeatedSyndromeShot,
    q_hat: np.ndarray,
    m_hat: np.ndarray,
    correction: np.ndarray,
) -> str:
    if not model.detector_slices_match(q_hat, m_hat, shot.detector_slices):
        return "syndrome_fail"
    return str(model.logical_status(shot.total_data, correction))


@dataclass
class FullBlockMinSumDecoder:
    model: GrossSideModel
    max_iter: int = 80
    alpha: float = 0.75
    beta: float = 0.0
    damp: float = 0.0
    schedule: str = "layered"
    label: str = "full_block_minsum"
    _decoder_cache: Dict[int, MinSumSyndromeDecoder] = field(default_factory=dict, init=False, repr=False)

    def _get_decoder(self, rounds: int) -> MinSumSyndromeDecoder:
        key = int(rounds)
        cached = self._decoder_cache.get(key)
        if cached is not None:
            return cached
        decoder = MinSumSyndromeDecoder(
            self.model.full_block_pcm(key),
            alpha=float(self.alpha),
            beta=float(self.beta),
            damp=float(self.damp),
            schedule=str(self.schedule),
            clamp_strength=float("inf"),
        )
        self._decoder_cache[key] = decoder
        return decoder

    def decode(self, shot: RepeatedSyndromeShot) -> DecoderRunResult:
        decoder = self._get_decoder(shot.rounds)
        prior_llr = self.model.full_block_prior_llr(rounds=shot.rounds, p_data=shot.p, p_meas=shot.p_meas)
        syndrome = np.asarray(shot.detector_slices, dtype=np.uint8).reshape(-1)
        t0 = time.perf_counter()
        try:
            estimate, _ok, _post = decoder.decode(
                syndrome,
                p=shot.p,
                max_iter=int(self.max_iter),
                prior_llr=prior_llr,
            )
            q_hat, m_hat = self.model.unpack_full_block_estimate(estimate, shot.rounds)
            correction = np.bitwise_xor.reduce(q_hat, axis=0).astype(np.uint8)
            status = _status_from_estimate(model=self.model, shot=shot, q_hat=q_hat, m_hat=m_hat, correction=correction)
        except Exception as exc:
            q_hat = np.zeros((shot.rounds, self.model.n_data), dtype=np.uint8)
            m_hat = np.zeros((shot.rounds, self.model.n_checks), dtype=np.uint8)
            correction = np.zeros(self.model.n_data, dtype=np.uint8)
            status = "exception_fail"
            return DecoderRunResult(
                decoder=self.label,
                status=status,
                correction=correction,
                data_fault_estimate=q_hat,
                measurement_fault_estimate=m_hat,
                runtime_ms=1000.0 * float(time.perf_counter() - t0),
                work_iterations=0,
                edge_updates=0,
                metadata={"exception": repr(exc)},
            )
        return DecoderRunResult(
            decoder=self.label,
            status=status,
            correction=correction,
            data_fault_estimate=q_hat,
            measurement_fault_estimate=m_hat,
            runtime_ms=1000.0 * float(time.perf_counter() - t0),
            work_iterations=int(decoder.last_decode_iterations),
            edge_updates=int(decoder.last_decode_edge_work_pess),
            metadata={"bp_converged": bool(int(decoder.last_decode_iterations) < int(self.max_iter))},
        )


@dataclass
class ImportedBpOsdDecoder:
    model: GrossSideModel
    max_iter: int = 80
    alpha: float = 0.75
    osd_order: int = 0
    label: str = "baseline_bposd"
    _decoder_cache: Dict[int, object] = field(default_factory=dict, init=False, repr=False)

    def _get_decoder(self, rounds: int) -> object:
        key = int(rounds)
        cached = self._decoder_cache.get(key)
        if cached is not None:
            return cached
        decoder = ldpc.BpOsdDecoder(
            self.model.full_block_pcm(key),
            error_rate=0.05,
            max_iter=int(self.max_iter),
            bp_method="minimum_sum",
            ms_scaling_factor=float(self.alpha),
            osd_order=int(self.osd_order),
        )
        self._decoder_cache[key] = decoder
        return decoder

    def decode(self, shot: RepeatedSyndromeShot) -> DecoderRunResult:
        decoder = self._get_decoder(shot.rounds)
        prior_probs = np.concatenate(
            [
                np.full(shot.rounds * self.model.n_data, shot.p, dtype=np.float64),
                np.full(shot.rounds * self.model.n_checks, shot.p_meas, dtype=np.float64),
            ]
        )
        syndrome = np.asarray(shot.detector_slices, dtype=np.uint8).reshape(-1)
        t0 = time.perf_counter()
        try:
            if hasattr(decoder, "update_channel_probs"):
                decoder.update_channel_probs(prior_probs)
            estimate = np.asarray(decoder.decode(syndrome.copy()), dtype=np.uint8).reshape(-1)
            q_hat, m_hat = self.model.unpack_full_block_estimate(estimate, shot.rounds)
            correction = np.bitwise_xor.reduce(q_hat, axis=0).astype(np.uint8)
            status = _status_from_estimate(model=self.model, shot=shot, q_hat=q_hat, m_hat=m_hat, correction=correction)
            edge_updates = int(getattr(decoder, "iter", 0)) * int(self.model.full_block_pcm(shot.rounds).nnz)
        except Exception as exc:
            q_hat = np.zeros((shot.rounds, self.model.n_data), dtype=np.uint8)
            m_hat = np.zeros((shot.rounds, self.model.n_checks), dtype=np.uint8)
            correction = np.zeros(self.model.n_data, dtype=np.uint8)
            status = "exception_fail"
            return DecoderRunResult(
                decoder=self.label,
                status=status,
                correction=correction,
                data_fault_estimate=q_hat,
                measurement_fault_estimate=m_hat,
                runtime_ms=1000.0 * float(time.perf_counter() - t0),
                work_iterations=0,
                edge_updates=0,
                metadata={"exception": repr(exc)},
            )
        return DecoderRunResult(
            decoder=self.label,
            status=status,
            correction=correction,
            data_fault_estimate=q_hat,
            measurement_fault_estimate=m_hat,
            runtime_ms=1000.0 * float(time.perf_counter() - t0),
            work_iterations=int(getattr(decoder, "iter", 0)),
            edge_updates=int(edge_updates),
            metadata={"bp_converged": bool(getattr(decoder, "converge", False))},
        )
