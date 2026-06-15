from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from grosscode.core import (
    DecoderConfig,
    SideContext,
    SideDecodeResult,
    TannerGraph,
    WindowConfig,
    WindowStepResult,
    run_message_passing,
)


@dataclass
class _BaseSideDecoder:
    context: SideContext
    config: DecoderConfig = DecoderConfig()
    algorithm: str = "minsum"

    def __post_init__(self) -> None:
        self.config.validate(self.algorithm)

    def decode(self, syndrome: np.ndarray, *, prior_llr: Optional[np.ndarray] = None) -> SideDecodeResult:
        prior = self.context.resolve_prior_llr(prior_llr)
        estimate, posterior_llr, converged, iterations, residual = run_message_passing(
            graph=self.context.graph,
            syndrome_bits=np.asarray(syndrome, dtype=np.uint8),
            prior_llr=prior,
            config=self.config,
            algorithm=self.algorithm,
        )
        return SideDecodeResult(
            estimate=np.asarray(estimate, dtype=np.uint8),
            posterior_llr=np.asarray(posterior_llr, dtype=np.float64),
            converged=bool(converged),
            iterations=int(iterations),
            unsatisfied_checks=int(np.count_nonzero(residual)),
            unsatisfied_vector=np.asarray(residual, dtype=np.uint8),
            logical_action=self.context.logical_action_for(estimate),
            window_steps=[],
        )


@dataclass
class _WindowedSideDecoder(_BaseSideDecoder):
    window: WindowConfig = WindowConfig(window_size=512, overlap_size=128)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.window.validate()

    def decode(self, syndrome: np.ndarray, *, prior_llr: Optional[np.ndarray] = None) -> SideDecodeResult:
        target = np.asarray(syndrome, dtype=np.uint8).reshape(-1) & 1
        if int(target.size) != self.context.m:
            raise ValueError(f"syndrome length mismatch: got {target.size}, expected {self.context.m}")
        prior = self.context.resolve_prior_llr(prior_llr)
        residual_syndrome = target.copy()
        estimate = np.zeros(self.context.n, dtype=np.uint8)
        posterior = np.full(self.context.n, np.nan, dtype=np.float64)
        carry_prior = prior.copy()
        window_steps: list[WindowStepResult] = []
        committed_prefix = 0
        total_iterations = 0

        while committed_prefix < self.context.n:
            start = int(committed_prefix)
            end = min(self.context.n, int(start + self.window.window_size))
            active_cols = np.arange(start, end, dtype=np.int32)

            active_submatrix = self.context.H[:, active_cols]
            row_has_active = np.asarray(active_submatrix.getnnz(axis=1)).reshape(-1) > 0
            row_closed = self.context.row_max_col < end
            active_rows = np.flatnonzero(row_has_active & row_closed)

            local_prior = np.asarray(carry_prior[active_cols], dtype=np.float64)
            if active_rows.size == 0:
                if end != self.context.n:
                    raise RuntimeError(
                        "window did not close any checks before the requested commit region; "
                        "increase window_size or overlap_size"
                    )
                local_estimate = (local_prior < 0.0).astype(np.uint8)
                local_posterior = local_prior.copy()
                local_converged = True
                local_iterations = 0
                local_residual = np.zeros(0, dtype=np.uint8)
            else:
                sub_graph = TannerGraph.from_csr(self.context.H[active_rows][:, active_cols])
                local_estimate, local_posterior, local_converged, local_iterations, local_residual = run_message_passing(
                    graph=sub_graph,
                    syndrome_bits=residual_syndrome[active_rows],
                    prior_llr=local_prior,
                    config=self.config,
                    algorithm=self.algorithm,
                )

            total_iterations += int(local_iterations)
            posterior[active_cols] = np.asarray(local_posterior, dtype=np.float64)

            requested_commit_end = self.context.n if end == self.context.n else min(
                self.context.n, int(start + self.window.effective_commit_size())
            )
            actual_commit_end = int(start)
            if end == self.context.n:
                actual_commit_end = int(end)
            else:
                while actual_commit_end < requested_commit_end:
                    if int(self.context.col_forward_reach[actual_commit_end]) >= int(end):
                        break
                    actual_commit_end += 1
                if actual_commit_end == start:
                    raise RuntimeError(
                        "window could not commit the next undecided column with the current overlap; "
                        "increase overlap_size or window_size"
                    )

            commit_count = int(actual_commit_end - start)
            commit_cols = active_cols[:commit_count]
            commit_vals = np.asarray(local_estimate[:commit_count], dtype=np.uint8)
            estimate[commit_cols] = commit_vals
            residual_syndrome = self.context.fold_columns_into_syndrome(
                residual_syndrome,
                columns=commit_cols.tolist(),
                values=commit_vals.tolist(),
            )
            if actual_commit_end < end:
                carry_prior[actual_commit_end:end] = np.asarray(local_posterior[commit_count:], dtype=np.float64)

            window_steps.append(
                WindowStepResult(
                    window_index=int(len(window_steps)),
                    column_start=int(start),
                    column_end=int(end),
                    commit_end_requested=int(requested_commit_end),
                    commit_end_actual=int(actual_commit_end),
                    overlap_start=int(actual_commit_end),
                    overlap_end=int(end),
                    active_row_count=int(active_rows.size),
                    iterations=int(local_iterations),
                    converged=bool(local_converged),
                    unsatisfied_checks=int(np.count_nonzero(local_residual)),
                )
            )
            committed_prefix = int(actual_commit_end)

        posterior = np.where(np.isfinite(posterior), posterior, prior)
        full_residual = (self.context.syndrome(estimate) ^ target).astype(np.uint8)
        return SideDecodeResult(
            estimate=estimate,
            posterior_llr=posterior,
            converged=bool(np.count_nonzero(full_residual) == 0),
            iterations=int(total_iterations),
            unsatisfied_checks=int(np.count_nonzero(full_residual)),
            unsatisfied_vector=full_residual,
            logical_action=self.context.logical_action_for(estimate),
            window_steps=window_steps,
        )
