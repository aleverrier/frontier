from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
from typing import Optional

import numpy as np

from grosscode.core import FrameDecodeResult, SideContext, SideDecodeResult


def load_worker_a_baseline_module(root: Optional[Path] = None):
    candidates = []
    env_root = os.environ.get("WORKER_A_BASELINE_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))
    if root is not None:
        candidates.append(Path(root))

    for candidate_root in candidates:
        module_path = candidate_root / "clamped_decoder.py"
        if not module_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("worker_a_clamped_decoder", str(module_path))
        if spec is None or spec.loader is None:
            continue
        if spec.name in sys.modules:
            return sys.modules[spec.name]
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return None


@dataclass
class WorkerABaselineSideDecoder:
    context: SideContext
    max_iter: int = 60
    alpha: float = 0.8
    beta: float = 0.05
    damping: float = 0.0
    schedule: str = "layered"
    module: object | None = None

    def __post_init__(self) -> None:
        module = self.module or load_worker_a_baseline_module()
        if module is None:
            raise RuntimeError("Worker A baseline module was not found")
        self.module = module
        decoder_cls = getattr(module, "MinSumSyndromeDecoder")
        self._decoder = decoder_cls(
            self.context.H,
            alpha=float(self.alpha),
            beta=float(self.beta),
            damp=float(self.damping),
            schedule=str(self.schedule),
        )

    def decode(self, syndrome: np.ndarray, *, prior_llr: Optional[np.ndarray] = None) -> SideDecodeResult:
        prior = self.context.resolve_prior_llr(prior_llr)
        avg_p = float(np.mean(self.context.priors))
        estimate, converged, post_llr = self._decoder.decode(
            np.asarray(syndrome, dtype=np.uint8),
            p=avg_p,
            max_iter=int(self.max_iter),
            prior_llr=prior,
        )
        estimate = np.asarray(estimate, dtype=np.uint8).reshape(-1)
        post_llr = np.asarray(post_llr, dtype=np.float64).reshape(-1)
        residual = (self.context.syndrome(estimate) ^ (np.asarray(syndrome, dtype=np.uint8).reshape(-1) & 1)).astype(np.uint8)
        return SideDecodeResult(
            estimate=estimate,
            posterior_llr=post_llr,
            converged=bool(converged),
            iterations=int(getattr(self._decoder, "last_decode_iterations", 0)),
            unsatisfied_checks=int(np.count_nonzero(residual)),
            unsatisfied_vector=residual,
            logical_action=self.context.logical_action_for(estimate),
            window_steps=[],
        )


@dataclass
class WorkerABaselineSplitDecoder:
    x_context: SideContext
    z_context: SideContext
    max_iter: int = 60
    alpha: float = 0.8
    beta: float = 0.05
    damping: float = 0.0
    schedule: str = "layered"
    module: object | None = None

    def __post_init__(self) -> None:
        self.x_decoder = WorkerABaselineSideDecoder(
            self.x_context,
            max_iter=int(self.max_iter),
            alpha=float(self.alpha),
            beta=float(self.beta),
            damping=float(self.damping),
            schedule=str(self.schedule),
            module=self.module,
        )
        self.z_decoder = WorkerABaselineSideDecoder(
            self.z_context,
            max_iter=int(self.max_iter),
            alpha=float(self.alpha),
            beta=float(self.beta),
            damping=float(self.damping),
            schedule=str(self.schedule),
            module=self.x_decoder.module,
        )

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
            logical_frame_action={"x": x_result.logical_action, "z": z_result.logical_action},
            unsatisfied_checks={"x": int(x_result.unsatisfied_checks), "z": int(z_result.unsatisfied_checks)},
            iterations={"x": int(x_result.iterations), "z": int(z_result.iterations)},
        )
