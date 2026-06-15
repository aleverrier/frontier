from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class DecoderRunResult:
    decoder: str
    status: str
    correction: np.ndarray
    data_fault_estimate: np.ndarray
    measurement_fault_estimate: np.ndarray
    runtime_ms: float
    work_iterations: int
    edge_updates: int
    metadata: Dict[str, object] = field(default_factory=dict)
