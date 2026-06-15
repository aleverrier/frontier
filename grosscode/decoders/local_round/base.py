from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class LocalRoundInput:
    detector_slice: np.ndarray
    data_prior_llr: np.ndarray
    left_interface_llr: np.ndarray
    right_interface_llr: np.ndarray
    left_clamp_mask: np.ndarray | None = None
    left_clamp_values: np.ndarray | None = None
    right_clamp_mask: np.ndarray | None = None
    right_clamp_values: np.ndarray | None = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class LocalRoundOutput:
    data_llr: np.ndarray
    left_interface_llr: np.ndarray
    right_interface_llr: np.ndarray
    data_estimate: np.ndarray
    left_interface_estimate: np.ndarray
    right_interface_estimate: np.ndarray
    success: bool
    work_iterations: int
    edge_updates: int
    metadata: Dict[str, object] = field(default_factory=dict)


class LocalRoundFactor(ABC):
    @abstractmethod
    def infer(self, round_input: LocalRoundInput) -> LocalRoundOutput:
        raise NotImplementedError
