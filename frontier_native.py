from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


try:
    import _frontier_native as _native
except ImportError:  # pragma: no cover - exercised when extension is not built.
    _native = None


def is_available() -> bool:
    return _native is not None


@dataclass(slots=True)
class NativeBinaryFrontierModel:
    """Small Python handle for the native C++ frontier binary model."""

    spec: Mapping[str, object]
    _capsule: Any = field(init=False, repr=False)
    _info: dict[str, object] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if _native is None:
            raise RuntimeError("native frontier extension is not built")
        self._capsule = _native.make_model(dict(self.spec))
        self._info = dict(_native.model_info(self._capsule))

    @property
    def info(self) -> dict[str, object]:
        return dict(self._info)

    def decode(
        self,
        syndrome_limbs: Sequence[int],
        K: int,
        Delta: float,
        score_alpha: float = 0.8,
        metric_mode: str = "logsumexp_float",
        int_metric_scale: int = 1024,
    ) -> dict[str, object]:
        return dict(
            _native.decode(
                self._capsule,
                tuple(int(value) for value in syndrome_limbs),
                int(K),
                float(Delta),
                float(score_alpha),
                str(metric_mode),
                int(int_metric_scale),
            )
        )

    def decode_many(
        self,
        syndrome_limb_rows: Sequence[Sequence[int]],
        K: int,
        Delta: float,
        score_alpha: float = 0.8,
        metric_mode: str = "logsumexp_float",
        int_metric_scale: int = 1024,
    ) -> list[dict[str, object]]:
        return [
            dict(payload)
            for payload in _native.decode_many(
                self._capsule,
                tuple(tuple(int(value) for value in row) for row in syndrome_limb_rows),
                int(K),
                float(Delta),
                float(score_alpha),
                str(metric_mode),
                int(int_metric_scale),
            )
        ]

    def decode_many_payloads(
        self,
        syndrome_limb_rows: Sequence[Sequence[int]],
        K: int,
        Delta: float,
        score_alpha: float = 0.8,
        metric_mode: str = "logsumexp_float",
        int_metric_scale: int = 1024,
    ) -> list[dict[str, object]]:
        return _native.decode_many(
            self._capsule,
            tuple(tuple(int(value) for value in row) for row in syndrome_limb_rows),
            int(K),
            float(Delta),
            float(score_alpha),
            str(metric_mode),
            int(int_metric_scale),
        )

    def decode_many_select(
        self,
        backward_model: "NativeBinaryFrontierModel",
        forward_syndrome_limb_rows: Sequence[Sequence[int]],
        backward_syndrome_limb_rows: Sequence[Sequence[int]],
        K: int,
        Delta: float,
        score_alpha: float = 0.8,
        metric_mode: str = "logsumexp_float",
        int_metric_scale: int = 1024,
    ) -> list[dict[str, object]]:
        return _native.decode_many_select(
            self._capsule,
            backward_model._capsule,
            tuple(tuple(int(value) for value in row) for row in forward_syndrome_limb_rows),
            tuple(tuple(int(value) for value in row) for row in backward_syndrome_limb_rows),
            int(K),
            float(Delta),
            float(score_alpha),
            str(metric_mode),
            int(int_metric_scale),
        )

    def decode_many_select_compact(
        self,
        backward_model: "NativeBinaryFrontierModel",
        forward_syndrome_limb_rows: Sequence[Sequence[int]],
        backward_syndrome_limb_rows: Sequence[Sequence[int]],
        K: int,
        Delta: float,
        score_alpha: float = 0.8,
        metric_mode: str = "logsumexp_float",
        int_metric_scale: int = 1024,
    ) -> list[dict[str, object]]:
        decode_many_select_compact = getattr(_native, "decode_many_select_compact", None)
        if decode_many_select_compact is None:
            return self.decode_many_select(
                backward_model,
                forward_syndrome_limb_rows,
                backward_syndrome_limb_rows,
                K,
                Delta,
                score_alpha,
                metric_mode,
                int_metric_scale,
            )
        return decode_many_select_compact(
            self._capsule,
            backward_model._capsule,
            tuple(tuple(int(value) for value in row) for row in forward_syndrome_limb_rows),
            tuple(tuple(int(value) for value in row) for row in backward_syndrome_limb_rows),
            int(K),
            float(Delta),
            float(score_alpha),
            str(metric_mode),
            int(int_metric_scale),
        )

    def decode_many_select_replay(
        self,
        backward_model: "NativeBinaryFrontierModel",
        forward_syndrome_limb_rows: Sequence[Sequence[int]],
        backward_syndrome_limb_rows: Sequence[Sequence[int]],
        K: int,
        Delta: float,
        score_alpha: float = 0.8,
        metric_mode: str = "logsumexp_float",
        int_metric_scale: int = 1024,
    ) -> list[dict[str, object]]:
        decode_many_select_replay = getattr(_native, "decode_many_select_replay", None)
        if decode_many_select_replay is None:
            raise RuntimeError("native frontier extension does not expose decode_many_select_replay")
        return decode_many_select_replay(
            self._capsule,
            backward_model._capsule,
            tuple(tuple(int(value) for value in row) for row in forward_syndrome_limb_rows),
            tuple(tuple(int(value) for value in row) for row in backward_syndrome_limb_rows),
            int(K),
            float(Delta),
            float(score_alpha),
            str(metric_mode),
            int(int_metric_scale),
        )

@dataclass(slots=True)
class NativeChoiceFrontierModel:
    """Small Python handle for the native C++ frontier multi-choice model."""

    spec: Mapping[str, object]
    _capsule: Any = field(init=False, repr=False)
    _info: dict[str, object] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if _native is None:
            raise RuntimeError("native frontier extension is not built")
        if not hasattr(_native, "make_choice_model"):
            raise RuntimeError("native frontier extension does not expose the choice model API")
        self._capsule = _native.make_choice_model(dict(self.spec))
        self._info = dict(_native.choice_model_info(self._capsule))

    @property
    def info(self) -> dict[str, object]:
        return dict(self._info)

    def decode(self, syndrome_limbs: Sequence[int], K: int, Delta: float, score_alpha: float = 0.8) -> dict[str, object]:
        return dict(
            _native.decode_choice(
                self._capsule,
                tuple(int(value) for value in syndrome_limbs),
                int(K),
                float(Delta),
                float(score_alpha),
            )
        )

    def decode_many(
        self,
        syndrome_limb_rows: Sequence[Sequence[int]],
        K: int,
        Delta: float,
        score_alpha: float = 0.8,
    ) -> list[dict[str, object]]:
        if not hasattr(_native, "decode_many_choice"):
            return [
                self.decode(row, K=int(K), Delta=float(Delta), score_alpha=float(score_alpha))
                for row in tuple(syndrome_limb_rows)
            ]
        return _native.decode_many_choice(
            self._capsule,
            tuple(tuple(int(value) for value in row) for row in syndrome_limb_rows),
            int(K),
            float(Delta),
            float(score_alpha),
        )
