from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True, init=False)
class PruneBlock:
    block_id: int
    start: int
    stop: int
    factor_ids: tuple[int, ...]
    size: int
    detector_union_weight: int
    has_detector_overlap: bool
    tick_min: Optional[int] = None
    tick_max: Optional[int] = None

    def __init__(
        self,
        *,
        block_id: int,
        start: int,
        stop: int,
        factor_ids: Sequence[int] | None = None,
        columns: Sequence[int] | None = None,
        size: int,
        detector_union_weight: int,
        has_detector_overlap: bool,
        tick_min: Optional[int] = None,
        tick_max: Optional[int] = None,
    ) -> None:
        if factor_ids is not None and columns is not None:
            if tuple(int(value) for value in factor_ids) != tuple(int(value) for value in columns):
                raise ValueError("factor_ids and columns must match when both are provided")
        ids = factor_ids if factor_ids is not None else columns
        if ids is None:
            raise ValueError("PruneBlock requires factor_ids")
        object.__setattr__(self, "block_id", int(block_id))
        object.__setattr__(self, "start", int(start))
        object.__setattr__(self, "stop", int(stop))
        object.__setattr__(self, "factor_ids", tuple(int(value) for value in ids))
        object.__setattr__(self, "size", int(size))
        object.__setattr__(self, "detector_union_weight", int(detector_union_weight))
        object.__setattr__(self, "has_detector_overlap", bool(has_detector_overlap))
        object.__setattr__(self, "tick_min", None if tick_min is None else int(tick_min))
        object.__setattr__(self, "tick_max", None if tick_max is None else int(tick_max))

    @property
    def columns(self) -> tuple[int, ...]:
        return tuple(self.factor_ids)


def _validate_n_columns(n_columns: int) -> int:
    n_columns = int(n_columns)
    if n_columns < 0:
        raise ValueError("n_columns must be >= 0")
    return int(n_columns)


def _validate_max_block_size(max_block_size: Optional[int]) -> Optional[int]:
    if max_block_size is None:
        return None
    max_block_size = int(max_block_size)
    if int(max_block_size) < 1:
        raise ValueError("max_block_size must be >= 1 when provided")
    return int(max_block_size)


def _matrix_shape(D_ordered) -> tuple[int, int]:
    shape = getattr(D_ordered, "shape", None)
    if shape is None or len(tuple(shape)) != 2:
        raise ValueError("D_ordered must be a detector x column matrix")
    return int(shape[0]), int(shape[1])


def _detector_masks_from_input(detector_masks_or_matrix) -> tuple[int, ...]:
    if sp.issparse(detector_masks_or_matrix):
        _num_detectors, n_items = _matrix_shape(detector_masks_or_matrix)
        return tuple(_column_detector_mask(detector_masks_or_matrix, column) for column in range(int(n_items)))
    arr = np.asarray(detector_masks_or_matrix)
    if arr.ndim == 2:
        _num_detectors, n_items = _matrix_shape(arr)
        return tuple(_column_detector_mask(arr, column) for column in range(int(n_items)))
    return tuple(int(value) for value in tuple(detector_masks_or_matrix))


def _column_detector_mask(D_ordered, column: int) -> int:
    column = int(column)
    if sp.issparse(D_ordered):
        col = D_ordered.getcol(int(column))
        rows = tuple(int(row) for row in col.indices)
    else:
        arr = np.asarray(D_ordered)
        if arr.ndim != 2:
            raise ValueError("D_ordered must be a detector x column matrix")
        rows = tuple(
            int(row)
            for row, value in enumerate(arr[:, int(column)].reshape(-1).tolist())
            if int(value) & 1
        )
    mask = 0
    for row in rows:
        mask |= 1 << int(row)
    return int(mask)


def _block_from_columns(
    *,
    block_id: int,
    columns: Sequence[int],
    detector_masks: Sequence[int] | None,
    tick_by_order_pos: Sequence[Optional[int]] | None,
) -> PruneBlock:
    cols = tuple(int(value) for value in columns)
    if not cols:
        raise ValueError("cannot build an empty prune block")
    expected = tuple(range(int(cols[0]), int(cols[-1]) + 1))
    if cols != expected:
        raise ValueError("prune blocks must contain consecutive ordered positions")
    detector_union = 0
    has_overlap = False
    if detector_masks is not None:
        for column in cols:
            mask = int(detector_masks[int(column)])
            if int(detector_union & mask) != 0:
                has_overlap = True
            detector_union |= int(mask)
    ticks: list[int] = []
    if tick_by_order_pos is not None:
        for column in cols:
            tick = tick_by_order_pos[int(column)]
            if tick is not None:
                ticks.append(int(tick))
    return PruneBlock(
        block_id=int(block_id),
        start=int(cols[0]),
        stop=int(cols[-1]) + 1,
        factor_ids=tuple(cols),
        size=int(len(cols)),
        detector_union_weight=int(detector_union.bit_count()),
        has_detector_overlap=bool(has_overlap),
        tick_min=(min(ticks) if ticks else None),
        tick_max=(max(ticks) if ticks else None),
    )


def build_singleton_prune_blocks(n_columns: int) -> list[PruneBlock]:
    n_columns = _validate_n_columns(int(n_columns))
    return [
        PruneBlock(
            block_id=int(column),
            start=int(column),
            stop=int(column) + 1,
            factor_ids=(int(column),),
            size=1,
            detector_union_weight=0,
            has_detector_overlap=False,
            tick_min=None,
            tick_max=None,
        )
        for column in range(int(n_columns))
    ]


def build_consecutive_detector_disjoint_prune_blocks(
    detector_masks,
    *,
    max_block_size: Optional[int] = None,
    tick_by_pos: Optional[Sequence[Optional[int]]] = None,
    tick_by_order_pos: Optional[Sequence[Optional[int]]] = None,
    require_same_tick: bool = False,
) -> list[PruneBlock]:
    detector_mask_tuple = _detector_masks_from_input(detector_masks)
    n_columns = int(len(detector_mask_tuple))
    max_block_size = _validate_max_block_size(max_block_size)
    if tick_by_pos is None:
        tick_by_pos = tick_by_order_pos
    if tick_by_order_pos is not None and len(tuple(tick_by_order_pos)) != int(n_columns):
        raise ValueError("tick_by_order_pos length must match the number of ordered columns")
    if tick_by_pos is not None and len(tuple(tick_by_pos)) != int(n_columns):
        raise ValueError("tick_by_pos length must match the number of ordered items")
    ticks = tuple(tick_by_pos) if tick_by_pos is not None else None
    blocks: list[PruneBlock] = []
    start = 0
    while int(start) < int(n_columns):
        stop = int(start) + 1
        union_mask = int(detector_mask_tuple[int(start)])
        block_tick = None if ticks is None else ticks[int(start)]
        while int(stop) < int(n_columns):
            next_mask = int(detector_mask_tuple[int(stop)])
            if int(union_mask & next_mask) != 0:
                break
            if max_block_size is not None and int(stop - start + 1) > int(max_block_size):
                break
            if bool(require_same_tick) and ticks is not None and ticks[int(stop)] != block_tick:
                break
            union_mask |= int(next_mask)
            stop += 1
        blocks.append(
            _block_from_columns(
                block_id=len(blocks),
                columns=tuple(range(int(start), int(stop))),
                detector_masks=detector_mask_tuple,
                tick_by_order_pos=ticks,
            )
        )
        start = int(stop)
    return blocks


def build_tick_prune_blocks(
    factors,
    tick_by_order_pos: Sequence[Optional[int]] | None = None,
    *,
    max_block_size: Optional[int] = None,
    split_on_detector_overlap: bool = True,
    detector_masks=None,
    D_ordered=None,
    strict_ticks: bool = False,
    strict_missing_tick: bool | None = None,
) -> list[PruneBlock]:
    if strict_missing_tick is not None:
        strict_ticks = bool(strict_missing_tick)
    if isinstance(factors, int):
        n_columns = _validate_n_columns(int(factors))
        ticks = tuple(tick_by_order_pos) if tick_by_order_pos is not None else tuple(None for _ in range(int(n_columns)))
    else:
        factor_tuple = tuple(factors)
        n_columns = _validate_n_columns(len(factor_tuple))
        if tick_by_order_pos is None:
            ticks = tuple(getattr(factor, "tick", None) for factor in factor_tuple)
        else:
            ticks = tuple(tick_by_order_pos)
    max_block_size = _validate_max_block_size(max_block_size)
    if len(ticks) != int(n_columns):
        raise ValueError("tick_by_order_pos length must match n_columns")
    detector_mask_tuple: tuple[int, ...] | None = None
    if detector_masks is not None:
        detector_mask_tuple = _detector_masks_from_input(detector_masks)
    elif D_ordered is not None:
        detector_mask_tuple = _detector_masks_from_input(D_ordered)
    elif bool(split_on_detector_overlap):
        raise ValueError("detector_masks is required when split_on_detector_overlap=True")
    if detector_mask_tuple is not None and len(tuple(detector_mask_tuple)) != int(n_columns):
        raise ValueError("detector_masks length must match n_columns")
    blocks: list[PruneBlock] = []
    start = 0
    while int(start) < int(n_columns):
        tick = ticks[int(start)]
        if tick is None:
            if bool(strict_ticks):
                raise ValueError(f"missing tick metadata at ordered position {int(start)}")
            stop = int(start) + 1
            while int(stop) < int(n_columns) and ticks[int(stop)] is None:
                stop += 1
            for column in range(int(start), int(stop)):
                blocks.append(
                    _block_from_columns(
                        block_id=len(blocks),
                        columns=(int(column),),
                        detector_masks=detector_mask_tuple,
                        tick_by_order_pos=ticks,
                    )
                )
            start = int(stop)
            continue
        stop = int(start) + 1
        while int(stop) < int(n_columns) and ticks[int(stop)] == tick:
            stop += 1
        if bool(split_on_detector_overlap):
            run_start = int(start)
            while int(run_start) < int(stop):
                run_stop = int(run_start) + 1
                if detector_mask_tuple is None:
                    raise AssertionError("detector masks unexpectedly missing")
                union_mask = int(detector_mask_tuple[int(run_start)])
                while int(run_stop) < int(stop):
                    next_mask = int(detector_mask_tuple[int(run_stop)])
                    if int(union_mask & next_mask) != 0:
                        break
                    if max_block_size is not None and int(run_stop - run_start + 1) > int(max_block_size):
                        break
                    union_mask |= int(next_mask)
                    run_stop += 1
                blocks.append(
                    _block_from_columns(
                        block_id=len(blocks),
                        columns=tuple(range(int(run_start), int(run_stop))),
                        detector_masks=detector_mask_tuple,
                        tick_by_order_pos=ticks,
                    )
                )
                run_start = int(run_stop)
        else:
            chunk_start = int(start)
            while int(chunk_start) < int(stop):
                chunk_stop = int(stop)
                if max_block_size is not None:
                    chunk_stop = min(int(chunk_stop), int(chunk_start) + int(max_block_size))
                blocks.append(
                    _block_from_columns(
                        block_id=len(blocks),
                        columns=tuple(range(int(chunk_start), int(chunk_stop))),
                        detector_masks=detector_mask_tuple,
                        tick_by_order_pos=ticks,
                    )
                )
                chunk_start = int(chunk_stop)
        start = int(stop)
    return blocks


def prune_boundary_mask_from_blocks(n_columns: int, blocks: Sequence[PruneBlock]) -> list[bool]:
    n_columns = _validate_n_columns(int(n_columns))
    mask = [False for _ in range(int(n_columns))]
    expected_start = 0
    for block in tuple(blocks):
        if int(block.start) != int(expected_start):
            raise ValueError("prune blocks must cover all columns consecutively from zero")
        if int(block.stop) <= int(block.start):
            raise ValueError("prune block stop must be greater than start")
        if int(block.stop) > int(n_columns):
            raise ValueError("prune block stop exceeds n_columns")
        expected_columns = tuple(range(int(block.start), int(block.stop)))
        if tuple(int(value) for value in block.columns) != expected_columns:
            raise ValueError("prune block columns must match range(start, stop)")
        mask[int(block.stop) - 1] = True
        expected_start = int(block.stop)
    if int(expected_start) != int(n_columns):
        raise ValueError("prune blocks must cover all columns")
    return mask


def summarize_prune_blocks(blocks: Sequence[PruneBlock]) -> dict[str, object]:
    block_tuple = tuple(blocks)
    sizes = tuple(int(block.size) for block in block_tuple)
    n_columns = int(sum(sizes))
    return {
        "n_blocks": int(len(block_tuple)),
        "n_columns": int(n_columns),
        "singleton_blocks": int(sum(1 for size in sizes if int(size) == 1)),
        "non_singleton_blocks": int(sum(1 for size in sizes if int(size) > 1)),
        "max_block_size": int(max(sizes, default=0)),
        "mean_block_size": (
            float(sum(sizes)) / float(len(sizes))
            if sizes
            else 0.0
        ),
        "detector_overlap_blocks": int(sum(1 for block in block_tuple if bool(block.has_detector_overlap))),
        "max_detector_union_weight": int(max((int(block.detector_union_weight) for block in block_tuple), default=0)),
        "ticked_blocks": int(
            sum(1 for block in block_tuple if block.tick_min is not None or block.tick_max is not None)
        ),
        "missing_tick_blocks": int(
            sum(1 for block in block_tuple if block.tick_min is None and block.tick_max is None)
        ),
    }
