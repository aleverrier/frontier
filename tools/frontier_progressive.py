"""Frontier column, layout, ordering, and scoring helpers.

Public entry points are the dataclasses and helper functions imported by
`tools.frontier_decoder` and `tools.dem_loader`. This is support code, not a
top-level user CLI.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field, replace
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ProgressiveColumn:
    family: str
    index: int
    label: str
    instruction_offset: int
    prior_probs: tuple[float, ...]
    detector_response_masks: tuple[int, ...]
    logical_response_masks: tuple[int, ...]
    detector_support_mask: int
    prior_log_probs: tuple[float, ...] | None = None
    detector_support_rows: tuple[int, ...] = ()
    correction_response_masks: tuple[int, ...] | None = None
    original_column_index: int = -1


@dataclass(frozen=True, slots=True)
class OutcomeTransition:
    probability: float
    detector_mask: int
    logical_mask: int
    label: str = ""


@dataclass(frozen=True, slots=True)
class FactorTransition:
    factor_id: int
    outcomes: tuple[OutcomeTransition, ...]
    tick: int | None = None
    instruction_offset: int | None = None
    label: str = ""


@dataclass(frozen=True, slots=True)
class FrontierRowUpdate:
    row: int
    row_mask: int
    active_before: bool
    active_after: bool
    touch_bonus_before: float
    touch_bonus_after: float
    urgency_before: float
    urgency_after: float
    parity_logodds_before: float
    parity_logodds_after: float


@dataclass(frozen=True, slots=True)
class ProgressiveFrontierLayout:
    detector_first_column: tuple[int, ...]
    detector_last_column: tuple[int, ...]
    closing_masks: tuple[int, ...]
    active_masks_after_column: tuple[int, ...]
    closure_block_start_by_column: tuple[int, ...]
    closure_block_end_by_column: tuple[int, ...]
    disjoint_detector_run_start_by_column: tuple[int, ...]
    disjoint_detector_run_end_by_column: tuple[int, ...]
    max_active_detectors: int
    active_width_profile: tuple[int, ...]
    column_row_updates: tuple[tuple[FrontierRowUpdate, ...], ...]
    row_touch_columns: tuple[tuple[int, ...], ...]
    row_suffix_even_probs: tuple[tuple[float, ...], ...]
    row_suffix_odd_probs: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class BinaryFrontierColumnPayload:
    no_error_log_const: float
    toggle_logodds: float
    toggle_detector_mask: int
    toggle_observable_mask: int
    before_row_bit_masks: tuple[int, ...] = tuple()
    before_touch_bonus_weights: tuple[float, ...] = tuple()
    before_urgency_weights: tuple[float, ...] = tuple()
    before_parity_logodds_weights: tuple[float, ...] = tuple()
    after_row_bit_masks: tuple[int, ...] = tuple()
    after_touch_bonus_weights: tuple[float, ...] = tuple()
    after_urgency_weights: tuple[float, ...] = tuple()
    after_parity_logodds_weights: tuple[float, ...] = tuple()


@dataclass(frozen=True, slots=True)
class FrontierMismatchFeatures:
    mismatch_count: int
    touch_bonus: float
    urgency_penalty: float
    parity_logodds: float


@dataclass(slots=True)
class FutureParityScorerStats:
    score_calls: int = 0
    payload_build_count: int = 0
    scorer_time_s: float = 0.0


@dataclass(frozen=True, slots=True)
class FutureParityRowMetric:
    row: int
    row_mask: int
    touch_bonus: float
    urgency_penalty: float
    parity_logodds: float


@dataclass(frozen=True, slots=True)
class FutureParityBoundaryPayload:
    boundary_column_index: int
    active_mask: int
    row_metric_by_mask: dict[int, FutureParityRowMetric]


@dataclass(slots=True)
class FutureParityScorer:
    frontier: ProgressiveFrontierLayout
    target_syndrome_int: int
    mode: str = "cached"
    stats: FutureParityScorerStats = field(default_factory=FutureParityScorerStats)
    payload_cache: dict[int, FutureParityBoundaryPayload] = field(default_factory=dict)

    def boundary_payload(self, *, boundary_column_index: int, active_mask: int) -> FutureParityBoundaryPayload:
        key = int(boundary_column_index)
        payload = self.payload_cache.get(key)
        if payload is None or int(payload.active_mask) != int(active_mask):
            payload = _build_future_parity_boundary_payload(
                frontier=self.frontier,
                boundary_column_index=int(boundary_column_index),
                active_mask=int(active_mask),
            )
            self.payload_cache[key] = payload
            self.stats.payload_build_count += 1
        return payload

    def score_from_detector_mask(self, *, det_mask: int, active_mask: int, boundary_column_index: int) -> float:
        payload = self.boundary_payload(
            boundary_column_index=int(boundary_column_index),
            active_mask=int(active_mask),
        )
        features = _future_mismatch_features_from_payload(
            det_mask=int(det_mask),
            target_syndrome_int=int(self.target_syndrome_int),
            payload=payload,
        )
        self.stats.score_calls += 1
        return float(features.parity_logodds)


def _support_rows_from_mask(mask: int) -> tuple[int, ...]:
    rows: list[int] = []
    value = int(mask)
    while value:
        lsb = int(value & -value)
        rows.append(int(lsb.bit_length() - 1))
        value ^= lsb
    return tuple(rows)


def _support_mask_from_detector_responses(detector_masks: Sequence[int]) -> int:
    support = 0
    for mask in tuple(detector_masks):
        support |= int(mask)
    return int(support)


def _column_detector_support_mask(column: ProgressiveColumn) -> int:
    support = int(column.detector_support_mask)
    if support:
        return int(support)
    return _support_mask_from_detector_responses(column.detector_response_masks)


def _column_log_priors(column: ProgressiveColumn) -> tuple[float, ...]:
    if column.prior_log_probs is not None:
        return tuple(float(value) for value in column.prior_log_probs)
    return tuple(float("-inf") if float(prob) <= 0.0 else float(math.log(float(prob))) for prob in column.prior_probs)


def _columns_from_factor_transitions(factor_transitions: Sequence[FactorTransition]) -> list[ProgressiveColumn]:
    columns: list[ProgressiveColumn] = []
    for order_index, factor in enumerate(tuple(factor_transitions)):
        outcomes = tuple(factor.outcomes)
        if not outcomes:
            raise ValueError(f"factor {int(factor.factor_id)} has no outcomes")
        probabilities = tuple(float(outcome.probability) for outcome in outcomes)
        total = float(sum(probabilities))
        if abs(total - 1.0) > 1e-10:
            raise ValueError(f"factor {int(factor.factor_id)} probabilities sum to {total:.18g}, not 1")
        detector_masks = tuple(int(outcome.detector_mask) for outcome in outcomes)
        logical_masks = tuple(int(outcome.logical_mask) for outcome in outcomes)
        support = _support_mask_from_detector_responses(detector_masks)
        instruction_offset = int(factor.instruction_offset) if factor.instruction_offset is not None else int(order_index)
        label = str(factor.label) if str(factor.label) else f"factor_{int(factor.factor_id)}"
        columns.append(
            ProgressiveColumn(
                family="factorized_transition",
                index=int(order_index),
                label=label,
                instruction_offset=int(instruction_offset),
                prior_probs=tuple(float(p) for p in probabilities),
                detector_response_masks=tuple(int(mask) for mask in detector_masks),
                logical_response_masks=tuple(int(mask) for mask in logical_masks),
                detector_support_mask=int(support),
                prior_log_probs=tuple(float("-inf") if p <= 0.0 else float(math.log(p)) for p in probabilities),
                detector_support_rows=_support_rows_from_mask(int(support)),
                original_column_index=int(factor.factor_id),
            )
        )
    return columns


def _reverse_progressive_columns(columns: Sequence[ProgressiveColumn]) -> list[ProgressiveColumn]:
    reversed_columns = list(reversed(tuple(columns)))
    return [replace(column, index=int(index)) for index, column in enumerate(reversed_columns)]


def optimize_column_order(
    columns: list[ProgressiveColumn],
    *,
    num_detectors: int,
) -> tuple[list[ProgressiveColumn], tuple[int, ...]]:
    if len(columns) <= 1:
        return list(columns), tuple(range(len(columns)))

    support_rows_by_column: list[tuple[int, ...]] = []
    first_touch_by_row = [-1 for _ in range(int(num_detectors))]
    last_touch_by_row = [-1 for _ in range(int(num_detectors))]
    for column_index, column in enumerate(columns):
        support_rows = tuple(int(row) for row in column.detector_support_rows)
        if not support_rows:
            support_rows = _support_rows_from_mask(_column_detector_support_mask(column))
        support_rows_by_column.append(tuple(int(row) for row in support_rows))
        for row in support_rows:
            if int(row) < 0 or int(row) >= int(num_detectors):
                raise ValueError(f"detector row {int(row)} out of range for {int(num_detectors)} detectors")
            if int(first_touch_by_row[int(row)]) < 0:
                first_touch_by_row[int(row)] = int(column_index)
            last_touch_by_row[int(row)] = int(column_index)

    def deadline_key(column_index: int) -> tuple[int, int, int, int, int]:
        support_rows = support_rows_by_column[int(column_index)]
        if not support_rows:
            sentinel = int(len(columns) + 1)
            return (
                sentinel,
                sentinel,
                sentinel,
                int(columns[int(column_index)].instruction_offset),
                int(columns[int(column_index)].index),
            )
        earliest_last = min(int(last_touch_by_row[int(row)]) for row in support_rows)
        latest_last = max(int(last_touch_by_row[int(row)]) for row in support_rows)
        earliest_first = min(int(first_touch_by_row[int(row)]) for row in support_rows)
        return (
            int(earliest_last),
            int(latest_last),
            int(earliest_first),
            int(columns[int(column_index)].instruction_offset),
            int(columns[int(column_index)].index),
        )

    ordering = tuple(sorted(range(len(columns)), key=deadline_key))
    reordered = [replace(columns[int(source_index)], index=int(target_index)) for target_index, source_index in enumerate(ordering)]
    return reordered, tuple(int(value) for value in ordering)


def _row_toggle_probability(column: ProgressiveColumn, row_mask: int) -> float:
    total = 0.0
    for probability, detector_mask in zip(column.prior_probs, column.detector_response_masks, strict=True):
        if int(detector_mask) & int(row_mask):
            total += float(probability)
    return float(min(1.0, max(0.0, total)))


def _build_row_future_statistics(
    columns: Sequence[ProgressiveColumn],
    *,
    num_detectors: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    row_touch_columns: list[list[int]] = [[] for _ in range(int(num_detectors))]
    row_touch_probs: list[list[float]] = [[] for _ in range(int(num_detectors))]
    for column_index, column in enumerate(tuple(columns)):
        support_rows = tuple(int(row) for row in column.detector_support_rows)
        if not support_rows:
            support_rows = _support_rows_from_mask(_column_detector_support_mask(column))
        for row in support_rows:
            row_mask = 1 << int(row)
            row_touch_columns[int(row)].append(int(column_index))
            row_touch_probs[int(row)].append(_row_toggle_probability(column, row_mask))

    suffix_even_rows: list[tuple[float, ...]] = []
    suffix_odd_rows: list[tuple[float, ...]] = []
    for probs in row_touch_probs:
        count = len(probs)
        even = [0.0 for _ in range(count + 1)]
        odd = [0.0 for _ in range(count + 1)]
        even[count] = 1.0
        odd[count] = 0.0
        for index in range(count - 1, -1, -1):
            q = float(probs[index])
            even[index] = float((1.0 - q) * even[index + 1] + q * odd[index + 1])
            odd[index] = float((1.0 - q) * odd[index + 1] + q * even[index + 1])
        suffix_even_rows.append(tuple(float(value) for value in even))
        suffix_odd_rows.append(tuple(float(value) for value in odd))
    return (
        tuple(tuple(int(value) for value in row) for row in row_touch_columns),
        tuple(suffix_even_rows),
        tuple(suffix_odd_rows),
    )


def build_frontier_layout(columns: list[ProgressiveColumn], *, num_detectors: int) -> ProgressiveFrontierLayout:
    if int(num_detectors) < 1:
        raise ValueError("num_detectors must be >= 1")
    columns_tuple = tuple(columns)
    if not columns_tuple:
        return ProgressiveFrontierLayout(
            detector_first_column=tuple(-1 for _ in range(int(num_detectors))),
            detector_last_column=tuple(-1 for _ in range(int(num_detectors))),
            closing_masks=tuple(),
            active_masks_after_column=tuple(),
            closure_block_start_by_column=tuple(),
            closure_block_end_by_column=tuple(),
            disjoint_detector_run_start_by_column=tuple(),
            disjoint_detector_run_end_by_column=tuple(),
            max_active_detectors=0,
            active_width_profile=(0,),
            column_row_updates=tuple(),
            row_touch_columns=tuple(tuple() for _ in range(int(num_detectors))),
            row_suffix_even_probs=tuple((1.0,) for _ in range(int(num_detectors))),
            row_suffix_odd_probs=tuple((0.0,) for _ in range(int(num_detectors))),
        )

    first = [-1 for _ in range(int(num_detectors))]
    last = [-1 for _ in range(int(num_detectors))]
    for column_index, column in enumerate(columns_tuple):
        support_rows = tuple(int(row) for row in column.detector_support_rows)
        if not support_rows:
            support_rows = _support_rows_from_mask(_column_detector_support_mask(column))
        for row in support_rows:
            if int(row) < 0 or int(row) >= int(num_detectors):
                raise ValueError(f"detector row {int(row)} out of range for {int(num_detectors)} detectors")
            if first[int(row)] < 0:
                first[int(row)] = int(column_index)
            last[int(row)] = int(column_index)

    rows_start: list[list[int]] = [[] for _ in range(len(columns_tuple))]
    rows_end: list[list[int]] = [[] for _ in range(len(columns_tuple))]
    for row in range(int(num_detectors)):
        if first[row] >= 0:
            rows_start[first[row]].append(row)
        if last[row] >= 0:
            rows_end[last[row]].append(row)

    closing_masks: list[int] = []
    active_masks_after_column: list[int] = []
    active_width_profile: list[int] = [0]
    active_mask = 0
    max_active_detectors = 0
    for column_index in range(len(columns_tuple)):
        close_mask = 0
        for row in rows_start[column_index]:
            if last[row] > column_index:
                active_mask |= 1 << row
        for row in rows_end[column_index]:
            close_mask |= 1 << row
            active_mask &= ~(1 << row)
        closing_masks.append(int(close_mask))
        active_masks_after_column.append(int(active_mask))
        active_width = int(active_mask.bit_count())
        active_width_profile.append(active_width)
        max_active_detectors = max(max_active_detectors, active_width)

    closure_block_start_by_column = [0 for _ in range(len(columns_tuple))]
    closure_block_end_by_column = [0 for _ in range(len(columns_tuple))]
    block_start = 0
    while block_start < len(columns_tuple):
        block_end = block_start
        while block_end + 1 < len(columns_tuple) and int(closing_masks[block_end]) == 0:
            block_end += 1
        for column_index in range(block_start, block_end + 1):
            closure_block_start_by_column[column_index] = int(block_start)
            closure_block_end_by_column[column_index] = int(block_end)
        block_start = block_end + 1

    disjoint_run_start_by_column = [0 for _ in range(len(columns_tuple))]
    disjoint_run_end_by_column = [0 for _ in range(len(columns_tuple))]
    run_start = 0
    while run_start < len(columns_tuple):
        run_end = run_start
        seen_support = _column_detector_support_mask(columns_tuple[run_start])
        while run_end + 1 < len(columns_tuple):
            next_support = _column_detector_support_mask(columns_tuple[run_end + 1])
            if int(seen_support & next_support):
                break
            seen_support |= int(next_support)
            run_end += 1
        for column_index in range(run_start, run_end + 1):
            disjoint_run_start_by_column[column_index] = int(run_start)
            disjoint_run_end_by_column[column_index] = int(run_end)
        run_start = run_end + 1

    row_touch_columns, row_suffix_even_probs, row_suffix_odd_probs = _build_row_future_statistics(
        columns_tuple,
        num_detectors=int(num_detectors),
    )
    tiny = float(2.2250738585072014e-308)
    column_row_updates: list[list[FrontierRowUpdate]] = [[] for _ in range(len(columns_tuple))]
    for row in range(int(num_detectors)):
        touch_columns = tuple(row_touch_columns[row])
        if not touch_columns:
            continue
        suffix_even = tuple(row_suffix_even_probs[row])
        suffix_odd = tuple(row_suffix_odd_probs[row])
        total_touches = len(touch_columns)
        row_mask = 1 << row
        for offset, column_index in enumerate(touch_columns):
            active_before = bool(offset > 0)
            active_after = bool(offset + 1 < total_touches)
            if active_before:
                remaining = total_touches - offset
                before_parity = float(math.log(max(suffix_odd[offset], tiny)) - math.log(max(suffix_even[offset], tiny)))
                before_touch = float(math.log(float(remaining)))
                before_urgency = float(1.0 / float(remaining))
            else:
                before_parity = 0.0
                before_touch = 0.0
                before_urgency = 0.0
            if active_after:
                remaining = total_touches - offset - 1
                after_parity = float(math.log(max(suffix_odd[offset + 1], tiny)) - math.log(max(suffix_even[offset + 1], tiny)))
                after_touch = float(math.log(float(remaining)))
                after_urgency = float(1.0 / float(remaining))
            else:
                after_parity = 0.0
                after_touch = 0.0
                after_urgency = 0.0
            column_row_updates[int(column_index)].append(
                FrontierRowUpdate(
                    row=int(row),
                    row_mask=int(row_mask),
                    active_before=active_before,
                    active_after=active_after,
                    touch_bonus_before=float(before_touch),
                    touch_bonus_after=float(after_touch),
                    urgency_before=float(before_urgency),
                    urgency_after=float(after_urgency),
                    parity_logodds_before=float(before_parity),
                    parity_logodds_after=float(after_parity),
                )
            )

    return ProgressiveFrontierLayout(
        detector_first_column=tuple(int(value) for value in first),
        detector_last_column=tuple(int(value) for value in last),
        closing_masks=tuple(int(value) for value in closing_masks),
        active_masks_after_column=tuple(int(value) for value in active_masks_after_column),
        closure_block_start_by_column=tuple(int(value) for value in closure_block_start_by_column),
        closure_block_end_by_column=tuple(int(value) for value in closure_block_end_by_column),
        disjoint_detector_run_start_by_column=tuple(int(value) for value in disjoint_run_start_by_column),
        disjoint_detector_run_end_by_column=tuple(int(value) for value in disjoint_run_end_by_column),
        max_active_detectors=int(max_active_detectors),
        active_width_profile=tuple(int(value) for value in active_width_profile),
        column_row_updates=tuple(tuple(update for update in updates) for updates in column_row_updates),
        row_touch_columns=tuple(tuple(int(value) for value in row) for row in row_touch_columns),
        row_suffix_even_probs=tuple(tuple(float(value) for value in row) for row in row_suffix_even_probs),
        row_suffix_odd_probs=tuple(tuple(float(value) for value in row) for row in row_suffix_odd_probs),
    )


def _build_future_parity_boundary_payload(
    *,
    frontier: ProgressiveFrontierLayout,
    boundary_column_index: int,
    active_mask: int,
) -> FutureParityBoundaryPayload:
    row_metric_by_mask: dict[int, FutureParityRowMetric] = {}
    tiny = float(2.2250738585072014e-308)
    for row in _support_rows_from_mask(int(active_mask)):
        touch_columns = tuple(frontier.row_touch_columns[int(row)])
        pos = bisect.bisect_right(touch_columns, int(boundary_column_index))
        remaining = int(len(touch_columns) - pos)
        if remaining <= 0:
            continue
        even = float(frontier.row_suffix_even_probs[int(row)][int(pos)])
        odd = float(frontier.row_suffix_odd_probs[int(row)][int(pos)])
        row_mask = 1 << int(row)
        row_metric_by_mask[int(row_mask)] = FutureParityRowMetric(
            row=int(row),
            row_mask=int(row_mask),
            touch_bonus=float(math.log(float(remaining))),
            urgency_penalty=float(1.0 / float(remaining)),
            parity_logodds=float(math.log(max(odd, tiny)) - math.log(max(even, tiny))),
        )
    return FutureParityBoundaryPayload(
        boundary_column_index=int(boundary_column_index),
        active_mask=int(active_mask),
        row_metric_by_mask=row_metric_by_mask,
    )


def _future_mismatch_features_from_payload(
    *,
    det_mask: int,
    target_syndrome_int: int,
    payload: FutureParityBoundaryPayload,
) -> FrontierMismatchFeatures:
    mismatch_mask = int(det_mask ^ target_syndrome_int) & int(payload.active_mask)
    if int(mismatch_mask) == 0:
        return FrontierMismatchFeatures(0, 0.0, 0.0, 0.0)

    mismatch_count = 0
    touch_bonus = 0.0
    urgency_penalty = 0.0
    parity_logodds = 0.0
    while mismatch_mask:
        lsb = int(mismatch_mask & -mismatch_mask)
        metric = payload.row_metric_by_mask.get(int(lsb))
        if metric is None:
            return FrontierMismatchFeatures(
                mismatch_count=int(mismatch_count) + 1,
                touch_bonus=float("-inf"),
                urgency_penalty=float("inf"),
                parity_logodds=float("-inf"),
            )
        mismatch_count += 1
        touch_bonus += float(metric.touch_bonus)
        urgency_penalty += float(metric.urgency_penalty)
        parity_logodds += float(metric.parity_logodds)
        mismatch_mask ^= lsb
    return FrontierMismatchFeatures(
        mismatch_count=int(mismatch_count),
        touch_bonus=float(touch_bonus),
        urgency_penalty=float(urgency_penalty),
        parity_logodds=float(parity_logodds),
    )


def _compile_binary_frontier_column_payload(
    column: ProgressiveColumn,
    row_updates: tuple[FrontierRowUpdate, ...],
) -> BinaryFrontierColumnPayload | None:
    if len(column.prior_probs) != 2 or len(column.detector_response_masks) != 2 or len(column.logical_response_masks) != 2:
        return None
    if int(column.detector_response_masks[0]) != 0 or int(column.logical_response_masks[0]) != 0:
        return None
    if column.correction_response_masks is not None:
        correction_masks = tuple(int(value) for value in column.correction_response_masks)
        if len(correction_masks) != 2 or int(correction_masks[0]) != 0:
            return None
    log_priors = _column_log_priors(column)
    if len(log_priors) != 2 or not math.isfinite(float(log_priors[0])):
        return None
    no_error_log_const = float(log_priors[0])
    toggle_log_prior = float(log_priors[1])
    toggle_logodds = float(toggle_log_prior - no_error_log_const) if math.isfinite(toggle_log_prior) else float("-inf")

    row_update_mask = 0
    before_masks: list[int] = []
    before_parity: list[float] = []
    after_masks: list[int] = []
    after_parity: list[float] = []
    for update in tuple(row_updates):
        row_mask = int(update.row_mask)
        row_update_mask |= row_mask
        if bool(update.active_before):
            before_masks.append(row_mask)
            before_parity.append(float(update.parity_logodds_before))
        if bool(update.active_after):
            after_masks.append(row_mask)
            after_parity.append(float(update.parity_logodds_after))
    if int(row_update_mask) != int(column.detector_response_masks[1]):
        return None
    return BinaryFrontierColumnPayload(
        no_error_log_const=float(no_error_log_const),
        toggle_logodds=float(toggle_logodds),
        toggle_detector_mask=int(column.detector_response_masks[1]),
        toggle_observable_mask=int(column.logical_response_masks[1]),
        before_row_bit_masks=tuple(int(value) for value in before_masks),
        before_parity_logodds_weights=tuple(float(value) for value in before_parity),
        after_row_bit_masks=tuple(int(value) for value in after_masks),
        after_parity_logodds_weights=tuple(float(value) for value in after_parity),
    )
