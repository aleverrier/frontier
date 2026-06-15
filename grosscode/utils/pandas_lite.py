from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


def isna(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _coerce_csv_value(raw: str) -> object:
    text = str(raw)
    if text == "":
        return ""
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


class SeriesILoc:
    def __init__(self, series: "Series") -> None:
        self._series = series

    def __getitem__(self, index: int) -> object:
        return self._series._values[int(index)]


class Series:
    def __init__(self, values: Iterable[object], *, name: str | None = None) -> None:
        self._values = list(values)
        self.name = name

    @property
    def iloc(self) -> SeriesILoc:
        return SeriesILoc(self)

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[object]:
        return iter(self._values)

    def __getitem__(self, index: int) -> object:
        return self._values[int(index)]

    def __eq__(self, other: object) -> "Series":
        return Series([value == other for value in self._values], name=self.name)

    def __ne__(self, other: object) -> "Series":
        return Series([value != other for value in self._values], name=self.name)

    def __and__(self, other: "Series") -> "Series":
        return Series([bool(a) and bool(b) for a, b in zip(self._values, other._values)], name=self.name)

    def __or__(self, other: "Series") -> "Series":
        return Series([bool(a) or bool(b) for a, b in zip(self._values, other._values)], name=self.name)

    def drop_duplicates(self) -> "Series":
        seen: list[object] = []
        out: list[object] = []
        for value in self._values:
            if value in seen:
                continue
            seen.append(value)
            out.append(value)
        return Series(out, name=self.name)

    def tolist(self) -> list[object]:
        return list(self._values)

    def to_numpy(self, dtype: Any | None = None) -> np.ndarray:
        arr = np.asarray(self._values)
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def mean(self) -> float:
        if not self._values:
            return float("nan")
        return float(np.mean(np.asarray(self._values, dtype=np.float64)))

    def quantile(self, q: float) -> float:
        if not self._values:
            return float("nan")
        return float(np.quantile(np.asarray(self._values, dtype=np.float64), float(q)))

    def sum(self) -> object:
        return sum(self._values)

    def max(self) -> object:
        return max(self._values)

    def nunique(self) -> int:
        return len(dict.fromkeys(self._values))


class DataFrameLoc:
    def __init__(self, frame: "LiteDataFrame") -> None:
        self._frame = frame

    def __getitem__(self, mask: Series | list[bool] | np.ndarray) -> "LiteDataFrame":
        if isinstance(mask, Series):
            values = [bool(value) for value in mask.tolist()]
        else:
            values = [bool(value) for value in list(mask)]
        rows = [dict(row) for row, keep in zip(self._frame._rows, values) if keep]
        return LiteDataFrame(rows, columns=self._frame.columns)


class DataFrameILoc:
    def __init__(self, frame: "LiteDataFrame") -> None:
        self._frame = frame

    def __getitem__(self, index: int) -> dict[str, object]:
        return dict(self._frame._rows[int(index)])


class LiteDataFrame:
    def __init__(self, rows: Iterable[dict[str, object]] | None = None, columns: list[str] | None = None) -> None:
        self._rows = [dict(row) for row in (rows or [])]
        if columns is not None:
            self.columns = list(columns)
        elif self._rows:
            ordered: list[str] = []
            for row in self._rows:
                for key in row.keys():
                    if key not in ordered:
                        ordered.append(str(key))
            self.columns = ordered
        else:
            self.columns = []
        for row in self._rows:
            for column in self.columns:
                row.setdefault(column, "")

    @property
    def empty(self) -> bool:
        return len(self._rows) == 0

    @property
    def loc(self) -> DataFrameLoc:
        return DataFrameLoc(self)

    @property
    def iloc(self) -> DataFrameILoc:
        return DataFrameILoc(self)

    def copy(self) -> "LiteDataFrame":
        return LiteDataFrame(self._rows, columns=self.columns)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key: str) -> Series:
        return Series([row.get(str(key), "") for row in self._rows], name=str(key))

    def sort_values(self, by: str | list[str], ascending: bool | list[bool] = True) -> "LiteDataFrame":
        columns = [str(by)] if isinstance(by, str) else [str(item) for item in by]
        ascend = [bool(ascending)] * len(columns) if isinstance(ascending, bool) else [bool(item) for item in ascending]
        if len(ascend) == 1 and len(columns) > 1:
            ascend = ascend * len(columns)
        rows = list(self._rows)
        for column, asc in reversed(list(zip(columns, ascend))):
            rows.sort(key=lambda row: row.get(column, ""), reverse=not bool(asc))
        return LiteDataFrame(rows, columns=self.columns)

    def reset_index(self, drop: bool = False) -> "LiteDataFrame":
        return LiteDataFrame(self._rows, columns=self.columns)

    def to_csv(self, path: str | Path, index: bool = False) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.columns))
            writer.writeheader()
            for row in self._rows:
                writer.writerow({column: row.get(column, "") for column in self.columns})

    def to_parquet(self, path: str | Path, index: bool = False) -> None:
        raise NotImplementedError("parquet output is unavailable in pandas_lite")

    def groupby(self, by: str | list[str], sort: bool = True) -> Iterator[tuple[object, "LiteDataFrame"]]:
        columns = [str(by)] if isinstance(by, str) else [str(item) for item in by]
        groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
        for row in self._rows:
            key = tuple(row.get(column, "") for column in columns)
            groups.setdefault(key, []).append(dict(row))
        keys = sorted(groups.keys()) if sort else list(groups.keys())
        for key in keys:
            payload = LiteDataFrame(groups[key], columns=self.columns)
            if len(columns) == 1:
                yield key[0], payload
            else:
                yield key, payload

    def iterrows(self) -> Iterator[tuple[int, dict[str, object]]]:
        for index, row in enumerate(self._rows):
            yield int(index), dict(row)

    def to_dict(self, orient: str) -> list[dict[str, object]]:
        if str(orient) != "records":
            raise ValueError("pandas_lite only supports orient='records'")
        return [dict(row) for row in self._rows]

    def equals(self, other: object) -> bool:
        if not isinstance(other, LiteDataFrame):
            return False
        return self.columns == other.columns and self._rows == other._rows


def read_csv(path: str | Path) -> LiteDataFrame:
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, object]] = []
        for row in reader:
            rows.append({str(key): _coerce_csv_value(value) for key, value in row.items()})
    return LiteDataFrame(rows)


DataFrame = LiteDataFrame
