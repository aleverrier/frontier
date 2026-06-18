# SPDX-License-Identifier: Apache-2.0
"""Reproduce paper plots from committed plot-ready data."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PLOTS_ROOT = REPO_ROOT / "paper" / "plots"
MANIFEST_PATH = PLOTS_ROOT / "manifest.csv"
DEFAULT_OUT_DIR = PLOTS_ROOT / "outputs"

REQUIRED_COLUMNS = [
    "figure_id",
    "panel_id",
    "title",
    "paper_reference",
    "data_file",
    "plotting_script",
    "output_file",
    "data_kind",
    "data_source",
    "generation_command",
    "environment",
    "status",
    "notes",
]

REPRODUCIBLE = "reproducible"
SCRIPT_MISSING = "script-missing"
SUPPORT_DATA = "support-data"


def _repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        return list(reader)


def _row_label(row: dict[str, str]) -> str:
    panel = row.get("panel_id", "")
    if panel:
        return f"{row.get('figure_id', '')}/{panel}"
    return row.get("figure_id", "")


def _print_list(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("No paper figure rows are declared in paper/plots/manifest.csv.")
        print("Paper figure list and plot data are data-missing in this checkout.")
        print("See paper/plots/README.md for the required data columns and sidecar schema.")
        return

    for row in rows:
        print(
            "\t".join(
                [
                    _row_label(row),
                    row["status"],
                    row["paper_reference"],
                    row["data_file"],
                    row["plotting_script"],
                    row["output_file"],
                ]
            )
        )


def _run_external_script(row: dict[str, str], out_dir: Path) -> int:
    data_path = _repo_path(row["data_file"])
    script_path = _repo_path(row["plotting_script"])
    output_path = out_dir / Path(row["output_file"]).name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script_path),
        "--data",
        str(data_path),
        "--output",
        str(output_path),
        "--figure-id",
        row["figure_id"],
        "--panel-id",
        row["panel_id"],
        "--manifest",
        str(MANIFEST_PATH),
    ]
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def _reproduce_rows(rows: list[dict[str, str]], out_dir: Path, strict: bool) -> int:
    if not rows:
        print("No paper plot rows selected; no outputs generated.")
        return 0

    status = 0
    for row in rows:
        label = _row_label(row)
        row_status = row["status"]
        if row_status == SUPPORT_DATA:
            print(f"Skipping {label or '<unknown>'}: status=support-data; used by another renderer.")
            continue
        if row_status != REPRODUCIBLE:
            missing = (
                "a committed figure-specific renderer is not available"
                if row_status == SCRIPT_MISSING
                else "committed paper data or scripts are not available"
            )
            print(
                f"Skipping {label or '<unknown>'}: status={row_status}; "
                f"{missing}."
            )
            if strict:
                status = 1
            continue

        data_path = _repo_path(row["data_file"])
        script_path = _repo_path(row["plotting_script"])
        missing = [str(path) for path in (data_path, script_path) if not path.exists()]
        if missing:
            print(f"Cannot reproduce {label}: missing {', '.join(missing)}.", file=sys.stderr)
            status = 1
            continue

        if script_path.resolve() == Path(__file__).resolve():
            print(
                f"Cannot reproduce {label}: no figure-specific renderer is registered.",
                file=sys.stderr,
            )
            status = 1
            continue

        result = _run_external_script(row, out_dir)
        if result != 0:
            status = result
    return status


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="List manifest rows and data status.")
    action.add_argument("--figure", help="Reproduce one figure_id from the manifest.")
    action.add_argument("--all", action="store_true", help="Reproduce all manifest rows.")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory for generated plot outputs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return nonzero when selected rows are not reproducible.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        rows = _load_manifest()
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.list:
        _print_list(rows)
        return 0

    if args.figure:
        selected = [row for row in rows if row["figure_id"] == args.figure]
        if not selected:
            print(f"No manifest rows found for figure_id={args.figure!r}.", file=sys.stderr)
            return 1 if args.strict else 0
    else:
        selected = rows

    return _reproduce_rows(selected, Path(args.out_dir), args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
