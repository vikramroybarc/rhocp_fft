#!/usr/bin/env python3
"""Compare every row of two numeric simulation CSV files (standard library only)."""

import argparse
import csv
import math
from pathlib import Path


DEFAULT_REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "examples/ddcp4p2/out_ddtfft_optiona.csv"
)


def read_csv(path):
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header or len(set(header)) != len(header):
            raise ValueError(f"{path}: missing or duplicate column names")
        rows = []
        for line, values in enumerate(reader, 2):
            if not values:
                continue
            if len(values) != len(header):
                raise ValueError(f"{path}:{line}: expected {len(header)} values")
            try:
                row = [float(value) for value in values]
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: non-numeric value") from exc
            if not all(math.isfinite(value) for value in row):
                raise ValueError(f"{path}:{line}: NaN or infinity")
            rows.append(dict(zip(header, row)))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return header, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="CSV result to compare")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-9)
    args = parser.parse_args()
    if not all(math.isfinite(x) and x >= 0 for x in (args.rtol, args.atol)):
        parser.error("tolerances must be finite and nonnegative")

    try:
        columns, reference = read_csv(args.reference)
        other_columns, result = read_csv(args.result)
        if set(columns) != set(other_columns):
            raise ValueError(
                f"Column mismatch: missing={sorted(set(columns) - set(other_columns))}, "
                f"extra={sorted(set(other_columns) - set(columns))}"
            )
        if len(reference) != len(result):
            raise ValueError(f"Row count mismatch: reference={len(reference)}, result={len(result)}")
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"Error: {exc}\n")

    print(f"Reference: {args.reference}\nResult:    {args.result}")
    print(f"Rows: {len(reference)}; rtol={args.rtol:g}; atol={args.atol:g}")
    print("Rows are paired in file order; time is compared like every other column.")
    print("Pass criterion: abs(result-reference) <= atol + rtol*abs(reference)")
    print(f"{'Column':<20} {'Max abs error':>15} {'Max rel error':>15} {'Failed rows':>12}")
    total_failed = 0
    for column in columns:
        max_abs = max_rel = 0.0
        failed = 0
        for left, right in zip(reference, result):
            expected, actual = left[column], right[column]
            error = abs(actual - expected)
            relative = error / abs(expected) if expected else (math.inf if error else 0.0)
            max_abs = max(max_abs, error)
            max_rel = max(max_rel, relative)
            failed += error > args.atol + args.rtol * abs(expected)
        total_failed += failed
        print(f"{column:<20} {max_abs:15.6g} {max_rel:15.6g} {failed:12d}")
    print(f"{'FAIL' if total_failed else 'PASS'}: {total_failed} values outside tolerance")
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
