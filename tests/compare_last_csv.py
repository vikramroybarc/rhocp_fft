#!/usr/bin/env python3
"""Compare final CSV rows of uninterrupted versus recovered RhoCP-FFT runs.
Usage: python tests/compare_last_csv.py clean.csv recovered.csv --rtol 1e-6
CSV headers must match; non-numeric metadata columns are ignored.
"""
import argparse
import csv
import math
from pathlib import Path


def last_row(path):
    with Path(path).open(newline="") as f:
        rows = csv.DictReader(f)
        row = None
        for candidate in rows:
            row = candidate
        if row is None:
            raise ValueError(f"No data rows in {path}")
        return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("uninterrupted")
    ap.add_argument("recovered")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-9)
    a = ap.parse_args()
    left, right = last_row(a.uninterrupted), last_row(a.recovered)
    if set(left) != set(right):
        raise SystemExit(f"CSV header mismatch: {set(left) ^ set(right)}")
    bad = []
    ncompared = 0
    for name in left:
        try:
            x, y = float(left[name]), float(right[name])
        except (ValueError, TypeError):
            continue
        ncompared += 1
        if not math.isclose(x, y, rel_tol=a.rtol, abs_tol=a.atol):
            bad.append((name, x, y))
    if ncompared == 0:
        raise SystemExit("No numeric CSV columns found")
    for name, x, y in bad:
        print(f"MISMATCH {name}: uninterrupted={x:.12g} recovered={y:.12g}")
    print(f"Compared {ncompared} numeric columns; mismatches: {len(bad)}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
