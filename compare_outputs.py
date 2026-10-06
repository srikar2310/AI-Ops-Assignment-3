#!/usr/bin/env python3
"""
compare_outputs.py - prove Spark and Ray produced identical results.

Usage:
  python compare_outputs.py results/spark_output results/ray_output

Checks (streamed in batches, so it works on multi-GB output):
  1. Row count
  2. Order-independent row checksum (exact match test; row order differs between engines)
  3. Per-column order-independent checksums (tells you WHICH column differs if (2) fails)
  4. Negative-zero counts for float columns (-0.0 == 0.0 in sums but not bit-for-bit)
  5. Per-column sums (tolerance 1e-9 relative, float summation order differs)
"""
import math
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

M = 2 ** 64


def fingerprint(path):
    dset = ds.dataset(path, format="parquet")
    rows, checksum = 0, 0
    sums, colhash, negzero = {}, {}, {}
    for batch in dset.to_batches(batch_size=500_000):
        df = batch.to_pandas()
        df = df[sorted(df.columns)]
        for c in df.columns:
            if pd.api.types.is_integer_dtype(df[c]):
                df[c] = df[c].astype("int64")
            elif pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].astype("float64")
        h = pd.util.hash_pandas_object(df, index=False).to_numpy(dtype="uint64")
        checksum = (checksum + int(h.sum(dtype=np.uint64))) % M
        rows += len(df)
        for c in df.columns:
            hc = pd.util.hash_pandas_object(df[c], index=False).to_numpy(dtype="uint64")
            colhash[c] = (colhash.get(c, 0) + int(hc.sum(dtype=np.uint64))) % M
            if pd.api.types.is_float_dtype(df[c]):
                v = df[c].to_numpy()
                negzero[c] = negzero.get(c, 0) + int((np.signbit(v) & (v == 0)).sum())
        for c in df.select_dtypes("number").columns:
            sums[c] = sums.get(c, 0.0) + float(df[c].sum())
    return rows, checksum, sums, colhash, negzero


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    ra, ca, sa, ha, za = fingerprint(sys.argv[1])
    rb, cb, sb, hb, zb = fingerprint(sys.argv[2])

    ok = True
    print(f"Rows      : {ra:,} vs {rb:,} -> {'OK' if ra == rb else 'MISMATCH'}")
    ok &= ra == rb
    print(f"Checksum  : {ca} vs {cb} -> {'OK' if ca == cb else 'MISMATCH'}")
    ok &= ca == cb

    print("\nPer-column checksums (which column differs?):")
    for c in sorted(set(ha) | set(hb)):
        print(f"  {c:18s} {'OK' if ha.get(c) == hb.get(c) else 'MISMATCH'}")

    print("\nNegative zeros (-0.0) per float column  [A vs B]:")
    any_nz = False
    for c in sorted(set(za) | set(zb)):
        if za.get(c, 0) or zb.get(c, 0):
            any_nz = True
            print(f"  {c:18s} {za.get(c, 0):>10,} vs {zb.get(c, 0):>10,}")
    if not any_nz:
        print("  none in either output")

    print("\nColumn sums:")
    for c in sorted(set(sa) | set(sb)):
        x, y = sa.get(c), sb.get(c)
        same = x is not None and y is not None and math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-6)
        ok &= same
        print(f"  {c:18s} {x!s:>24} {y!s:>24}  {'OK' if same else 'MISMATCH'}")
    print("\nRESULT:", "IDENTICAL" if ok else "DIFFERENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()