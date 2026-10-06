#!/usr/bin/env python3
"""
ray_clean.py - NYC Yellow Taxi cleaning pipeline (Ray Data)

Same steps, filters, columns and UDF logic as spark_clean.py.

Example (cluster, after `ray start --head` / `ray start --address=...`):
  python ray_clean.py --address auto \
      --data-dir /abs/shared/Data --output /abs/shared/out/ray_output --mode staged

--mode staged : materializes after every stage so each stage (and the UDF) is timed.
--mode e2e    : one lazy end-to-end run; only total wall-clock is meaningful.
"""
import argparse
import glob
import json
import os
import shutil
import time
from datetime import datetime

import pandas as pd
import ray

# ----------------------------------------------------------------------------
# Shared pipeline definition. Keep IDENTICAL in spark_clean.py.
# ----------------------------------------------------------------------------
COLS = [
    "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
    "trip_distance", "PULocationID", "DOLocationID", "fare_amount", "tip_amount",
    "total_amount",
]
OUT_COLS = [
    "VendorID", "passenger_count", "trip_distance", "fare_amount", "tip_amount",
    "total_amount", "PULocationID", "DOLocationID", "pickup_borough", "pickup_zone",
    "dropoff_borough", "dropoff_zone", "pickup_ts", "dropoff_ts", "pickup_hour",
    "pickup_dow", "duration_sec", "avg_speed_mph",
]
TS_FMT = "%Y-%m-%d %H:%M:%S"
VALID_START = "2026-01-01 00:00:00"      # default window: Jan 2025 .. Aug 2026 (end exclusive)
VALID_END = "2026-04-01 00:00:00"      # override with --valid-start / --valid-end
CANON_DTYPES = {                         # canonical types, same as Spark TARGET_TYPES
    "VendorID": "int32", "tpep_pickup_datetime": "datetime64[us]", "tpep_dropoff_datetime": "datetime64[us]",
    "passenger_count": "int64", "trip_distance": "float64", "PULocationID": "int32",
    "DOLocationID": "int32", "fare_amount": "float64", "tip_amount": "float64", "total_amount": "float64",
}
MAX_DISTANCE = 100.0
MIN_DURATION = 60
MAX_DURATION = 6 * 3600
MAX_SPEED = 100.0


def avg_speed(distance, duration_sec):
    """Custom Python UDF logic: average speed in mph (same function as Spark)."""
    if distance is None or duration_sec is None or duration_sec <= 0:
        return None
    return round(distance / (duration_sec / 3600.0), 4)


# ----------------------------- batch functions -----------------------------
def drop_nulls(df):
    df = df.dropna(subset=COLS)
    return df.astype(CANON_DTYPES)       # after dropna, so float->int casts are safe


def derive_and_filter(df, valid_start, valid_end):
    p, d = df["tpep_pickup_datetime"], df["tpep_dropoff_datetime"]
    df = df.assign(
        pickup_ts=p.dt.strftime(TS_FMT),
        dropoff_ts=d.dt.strftime(TS_FMT),
        duration_sec=(d.dt.floor("s") - p.dt.floor("s")).dt.total_seconds().astype("int64"),
        pickup_hour=p.dt.hour.astype("int32"),
        pickup_dow=p.dt.dayofweek.astype("int32"),          # Mon=0 .. Sun=6
    )
    keep = (
        (df["passenger_count"] > 0)
        & (df["trip_distance"] > 0) & (df["trip_distance"] < MAX_DISTANCE)
        & (df["fare_amount"] > 0) & (df["total_amount"] > 0)
        & (df["duration_sec"] >= MIN_DURATION) & (df["duration_sec"] <= MAX_DURATION)
        & (df["pickup_ts"] >= valid_start) & (df["pickup_ts"] < valid_end)
    )
    return df[keep]


def make_join(pu, do):
    def _join(df):
        df = df.merge(pu, on="PULocationID", how="inner")
        return df.merge(do, on="DOLocationID", how="inner")
    return _join


def add_speed(df):
    # .tolist() -> native Python floats/ints, so round() behaves exactly like in Spark's UDF
    speeds = [avg_speed(x, y) for x, y in
              zip(df["trip_distance"].tolist(), df["duration_sec"].tolist())]
    df = df.assign(avg_speed_mph=pd.Series(speeds, index=df.index, dtype="float64"))
    return df[(df["avg_speed_mph"] > 0) & (df["avg_speed_mph"] <= MAX_SPEED)]


def add_speed_native(df):
    s = (df["trip_distance"] / (df["duration_sec"] / 3600.0)).round(4)
    df = df.assign(avg_speed_mph=s)
    return df[(df["avg_speed_mph"] > 0) & (df["avg_speed_mph"] <= MAX_SPEED)]


HASH_COL = "_row_hash"


def add_row_hash(df):
    h = pd.util.hash_pandas_object(df[COLS], index=False).to_numpy(dtype="uint64")
    return df.assign(**{HASH_COL: h.astype("int64")})


def drop_dups_block(df):
    # Identical rows share a hash -> same partition, so exact dedup per block is enough.
    return df.drop_duplicates(subset=COLS).drop(columns=[HASH_COL])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default=None, help="'auto' to join a running cluster; omit for local")
    ap.add_argument("--data-dir", default="Data")
    ap.add_argument("--valid-start", default=VALID_START, help="keep trips with pickup >= this")
    ap.add_argument("--valid-end", default=VALID_END, help="keep trips with pickup < this")
    ap.add_argument("--output", default="results/ray_output")
    ap.add_argument("--mode", choices=["staged", "e2e"], default="staged")
    ap.add_argument("--parallelism", type=int, default=None,
                    help="optional override_num_blocks for read_parquet")
    ap.add_argument("--dedup-partitions", type=int, default=None,
                    help="num_partitions for the dedup groupby (Ray default is 200); "
                         "use the same value as Spark's --shuffle-partitions for a fair test")
    ap.add_argument("--dedup-mode", choices=["groupby", "hash"], default="groupby",
                    help="groupby = 10-column groupby (default); hash = hash-partition on a "
                         "row hash, then exact drop_duplicates inside each partition")
    ap.add_argument("--udf-baseline", action="store_true",
                    help="(staged only) also time the same math as a vectorized pandas expression")
    ap.add_argument("--hold", type=int, default=0,
                    help="seconds to keep the driver alive at the end (for dashboard screenshots)")
    ap.add_argument("--metrics-file", default="results/benchmark_runs.jsonl")
    return ap.parse_args()


def main():
    a = parse_args()
    data_dir = os.path.abspath(a.data_dir)
    out_path = os.path.abspath(a.output)
    files = sorted(glob.glob(os.path.join(data_dir, "yellow_tripdata_*.parquet")))
    if not files:
        raise SystemExit(f"No yellow_tripdata_*.parquet in {data_dir}")
    in_bytes = sum(os.path.getsize(f) for f in files)
    staged = a.mode == "staged"

    ray.init(address=a.address, ignore_reinit_error=True)
    resources = ray.cluster_resources()
    print("Ray cluster resources:", resources)

    zones = pd.read_csv(os.path.join(data_dir, "taxi_zone_lookup.csv"), keep_default_na=False)
    zones = zones[["LocationID", "Borough", "Zone"]]
    zones["LocationID"] = zones["LocationID"].astype("int32")
    pu = zones.rename(columns={"LocationID": "PULocationID", "Borough": "pickup_borough",
                               "Zone": "pickup_zone"})
    do = zones.rename(columns={"LocationID": "DOLocationID", "Borough": "dropoff_borough",
                               "Zone": "dropoff_zone"})

    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    read_kwargs = {}
    if a.parallelism:
        read_kwargs["override_num_blocks"] = a.parallelism

    times, counts = {}, {}
    wall_start = time.perf_counter()

    # ---------------- Stage 1: ingest + cleanse ----------------
    s = time.perf_counter()
    ds = ray.data.read_parquet(files, **read_kwargs).select_columns(COLS)
    ds = ds.map_batches(drop_nulls, batch_format="pandas")
    if a.dedup_mode == "hash":
        ds = ds.map_batches(add_row_hash, batch_format="pandas")
        ds = ds.repartition(a.dedup_partitions or 16, keys=[HASH_COL])
        ds = ds.map_batches(drop_dups_block, batch_format="pandas", batch_size=None)
    else:
        gb_kwargs = {"num_partitions": a.dedup_partitions} if a.dedup_partitions else {}
        ds = ds.groupby(COLS, **gb_kwargs).count().drop_columns(["count()"])   # == dropDuplicates()
    ds = ds.map_batches(derive_and_filter, batch_format="pandas",
                        fn_kwargs={"valid_start": a.valid_start, "valid_end": a.valid_end})
    if staged:
        ds = ds.materialize()
        counts["rows_clean"] = ds.count()
    times["ingest_clean"] = time.perf_counter() - s

    # ---------------- Stage 2: heavy join (2x broadcast-style merge) ----------------
    s = time.perf_counter()
    ds = ds.map_batches(make_join(pu, do), batch_format="pandas")
    if staged:
        ds = ds.materialize()
        counts["rows_joined"] = ds.count()
    times["join"] = time.perf_counter() - s
    joined = ds

    # ---------------- Stage 3: Python UDF ----------------
    s = time.perf_counter()
    ds = joined.map_batches(add_speed, batch_format="pandas")
    if staged:
        ds = ds.materialize()
        counts["rows_out"] = ds.count()
    times["udf"] = time.perf_counter() - s

    if staged and a.udf_baseline:
        s = time.perf_counter()
        joined.map_batches(add_speed_native, batch_format="pandas").materialize()
        times["udf_native_baseline"] = time.perf_counter() - s

    # ---------------- Stage 4: export ----------------
    s = time.perf_counter()
    ds.select_columns(OUT_COLS).write_parquet(out_path)
    times["export"] = time.perf_counter() - s

    total = time.perf_counter() - wall_start
    if not staged:
        counts["rows_out"] = ray.data.read_parquet(out_path).count()   # outside the timer

    record = {
        "dedup_mode": a.dedup_mode,
        "dedup_partitions": a.dedup_partitions or ("default(200)" if a.dedup_mode == "groupby" else 16),
        "framework": "ray", "mode": a.mode, "address": a.address or "local",
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "ray_version": ray.__version__, "input_files": len(files),
        "input_mb": round(in_bytes / 1e6, 1),
        "valid_start": a.valid_start, "valid_end": a.valid_end,
        "cluster_cpus": resources.get("CPU"),
        "times_sec": {k: round(v, 3) for k, v in times.items()},
        "total_sec": round(total, 3), "counts": counts,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.metrics_file)), exist_ok=True)
    with open(a.metrics_file, "a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))

    if a.hold:
        print(f"Holding {a.hold}s - take your Ray Dashboard screenshot now (port 8265).")
        time.sleep(a.hold)
    ray.shutdown()


if __name__ == "__main__":
    main()