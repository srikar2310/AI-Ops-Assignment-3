#!/usr/bin/env python3
"""
spark_clean.py - NYC Yellow Taxi cleaning pipeline (PySpark)

Pipeline (identical to ray_clean.py):
  1. Ingest   : read yellow_tripdata_*.parquet, keep 10 columns
  2. Cleanse  : drop nulls -> drop duplicates -> format timestamps -> sanity filters
  3. Transform: join zone lookup twice (pickup + dropoff), add hour/day-of-week,
                compute avg speed (mph) with a custom Python UDF
  4. Export   : write Parquet

Example (cluster):
  python spark_clean.py --master spark://<MASTER_IP>:7077 \
      --data-dir /abs/shared/Data --output /abs/shared/out/spark_output --mode staged

--mode staged : materializes after every stage so each stage (and the UDF) can be
                timed on its own.
--mode e2e    : one lazy end-to-end run; only total wall-clock is meaningful.
Report both: staged for UDF overhead, e2e for the headline execution time.
"""
import argparse
import glob
import json
import os
import time
from datetime import datetime
from functools import reduce

from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import DoubleType

# ----------------------------------------------------------------------------
# Shared pipeline definition. Keep IDENTICAL in ray_clean.py.
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
TS_FMT = "yyyy-MM-dd HH:mm:ss"          # Python equivalent: %Y-%m-%d %H:%M:%S
VALID_START = "2026-01-01 00:00:00"      # default window: Jan 2025 .. Aug 2026 (end exclusive)
VALID_END = "2026-04-01 00:00:00"      # override with --valid-start / --valid-end
TARGET_TYPES = {                         # canonical types so files with different schemas can be unioned
    "VendorID": "int", "tpep_pickup_datetime": "timestamp", "tpep_dropoff_datetime": "timestamp",
    "passenger_count": "bigint", "trip_distance": "double", "PULocationID": "int",
    "DOLocationID": "int", "fare_amount": "double", "tip_amount": "double", "total_amount": "double",
}
MAX_DISTANCE = 100.0
MIN_DURATION = 60
MAX_DURATION = 6 * 3600
MAX_SPEED = 100.0


def avg_speed(distance, duration_sec):
    """Custom Python UDF logic: average speed in mph."""
    if distance is None or duration_sec is None or duration_sec <= 0:
        return None
    return round(distance / (duration_sec / 3600.0), 4)


def read_normalized(spark, files):
    """Read each file on its own, cast only the columns whose type differs, then union.
    (Spark's vectorized Parquet reader fails if the same column is int64 in one file and double in another.)"""
    parts = []
    for f in files:
        d = spark.read.parquet(f)
        src = {fld.name: fld.dataType.simpleString() for fld in d.schema.fields}
        cols = []
        for c in COLS:
            col = F.col(c)
            if src[c] != TARGET_TYPES[c]:
                col = col.try_cast(TARGET_TYPES[c])
            cols.append(col.alias(c))
        parts.append(d.select(*cols))
    return reduce(DataFrame.unionByName, parts)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="local[*]")
    ap.add_argument("--data-dir", default="Data")
    ap.add_argument("--output", default="results/spark_output")
    ap.add_argument("--mode", choices=["staged", "e2e"], default="staged")
    ap.add_argument("--shuffle-partitions", type=int, default=16)
    ap.add_argument("--valid-start", default=VALID_START, help="keep trips with pickup >= this")
    ap.add_argument("--valid-end", default=VALID_END, help="keep trips with pickup < this")
    ap.add_argument("--executor-memory", default=None, help="e.g. 2g")
    ap.add_argument("--udf-baseline", action="store_true",
                    help="(staged only) also time the same math as a native Spark expression")
    ap.add_argument("--hold", type=int, default=0,
                    help="seconds to keep the app alive at the end (for Spark UI :4040 screenshots)")
    ap.add_argument("--metrics-file", default="results/benchmark_runs.jsonl")
    return ap.parse_args()


def main():
    a = parse_args()
    data_dir = os.path.abspath(a.data_dir)
    out_path = os.path.abspath(a.output)
    pattern = os.path.join(data_dir, "yellow_tripdata_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files match {pattern}")
    in_bytes = sum(os.path.getsize(f) for f in files)
    staged = a.mode == "staged"

    builder = (
        SparkSession.builder.appName("a3-spark-clean").master(a.master)
        .config("spark.sql.shuffle.partitions", str(a.shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.inferTimestampNTZ.enabled", "false")
    )
    if a.executor_memory:
        builder = builder.config("spark.executor.memory", a.executor_memory)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    zones = (
        spark.read.option("header", True).csv(os.path.join(data_dir, "taxi_zone_lookup.csv"))
        .select(F.col("LocationID").cast("int").alias("LocationID"), "Borough", "Zone")
        .na.fill("")
    )
    pu = zones.select(F.col("LocationID").alias("PULocationID"),
                      F.col("Borough").alias("pickup_borough"),
                      F.col("Zone").alias("pickup_zone"))
    do = zones.select(F.col("LocationID").alias("DOLocationID"),
                      F.col("Borough").alias("dropoff_borough"),
                      F.col("Zone").alias("dropoff_zone"))

    times, counts = {}, {}
    p, d = "tpep_pickup_datetime", "tpep_dropoff_datetime"
    wall_start = time.perf_counter()

    # ---------------- Stage 1: ingest + cleanse ----------------
    s = time.perf_counter()
    df = read_normalized(spark, files)
    df = df.dropna(subset=COLS).dropDuplicates()
    df = (
        df.withColumn("pickup_ts", F.date_format(p, TS_FMT))
          .withColumn("dropoff_ts", F.date_format(d, TS_FMT))
          .withColumn("duration_sec", F.col(d).cast("long") - F.col(p).cast("long"))
          .withColumn("pickup_hour", F.hour(p))
          .withColumn("pickup_dow", (F.dayofweek(p) + 5) % 7) 
    )
    df = df.filter(
        (F.col("passenger_count") > 0)
        & (F.col("trip_distance") > 0) & (F.col("trip_distance") < MAX_DISTANCE)
        & (F.col("fare_amount") > 0) & (F.col("total_amount") > 0)
        & (F.col("duration_sec") >= MIN_DURATION) & (F.col("duration_sec") <= MAX_DURATION)
        & (F.col("pickup_ts") >= a.valid_start) & (F.col("pickup_ts") < a.valid_end)
    )
    if staged:
        df = df.cache()
        counts["rows_clean"] = df.count()
    times["ingest_clean"] = time.perf_counter() - s
    clean = df

    # ---------------- Stage 2: heavy join (2x broadcast join) ----------------
    s = time.perf_counter()
    df = clean.join(F.broadcast(pu), "PULocationID", "inner") \
              .join(F.broadcast(do), "DOLocationID", "inner")
    if staged:
        df = df.cache()
        counts["rows_joined"] = df.count()
        clean.unpersist()
    times["join"] = time.perf_counter() - s
    joined = df

    # ---------------- Stage 3: Python UDF ----------------
    s = time.perf_counter()
    speed_udf = F.udf(avg_speed, DoubleType())
    df = joined.withColumn("avg_speed_mph", speed_udf("trip_distance", "duration_sec"))
    df = df.filter((F.col("avg_speed_mph") > 0) & (F.col("avg_speed_mph") <= MAX_SPEED))
    if staged:
        df = df.cache()
        counts["rows_out"] = df.count()
    times["udf"] = time.perf_counter() - s
    final = df

    if staged and a.udf_baseline:
        s = time.perf_counter()
        native = F.round(F.col("trip_distance") / (F.col("duration_sec") / 3600.0), 4)
        joined.withColumn("avg_speed_mph", native) \
              .filter((F.col("avg_speed_mph") > 0) & (F.col("avg_speed_mph") <= MAX_SPEED)).count()
        times["udf_native_baseline"] = time.perf_counter() - s
    if staged:
        joined.unpersist()

    # ---------------- Stage 4: export ----------------
    s = time.perf_counter()
    final.select(*OUT_COLS).write.mode("overwrite").parquet(out_path)
    times["export"] = time.perf_counter() - s

    total = time.perf_counter() - wall_start
    if not staged:
        counts["rows_out"] = spark.read.parquet(out_path).count()

    record = {
        "framework": "spark", "mode": a.mode, "master": a.master,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "spark_version": spark.version, "input_files": len(files),
        "input_mb": round(in_bytes / 1e6, 1),
        "valid_start": a.valid_start, "valid_end": a.valid_end,
        "shuffle_partitions": a.shuffle_partitions,
        "times_sec": {k: round(v, 3) for k, v in times.items()},
        "total_sec": round(total, 3), "counts": counts,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.metrics_file)), exist_ok=True)
    with open(a.metrics_file, "a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))

    if a.hold:
        print(f"Holding {a.hold}s - take your Spark UI screenshot now (port 4040).")
        time.sleep(a.hold)
    spark.stop()


if __name__ == "__main__":
    main()