"""
Scalability and Performance Benchmarking Engine.
Evaluates data-processing speed, storage efficiency, and Spark scaling
across increasing dataset volumes (up to 1,000,000+ records).
"""

import os
import time
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from pyspark.sql import DataFrame

from config import BENCHMARK_DIR, DATA_DIR
from src.big_data_pipeline.spark_manager import SparkSessionManager
from src.big_data_pipeline.historical_generator import HistoricalOrbitalGenerator
from src.big_data_pipeline.parquet_lake import ORBITAL_RECORD_SCHEMA

logger = logging.getLogger(__name__)


class ScalabilityBenchmark:
    """Runs automated benchmarks measuring Big Data scalability, throughput, and storage efficiency."""

    def __init__(self, output_dir: Path = BENCHMARK_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.spark = SparkSessionManager.get_spark_session()
        self.generator = HistoricalOrbitalGenerator()

    def run_storage_efficiency_test(self, record_count: int = 50000) -> Dict[str, Any]:
        """
        Compares storage footprint across:
        1. Raw Uncompressed CSV
        2. Parquet (Snappy compressed, partitioned)
        3. Parquet (Gzip compressed, partitioned)
        """
        temp_dir = DATA_DIR / "temp_benchmark_storage"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {record_count} records for storage efficiency test...")
        records = self.generator._generate_synthetic_chunk(record_count, time.time())
        pdf = pd.DataFrame(records)
        df: DataFrame = self.spark.createDataFrame(pdf, schema=ORBITAL_RECORD_SCHEMA)

        # 1. Raw CSV
        csv_path = temp_dir / "raw_records.csv"
        t0 = time.time()
        pdf.to_csv(csv_path, index=False)
        csv_write_time = time.time() - t0
        csv_size_bytes = csv_path.stat().st_size

        # 2. Parquet Snappy (default)
        parquet_snappy_path = temp_dir / "parquet_snappy"
        t0 = time.time()
        (
            df.write.mode("overwrite")
            .option("compression", "snappy")
            .partitionBy("orbit_regime", "object_type")
            .parquet(str(parquet_snappy_path))
        )
        snappy_write_time = time.time() - t0
        snappy_size_bytes = sum(f.stat().st_size for f in parquet_snappy_path.rglob("*.parquet"))

        # 3. Parquet Gzip
        parquet_gzip_path = temp_dir / "parquet_gzip"
        t0 = time.time()
        (
            df.write.mode("overwrite")
            .option("compression", "gzip")
            .partitionBy("orbit_regime", "object_type")
            .parquet(str(parquet_gzip_path))
        )
        gzip_write_time = time.time() - t0
        gzip_size_bytes = sum(f.stat().st_size for f in parquet_gzip_path.rglob("*.parquet"))

        # Clean up temporary test files
        shutil.rmtree(temp_dir, ignore_errors=True)

        csv_mb = round(csv_size_bytes / (1024 * 1024), 2)
        snappy_mb = round(snappy_size_bytes / (1024 * 1024), 2)
        gzip_mb = round(gzip_size_bytes / (1024 * 1024), 2)

        results = {
            "record_count": record_count,
            "csv_size_mb": csv_mb,
            "parquet_snappy_size_mb": snappy_mb,
            "parquet_gzip_size_mb": gzip_mb,
            "snappy_compression_ratio": round(csv_size_bytes / max(1, snappy_size_bytes), 2),
            "gzip_compression_ratio": round(csv_size_bytes / max(1, gzip_size_bytes), 2),
            "csv_write_time_sec": round(csv_write_time, 2),
            "snappy_write_time_sec": round(snappy_write_time, 2),
            "gzip_write_time_sec": round(gzip_write_time, 2),
            "storage_savings_pct": round((1.0 - (snappy_size_bytes / max(1, csv_size_bytes))) * 100, 1),
        }

        # Save to benchmark directory
        out_file = self.output_dir / "storage_efficiency.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def run_scaling_benchmark(
        self, scale_points: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Executes scalability suite across varying record sizes.
        Measures: Ingestion throughput, Spark aggregation query latency, Memory usage.
        """
        if scale_points is None:
            scale_points = [20000, 50000, 100000, 250000]

        temp_lake = DATA_DIR / "temp_benchmark_lake"
        records_history = []

        for count in scale_points:
            logger.info(f"Running scaling benchmark for N={count:,}...")
            if temp_lake.exists():
                shutil.rmtree(temp_lake)

            # Measure Data Generation Time
            t_gen_start = time.time()
            data_chunks = self.generator._generate_synthetic_chunk(count, time.time())
            gen_sec = time.time() - t_gen_start

            # Measure Spark DataFrame Creation & Ingestion
            t_spark_start = time.time()
            pdf = pd.DataFrame(data_chunks)
            df = self.spark.createDataFrame(pdf, schema=ORBITAL_RECORD_SCHEMA)
            df.write.mode("overwrite").partitionBy("orbit_regime").parquet(str(temp_lake))
            ingest_sec = time.time() - t_spark_start

            # Measure Spark SQL Query Latency (Aggregation over partitions)
            bench_df = self.spark.read.parquet(str(temp_lake))
            bench_df.createOrReplaceTempView("bench_orbital")

            t_query1_start = time.time()
            self.spark.sql("""
                SELECT orbit_regime, AVG(altitude_km), AVG(velocity_km_s), COUNT(*) 
                FROM bench_orbital 
                GROUP BY orbit_regime
            """).collect()
            query1_sec = time.time() - t_query1_start

            t_query2_start = time.time()
            self.spark.sql("""
                SELECT CAST(FLOOR(altitude_km / 50) * 50 AS INT) as alt_bin, COUNT(*) 
                FROM bench_orbital 
                WHERE object_type = 'DEBRIS'
                GROUP BY CAST(FLOOR(altitude_km / 50) * 50 AS INT)
            """).collect()
            query2_sec = time.time() - t_query2_start

            # Calculate metrics
            total_sec = gen_sec + ingest_sec
            throughput = int(count / max(0.001, total_sec))
            parquet_size_mb = sum(f.stat().st_size for f in temp_lake.rglob("*.parquet")) / (1024 * 1024)

            records_history.append({
                "record_count": count,
                "generation_time_s": round(gen_sec, 3),
                "ingest_parquet_time_s": round(ingest_sec, 3),
                "total_pipeline_time_s": round(total_sec, 3),
                "throughput_records_s": throughput,
                "aggregation_query_s": round(query1_sec, 4),
                "filtered_query_s": round(query2_sec, 4),
                "parquet_size_mb": round(parquet_size_mb, 2),
            })

        # Clean up
        shutil.rmtree(temp_lake, ignore_errors=True)

        result_df = pd.DataFrame(records_history)
        result_df.to_json(self.output_dir / "scaling_benchmark.json", orient="records", indent=2)
        return result_df

    def get_latest_benchmark_results(self) -> Dict[str, Any]:
        """Returns pre-saved or cached benchmark metrics."""
        storage_file = self.output_dir / "storage_efficiency.json"
        scaling_file = self.output_dir / "scaling_benchmark.json"

        storage_data = None
        scaling_data = None

        if storage_file.exists():
            try:
                with open(storage_file) as f:
                    storage_data = json.load(f)
            except Exception:
                pass

        if scaling_file.exists():
            try:
                scaling_data = pd.read_json(scaling_file)
            except Exception:
                pass

        return {
            "storage_efficiency": storage_data,
            "scaling_benchmark": scaling_data,
        }
