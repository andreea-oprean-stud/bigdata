"""
Partitioned Parquet Data Lake Manager.
Handles writing, partitioning, and querying large orbital position datasets.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    BooleanType,
    TimestampType,
)

from config import PARQUET_LAKE_DIR
from src.big_data_pipeline.spark_manager import SparkSessionManager

logger = logging.getLogger(__name__)

# Canonical Orbital Position Schema
ORBITAL_RECORD_SCHEMA = StructType([
    StructField("record_id", StringType(), False),
    StructField("norad_id", IntegerType(), False),
    StructField("object_name", StringType(), False),
    StructField("object_type", StringType(), False),  # PAYLOAD, DEBRIS, ROCKET BODY
    StructField("orbit_regime", StringType(), False),  # LEO, MEO, GEO, HEO
    StructField("owner", StringType(), True),
    StructField("timestamp", TimestampType(), False),
    StructField("epoch_year", IntegerType(), False),
    StructField("latitude", DoubleType(), False),
    StructField("longitude", DoubleType(), False),
    StructField("altitude_km", DoubleType(), False),
    StructField("velocity_km_s", DoubleType(), False),
    StructField("footprint_radius_km", DoubleType(), False),
    StructField("is_sunlit", BooleanType(), False),
    StructField("solar_elevation_subpoint", DoubleType(), True),
])


class ParquetLakeManager:
    """Manages Parquet Lake storage with dual partitioning (orbit_regime, object_type)."""

    def __init__(self, lake_dir: Path = PARQUET_LAKE_DIR):
        self.lake_dir = Path(lake_dir)
        self.lake_dir.mkdir(parents=True, exist_ok=True)
        self.spark = SparkSessionManager.get_spark_session()

    def write_dataframe(
        self,
        df: DataFrame,
        mode: str = "overwrite",
        partition_cols: Optional[List[str]] = None,
    ) -> Path:
        """Writes Spark DataFrame to partitioned Parquet lake."""
        if partition_cols is None:
            partition_cols = ["orbit_regime", "object_type"]

        logger.info(f"Writing DataFrame to Parquet Lake at {self.lake_dir} (partitions: {partition_cols})...")
        (
            df.write.mode(mode)
            .partitionBy(*partition_cols)
            .parquet(str(self.lake_dir))
        )
        return self.lake_dir

    def read_lake(self) -> DataFrame:
        """Reads complete Parquet Lake into Spark DataFrame."""
        return self.spark.read.parquet(str(self.lake_dir))

    def get_lake_stats(self) -> Dict[str, Any]:
        """Calculates statistics on current Parquet Lake (file size, total records, partitions)."""
        if not self.lake_dir.exists() or not any(self.lake_dir.iterdir()):
            return {
                "exists": False,
                "total_records": 0,
                "total_size_mb": 0.0,
                "partitions": [],
                "file_count": 0,
            }

        total_size = sum(f.stat().st_size for f in self.lake_dir.rglob("*.parquet"))
        file_count = len(list(self.lake_dir.rglob("*.parquet")))
        
        # Discover partition folders
        partition_dirs = [
            str(p.relative_to(self.lake_dir))
            for p in self.lake_dir.glob("*/*")
            if p.is_dir() and "=" in p.name
        ]

        # Read row count via Spark
        try:
            df = self.read_lake()
            row_count = df.count()
        except Exception as e:
            logger.warning(f"Could not count rows in lake: {e}")
            row_count = 0

        return {
            "exists": True,
            "total_records": row_count,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_count": file_count,
            "partitions": partition_dirs,
            "lake_path": str(self.lake_dir),
        }

    def clear_lake(self):
        """Removes all data from Parquet Lake."""
        if self.lake_dir.exists():
            shutil.rmtree(self.lake_dir)
            self.lake_dir.mkdir(parents=True, exist_ok=True)
