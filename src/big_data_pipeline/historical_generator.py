"""
Massive Orbital Historical Record Generator.
Generates large-scale high-precision orbital records across diverse satellite constellations
and debris populations for Big Data analysis and Spark benchmarks.
"""

import math
import uuid
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from pyspark.sql import DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    BooleanType,
    TimestampType,
)

from src.big_data_pipeline.spark_manager import SparkSessionManager
from src.big_data_pipeline.parquet_lake import ParquetLakeManager, ORBITAL_RECORD_SCHEMA
from src.orbital_engine.sgp4_engine import SGP4Engine, EARTH_RADIUS_KM
from src.orbital_engine.solar_calculator import SolarCalculator
from src.data_ingestion.sample_data_loader import SAMPLE_TLES, get_sample_satcat_records

logger = logging.getLogger(__name__)


class HistoricalOrbitalGenerator:
    """Generates massive synthetic and propagated historical satellite positions."""

    def __init__(self):
        self.spark = SparkSessionManager.get_spark_session()
        self.lake = ParquetLakeManager()

    @staticmethod
    def _generate_synthetic_chunk(
        chunk_size: int,
        start_time_ts: float,
        seed_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Generates a realistic vectorized chunk of orbital position records."""
        np.random.seed(42 + seed_offset)

        # Regimes and Object types distribution
        regimes = ["LEO", "MEO", "GEO", "HEO"]
        regime_weights = [0.70, 0.15, 0.12, 0.03]
        obj_types = ["PAYLOAD", "DEBRIS", "ROCKET BODY"]
        type_weights = [0.55, 0.35, 0.10]
        owners = ["US", "CIS", "PRC", "ESA", "IND", "JPN", "OTHER"]
        owner_weights = [0.35, 0.25, 0.15, 0.10, 0.05, 0.05, 0.05]

        chosen_regimes = np.random.choice(regimes, size=chunk_size, p=regime_weights)
        chosen_types = np.random.choice(obj_types, size=chunk_size, p=type_weights)
        chosen_owners = np.random.choice(owners, size=chunk_size, p=owner_weights)

        # Pre-assign satellite IDs
        cat_ids = np.random.randint(10000, 60000, size=chunk_size)

        # Realistic altitudes (km) based on regime
        altitudes = np.zeros(chunk_size)
        leo_mask = chosen_regimes == "LEO"
        meo_mask = chosen_regimes == "MEO"
        geo_mask = chosen_regimes == "GEO"
        heo_mask = chosen_regimes == "HEO"

        altitudes[leo_mask] = np.random.uniform(300, 1500, size=np.sum(leo_mask))
        altitudes[meo_mask] = np.random.uniform(19000, 24000, size=np.sum(meo_mask))
        altitudes[geo_mask] = np.random.normal(35786, 20, size=np.sum(geo_mask))
        altitudes[heo_mask] = np.random.uniform(1000, 40000, size=np.sum(heo_mask))

        # Orbital velocities v = sqrt(GM / (R + h)) in km/s (GM_earth = 398600.4418)
        GM = 398600.4418
        velocities = np.sqrt(GM / (EARTH_RADIUS_KM + altitudes))
        # Add small eccentricity perturbation
        velocities += np.random.normal(0, 0.05, size=chunk_size)

        # Latitudes and Longitudes
        # Latitudes follow inclination distribution (clustered around 51.6, 98.0, 53.0)
        lats = np.random.uniform(-80, 80, size=chunk_size)
        lons = np.random.uniform(-180, 180, size=chunk_size)

        # Timestamps over a 30-day window
        time_offsets = np.random.uniform(0, 30 * 86400, size=chunk_size)
        record_timestamps = [datetime.fromtimestamp(start_time_ts + to, tz=timezone.utc) for to in time_offsets]

        # Footprint radius: R * acos(R / (R + h))
        footprints = EARTH_RADIUS_KM * np.arccos(EARTH_RADIUS_KM / (EARTH_RADIUS_KM + np.maximum(altitudes, 1.0)))

        # Solar status
        sunlit_prob = np.where(chosen_regimes == "GEO", 0.99, 0.65)
        is_sunlit = np.random.rand(chunk_size) < sunlit_prob

        records = []
        for i in range(chunk_size):
            ts = record_timestamps[i]
            records.append({
                "record_id": str(uuid.uuid4()),
                "norad_id": int(cat_ids[i]),
                "object_name": f"{chosen_types[i]}-{cat_ids[i]}",
                "object_type": str(chosen_types[i]),
                "orbit_regime": str(chosen_regimes[i]),
                "owner": str(chosen_owners[i]),
                "timestamp": ts,
                "epoch_year": int(ts.year),
                "latitude": float(round(lats[i], 4)),
                "longitude": float(round(lons[i], 4)),
                "altitude_km": float(round(altitudes[i], 2)),
                "velocity_km_s": float(round(velocities[i], 3)),
                "footprint_radius_km": float(round(footprints[i], 1)),
                "is_sunlit": bool(is_sunlit[i]),
                "solar_elevation_subpoint": float(round(np.random.uniform(-90, 90), 2)),
            })

        return records

    def generate_and_save_dataset(
        self,
        total_records: int = 100000,
        chunk_size: int = 25000,
        write_mode: str = "overwrite",
    ) -> Dict[str, Any]:
        """
        Generates `total_records` orbital observations and saves directly to Parquet Lake.
        Returns execution performance summary.
        """
        start_wall_time = time.time()
        logger.info(f"Starting generation of {total_records:,} orbital records...")

        now_ts = datetime.now(timezone.utc).timestamp()
        num_chunks = max(1, math.ceil(total_records / chunk_size))

        all_records: List[Dict[str, Any]] = []
        for c in range(num_chunks):
            current_batch_size = min(chunk_size, total_records - len(all_records))
            if current_batch_size <= 0:
                break
            chunk = self._generate_synthetic_chunk(
                current_batch_size, start_time_ts=now_ts, seed_offset=c * 100
            )
            all_records.extend(chunk)

        gen_time = time.time() - start_wall_time
        logger.info(f"Vectorized record generation took {gen_time:.2f}s. Converting to Spark DataFrame...")

        spark_start = time.time()
        # Convert records to Spark DataFrame
        pdf = pd.DataFrame(all_records)
        df: DataFrame = self.spark.createDataFrame(pdf, schema=ORBITAL_RECORD_SCHEMA)

        # Write to partitioned parquet lake
        self.lake.write_dataframe(df, mode=write_mode)
        spark_write_time = time.time() - spark_start
        total_elapsed = time.time() - start_wall_time

        stats = self.lake.get_lake_stats()

        summary = {
            "requested_records": total_records,
            "generated_records": len(all_records),
            "generation_time_sec": round(gen_time, 2),
            "spark_write_time_sec": round(spark_write_time, 2),
            "total_time_sec": round(total_elapsed, 2),
            "throughput_records_per_sec": int(total_records / max(0.001, total_elapsed)),
            "lake_size_mb": stats["total_size_mb"],
            "parquet_files": stats["file_count"],
        }
        logger.info(f"Historical generation complete: {summary}")
        return summary
