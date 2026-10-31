"""
Apache Spark SQL Analytics Engine.
Performs scalable distributed analytics on partitioned Parquet orbital datasets.
"""

import time
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from src.big_data_pipeline.spark_manager import SparkSessionManager
from src.big_data_pipeline.parquet_lake import ParquetLakeManager

logger = logging.getLogger(__name__)


class SparkOrbitalAnalytics:
    """Analytical queries and aggregations over the orbital Parquet data lake."""

    def __init__(self):
        self.spark = SparkSessionManager.get_spark_session()
        self.lake = ParquetLakeManager()
        self._ensure_table_registered()

    def _ensure_table_registered(self):
        """Registers Parquet Lake as a temporary SQL view."""
        try:
            df = self.lake.read_lake()
            df.createOrReplaceTempView("orbital_records")
        except Exception as e:
            logger.warning(f"Could not register temporary view orbital_records: {e}")

    def refresh_view(self):
        """Refreshes the temporary view after new data is written."""
        self._ensure_table_registered()

    def get_regime_and_type_distribution(self) -> pd.DataFrame:
        """
        Computes object counts and percentages across orbital regimes and object types.
        """
        self.refresh_view()
        query = """
            SELECT 
                orbit_regime,
                object_type,
                COUNT(*) as count,
                AVG(altitude_km) as avg_altitude_km,
                AVG(velocity_km_s) as avg_velocity_km_s
            FROM orbital_records
            GROUP BY orbit_regime, object_type
            ORDER BY count DESC
        """
        return self.spark.sql(query).toPandas()

    def get_debris_altitude_density(self, bin_size_km: int = 50) -> pd.DataFrame:
        """
        Calculates spatial density of space debris vs active payloads across altitude shells.
        """
        self.refresh_view()
        query = f"""
            SELECT 
                CAST(FLOOR(altitude_km / {bin_size_km}) * {bin_size_km} AS INT) as altitude_bin_km,
                SUM(CASE WHEN object_type = 'DEBRIS' THEN 1 ELSE 0 END) as debris_count,
                SUM(CASE WHEN object_type = 'PAYLOAD' THEN 1 ELSE 0 END) as payload_count,
                SUM(CASE WHEN object_type = 'ROCKET BODY' THEN 1 ELSE 0 END) as rocket_body_count,
                COUNT(*) as total_objects
            FROM orbital_records
            WHERE altitude_km <= 2000
            GROUP BY CAST(FLOOR(altitude_km / {bin_size_km}) * {bin_size_km} AS INT)
            ORDER BY altitude_bin_km ASC
        """
        return self.spark.sql(query).toPandas()

    def get_fleet_by_owner(self) -> pd.DataFrame:
        """
        Aggregates space assets and debris by owning nation/agency.
        """
        self.refresh_view()
        query = """
            SELECT 
                owner,
                COUNT(*) as total_records,
                SUM(CASE WHEN object_type = 'PAYLOAD' THEN 1 ELSE 0 END) as payloads,
                SUM(CASE WHEN object_type = 'DEBRIS' THEN 1 ELSE 0 END) as debris,
                SUM(CASE WHEN object_type = 'ROCKET BODY' THEN 1 ELSE 0 END) as rocket_bodies,
                ROUND(AVG(velocity_km_s), 2) as avg_speed_km_s
            FROM orbital_records
            GROUP BY owner
            ORDER BY total_records DESC
        """
        return self.spark.sql(query).toPandas()

    def get_velocity_vs_altitude_profile(self, sample_size: int = 5000) -> pd.DataFrame:
        """
        Retrieves a sample of altitude vs velocity records to validate Keplerian mechanics.
        """
        self.refresh_view()
        query = f"""
            SELECT 
                norad_id,
                object_name,
                orbit_regime,
                object_type,
                altitude_km,
                velocity_km_s,
                is_sunlit
            FROM orbital_records
            ORDER BY RAND(42)
            LIMIT {sample_size}
        """
        return self.spark.sql(query).toPandas()

    def get_sunlit_ratio_by_regime(self) -> pd.DataFrame:
        """
        Calculates percentage of time objects spend illuminated by sunlight per regime.
        """
        self.refresh_view()
        query = """
            SELECT 
                orbit_regime,
                COUNT(*) as total_samples,
                SUM(CASE WHEN is_sunlit THEN 1 ELSE 0 END) as sunlit_samples,
                ROUND(SUM(CASE WHEN is_sunlit THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) as sunlit_pct
            FROM orbital_records
            GROUP BY orbit_regime
            ORDER BY sunlit_pct DESC
        """
        return self.spark.sql(query).toPandas()

    def execute_custom_sql(self, sql_query: str) -> Dict[str, Any]:
        """
        Executes arbitrary Spark SQL query for interactive data lake exploration.
        """
        self.refresh_view()
        t0 = time.time()
        try:
            df = self.spark.sql(sql_query)
            pdf = df.limit(500).toPandas()
            elapsed = time.time() - t0
            return {
                "success": True,
                "data": pdf,
                "rows_returned": len(pdf),
                "execution_time_sec": round(elapsed, 4),
                "error": None,
            }
        except Exception as e:
            elapsed = time.time() - t0
            return {
                "success": False,
                "data": pd.DataFrame(),
                "rows_returned": 0,
                "execution_time_sec": round(elapsed, 4),
                "error": str(e),
            }
