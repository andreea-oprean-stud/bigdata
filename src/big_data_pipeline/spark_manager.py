"""
Apache Spark Session Manager.
Configures and initializes a high-performance PySpark Session for local Big Data analytics.
"""

import os
import sys
import logging
from typing import Optional
from pyspark.sql import SparkSession
from config import SPARK_APP_NAME, SPARK_MASTER, SPARK_DRIVER_MEMORY

logger = logging.getLogger(__name__)

# Ensure python executable matches spark worker python
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


class SparkSessionManager:
    """Singleton SparkSession manager."""

    _instance: Optional[SparkSession] = None

    @classmethod
    def get_spark_session(cls, app_name: str = SPARK_APP_NAME) -> SparkSession:
        """Returns or creates active SparkSession."""
        if cls._instance is None or cls._instance._jsc is None:
            logger.info("Initializing Apache Spark Session...")
            builder = (
                SparkSession.builder.appName(app_name)
                .master(SPARK_MASTER)
                .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
                .config("spark.sql.shuffle.partitions", "4")
                .config("spark.default.parallelism", "4")
                .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                .config("spark.ui.enabled", "false")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .config("spark.driver.host", "127.0.0.1")
            )
            cls._instance = builder.getOrCreate()
            # Set log level to WARN to reduce clutter
            cls._instance.sparkContext.setLogLevel("WARN")
            logger.info("Apache Spark Session successfully initialized.")
        return cls._instance

    @classmethod
    def stop_session(cls):
        """Stops SparkSession if active."""
        if cls._instance is not None:
            try:
                cls._instance.stop()
            except Exception:
                pass
            cls._instance = None
