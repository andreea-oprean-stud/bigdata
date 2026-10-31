"""
Unit Tests for Apache Spark Big Data Pipeline & Analytics.
"""

import unittest
from src.big_data_pipeline.spark_manager import SparkSessionManager
from src.big_data_pipeline.parquet_lake import ParquetLakeManager
from src.big_data_pipeline.historical_generator import HistoricalOrbitalGenerator
from src.big_data_pipeline.analytics import SparkOrbitalAnalytics


class TestSparkPipeline(unittest.TestCase):
    """Tests PySpark session, data generation, partitioned Parquet lake, and analytics queries."""

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSessionManager.get_spark_session("TestSparkSuite")
        cls.lake = ParquetLakeManager()
        cls.generator = HistoricalOrbitalGenerator()
        # Seed small dataset for tests
        cls.generator.generate_and_save_dataset(total_records=2000, chunk_size=1000)
        cls.analytics = SparkOrbitalAnalytics()

    def test_spark_session_active(self):
        """Tests that SparkSession is active and healthy."""
        self.assertIsNotNone(self.spark)
        self.assertEqual(self.spark.version.split(".")[0], "4" or "3")

    def test_parquet_lake_read(self):
        """Tests reading partitioned Parquet lake."""
        df = self.lake.read_lake()
        self.assertGreaterEqual(df.count(), 2000)

    def test_analytics_regime_distribution(self):
        """Tests Spark SQL regime distribution aggregation."""
        df = self.analytics.get_regime_and_type_distribution()
        self.assertFalse(df.empty)
        self.assertIn("orbit_regime", df.columns)
        self.assertIn("count", df.columns)

    def test_analytics_debris_density(self):
        """Tests Spark SQL debris altitude binning query."""
        df = self.analytics.get_debris_altitude_density(bin_size_km=100)
        self.assertFalse(df.empty)
        self.assertIn("altitude_bin_km", df.columns)
        self.assertIn("debris_count", df.columns)

    def test_custom_sql_execution(self):
        """Tests interactive SQL console engine."""
        res = self.analytics.execute_custom_sql("SELECT COUNT(*) as total FROM orbital_records")
        self.assertTrue(res["success"])
        self.assertEqual(len(res["data"]), 1)


if __name__ == "__main__":
    unittest.main()
