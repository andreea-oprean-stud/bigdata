"""
Unit Tests for Data Ingestion and Quality Validation.
"""

import unittest
import pandas as pd
from src.data_ingestion.data_quality import DataQualityValidator
from src.data_ingestion.sample_data_loader import SAMPLE_TLES, get_sample_satcat_df
from src.data_ingestion.celestrak_client import CelesTrakClient


class TestDataQuality(unittest.TestCase):
    """Tests TLE validation, checksum computation, and SATCAT data cleansing."""

    def test_tle_checksum_valid(self):
        """Tests checksum validation for authentic ISS TLE."""
        iss = SAMPLE_TLES[25544]
        is_valid, msg = DataQualityValidator.validate_tle_pair(iss["line1"], iss["line2"])
        self.assertTrue(is_valid, f"Validation failed: {msg}")

    def test_tle_checksum_corrupted(self):
        """Tests detection of checksum error in corrupted TLE."""
        line1 = "1 25544U 98067A   24089.51680556  .00014321  00000+0  25841-3 0  9990"  # Corrupted checksum
        line2 = "2 25544  51.6416 195.4261 0004561  41.5123 318.6185 15.49815344446362"
        is_valid, msg = DataQualityValidator.validate_tle_pair(line1, line2)
        self.assertFalse(is_valid)
        self.assertIn("checksum error", msg)

    def test_satcat_cleaning_and_regimes(self):
        """Tests SATCAT cleaning and orbital regime classification."""
        df = get_sample_satcat_df()
        self.assertIn("orbit_regime", df.columns)
        self.assertIn("norad_id", df.columns)
        
        # Check ISS is classified as LEO
        iss_row = df[df["norad_id"] == 25544]
        self.assertFalse(iss_row.empty)
        self.assertEqual(iss_row.iloc[0]["orbit_regime"], "LEO")

        # Check GOES 16 is classified as GEO
        goes_row = df[df["norad_id"] == 41866]
        self.assertFalse(goes_row.empty)
        self.assertEqual(goes_row.iloc[0]["orbit_regime"], "GEO")

    def test_celestrak_client_offline_fallback(self):
        """Tests CelesTrak client fallback behavior."""
        client = CelesTrakClient()
        tle = client.get_tle_by_catnr(25544)
        self.assertIsNotNone(tle)
        self.assertIn("line1", tle)
        self.assertIn("line2", tle)


if __name__ == "__main__":
    unittest.main()
