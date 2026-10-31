"""
Unit Tests for Validation Engine.
"""

import unittest
from src.validation.comparator import OrbitalValidator


class TestValidation(unittest.TestCase):
    """Tests position validation and error residual evaluation."""

    def setUp(self):
        self.validator = OrbitalValidator()

    def test_satellite_position_validation(self):
        """Tests orbital position comparator output."""
        res = self.validator.validate_satellite_positions(
            norad_id=25544,
            obs_lat=44.4268,
            obs_lon=26.1025,
            obs_alt_m=75.0,
            seconds=5,
        )
        self.assertNotIn("error", res)
        self.assertIn("metrics", res)
        self.assertIn("comparison_df", res)
        self.assertTrue(res["metrics"]["validation_passed"])
        self.assertLess(res["metrics"]["mae_latitude_deg"], 0.1)


if __name__ == "__main__":
    unittest.main()
