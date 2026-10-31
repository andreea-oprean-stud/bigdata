"""
Unit Tests for Orbital Mechanics Engine and Pass Predictor.
"""

import unittest
from datetime import datetime, timezone, timedelta
from src.data_ingestion.sample_data_loader import SAMPLE_TLES
from src.orbital_engine.sgp4_engine import SGP4Engine
from src.orbital_engine.pass_predictor import PassPredictor
from src.orbital_engine.solar_calculator import SolarCalculator


class TestOrbitalEngine(unittest.TestCase):
    """Tests SGP4 propagation accuracy, topocentric coordinate transforms, and pass predictions."""

    def setUp(self):
        iss_tle = SAMPLE_TLES[25544]
        self.engine = SGP4Engine(iss_tle["line1"], iss_tle["line2"], name="ISS")
        self.dt = datetime(2024, 3, 29, 12, 0, 0, tzinfo=timezone.utc)

    def test_geodetic_position(self):
        """Tests geodetic coordinates calculation."""
        pos = self.engine.get_geodetic_position(self.dt)
        self.assertNotIn("error", pos)
        self.assertTrue(-90.0 <= pos["latitude"] <= 90.0)
        self.assertTrue(-180.0 <= pos["longitude"] <= 180.0)
        self.assertTrue(350.0 <= pos["altitude_km"] <= 450.0)
        self.assertTrue(7.0 <= pos["velocity_km_s"] <= 8.0)

    def test_topocentric_coordinates(self):
        """Tests Azimuth, Elevation, Range calculations."""
        topo = self.engine.get_topocentric_coordinates(
            self.dt, obs_lat=44.4268, obs_lon=26.1025, obs_alt_m=75.0
        )
        self.assertTrue(0.0 <= topo["azimuth_deg"] <= 360.0)
        self.assertTrue(-90.0 <= topo["elevation_deg"] <= 90.0)
        self.assertTrue(topo["range_km"] > 0)

    def test_solar_calculator(self):
        """Tests solar elevation and sub-solar calculation."""
        dec, sub_lon = SolarCalculator.get_solar_subpoint(self.dt)
        self.assertTrue(-25.0 <= dec <= 25.0)
        self.assertTrue(-180.0 <= sub_lon <= 180.0)

        # Solar elevation at noon near sub-solar point should be high
        elev = SolarCalculator.get_solar_elevation(dec, sub_lon, self.dt)
        self.assertTrue(elev > 80.0)

    def test_pass_predictor(self):
        """Tests upcoming pass prediction for Bucharest."""
        predictor = PassPredictor(self.engine)
        passes = predictor.predict_passes(
            obs_lat=44.4268,
            obs_lon=26.1025,
            obs_alt_m=75.0,
            start_dt=self.dt,
            duration_days=3.0,
            min_elevation_deg=10.0,
        )
        self.assertIsInstance(passes, list)
        self.assertTrue(len(passes) > 0, "Should find at least 1 pass in 3 days for ISS")

        p0 = passes[0]
        self.assertIn("aos_time_utc", p0)
        self.assertIn("max_elevation_deg", p0)
        self.assertTrue(p0["max_elevation_deg"] >= 10.0)


if __name__ == "__main__":
    unittest.main()
