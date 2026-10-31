"""
Configuration Module for Big Data Satellite Pass Prediction Platform.
Centralizes API endpoints, directory paths, default observer locations, and constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PARQUET_LAKE_DIR = DATA_DIR / "parquet_lake"
BENCHMARK_DIR = DATA_DIR / "benchmarks"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR, PARQUET_LAKE_DIR, BENCHMARK_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configurations
N2YO_API_KEY = os.getenv("N2YO_API_KEY", "")
N2YO_BASE_URL = "https://api.n2yo.com/rest/v1/satellite"

CELESTRAK_BASE_URL = "https://celestrak.org/NORAD/elements/gp.php"
CELESTRAK_SATCAT_URL = "https://celestrak.org/pub/satcat.csv"

# Pre-defined Satellite groups on CelesTrak
CELESTRAK_GROUPS = {
    "Space Stations (ISS, Tiangong)": "stations",
    "Brightest Satellites (Visual)": "visual",
    "Active Satellites (Full)": "active",
    "Starlink": "starlink",
    "OneWeb": "oneweb",
    "GPS Constellation": "gps-ops",
    "Galileo Constellation": "galileo",
    "Weather Satellites": "weather",
    "Earth Resources (NOAA, Landsat)": "resource",
    "Disaster Monitoring": "dmc",
    "Amateur Radio": "amateur",
}

# Default Popular Satellites for Quick Tracking
POPULAR_SATELLITES = {
    25544: "ISS (ZARYA)",
    48274: "CSS (TIANHE)",
    20580: "HUBBLE SPACE TELESCOPE",
    25338: "NOAA 15",
    28654: "NOAA 18",
    33591: "NOAA 19",
    43013: "NOAA 20 (JPSS-1)",
    54234: "NOAA 21 (JPSS-2)",
    37849: "SUOMI NPP",
    25994: "TERRA",
    27424: "AQUA",
    44713: "STARLINK-1007",
    44714: "STARLINK-1008",
    41866: "GOES 16",
    43226: "GOES 17",
    39086: "LANDSAT 8",
    49260: "LANDSAT 9",
    24876: "GPS BIIR-2 (PRN 13)",
    28874: "GPS IIR-M-1 (PRN 17)",
    39166: "GALILEO 1 (GSAT0101)",
}

# Pre-defined Observer Locations (Latitude, Longitude, Altitude in meters)
OBSERVER_PRESETS = {
    "Bucharest, Romania": {"lat": 44.4268, "lon": 26.1025, "alt": 75.0},
    "Cluj-Napoca, Romania": {"lat": 46.7712, "lon": 23.6236, "alt": 360.0},
    "Timisoara, Romania": {"lat": 45.7489, "lon": 21.2087, "alt": 90.0},
    "London, United Kingdom": {"lat": 51.5074, "lon": -0.1278, "alt": 25.0},
    "New York, USA": {"lat": 40.7128, "lon": -74.0060, "alt": 10.0},
    "Tokyo, Japan": {"lat": 35.6762, "lon": 139.6503, "alt": 40.0},
    "Sydney, Australia": {"lat": -33.8688, "lon": 151.2093, "alt": 20.0},
    "Berlin, Germany": {"lat": 52.5200, "lon": 13.4050, "alt": 34.0},
    "Paris, France": {"lat": 48.8566, "lon": 2.3522, "alt": 35.0},
    "Cape Town, South Africa": {"lat": -33.9249, "lon": 18.4241, "alt": 20.0},
}

# Spark Configuration Defaults
SPARK_APP_NAME = "SatelliteOrbitalAnalytics"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "2g")
