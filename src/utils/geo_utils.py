"""
Geographical and Coordinate Utilities.
Provides distance calculations, observer coordinate handling, and formatting helpers.
"""

import math
from typing import Tuple, Dict, Any


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two points on Earth in km."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return r * c


def format_lat_lon(lat: float, lon: float) -> str:
    """Formats latitude and longitude into human-readable string (e.g. 44.42° N, 26.10° E)."""
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}° {lat_dir}, {abs(lon):.2f}° {lon_dir}"
