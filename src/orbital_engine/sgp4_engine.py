"""
SGP4 Orbital Propagation and Coordinate Transformation Engine.
Provides high-accuracy orbit propagation, ground track generation,
and topocentric (Azimuth, Elevation, Slant Range) observer transformations.
"""

import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sgp4.api import Satrec, jday
from sgp4 import omm

EARTH_RADIUS_KM = 6378.137
EARTH_FLATTENING = 1.0 / 298.257223563


class SGP4Engine:
    """Core SGP4 propagation wrapper."""

    def __init__(self, line1: str, line2: str, name: str = "SATELLITE"):
        self.name = name
        self.line1 = line1.strip()
        self.line2 = line2.strip()
        self.satrec = Satrec.twoline2rv(self.line1, self.line2)
        if self.satrec.error != 0:
            raise ValueError(f"SGP4 Satrec initialization error code {self.satrec.error}")

    @staticmethod
    def _datetime_to_jday(dt: datetime) -> Tuple[float, float]:
        """Converts UTC datetime to Julian Day and fraction."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)

    @staticmethod
    def _gmst_radians(jd_ut1: float) -> float:
        """Computes Greenwich Mean Sidereal Time in radians for given Julian Day."""
        t_ut1 = (jd_ut1 - 2451545.0) / 36525.0
        # IAU 1982 GMST polynomial
        gmst_sec = (
            24110.54841
            + 8640184.812866 * t_ut1
            + 0.093104 * (t_ut1 ** 2)
            - 0.0000062 * (t_ut1 ** 3)
        )
        gmst_rad = math.radians((gmst_sec % 86400.0) / 240.0)
        return gmst_rad

    def propagate_teme(self, dt: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagates satellite state at UTC datetime.
        Returns (position_km, velocity_km_s) in TEME frame.
        """
        jd, fr = self._datetime_to_jday(dt)
        err, r, v = self.satrec.sgp4(jd, fr)
        if err != 0:
            raise RuntimeError(f"SGP4 propagation error code {err}")
        return np.array(r), np.array(v)

    def get_geodetic_position(self, dt: datetime) -> Dict[str, Any]:
        """
        Calculates sub-satellite geodetic latitude (deg), longitude (deg),
        altitude (km), velocity magnitude (km/s).
        """
        jd, fr = self._datetime_to_jday(dt)
        err, r, v = self.satrec.sgp4(jd, fr)
        if err != 0:
            return {"error": f"SGP4 error {err}"}

        r = np.array(r)
        v = np.array(v)
        vel_mag = float(np.linalg.norm(v))

        # Rotate from TEME to ECEF using GMST
        gmst = self._gmst_radians(jd + fr)
        cos_g, sin_g = math.cos(gmst), math.sin(gmst)
        x_ecef = cos_g * r[0] + sin_g * r[1]
        y_ecef = -sin_g * r[0] + cos_g * r[1]
        z_ecef = r[2]

        # Calculate Geodetic Latitude and Longitude using Bowring's method
        lon_rad = math.atan2(y_ecef, x_ecef)
        lon_deg = math.degrees(lon_rad)

        p = math.hypot(x_ecef, y_ecef)
        e2 = 2 * EARTH_FLATTENING - EARTH_FLATTENING ** 2
        lat_rad = math.atan2(z_ecef, p * (1 - e2))

        # Iterate 3 times for sub-millimeter convergence
        for _ in range(4):
            n = EARTH_RADIUS_KM / math.sqrt(1 - e2 * (math.sin(lat_rad) ** 2))
            alt_km = p / math.cos(lat_rad) - n
            lat_rad = math.atan2(z_ecef, p * (1 - e2 * (n / (n + alt_km))))

        lat_deg = math.degrees(lat_rad)

        # Footprint radius: angular distance theta = acos(R / (R + alt))
        footprint_rad_km = EARTH_RADIUS_KM * math.acos(EARTH_RADIUS_KM / (EARTH_RADIUS_KM + max(1.0, alt_km)))

        return {
            "datetime_utc": dt.isoformat(),
            "latitude": round(lat_deg, 4),
            "longitude": round(lon_deg, 4),
            "altitude_km": round(alt_km, 2),
            "velocity_km_s": round(vel_mag, 3),
            "footprint_radius_km": round(footprint_rad_km, 1),
            "r_ecef": (x_ecef, y_ecef, z_ecef),
        }

    def get_topocentric_coordinates(
        self,
        dt: datetime,
        obs_lat: float,
        obs_lon: float,
        obs_alt_m: float = 0.0,
    ) -> Dict[str, float]:
        """
        Computes Azimuth (0-360 deg), Elevation (-90 to +90 deg), and Slant Range (km)
        as seen from observer location at UTC datetime.
        """
        pos_info = self.get_geodetic_position(dt)
        if "error" in pos_info:
            return {"azimuth_deg": 0.0, "elevation_deg": -90.0, "range_km": 0.0}

        x_sat, y_sat, z_sat = pos_info["r_ecef"]

        # Observer ECEF coordinates
        obs_lat_rad = math.radians(obs_lat)
        obs_lon_rad = math.radians(obs_lon)
        obs_alt_km = obs_alt_m / 1000.0

        e2 = 2 * EARTH_FLATTENING - EARTH_FLATTENING ** 2
        n_obs = EARTH_RADIUS_KM / math.sqrt(1 - e2 * (math.sin(obs_lat_rad) ** 2))
        x_obs = (n_obs + obs_alt_km) * math.cos(obs_lat_rad) * math.cos(obs_lon_rad)
        y_obs = (n_obs + obs_alt_km) * math.cos(obs_lat_rad) * math.sin(obs_lon_rad)
        z_obs = (n_obs * (1 - e2) + obs_alt_km) * math.sin(obs_lat_rad)

        # Topocentric vector (Range vector in ECEF)
        dx = x_sat - x_obs
        dy = y_sat - y_obs
        dz = z_sat - z_obs
        slant_range = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # Convert to Topocentric Horizon system (SEZ or ENU)
        # ENU (East, North, Up):
        sin_lat, cos_lat = math.sin(obs_lat_rad), math.cos(obs_lat_rad)
        sin_lon, cos_lon = math.sin(obs_lon_rad), math.cos(obs_lon_rad)

        e = -sin_lon * dx + cos_lon * dy
        n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

        # Azimuth from North (clockwise): atan2(East, North)
        az_rad = math.atan2(e, n)
        az_deg = (math.degrees(az_rad) + 360.0) % 360.0

        # Elevation angle: asin(Up / Slant Range)
        el_rad = math.asin(max(-1.0, min(1.0, u / max(1e-6, slant_range))))
        el_deg = math.degrees(el_rad)

        return {
            "azimuth_deg": round(az_deg, 2),
            "elevation_deg": round(el_deg, 2),
            "range_km": round(slant_range, 2),
            "sat_lat": pos_info["latitude"],
            "sat_lon": pos_info["longitude"],
            "sat_alt_km": pos_info["altitude_km"],
            "sat_vel_km_s": pos_info["velocity_km_s"],
        }

    def compute_ground_track(
        self,
        start_dt: datetime,
        duration_minutes: int = 100,
        step_seconds: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Computes orbital path coordinates for map plotting over a time window.
        """
        track = []
        n_steps = int((duration_minutes * 60) / step_seconds)
        for i in range(n_steps):
            t = start_dt + timedelta(seconds=i * step_seconds)
            pos = self.get_geodetic_position(t)
            if "error" not in pos:
                track.append({
                    "time": t.strftime("%H:%M:%S"),
                    "dt": t,
                    "lat": pos["latitude"],
                    "lon": pos["longitude"],
                    "alt_km": pos["altitude_km"],
                })
        return track
