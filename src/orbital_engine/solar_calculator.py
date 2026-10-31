"""
Solar Illumination & Observer Twilight Calculator.
Determines whether an observer is in daylight, twilight, or night,
and computes whether an orbiting satellite is in Earth's shadow (eclipse) or illuminated by sunlight.
"""

import math
from datetime import datetime, timezone
from typing import Tuple, Dict, Any
import numpy as np

EARTH_RADIUS_KM = 6378.137


class SolarCalculator:
    """Calculates solar positions, eclipse conditions, and atmospheric twilight."""

    @staticmethod
    def get_solar_subpoint(dt: datetime) -> Tuple[float, float]:
        """
        Calculates sub-solar latitude and longitude (deg) for a given UTC datetime.
        Accurate to within ~0.2 degrees.
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        # Day of year
        day_of_year = dt.timetuple().tm_yday
        hour_fraction = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

        # Solar declination approximation (deg)
        declination = -23.44 * math.cos(math.radians((360 / 365.25) * (day_of_year + 10)))

        # Equation of time approximation (minutes)
        b = math.radians((360 / 365.0) * (day_of_year - 81))
        eot = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)

        # Greenwich solar time -> sub-solar longitude
        solar_noon_utc = 12.0 - (eot / 60.0)
        sub_solar_lon = (solar_noon_utc - hour_fraction) * 15.0
        # Normalize longitude between -180 and 180
        sub_solar_lon = (sub_solar_lon + 180) % 360 - 180

        return declination, sub_solar_lon

    @classmethod
    def get_solar_elevation(cls, obs_lat: float, obs_lon: float, dt: datetime) -> float:
        """
        Calculates solar elevation angle in degrees for an observer.
        Positive = Sun above horizon (Daylight)
        0 to -6 deg = Civil Twilight
        -6 to -12 deg = Nautical Twilight
        -12 to -18 deg = Astronomical Twilight
        < -18 deg = Full Astronomical Night
        """
        solar_lat, solar_lon = cls.get_solar_subpoint(dt)

        lat1 = math.radians(obs_lat)
        lat2 = math.radians(solar_lat)
        dlon = math.radians(solar_lon - obs_lon)

        # Great-circle angular distance from sub-solar point (Zenith Angle)
        cos_zenith = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(dlon)
        cos_zenith = max(-1.0, min(1.0, cos_zenith))
        zenith_deg = math.degrees(math.acos(cos_zenith))

        solar_elevation = 90.0 - zenith_deg
        return solar_elevation

    @classmethod
    def get_twilight_category(cls, solar_elev: float) -> str:
        """Returns human-readable lighting condition."""
        if solar_elev > 0:
            return "Daylight"
        elif solar_elev > -6.0:
            return "Civil Twilight"
        elif solar_elev > -12.0:
            return "Nautical Twilight"
        elif solar_elev > -18.0:
            return "Astronomical Twilight"
        else:
            return "Night"

    @classmethod
    def is_satellite_sunlit(
        cls, sat_lat: float, sat_lon: float, sat_alt_km: float, dt: datetime
    ) -> bool:
        """
        Determines whether the satellite is in direct sunlight or in Earth's shadow (Umbra).
        """
        solar_lat, solar_lon = cls.get_solar_subpoint(dt)

        # Vector from Earth center to Sun (unit vector approx)
        sol_lat_rad = math.radians(solar_lat)
        sol_lon_rad = math.radians(solar_lon)
        s_hat = np.array([
            math.cos(sol_lat_rad) * math.cos(sol_lon_rad),
            math.cos(sol_lat_rad) * math.sin(sol_lon_rad),
            math.sin(sol_lat_rad),
        ])

        # Vector from Earth center to Satellite (ECEF-approx)
        r_mag = EARTH_RADIUS_KM + sat_alt_km
        sat_lat_rad = math.radians(sat_lat)
        sat_lon_rad = math.radians(sat_lon)
        r_vec = np.array([
            r_mag * math.cos(sat_lat_rad) * math.cos(sat_lon_rad),
            r_mag * math.cos(sat_lat_rad) * math.sin(sat_lon_rad),
            r_mag * math.sin(sat_lat_rad),
        ])

        # Dot product with sun direction
        proj = np.dot(r_vec, s_hat)

        if proj > 0:
            # Satellite is on the sunward side of the Earth
            return True

        # Satellite is on the dark side of Earth -> Check perpendicular distance to shadow axis
        perp_dist = np.linalg.norm(r_vec - proj * s_hat)
        return bool(perp_dist > EARTH_RADIUS_KM)

    @classmethod
    def evaluate_pass_visibility(
        cls, obs_lat: float, obs_lon: float, sat_lat: float, sat_lon: float, sat_alt_km: float, dt: datetime
    ) -> Dict[str, Any]:
        """
        Evaluates optical pass visibility criteria:
        1. Is observer in twilight or night? (Solar elevation <= -6 deg)
        2. Is satellite in direct sunlight? (Not eclipsed)
        """
        solar_elev = cls.get_solar_elevation(obs_lat, obs_lon, dt)
        sunlit = cls.is_satellite_sunlit(sat_lat, sat_lon, sat_alt_km, dt)
        twilight_cat = cls.get_twilight_category(solar_elev)

        # Visually observable if the sky is sufficiently dark and satellite reflects sunlight
        is_visual = (solar_elev <= -6.0) and sunlit

        return {
            "solar_elevation_deg": round(solar_elev, 2),
            "twilight_category": twilight_cat,
            "satellite_sunlit": sunlit,
            "is_visually_visible": is_visual,
        }
