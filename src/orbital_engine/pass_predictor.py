"""
Satellite Pass Prediction Engine.
Computes upcoming satellite passes over an observer location,
with high-precision Acquisition of Signal (AOS), Time of Closest Approach (TCA),
Loss of Signal (LOS), compass trajectories, and optical visibility determination.
"""

import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional

from src.orbital_engine.sgp4_engine import SGP4Engine
from src.orbital_engine.solar_calculator import SolarCalculator


def azimuth_to_compass(az_deg: float) -> str:
    """Converts azimuth degrees (0-360) to 16-point compass direction."""
    directions = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    ]
    idx = int((az_deg + 11.25) / 22.5) % 16
    return directions[idx]


class PassPredictor:
    """Predicts upcoming satellite passes for a ground observer."""

    def __init__(self, engine: SGP4Engine):
        self.engine = engine

    def predict_passes(
        self,
        obs_lat: float,
        obs_lon: float,
        obs_alt_m: float = 0.0,
        start_dt: Optional[datetime] = None,
        duration_days: float = 3.0,
        min_elevation_deg: float = 10.0,
        coarse_step_sec: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Calculates all passes above `min_elevation_deg` within `duration_days` from `start_dt`.
        """
        if start_dt is None:
            start_dt = datetime.now(timezone.utc)
        elif start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)

        total_seconds = int(duration_days * 86400)
        passes: List[Dict[str, Any]] = []

        in_pass = False
        current_pass_points: List[Dict[str, Any]] = []
        pass_index = 1

        t = start_dt
        end_dt = start_dt + timedelta(seconds=total_seconds)

        while t <= end_dt:
            topo = self.engine.get_topocentric_coordinates(t, obs_lat, obs_lon, obs_alt_m)
            el = topo["elevation_deg"]

            if el >= 0.0:
                # Satellite is above physical horizon
                in_pass = True
                sol_info = SolarCalculator.evaluate_pass_visibility(
                    obs_lat, obs_lon, topo["sat_lat"], topo["sat_lon"], topo["sat_alt_km"], t
                )
                current_pass_points.append({
                    "dt": t,
                    "elevation": el,
                    "azimuth": topo["azimuth_deg"],
                    "range_km": topo["range_km"],
                    "sat_lat": topo["sat_lat"],
                    "sat_lon": topo["sat_lon"],
                    "sat_alt_km": topo["sat_alt_km"],
                    **sol_info,
                })
            else:
                if in_pass and len(current_pass_points) > 1:
                    # Pass completed -> process and record if max elevation >= min_elevation_deg
                    max_pt = max(current_pass_points, key=lambda p: p["elevation"])
                    if max_pt["elevation"] >= min_elevation_deg:
                        pass_dict = self._format_pass_record(
                            pass_index, current_pass_points, obs_lat, obs_lon, obs_alt_m
                        )
                        passes.append(pass_dict)
                        pass_index += 1

                    in_pass = False
                    current_pass_points = []

            # Step forward
            t += timedelta(seconds=coarse_step_sec)

        # Handle edge case where last pass extends beyond end_dt
        if in_pass and len(current_pass_points) > 1:
            max_pt = max(current_pass_points, key=lambda p: p["elevation"])
            if max_pt["elevation"] >= min_elevation_deg:
                pass_dict = self._format_pass_record(
                    pass_index, current_pass_points, obs_lat, obs_lon, obs_alt_m
                )
                passes.append(pass_dict)

        return passes

    def _format_pass_record(
        self,
        pass_id: int,
        points: List[Dict[str, Any]],
        obs_lat: float,
        obs_lon: float,
        obs_alt_m: float,
    ) -> Dict[str, Any]:
        """Refines timestamps and generates trajectory representation for a pass."""
        start_pt = points[0]
        end_pt = points[-1]
        max_pt = max(points, key=lambda p: p["elevation"])

        aos_dt: datetime = start_pt["dt"]
        tca_dt: datetime = max_pt["dt"]
        los_dt: datetime = end_pt["dt"]

        # Refine TCA using 5-second sampling around maximum
        fine_max_el = max_pt["elevation"]
        fine_tca_dt = tca_dt
        for offset_sec in range(-60, 65, 5):
            test_dt = tca_dt + timedelta(seconds=offset_sec)
            test_topo = self.engine.get_topocentric_coordinates(test_dt, obs_lat, obs_lon, obs_alt_m)
            if test_topo["elevation_deg"] > fine_max_el:
                fine_max_el = test_topo["elevation_deg"]
                fine_tca_dt = test_dt

        duration_sec = int((los_dt - aos_dt).total_seconds())

        # Generate smooth trajectory every 15s for sky charts
        trajectory_samples = []
        cur_t = aos_dt
        while cur_t <= los_dt:
            tp = self.engine.get_topocentric_coordinates(cur_t, obs_lat, obs_lon, obs_alt_m)
            vis = SolarCalculator.evaluate_pass_visibility(
                obs_lat, obs_lon, tp["sat_lat"], tp["sat_lon"], tp["sat_alt_km"], cur_t
            )
            trajectory_samples.append({
                "time_utc": cur_t.strftime("%H:%M:%S"),
                "dt_iso": cur_t.isoformat(),
                "azimuth": tp["azimuth_deg"],
                "elevation": max(0.0, tp["elevation_deg"]),
                "range_km": tp["range_km"],
                "is_visual": vis["is_visually_visible"],
            })
            cur_t += timedelta(seconds=15)

        # Check if pass has any visually visible segment
        is_visual_pass = any(p["is_visually_visible"] for p in points)

        return {
            "pass_id": pass_id,
            "satellite_name": self.engine.name,
            "aos_time_utc": aos_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "tca_time_utc": fine_tca_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "los_time_utc": los_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_minutes": round(duration_sec / 60.0, 1),
            "duration_seconds": duration_sec,
            "max_elevation_deg": round(fine_max_el, 1),
            "start_azimuth_deg": round(start_pt["azimuth"], 1),
            "start_compass": azimuth_to_compass(start_pt["azimuth"]),
            "max_azimuth_deg": round(max_pt["azimuth"], 1),
            "max_compass": azimuth_to_compass(max_pt["azimuth"]),
            "end_azimuth_deg": round(end_pt["azimuth"], 1),
            "end_compass": azimuth_to_compass(end_pt["azimuth"]),
            "is_visual": is_visual_pass,
            "visibility_type": "Visual Pass (Visible to Naked Eye/Binoculars)" if is_visual_pass else "Radio / Daylight Pass",
            "twilight_condition": max_pt.get("twilight_category", "Daylight"),
            "solar_elevation_at_max": max_pt.get("solar_elevation_deg", 0.0),
            "trajectory": trajectory_samples,
        }
