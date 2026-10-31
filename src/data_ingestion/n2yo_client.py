"""
N2YO REST API Client.
Provides methods to query positions, visual passes, radio passes, and sky search.
Gracefully handles API key presence, rate limits, and fallback mocks.
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import requests

from config import N2YO_BASE_URL, N2YO_API_KEY, RAW_DIR

logger = logging.getLogger(__name__)


class N2YOClient:
    """Interface to N2YO.com REST API for satellite tracking and pass predictions."""

    def __init__(self, api_key: Optional[str] = None, cache_dir: Path = RAW_DIR):
        self.api_key = api_key or N2YO_API_KEY
        self.base_url = N2YO_BASE_URL
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_call_time = 0.0
        self.min_interval = 0.5  # Rate-limit safety: 2 calls/sec

    def _rate_limit(self):
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()

    def is_configured(self) -> bool:
        """Checks if a non-placeholder API key is configured."""
        return bool(self.api_key and self.api_key not in ["DEMO_KEY_OR_YOUR_KEY", "YOUR_KEY_HERE", ""])

    def get_tle(self, norad_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves TLE for satellite via N2YO API."""
        if not self.is_configured():
            return None

        url = f"{self.base_url}/tle/{norad_id}&apiKey={self.api_key}"
        self._rate_limit()
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if "tle" in data:
                    lines = data["tle"].split("\r\n")
                    if len(lines) >= 2:
                        return {
                            "norad_id": norad_id,
                            "name": data.get("info", {}).get("satname", f"SAT-{norad_id}"),
                            "line1": lines[0],
                            "line2": lines[1],
                            "source": "n2yo",
                        }
        except Exception as e:
            logger.error(f"N2YO TLE request error: {e}")
        return None

    def get_positions(
        self,
        norad_id: int,
        observer_lat: float,
        observer_lon: float,
        observer_alt: float = 0.0,
        seconds: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves real-time predicted positions for next N seconds.
        """
        if not self.is_configured():
            return None

        url = (
            f"{self.base_url}/positions/{norad_id}/{observer_lat}/{observer_lon}/"
            f"{observer_alt}/{seconds}&apiKey={self.api_key}"
        )
        self._rate_limit()
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"N2YO positions request error: {e}")
        return None

    def get_visual_passes(
        self,
        norad_id: int,
        observer_lat: float,
        observer_lon: float,
        observer_alt: float = 0.0,
        days: int = 5,
        min_visibility: int = 60,
    ) -> Optional[Dict[str, Any]]:
        """
        Predicts visual passes above horizon for observer.
        """
        if not self.is_configured():
            return None

        url = (
            f"{self.base_url}/visualpasses/{norad_id}/{observer_lat}/{observer_lon}/"
            f"{observer_alt}/{days}/{min_visibility}&apiKey={self.api_key}"
        )
        self._rate_limit()
        try:
            resp = requests.get(url, timeout=12)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"N2YO visual passes request error: {e}")
        return None

    def get_radiopasses(
        self,
        norad_id: int,
        observer_lat: float,
        observer_lon: float,
        observer_alt: float = 0.0,
        days: int = 5,
        min_elevation: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """
        Predicts geometric radio passes above a minimum elevation.
        """
        if not self.is_configured():
            return None

        url = (
            f"{self.base_url}/radiopasses/{norad_id}/{observer_lat}/{observer_lon}/"
            f"{observer_alt}/{days}/{min_elevation}&apiKey={self.api_key}"
        )
        self._rate_limit()
        try:
            resp = requests.get(url, timeout=12)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"N2YO radiopasses request error: {e}")
        return None
