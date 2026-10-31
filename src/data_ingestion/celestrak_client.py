"""
CelesTrak REST and Data Ingestion Client.
Fetches general perturbations (GP) orbital data, TLEs, and full SATCAT catalog.
Includes smart caching and offline fallback.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import requests
import pandas as pd

from config import CELESTRAK_BASE_URL, CELESTRAK_SATCAT_URL, RAW_DIR, CELESTRAK_GROUPS
from src.data_ingestion.data_quality import DataQualityValidator
from src.data_ingestion.sample_data_loader import SAMPLE_TLES, get_sample_satcat_df

logger = logging.getLogger(__name__)


class CelesTrakClient:
    """Client for retrieving TLEs and SATCAT metadata from CelesTrak."""

    def __init__(self, cache_dir: Path = RAW_DIR, cache_ttl_seconds: int = 86400):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl_seconds

    def _is_cache_valid(self, cache_file: Path) -> bool:
        if not cache_file.exists():
            return False
        age = time.time() - cache_file.stat().st_mtime
        return age < self.cache_ttl

    def get_tle_by_catnr(self, norad_id: int) -> Optional[Dict[str, str]]:
        """
        Retrieves TLE for a given NORAD catalog number.
        Returns dict with keys: 'name', 'line1', 'line2', 'norad_id'
        """
        cache_file = self.cache_dir / f"tle_{norad_id}.json"

        # Check local cache
        if self._is_cache_valid(cache_file):
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading cache for {norad_id}: {e}")

        # Fetch from CelesTrak REST API
        params = {"CATNR": norad_id, "FORMAT": "TLE"}
        try:
            resp = requests.get(CELESTRAK_BASE_URL, params=params, timeout=10)
            if resp.status_code == 200 and resp.text.strip():
                lines = [line.strip() for line in resp.text.strip().splitlines() if line.strip()]
                if len(lines) >= 3:
                    name = lines[0]
                    line1 = lines[1]
                    line2 = lines[2]
                elif len(lines) == 2:
                    name = f"OBJECT {norad_id}"
                    line1 = lines[0]
                    line2 = lines[1]
                else:
                    raise ValueError(f"Incomplete TLE response: {lines}")

                valid, msg = DataQualityValidator.validate_tle_pair(line1, line2)
                if not valid:
                    logger.warning(f"Downloaded TLE validation warning for {norad_id}: {msg}")

                tle_data = {
                    "norad_id": norad_id,
                    "name": name,
                    "line1": line1,
                    "line2": line2,
                    "timestamp": time.time(),
                }

                # Save cache
                with open(cache_file, "w") as f:
                    json.dump(tle_data, f, indent=2)

                return tle_data
        except Exception as e:
            logger.warning(f"CelesTrak online fetch failed for {norad_id}: {e}")

        # Fallback to sample TLEs if available
        if norad_id in SAMPLE_TLES:
            data = SAMPLE_TLES[norad_id].copy()
            data["norad_id"] = norad_id
            data["timestamp"] = time.time()
            return data

        # Check stale cache as last resort
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass

        return None

    def get_group_tles(self, group_key: str = "stations") -> List[Dict[str, Any]]:
        """
        Retrieves all TLEs for a given group (e.g., 'stations', 'visual', 'active', 'starlink').
        """
        cache_file = self.cache_dir / f"group_{group_key}.json"

        if self._is_cache_valid(cache_file):
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass

        params = {"GROUP": group_key, "FORMAT": "TLE"}
        try:
            resp = requests.get(CELESTRAK_BASE_URL, params=params, timeout=15)
            if resp.status_code == 200 and resp.text.strip():
                lines = [line.strip() for line in resp.text.strip().splitlines() if line.strip()]
                tles = []
                # Process in 3-line chunks (Name, Line1, Line2)
                i = 0
                while i < len(lines):
                    if i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
                        name = lines[i]
                        l1 = lines[i + 1]
                        l2 = lines[i + 2]
                        try:
                            nid = int(l1[2:7].strip())
                        except ValueError:
                            nid = 0
                        tles.append({"norad_id": nid, "name": name, "line1": l1, "line2": l2})
                        i += 3
                    elif i + 1 < len(lines) and lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
                        l1 = lines[i]
                        l2 = lines[i + 1]
                        try:
                            nid = int(l1[2:7].strip())
                        except ValueError:
                            nid = 0
                        tles.append({"norad_id": nid, "name": f"OBJECT {nid}", "line1": l1, "line2": l2})
                        i += 2
                    else:
                        i += 1

                if tles:
                    with open(cache_file, "w") as f:
                        json.dump(tles, f, indent=2)
                    return tles
        except Exception as e:
            logger.warning(f"Could not fetch group {group_key} online: {e}")

        # Fallback to sample data matching group
        return [{"norad_id": k, **v} for k, v in SAMPLE_TLES.items()]

    def get_satcat_catalog(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Downloads or loads cached SATCAT (Satellite Catalog) metadata.
        Returns cleaned pandas DataFrame.
        """
        cache_file = self.cache_dir / "satcat.csv"

        if not force_refresh and self._is_cache_valid(cache_file):
            try:
                df = pd.read_csv(cache_file)
                return DataQualityValidator.clean_satcat_data(df)
            except Exception as e:
                logger.warning(f"Failed to read cached SATCAT: {e}")

        try:
            logger.info("Downloading latest SATCAT from CelesTrak...")
            resp = requests.get(CELESTRAK_SATCAT_URL, timeout=20)
            if resp.status_code == 200 and resp.text:
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                df = pd.read_csv(cache_file)
                return DataQualityValidator.clean_satcat_data(df)
        except Exception as e:
            logger.warning(f"Failed to download SATCAT: {e}. Using fallback dataset.")

        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file)
                return DataQualityValidator.clean_satcat_data(df)
            except Exception:
                pass

        # Return rich offline sample dataframe
        return get_sample_satcat_df()
