"""
Data Quality and Validation Module.
Checks TLE checksums, schema completeness, coordinate validity, and cleans records.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd


class DataQualityValidator:
    """Validates orbital data, TLEs, and catalog schemas."""

    @staticmethod
    def calculate_tle_checksum(line: str) -> int:
        """
        Calculates TLE standard checksum.
        Letters, blanks, periods, and plus signs count as 0.
        Minus signs count as 1.
        Digits 0-9 count as their numeric value.
        Sum modulo 10 gives the checksum.
        """
        if len(line) < 68:
            return -1
        
        # Calculate over characters 0..67 (68 chars total)
        checksum = 0
        for ch in line[:68]:
            if ch.isdigit():
                checksum += int(ch)
            elif ch == "-":
                checksum += 1
        return checksum % 10

    @classmethod
    def validate_tle_pair(cls, line1: str, line2: str) -> Tuple[bool, str]:
        """Validates a Two-Line Element pair structure and checksums."""
        line1 = line1.strip()
        line2 = line2.strip()

        if len(line1) != 69:
            return False, f"Line 1 length is {len(line1)}, expected 69"
        if len(line2) != 69:
            return False, f"Line 2 length is {len(line2)}, expected 69"

        if not line1.startswith("1 "):
            return False, "Line 1 does not start with '1 '"
        if not line2.startswith("2 "):
            return False, "Line 2 does not start with '2 '"

        # Check line numbers and catalog ID consistency
        try:
            norad1 = int(line1[2:7].strip())
            norad2 = int(line2[2:7].strip())
            if norad1 != norad2:
                return False, f"NORAD ID mismatch: Line 1 ({norad1}) != Line 2 ({norad2})"
        except ValueError:
            return False, "Non-numeric NORAD catalog ID in TLE"

        # Checksum checks
        expected_chk1 = int(line1[68])
        calc_chk1 = cls.calculate_tle_checksum(line1)
        if calc_chk1 != expected_chk1:
            return False, f"Line 1 checksum error: computed {calc_chk1}, expected {expected_chk1}"

        expected_chk2 = int(line2[68])
        calc_chk2 = cls.calculate_tle_checksum(line2)
        if calc_chk2 != expected_chk2:
            return False, f"Line 2 checksum error: computed {calc_chk2}, expected {expected_chk2}"

        return True, "Valid TLE"

    @classmethod
    def validate_satcat_dataframe(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates SATCAT DataFrame quality metrics:
        Missing columns, null counts, duplicate NORAD IDs, and valid orbital periods.
        """
        report = {
            "total_rows": len(df),
            "missing_columns": [],
            "null_counts": {},
            "duplicate_norad_ids": 0,
            "invalid_periods": 0,
            "is_valid": True,
            "quality_score": 100.0,
        }

        required_cols = ["NORAD_CAT_ID", "OBJECT_NAME"]
        for col in required_cols:
            if col not in df.columns:
                report["missing_columns"].append(col)
                report["is_valid"] = False

        if "NORAD_CAT_ID" in df.columns:
            dups = df["NORAD_CAT_ID"].duplicated().sum()
            report["duplicate_norad_ids"] = int(dups)
            nulls = df["NORAD_CAT_ID"].isnull().sum()
            report["null_counts"]["NORAD_CAT_ID"] = int(nulls)
            if dups > 0 or nulls > 0:
                report["quality_score"] -= 10.0

        if "PERIOD" in df.columns:
            # Periods should be positive numbers, normally 80 to 2000 minutes for Earth orbits
            invalid_p = df[df["PERIOD"].notnull() & (df["PERIOD"] <= 0)].shape[0]
            report["invalid_periods"] = int(invalid_p)

        # Check completeness of primary attributes
        for col in ["OBJECT_TYPE", "OPS_STATUS_CODE", "LAUNCH_DATE", "INCLINATION"]:
            if col in df.columns:
                n_null = int(df[col].isnull().sum())
                report["null_counts"][col] = n_null
                if len(df) > 0:
                    pct_null = (n_null / len(df)) * 100
                    report["quality_score"] -= min(5.0, pct_null * 0.1)

        report["quality_score"] = max(0.0, round(report["quality_score"], 2))
        return report

    @staticmethod
    def clean_satcat_data(df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and standardizes SATCAT DataFrame."""
        cleaned = df.copy()

        # Rename common variations of columns
        col_mappings = {
            "NORAD_CAT_ID": "norad_id",
            "OBJECT_NAME": "object_name",
            "OBJECT_TYPE": "object_type",
            "OPS_STATUS_CODE": "status_code",
            "OWNER": "owner",
            "LAUNCH_DATE": "launch_date",
            "DECAY_DATE": "decay_date",
            "PERIOD": "period_min",
            "INCLINATION": "inclination_deg",
            "APOGEE": "apogee_km",
            "PERIGEE": "perigee_km",
            "RCS_SIZE": "rcs_size",
        }
        cleaned = cleaned.rename(columns={k: v for k, v in col_mappings.items() if k in cleaned.columns})

        if "norad_id" in cleaned.columns:
            cleaned = cleaned.dropna(subset=["norad_id"])
            cleaned["norad_id"] = pd.to_numeric(cleaned["norad_id"], errors="coerce")
            cleaned = cleaned.dropna(subset=["norad_id"])
            cleaned["norad_id"] = cleaned["norad_id"].astype(int)
            cleaned = cleaned.drop_duplicates(subset=["norad_id"])

        # Fill text NA
        for text_col in ["object_name", "object_type", "status_code", "owner"]:
            if text_col in cleaned.columns:
                cleaned[text_col] = cleaned[text_col].fillna("UNKNOWN").astype(str).str.strip()

        # Classify orbital regime (LEO, MEO, GEO, HEO)
        if "perigee_km" in cleaned.columns and "apogee_km" in cleaned.columns:
            def get_regime(row):
                p = row.get("perigee_km")
                a = row.get("apogee_km")
                if pd.isna(p) or pd.isna(a):
                    return "UNKNOWN"
                if a <= 2000:
                    return "LEO"
                elif 35000 <= a <= 36500 and 35000 <= p <= 36500:
                    return "GEO"
                elif 2000 < p and a < 35000:
                    return "MEO"
                elif a >= 35000:
                    return "HEO"
                return "OTHER"

            cleaned["orbit_regime"] = cleaned.apply(get_regime, axis=1)

        return cleaned
