"""Join the visitation panel with matched-station monthly weather.

Result: one row per (unit_code, year, month) with the visitation outcome,
status/event flags, static park/station attributes, and the station's weather
for that same month. Rows keep NaN weather when no station or month is
available; `weather_complete` marks rows with all core weather variables.

Main_Data.csv carries no park names or designations, so `park_name` and
`designation` come from the NPS API when the visitation file lacks them.
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from . import config, weather_monthly

log = logging.getLogger(__name__)

MATCH_KEEP = [
    "unit_code", "station_id", "station_name", "station_elev_m",
    "distance_km", "coverage", "match_quality",
]
COORD_KEEP = [
    "unit_code", "latitude", "longitude", "api_full_name",
    "api_designation", "api_states", "coord_source",
]


def _fill_from_api(pm: pd.DataFrame, target: str, source: str) -> pd.DataFrame:
    if source not in pm.columns:
        return pm
    if target in pm.columns:
        pm[target] = pm[target].fillna(pm[source])
    else:
        pm[target] = pm[source]
    return pm


def build_park_month(
    visits: pd.DataFrame,
    units: pd.DataFrame,
    coords: pd.DataFrame,
    match: pd.DataFrame,
) -> pd.DataFrame:
    station_ids = match["station_id"].dropna().unique().tolist()
    frames = [weather_monthly.monthly_for_station(s) for s in station_ids]
    weather = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["station_id", "year", "month", *weather_monthly.CORE_MONTHLY]
    )
    weather = weather.drop(columns=["days_in_month"], errors="ignore")

    attr_cols = [c for c in ("unit_code", "park_name", "park_type", "region", "state") if c in units.columns]
    coord_cols = [c for c in COORD_KEEP if c in coords.columns]

    pm = (
        visits.merge(units[attr_cols], on="unit_code", how="left")
        .merge(coords[coord_cols], on="unit_code", how="left")
        .merge(match[MATCH_KEEP], on="unit_code", how="left")
        .merge(weather, on=["station_id", "year", "month"], how="left")
    )
    pm = _fill_from_api(pm, "park_name", "api_full_name")
    pm = _fill_from_api(pm, "designation", "api_designation")
    pm["weather_complete"] = pm[weather_monthly.CORE_MONTHLY].notna().all(axis=1)
    return pm.sort_values(["unit_code", "date"]).reset_index(drop=True)


def quality_report(pm: pd.DataFrame, units: pd.DataFrame, coords: pd.DataFrame, match: pd.DataFrame) -> dict:
    """Headline data-quality numbers for the report and check-ins."""
    dist_q = match["distance_km"].quantile([0.5, 0.9, 0.99]).round(1)
    positive = pm["status"].eq("positive")
    report = {
        "study_period": f"{config.STUDY_START_YEAR}-{config.STUDY_END_YEAR}",
        "units_with_visitation": int(len(units)),
        "units_with_coordinates": int(coords["latitude"].notna().sum()),
        "match_quality": {k: int(v) for k, v in match["match_quality"].value_counts().items()},
        "station_distance_km_quantiles": {f"q{int(q * 100)}": float(v) for q, v in dist_q.items()},
        "park_months": int(len(pm)),
        "status_counts": {k: int(v) for k, v in pm["status"].value_counts().items()},
        "share_rows_weather_complete": round(float(pm["weather_complete"].mean()), 4),
        "share_positive_rows_weather_complete": round(float(pm.loc[positive, "weather_complete"].mean()), 4)
        if positive.any() else None,
        "covid_period_rows": int(pm["covid_period"].sum()),
        "shutdown_affected_rows": int((pm["shutdown_days"] > 0).sum()),
    }
    config.QUALITY_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.QUALITY_REPORT_FILE.write_text(json.dumps(report, indent=2))
    return report