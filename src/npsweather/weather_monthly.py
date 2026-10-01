"""Aggregate daily station weather to calendar months.

For each station-month we compute means, totals, extremes, and threshold-day
counts. Totals and day counts are scaled to the full month from the valid
days (e.g. 28 valid days of rain in a 31-day month -> mean x 31). Any value is
set to NaN when fewer than MIN_DAY_COVERAGE of the month's days are valid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config, ghcn

CORE_MONTHLY = ["tmax_mean_c", "tmin_mean_c", "prcp_total_mm"]

# Which daily element's coverage governs each monthly column
COVERAGE_RULES = {
    "TMAX": ["tmax_mean_c", "tmax_max_c", "hot_days", "extreme_heat_days", "mild_days", "ice_days"],
    "TMIN": ["tmin_mean_c", "tmin_min_c", "freeze_days"],
    "TAVG": ["tavg_mean_c", "dtr_mean_c"],  # needs both TMAX and TMIN on the same day
    "PRCP": ["prcp_total_mm", "prcp_max_mm", "wet_days", "heavy_rain_days"],
    "SNOW": ["snow_total_mm", "snowfall_days"],
    "SNWD": ["snwd_mean_mm", "snow_cover_days"],
}


def aggregate_monthly(daily: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Daily table (DatetimeIndex; TMAX, TMIN, PRCP, SNOW, SNWD) -> one row per month."""
    idx = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D", name="date")
    d = daily.reindex(idx)
    for el in config.ALL_ELEMENTS:
        if el not in d.columns:
            d[el] = np.nan
    tmax, tmin, prcp, snow, snwd = (d[el].astype(float) for el in ("TMAX", "TMIN", "PRCP", "SNOW", "SNWD"))
    tavg = (tmax + tmin) / 2

    key = d.index.to_period("M")

    def grp(s: pd.Series):
        return s.groupby(key)

    days = grp(tmax).size().astype(float)
    n = {
        "TMAX": grp(tmax).count(),
        "TMIN": grp(tmin).count(),
        "TAVG": grp(tavg).count(),
        "PRCP": grp(prcp).count(),
        "SNOW": grp(snow).count(),
        "SNWD": grp(snwd).count(),
    }

    def scaled_days(condition: pd.Series, element: str) -> pd.Series:
        hits = grp(condition.astype(float)).sum()
        return hits / n[element].replace(0, np.nan) * days

    out = pd.DataFrame(index=days.index)
    out["days_in_month"] = days.astype(int)
    for el in ("TMAX", "TMIN", "PRCP", "SNOW", "SNWD"):
        out[f"n_{el.lower()}"] = n[el].astype(int)

    out["tmax_mean_c"] = grp(tmax).mean()
    out["tmin_mean_c"] = grp(tmin).mean()
    out["tavg_mean_c"] = grp(tavg).mean()
    out["dtr_mean_c"] = grp(tmax - tmin).mean()
    out["tmax_max_c"] = grp(tmax).max()
    out["tmin_min_c"] = grp(tmin).min()
    out["prcp_total_mm"] = grp(prcp).mean() * days
    out["prcp_max_mm"] = grp(prcp).max()
    out["snow_total_mm"] = grp(snow).mean() * days
    out["snwd_mean_mm"] = grp(snwd).mean()

    out["hot_days"] = scaled_days(tmax >= config.HOT_TMAX_C, "TMAX")
    out["extreme_heat_days"] = scaled_days(tmax >= config.EXTREME_HEAT_TMAX_C, "TMAX")
    out["mild_days"] = scaled_days(tmax.between(*config.MILD_TMAX_RANGE_C), "TMAX")
    out["ice_days"] = scaled_days(tmax <= config.ICE_TMAX_C, "TMAX")
    out["freeze_days"] = scaled_days(tmin <= config.FREEZE_TMIN_C, "TMIN")
    out["wet_days"] = scaled_days(prcp >= config.WET_PRCP_MM, "PRCP")
    out["heavy_rain_days"] = scaled_days(prcp >= config.HEAVY_PRCP_MM, "PRCP")
    out["snowfall_days"] = scaled_days(snow >= config.SNOWFALL_DAY_MM, "SNOW")
    out["snow_cover_days"] = scaled_days(snwd >= config.SNOW_COVER_MM, "SNWD")

    for element, cols in COVERAGE_RULES.items():
        low = (n[element] / days) < config.MIN_DAY_COVERAGE
        out.loc[low, cols] = np.nan

    out = out.reset_index(names="period")
    out.insert(0, "year", out["period"].dt.year)
    out.insert(1, "month", out["period"].dt.month)
    return out.drop(columns="period")


def monthly_cache_path(
    station_id: str,
    start_year: int = config.STUDY_START_YEAR,
    end_year: int = config.STUDY_END_YEAR,
) -> Path:
    return config.GHCN_MONTHLY_DIR / f"{station_id}_{start_year}_{end_year}.parquet"


def monthly_for_station(
    station_id: str,
    start_year: int = config.STUDY_START_YEAR,
    end_year: int = config.STUDY_END_YEAR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Monthly weather for one station over the study period (cached)."""
    cache = monthly_cache_path(station_id, start_year, end_year)
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    daily = ghcn.fetch_station_daily(station_id, start_year, end_year, refresh=refresh)
    monthly = aggregate_monthly(daily, start_year, end_year)
    monthly.insert(0, "station_id", station_id)
    cache.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(cache, index=False)
    return monthly


def coverage(monthly: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Share of months in [start, end] with all core variables present."""
    dates = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
    sel = monthly.loc[(dates >= start) & (dates <= end), CORE_MONTHLY]
    if sel.empty:
        return 0.0
    return float(sel.notna().all(axis=1).mean())
