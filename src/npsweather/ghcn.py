"""NOAA GHCN-Daily: station list, inventory, and per-station .dly files.

File formats follow NCEI's GHCN-Daily readme. Raw values are integers in
tenths of deg C (TMAX, TMIN), tenths of mm (PRCP), and mm (SNOW, SNWD);
missing = -9999. Any value with a non-blank quality flag failed NOAA's QC
checks and is dropped.
"""
from __future__ import annotations

import calendar
import logging
from pathlib import Path

import pandas as pd

from . import config
from .io_utils import download

log = logging.getLogger(__name__)

STATION_COLSPECS = [(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71), (72, 75), (76, 79), (80, 85)]
STATION_NAMES = ["station_id", "lat", "lon", "elev_m", "state", "name", "gsn", "hcn_crn", "wmo_id"]
INVENTORY_COLSPECS = [(0, 11), (12, 20), (21, 30), (31, 35), (36, 40), (41, 45)]
INVENTORY_NAMES = ["station_id", "lat", "lon", "element", "first_year", "last_year"]

SCALE = {"TMAX": 0.1, "TMIN": 0.1, "PRCP": 0.1, "SNOW": 1.0, "SNWD": 1.0}
DLY_LINE_LEN = 269  # 21 header chars + 31 days x 8 chars
MISSING = -9999


def _us_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["station_id"].str[:2].isin(config.STATION_ID_PREFIXES)]


def load_stations(refresh: bool = False) -> pd.DataFrame:
    """US + territory stations from ghcnd-stations.txt."""
    path = download(config.GHCN_STATIONS_URL, config.GHCN_RAW_DIR / "ghcnd-stations.txt", overwrite=refresh)
    df = pd.read_fwf(
        path,
        colspecs=STATION_COLSPECS,
        names=STATION_NAMES,
        header=None,
        dtype={c: str for c in ("station_id", "state", "name", "gsn", "hcn_crn", "wmo_id")},
        keep_default_na=False,
        encoding="latin-1",
    )
    df = _us_only(df).copy()
    for col in ("lat", "lon", "elev_m"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df["elev_m"] <= -999, "elev_m"] = float("nan")
    df["name"] = df["name"].str.strip()
    return df.reset_index(drop=True)


def load_inventory(refresh: bool = False) -> pd.DataFrame:
    """Per-station first/last year for each element we use (wide table)."""
    path = download(config.GHCN_INVENTORY_URL, config.GHCN_RAW_DIR / "ghcnd-inventory.txt", overwrite=refresh)
    df = pd.read_fwf(
        path,
        colspecs=INVENTORY_COLSPECS,
        names=INVENTORY_NAMES,
        header=None,
        dtype={"station_id": str, "element": str},
        keep_default_na=False,
    )
    df = _us_only(df)
    df = df[df["element"].isin(config.ALL_ELEMENTS)]
    first = df.pivot_table(index="station_id", columns="element", values="first_year", aggfunc="min")
    last = df.pivot_table(index="station_id", columns="element", values="last_year", aggfunc="max")
    first.columns = [f"first_{c}" for c in first.columns]
    last.columns = [f"last_{c}" for c in last.columns]
    return first.join(last).reset_index()


def parse_dly(path: Path, start_year: int, end_year: int, elements=config.ALL_ELEMENTS) -> pd.DataFrame:
    """Parse a .dly file into a daily table (DatetimeIndex, one column per element)."""
    wanted = set(elements)
    years, months, days, els, vals = [], [], [], [], []
    with open(path, encoding="latin-1") as fh:
        for raw in fh:
            element = raw[17:21]
            if element not in wanted:
                continue
            year = int(raw[11:15])
            if year < start_year or year > end_year:
                continue
            month = int(raw[15:17])
            line = raw.rstrip("\r\n").ljust(DLY_LINE_LEN)
            for d in range(calendar.monthrange(year, month)[1]):
                base = 21 + 8 * d
                field = line[base:base + 5].strip()
                if not field:
                    continue
                value = int(field)
                if value == MISSING or line[base + 6] != " ":  # missing or failed QC
                    continue
                years.append(year)
                months.append(month)
                days.append(d + 1)
                els.append(element)
                vals.append(value)

    columns = list(elements)
    if not vals:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"), dtype=float)
    long = pd.DataFrame({"year": years, "month": months, "day": days, "element": els, "value": vals})
    long["value"] = long["value"] * long["element"].map(SCALE)
    long["date"] = pd.to_datetime(long[["year", "month", "day"]])
    wide = long.pivot_table(index="date", columns="element", values="value", aggfunc="first")
    wide = wide.reindex(columns=columns).astype(float).sort_index()
    wide.columns.name = None
    return wide


def daily_cache_path(station_id: str, start_year: int, end_year: int) -> Path:
    return config.GHCN_DAILY_DIR / f"{station_id}_{start_year}_{end_year}.parquet"


def fetch_station_daily(
    station_id: str,
    start_year: int = config.STUDY_START_YEAR,
    end_year: int = config.STUDY_END_YEAR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Daily TMAX/TMIN/PRCP/SNOW/SNWD in deg C / mm for one station (cached)."""
    cache = daily_cache_path(station_id, start_year, end_year)
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    dly = config.GHCN_DLY_DIR / f"{station_id}.dly"
    download(config.GHCN_DLY_URL.format(station_id=station_id), dly, overwrite=refresh)
    daily = parse_dly(dly, start_year, end_year)
    cache.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(cache)
    if not config.KEEP_RAW_DLY:
        dly.unlink(missing_ok=True)
    return daily
