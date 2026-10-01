"""Project-wide configuration.

Every judgment call that shapes the data (study period, station-matching
thresholds, weather thresholds, event flags) lives here, so it can be cited in
the report and changed in one place.

The data directory can be redirected with the NPSW_DATA_DIR environment
variable. The Colab notebook uses this to keep data on Google Drive.
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv is optional; Colab passes settings as env vars instead
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_DIR = Path(os.environ.get("NPSW_DATA_DIR") or PROJECT_ROOT / "data").expanduser()
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
MANUAL_DIR = PROJECT_ROOT / "manual"  # hand-edited inputs, kept in git

NPS_RAW_DIR = RAW_DIR / "nps"
GHCN_RAW_DIR = RAW_DIR / "ghcn"
GHCN_DLY_DIR = GHCN_RAW_DIR / "dly"
GHCN_DAILY_DIR = INTERIM_DIR / "ghcn_daily"
GHCN_MONTHLY_DIR = INTERIM_DIR / "ghcn_monthly"

NPS_MAIN_DATA_FILE = NPS_RAW_DIR / "Main_Data.csv"
NPS_API_CACHE_FILE = NPS_RAW_DIR / "nps_api_parks.json"
COORDS_OVERRIDE_FILE = MANUAL_DIR / "park_coords_override.csv"

VISITS_FILE = INTERIM_DIR / "visits_clean.parquet"
UNITS_FILE = INTERIM_DIR / "units.parquet"
COORDS_FILE = INTERIM_DIR / "park_coords.parquet"
MATCH_FILE = INTERIM_DIR / "park_station_match.parquet"
PARK_MONTH_FILE = PROCESSED_DIR / "park_month.parquet"
QUALITY_REPORT_FILE = REPORTS_DIR / "data_quality.json"
MISSING_COORDS_FILE = REPORTS_DIR / "park_coords_missing.csv"


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for d in (
        NPS_RAW_DIR,
        GHCN_DLY_DIR,
        GHCN_DAILY_DIR,
        GHCN_MONTHLY_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        REPORTS_DIR,
        MANUAL_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Study period
# ---------------------------------------------------------------------------
# 1990 onward: denser GHCN temperature coverage near parks and fewer legacy
# changes in NPS counting procedures, while still giving 30+ years.
STUDY_START_YEAR = 1990
STUDY_END_YEAR = 2025

# Last year used to estimate station climatologies and other "training-only"
# statistics in the feature-engineering stage (prevents temporal leakage).
TRAIN_END_YEAR = 2017

# ---------------------------------------------------------------------------
# Source URLs
# ---------------------------------------------------------------------------
NPS_DATASTORE_PROFILE = "https://irma.nps.gov/DataStore/Reference/Profile/2317666"
NPS_MAIN_DATA_URL = "https://irma.nps.gov/DataStore/DownloadFile/756959?Reference=2317666"
NPS_API_PARKS_URL = "https://developer.nps.gov/api/v1/parks"
NPS_API_SIGNUP_URL = "https://www.nps.gov/subjects/developer/get-started.htm"

GHCN_BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
GHCN_STATIONS_URL = f"{GHCN_BASE_URL}/ghcnd-stations.txt"
GHCN_INVENTORY_URL = f"{GHCN_BASE_URL}/ghcnd-inventory.txt"
GHCN_DLY_URL = GHCN_BASE_URL + "/all/{station_id}.dly"

# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
USER_AGENT = "CIS635-student-project/0.1 (academic use; npsweather)"
HTTP_TIMEOUT = 120  # seconds
DOWNLOAD_WORKERS = 4  # parallel station downloads; keep small to be polite
KEEP_RAW_DLY = False  # delete raw .dly files after parsing to save Drive space

# ---------------------------------------------------------------------------
# GHCN-Daily stations and elements
# ---------------------------------------------------------------------------
# FIPS country prefixes of GHCN station IDs: US states + US territories
# (Puerto Rico, US Virgin Islands, Guam, American Samoa, N. Mariana Islands).
STATION_ID_PREFIXES = ("US", "RQ", "VQ", "GQ", "AQ", "CQ")
REQUIRED_ELEMENTS = ("TMAX", "TMIN", "PRCP")
OPTIONAL_ELEMENTS = ("SNOW", "SNWD")
ALL_ELEMENTS = REQUIRED_ELEMENTS + OPTIONAL_ELEMENTS

# ---------------------------------------------------------------------------
# Park -> station matching
# ---------------------------------------------------------------------------
MAX_SEARCH_KM = 150.0  # ignore stations farther than this
GOOD_MATCH_KM = 50.0  # beyond this a match is labelled "far"
MAX_CANDIDATES = 6  # nearest eligible stations to try per park
MIN_INVENTORY_OVERLAP = 0.85  # share of park's years covered per inventory
MIN_STATION_MONTH_COVERAGE = 0.85  # share of park months with complete core weather

# ---------------------------------------------------------------------------
# Daily -> monthly weather aggregation
# ---------------------------------------------------------------------------
MIN_DAY_COVERAGE = 0.80  # month-level value is NaN if fewer valid days

HOT_TMAX_C = 32.2  # 90 F
EXTREME_HEAT_TMAX_C = 37.8  # 100 F
MILD_TMAX_RANGE_C = (15.6, 26.7)  # 60-80 F, "pleasant" days
FREEZE_TMIN_C = 0.0
ICE_TMAX_C = 0.0  # max temp at/below freezing
WET_PRCP_MM = 1.0
HEAVY_PRCP_MM = 25.4  # 1 inch
SNOWFALL_DAY_MM = 25.4  # 1 inch of new snow
SNOW_COVER_MM = 25.4  # 1 inch on the ground

# ---------------------------------------------------------------------------
# Visitation cleaning
# ---------------------------------------------------------------------------
# A zero (or absent) month is a seasonal closure if the park reports zero in
# that calendar month in at least this share of its observed years.
SEASONAL_ZERO_SHARE = 0.5
# If a park's last positive month is followed by more than this many months of
# zeros, the trailing zeros are treated as "stopped reporting" and dropped.
MAX_TRAILING_ZERO_MONTHS = 12

# Structural shocks (inclusive dates; +/- a day is irrelevant at monthly scale)
COVID_CLOSURE = ("2020-03-01", "2020-06-30")  # widespread park closures
COVID_PERIOD = ("2020-03-01", "2021-12-31")  # broader pandemic disruption

# Federal funding lapses inside the study period. Verify against the
# Congressional Research Service list before citing in the report.
FEDERAL_SHUTDOWNS = (
    ("1990-10-06", "1990-10-08"),
    ("1995-11-14", "1995-11-19"),
    ("1995-12-16", "1996-01-06"),
    ("2013-10-01", "2013-10-16"),
    ("2018-01-20", "2018-01-22"),
    ("2018-12-22", "2019-01-25"),
    ("2025-10-01", "2025-11-12"),
)
