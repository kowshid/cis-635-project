"""Load, standardise, and clean NPS monthly visitation (Main_Data.csv).

The exact column layout of Main_Data.csv is detected at run time, so the loader
handles either a wide file (one RecreationVisits column) or a long file (one
row per statistic, with a statistic-code column and a value column). If
detection fails, the error message lists the columns found; run
`npsw inspect-visits` and adjust FIELD_CANDIDATES below.

Output of clean_visits():
  panel   one row per (unit_code, month) inside each unit's active window,
          with a `status` label and event flags
  units   one row per unit with reporting window and data-quality counts
"""
from __future__ import annotations

import calendar
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .io_utils import download, looks_like_html

log = logging.getLogger(__name__)

STATUSES = ("positive", "seasonal_zero", "isolated_zero", "missing", "invalid")

# Normalised (lower-case, alphanumeric only) column-name candidates
FIELD_CANDIDATES: dict[str, list[str]] = {
    "unit_code": ["unitcode", "parkcode", "parkunitcode", "unitalphacode", "alphacode", "unit"],
    "park_name": ["parkname", "unitname", "parkfullname", "park", "name"],
    "park_type": ["parktype", "unittype", "designation", "parkdesignation", "unitdesignation"],
    "region": ["region", "regionname", "npsregion", "regioncode"],
    "state": ["state", "states", "statecode", "stateabbr"],
    "year": ["year", "yr", "calendaryear"],
    "month": ["month", "mo", "monthnumber", "monthnum"],
    "rec_visits": [
        "recreationvisits",
        "recreationvisitors",
        "totalrecreationvisits",
        "recvisits",
        "trv",
    ],
}
DATE_CANDIDATES = ["date", "yearmonth", "monthyear", "period", "reportdate"]
STAT_COL_CANDIDATES = [
    "statistic",
    "statisticcode",
    "statcode",
    "field",
    "fieldname",
    "fieldcode",
    "variable",
    "metric",
    "measure",
    "code",
    "statistictype",
    "type",
]
VALUE_COL_CANDIDATES = ["value", "statvalue", "statisticvalue", "count", "amount", "total"]
REC_VISIT_CODES = {
    "trv",
    "recreationvisits",
    "recreationvisit",
    "totalrecreationvisits",
    "recvisits",
}

MONTH_NAMES = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTH_NAMES |= {name.lower(): i for i, name in enumerate(calendar.month_abbr) if name}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _norm(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _find(columns, candidates) -> str | None:
    lookup = {_norm(c): c for c in columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    return None


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kwargs)


def _layout_error(columns, reason: str) -> ValueError:
    return ValueError(
        f"Could not interpret Main_Data.csv ({reason}).\n"
        f"Columns found: {list(columns)}\n"
        "Run `npsw inspect-visits` and adjust FIELD_CANDIDATES / STAT_COL_CANDIDATES "
        "in nps_visits.py to match."
    )


def ensure_main_data(refresh: bool = False) -> Path:
    """Return the path to Main_Data.csv, downloading it if needed."""
    path = config.NPS_MAIN_DATA_FILE
    if path.exists() and not refresh:
        return path
    manual = (
        f"Download Main_Data.csv manually from {config.NPS_DATASTORE_PROFILE} "
        f"and save it as {path}"
    )
    try:
        download(config.NPS_MAIN_DATA_URL, path, overwrite=refresh)
    except Exception as exc:  # network errors, 4xx/5xx
        raise RuntimeError(f"Automatic download failed ({exc}). {manual}") from exc
    if looks_like_html(path):
        path.unlink()
        raise RuntimeError(f"The DataStore returned an HTML page instead of CSV. {manual}")
    return path


def detect_layout(path: Path, sample_rows: int = 5000) -> dict:
    """Work out which columns hold the unit code, date parts, and visits."""
    sample = _read_csv(path, nrows=sample_rows, low_memory=False)
    cols = list(sample.columns)
    found = {key: _find(cols, cands) for key, cands in FIELD_CANDIDATES.items()}
    found["date"] = None
    if found["unit_code"] is None:
        raise _layout_error(cols, "no park/unit code column")
    if found["year"] is None or found["month"] is None:
        found["date"] = _find(cols, DATE_CANDIDATES)
        if found["date"] is None:
            raise _layout_error(cols, "no year/month or date column")

    found["stat"] = found["value"] = None
    if found["rec_visits"] is not None:
        found["layout"] = "wide"
        return found

    # Long layout: prefer a code column whose values include a rec-visit code
    present = [c for c in (_find(cols, [cand]) for cand in STAT_COL_CANDIDATES) if c]
    present = list(dict.fromkeys(present))
    for col in present:
        if sample[col].map(_norm).isin(REC_VISIT_CODES).any():
            found["stat"] = col
            break
    if found["stat"] is None and present:
        found["stat"] = present[0]  # verified after the full read
    found["value"] = _find(cols, VALUE_COL_CANDIDATES)
    if found["stat"] is None or found["value"] is None:
        raise _layout_error(cols, "no RecreationVisits column and no statistic/value pair")
    found["layout"] = "long"
    return found


def inspect(path: Path, n: int = 5) -> None:
    """Print the columns, first rows, and detected layout of Main_Data.csv."""
    path = Path(path)
    sample = _read_csv(path, nrows=n, low_memory=False)
    print(f"File: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"Columns ({len(sample.columns)}): {list(sample.columns)}\n")
    with pd.option_context("display.width", 220, "display.max_columns", 60):
        print(sample.to_string())
    try:
        layout = detect_layout(path)
        print("\nDetected layout:", {k: v for k, v in layout.items() if v})
    except ValueError as exc:
        print(f"\nLayout detection FAILED:\n{exc}")


def _to_number(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def _parse_month(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all():
        return num
    names = s.astype(str).str.strip().str.lower().map(MONTH_NAMES)
    return num.fillna(names)


def load_raw_visits(path: Path | None = None) -> pd.DataFrame:
    """Read Main_Data.csv into a standard long table.

    Columns: unit_code, year, month, rec_visits, plus park_name / park_type /
    region / state when present. Restricted to the study period.
    """
    path = Path(path) if path else ensure_main_data()
    lay = detect_layout(path)
    attr_keys = ["park_name", "park_type", "region", "state"]
    keys = ["unit_code", *attr_keys]
    keys += ["year", "month"] if lay["date"] is None else ["date"]
    keys += ["rec_visits"] if lay["layout"] == "wide" else ["stat", "value"]
    usecols = list(dict.fromkeys(lay[k] for k in keys if lay.get(k)))

    df = _read_csv(path, usecols=usecols, thousands=",", low_memory=False)
    if lay["layout"] == "long":
        mask = df[lay["stat"]].map(_norm).isin(REC_VISIT_CODES)
        if not mask.any():
            raise _layout_error(df.columns, f"no recreation-visit rows in column {lay['stat']!r}")
        df = df.loc[mask].drop(columns=lay["stat"]).rename(columns={lay["value"]: "rec_visits"})
    else:
        df = df.rename(columns={lay["rec_visits"]: "rec_visits"})

    rename = {lay[k]: k for k in ("unit_code", *attr_keys, "year", "month") if lay.get(k)}
    df = df.rename(columns=rename)
    if lay["date"] is not None:
        dt = pd.to_datetime(df.pop(lay["date"]), errors="coerce")
        df["year"], df["month"] = dt.dt.year, dt.dt.month

    df["unit_code"] = df["unit_code"].astype(str).str.strip().str.upper()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = _parse_month(df["month"])
    df["rec_visits"] = _to_number(df["rec_visits"])
    df = df.dropna(subset=["year", "month"])
    df = df.astype({"year": int, "month": int})
    in_period = df["year"].between(config.STUDY_START_YEAR, config.STUDY_END_YEAR)
    df = df.loc[in_period & df["month"].between(1, 12)].drop_duplicates()

    dup = df.duplicated(["unit_code", "year", "month"], keep=False)
    if dup.any():
        log.warning(
            "%d rows share a (unit, year, month) key; summing them. "
            "Check whether Main_Data.csv contains sub-unit rows.",
            int(dup.sum()),
        )
        present_attrs = [c for c in attr_keys if c in df.columns]
        agg = {"rec_visits": lambda s: s.sum(min_count=1)} | {c: "last" for c in present_attrs}
        df = df.groupby(["unit_code", "year", "month"], as_index=False).agg(agg)

    return df.sort_values(["unit_code", "year", "month"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def _month_start(year: pd.Series, month: pd.Series) -> pd.Series:
    return pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": 1}))


def _active_windows(df: pd.DataFrame) -> pd.DataFrame:
    """First positive month to last reported month, per unit."""
    obs = df[df["rec_visits"].notna()]
    pos = obs[obs["rec_visits"] > 0]
    first_pos = pos.groupby("unit_code")["date"].min()
    last_pos = pos.groupby("unit_code")["date"].max()
    last_obs = obs.groupby("unit_code")["date"].max().reindex(last_pos.index)
    gap = (last_obs.dt.year - last_pos.dt.year) * 12 + (last_obs.dt.month - last_pos.dt.month)
    end = last_obs.where(gap <= config.MAX_TRAILING_ZERO_MONTHS, last_pos)
    return pd.DataFrame({"start": first_pos, "end": end})


def _complete_panel(windows: pd.DataFrame) -> pd.DataFrame:
    frames = [
        pd.DataFrame({"unit_code": unit, "date": pd.date_range(w["start"], w["end"], freq="MS")})
        for unit, w in windows.iterrows()
    ]
    if not frames:
        return pd.DataFrame({"unit_code": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]")})
    return pd.concat(frames, ignore_index=True)


def classify_status(panel: pd.DataFrame) -> pd.Series:
    """Label each park-month: positive / seasonal_zero / isolated_zero / missing / invalid."""
    v = panel["rec_visits"]
    is_zero = v.eq(0)
    zero_share = (
        is_zero.astype(float)
        .where(v.notna())
        .groupby([panel["unit_code"], panel["month"]])
        .transform("mean")
    )
    seasonal = zero_share.ge(config.SEASONAL_ZERO_SHARE)
    conditions = [
        v.lt(0),
        v.gt(0),
        is_zero & seasonal,
        v.isna() & seasonal,  # closed months the park simply doesn't report
        v.isna(),
    ]
    choices = ["invalid", "positive", "seasonal_zero", "seasonal_zero", "missing"]
    return pd.Series(np.select(conditions, choices, default="isolated_zero"), index=panel.index)


def shutdown_days_per_month(periods=config.FEDERAL_SHUTDOWNS) -> pd.Series:
    """Number of shutdown days in each calendar month (index = month start)."""
    if not periods:
        return pd.Series(dtype=int)
    days = pd.DatetimeIndex(
        np.concatenate([pd.date_range(start, end, freq="D").values for start, end in periods])
    ).unique()
    counts = pd.Series(1, index=days).groupby(days.to_period("M")).sum()
    counts.index = counts.index.to_timestamp()
    return counts


def add_event_flags(panel: pd.DataFrame) -> pd.DataFrame:
    """COVID and federal-shutdown flags; these months are not ordinary weather effects."""
    out = panel.copy()
    d = out["date"]
    out["covid_closure"] = d.between(*(pd.Timestamp(x) for x in config.COVID_CLOSURE))
    out["covid_period"] = d.between(*(pd.Timestamp(x) for x in config.COVID_PERIOD))
    out["shutdown_days"] = d.map(shutdown_days_per_month()).fillna(0).astype(int)
    out["shutdown_frac"] = out["shutdown_days"] / d.dt.days_in_month
    return out


def _unit_summary(panel: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    g = panel.groupby("unit_code")
    units = pd.DataFrame(
        {
            "first_month": g["date"].min(),
            "last_month": g["date"].max(),
            "n_months": g.size(),
        }
    )
    counts = pd.crosstab(panel["unit_code"], panel["status"]).reindex(columns=STATUSES, fill_value=0)
    for status in STATUSES:
        units[f"n_{status}"] = counts[status].reindex(units.index).fillna(0).astype(int)

    # Mean annual visits over complete years only (12 reported months)
    yearly = panel.groupby(["unit_code", "year"]).agg(
        visits=("rec_visits", lambda s: s.sum(min_count=1)),
        n_reported=("rec_visits", "count"),
    )
    full = yearly[yearly["n_reported"] == 12]
    units["mean_annual_visits"] = full.groupby(level="unit_code")["visits"].mean()
    units["n_full_years"] = full.groupby(level="unit_code").size()
    units["n_full_years"] = units["n_full_years"].fillna(0).astype(int)

    attr_cols = [c for c in ("park_name", "park_type", "region", "state") if c in raw.columns]
    if attr_cols:
        units = units.join(raw.groupby("unit_code")[attr_cols].last())
    return units.reset_index()


def clean_visits(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the complete park-month panel with status labels and flags."""
    df = raw.copy()
    df["date"] = _month_start(df["year"], df["month"])
    windows = _active_windows(df)
    panel = _complete_panel(windows).merge(
        df[["unit_code", "date", "rec_visits"]], on=["unit_code", "date"], how="left"
    )
    panel["year"] = panel["date"].dt.year
    panel["month"] = panel["date"].dt.month
    panel["days_in_month"] = panel["date"].dt.days_in_month
    panel["status"] = classify_status(panel)
    panel = add_event_flags(panel)
    units = _unit_summary(panel, df)

    dropped = sorted(set(df["unit_code"]) - set(units["unit_code"]))
    if dropped:
        log.info("%d units have no positive visits in the study period: %s", len(dropped), dropped)
    cols = ["unit_code", "date", "year", "month", "days_in_month", "rec_visits", "status",
            "covid_closure", "covid_period", "shutdown_days", "shutdown_frac"]
    return panel[cols], units
