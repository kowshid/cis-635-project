"""Park coordinates from the NPS Data API, with manual overrides.

The API (free key required) returns one representative lat/long per park
code. Visitation unit codes that the API doesn't know, or that come back
without coordinates, are written to reports/park_coords_missing.csv; copy any
you can resolve into manual/park_coords_override.csv and rerun the stage.

Source order for the API records:
  1. data/raw/nps/nps_api_parks.json     local cache from an earlier run
  2. the live API                        when NPS_API_KEY is set
  3. manual/nps_api_parks_snapshot.json  committed copy, so the Colab notebook
                                         runs for anyone without a key
"""
from __future__ import annotations

import json
import logging
import os

import pandas as pd

from . import config
from .io_utils import get_session

log = logging.getLogger(__name__)

API_FIELDS = ("parkCode", "fullName", "designation", "states", "latitude", "longitude")
SNAPSHOT_FILE = config.MANUAL_DIR / "nps_api_parks_snapshot.json"


def _fetch_from_api(key: str) -> list[dict]:
    records, start, session = [], 0, get_session()
    while True:
        resp = session.get(
            config.NPS_API_PARKS_URL,
            params={"limit": 500, "start": start},
            headers={"X-Api-Key": key},  # header, so the key never appears in URLs/logs
            timeout=config.HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        batch = payload.get("data", [])
        records.extend({f: p.get(f) for f in API_FIELDS} for p in batch)
        start += len(batch)
        total = int(payload.get("total") or 0)
        if not batch or start >= total:
            return records


def fetch_nps_api_parks(refresh: bool = False) -> pd.DataFrame:
    """All parks from the NPS API (cache -> live API -> committed snapshot)."""
    cache = config.NPS_API_CACHE_FILE
    key = os.environ.get("NPS_API_KEY", "").strip()

    if cache.exists() and not refresh:
        records = json.loads(cache.read_text())
    elif key:
        records = _fetch_from_api(key)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(records, indent=1))
        log.info("NPS API: cached %d parks to %s", len(records), cache)
    elif SNAPSHOT_FILE.exists():
        log.info("NPS_API_KEY not set; using committed snapshot %s", SNAPSHOT_FILE)
        records = json.loads(SNAPSHOT_FILE.read_text())
    else:
        raise RuntimeError(
            "NPS_API_KEY is not set and no snapshot exists. Get a free key at "
            f"{config.NPS_API_SIGNUP_URL} and put it in .env (local) or Colab "
            f"Secrets, or commit a snapshot to {SNAPSHOT_FILE}."
        )

    api = pd.DataFrame.from_records(records, columns=list(API_FIELDS)).rename(
        columns={
            "parkCode": "unit_code",
            "fullName": "api_full_name",
            "designation": "api_designation",
            "states": "api_states",
        }
    )
    api["unit_code"] = api["unit_code"].astype(str).str.strip().str.upper()
    for col in ("latitude", "longitude"):
        api[col] = pd.to_numeric(api[col], errors="coerce")
    return api.drop_duplicates("unit_code")


def load_overrides() -> pd.DataFrame:
    path = config.COORDS_OVERRIDE_FILE
    cols = ["unit_code", "latitude", "longitude", "note"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    ov = pd.read_csv(path, dtype={"unit_code": str})
    ov["unit_code"] = ov["unit_code"].str.strip().str.upper()
    for col in ("latitude", "longitude"):
        ov[col] = pd.to_numeric(ov[col], errors="coerce")
    return ov.dropna(subset=["latitude", "longitude"]).drop_duplicates("unit_code", keep="last")


def build_park_coords(units: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """One row per visitation unit with latitude/longitude and their source."""
    base_cols = [c for c in ("unit_code", "park_name", "park_type") if c in units.columns]
    coords = units[base_cols].merge(fetch_nps_api_parks(refresh), on="unit_code", how="left")
    coords["coord_source"] = coords["latitude"].notna().map({True: "nps_api", False: pd.NA})

    ov = load_overrides().set_index("unit_code")
    hit = coords["unit_code"].isin(ov.index)
    if hit.any():
        idx = coords.loc[hit, "unit_code"]
        coords.loc[hit, "latitude"] = idx.map(ov["latitude"]).to_numpy()
        coords.loc[hit, "longitude"] = idx.map(ov["longitude"]).to_numpy()
        coords.loc[hit, "coord_source"] = "manual"
        log.info("Applied %d manual coordinate overrides", int(hit.sum()))

    missing = coords[coords["latitude"].isna() | coords["longitude"].isna()]
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    template = missing[base_cols].assign(latitude="", longitude="", note="")
    template.to_csv(config.MISSING_COORDS_FILE, index=False)
    if len(missing):
        log.warning(
            "%d of %d units have no coordinates (listed in %s). They will be "
            "excluded from the weather join unless added to %s.",
            len(missing), len(coords), config.MISSING_COORDS_FILE, config.COORDS_OVERRIDE_FILE,
        )
    return coords