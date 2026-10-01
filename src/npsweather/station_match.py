"""Match each park unit to a GHCN-Daily station.

For each park:
  1. Candidate pool = US/territory stations that report TMAX, TMIN, and PRCP
     and whose inventory spans >= MIN_INVENTORY_OVERLAP of the park's active
     years, within MAX_SEARCH_KM.
  2. Walk candidates from nearest to farthest. Download each one and compute
     the share of the park's months with complete core weather. Take the first
     station with coverage >= MIN_STATION_MONTH_COVERAGE; otherwise keep the
     best-covered station and label the match "low_coverage".

match_quality is one of: good, far, low_coverage, no_station, no_coords,
download_failed.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

from . import config, ghcn, weather_monthly

log = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0088
MATCH_COLUMNS = [
    "unit_code", "station_id", "station_name", "station_state", "station_lat", "station_lon",
    "station_elev_m", "station_hcn_crn", "distance_km", "inventory_overlap", "candidate_rank",
    "coverage", "n_candidates", "match_quality",
]


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km (vectorised)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def station_pool(stations: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    """Stations that have every required element in the inventory."""
    need = [f"{p}_{e}" for e in config.REQUIRED_ELEMENTS for p in ("first", "last")]
    inv = inventory.dropna(subset=[c for c in need if c in inventory.columns])
    missing_cols = [c for c in need if c not in inv.columns]
    if missing_cols:
        raise ValueError(f"Inventory lacks columns {missing_cols}")
    return stations.merge(inv, on="station_id", how="inner").reset_index(drop=True)


def inventory_overlap(pool: pd.DataFrame, start_year: int, end_year: int) -> np.ndarray:
    """Minimum (over required elements) share of [start_year, end_year] in the inventory."""
    span = end_year - start_year + 1
    shares = []
    for el in config.REQUIRED_ELEMENTS:
        lo = np.maximum(pool[f"first_{el}"].to_numpy(dtype=float), start_year)
        hi = np.minimum(pool[f"last_{el}"].to_numpy(dtype=float), end_year)
        shares.append(np.clip(hi - lo + 1, 0, None) / span)
    return np.min(shares, axis=0)


def rank_candidates(lat: float, lon: float, start_year: int, end_year: int, pool: pd.DataFrame) -> pd.DataFrame:
    dist = haversine_km(lat, lon, pool["lat"], pool["lon"])
    overlap = inventory_overlap(pool, start_year, end_year)
    ok = (dist <= config.MAX_SEARCH_KM) & (overlap >= config.MIN_INVENTORY_OVERLAP)
    cols = ["station_id", "name", "state", "lat", "lon", "elev_m", "hcn_crn"]
    cands = pool.loc[ok, cols].assign(distance_km=dist[ok], inventory_overlap=overlap[ok])
    return cands.nsmallest(config.MAX_CANDIDATES, "distance_km").reset_index(drop=True)


def _prefetch(station_ids: list[str]) -> None:
    """Download + aggregate stations in parallel (network-bound)."""
    todo = [s for s in station_ids if not weather_monthly.monthly_cache_path(s).exists()]
    if not todo:
        return
    with ThreadPoolExecutor(max_workers=config.DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(weather_monthly.monthly_for_station, s): s for s in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading stations"):
            try:
                fut.result()
            except Exception as exc:  # logged; the matcher will skip this station
                log.warning("Prefetch failed for %s: %s", futures[fut], exc)


def _choose_station(park, cands: pd.DataFrame | None) -> dict:
    result = {"unit_code": park.unit_code, "n_candidates": 0 if cands is None else len(cands)}
    if cands is None:
        return result | {"match_quality": "no_coords"}
    if cands.empty:
        return result | {"match_quality": "no_station"}

    best = None
    for rank, st in enumerate(cands.itertuples(index=False), start=1):
        try:
            monthly = weather_monthly.monthly_for_station(st.station_id)
        except (requests.RequestException, OSError, ValueError) as exc:
            log.warning("%s: station %s unusable (%s)", park.unit_code, st.station_id, exc)
            continue
        cov = weather_monthly.coverage(monthly, park.window_start, park.window_end)
        info = {
            "station_id": st.station_id,
            "station_name": st.name,
            "station_state": st.state,
            "station_lat": st.lat,
            "station_lon": st.lon,
            "station_elev_m": st.elev_m,
            "station_hcn_crn": st.hcn_crn,
            "distance_km": round(float(st.distance_km), 2),
            "inventory_overlap": round(float(st.inventory_overlap), 3),
            "candidate_rank": rank,
            "coverage": round(cov, 3),
        }
        if best is None or cov > best["coverage"]:
            best = info
        if cov >= config.MIN_STATION_MONTH_COVERAGE:
            best = info
            break

    if best is None:
        return result | {"match_quality": "download_failed"}
    if best["coverage"] < config.MIN_STATION_MONTH_COVERAGE:
        quality = "low_coverage"
    elif best["distance_km"] > config.GOOD_MATCH_KM:
        quality = "far"
    else:
        quality = "good"
    return result | best | {"match_quality": quality}


def match_parks_to_stations(coords: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """One row per unit: chosen station, distance, coverage, match_quality."""
    parks = coords[["unit_code", "latitude", "longitude"]].merge(
        units[["unit_code", "first_month", "last_month"]], on="unit_code", how="inner"
    ).rename(columns={"first_month": "window_start", "last_month": "window_end"})

    pool = station_pool(ghcn.load_stations(), ghcn.load_inventory())
    log.info("Station pool: %d US/territory stations with %s", len(pool), "/".join(config.REQUIRED_ELEMENTS))

    candidates: dict[str, pd.DataFrame | None] = {}
    for park in parks.itertuples(index=False):
        if pd.isna(park.latitude) or pd.isna(park.longitude):
            candidates[park.unit_code] = None
        else:
            candidates[park.unit_code] = rank_candidates(
                park.latitude, park.longitude, park.window_start.year, park.window_end.year, pool
            )

    nearest = sorted({c["station_id"].iat[0] for c in candidates.values() if c is not None and len(c)})
    log.info("Prefetching %d nearest-candidate stations", len(nearest))
    _prefetch(nearest)

    rows = [
        _choose_station(park, candidates[park.unit_code])
        for park in tqdm(parks.itertuples(index=False), total=len(parks), desc="Matching parks")
    ]
    match = pd.DataFrame(rows).reindex(columns=MATCH_COLUMNS)
    log.info("Match quality: %s", match["match_quality"].value_counts().to_dict())
    return match
