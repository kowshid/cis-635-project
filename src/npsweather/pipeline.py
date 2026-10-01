"""Pipeline stages and the `npsw` command-line entry point.

Stages (run in this order; each reads the previous stage's outputs):
  visits    download + clean NPS visitation        -> interim/visits_clean, units
  coords    park coordinates (NPS API + overrides)  -> interim/park_coords
  stations  GHCN station list + inventory            -> raw/ghcn/*.txt
  match     park -> station matching (downloads)     -> interim/park_station_match
  join      park-month analysis table + QA report    -> processed/park_month

Terminal:   npsw inspect-visits
            npsw run all
            npsw run match join
Notebook:   from npsweather import pipeline; pipeline.run(["all"])

--refresh re-downloads a stage's source files. Per-station weather caches in
interim/ghcn_daily and interim/ghcn_monthly are always reused; delete those
folders to rebuild them.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from . import config, ghcn, integrate, nps_parks, nps_visits, station_match
from .io_utils import require, save_table

log = logging.getLogger("npsweather")

STAGES = ("visits", "coords", "stations", "match", "join")


def stage_visits(refresh: bool = False) -> None:
    path = nps_visits.ensure_main_data(refresh=refresh)
    raw = nps_visits.load_raw_visits(path)
    visits, units = nps_visits.clean_visits(raw)
    save_table(visits, config.VISITS_FILE)
    save_table(units, config.UNITS_FILE, csv_copy=True)
    log.info(
        "visits: %d units, %d park-months; status %s",
        len(units), len(visits), visits["status"].value_counts().to_dict(),
    )


def stage_coords(refresh: bool = False) -> None:
    units = pd.read_parquet(require(config.UNITS_FILE, "visits"))
    coords = nps_parks.build_park_coords(units, refresh=refresh)
    save_table(coords, config.COORDS_FILE, csv_copy=True)
    log.info("coords: %d of %d units located", int(coords["latitude"].notna().sum()), len(coords))


def stage_stations(refresh: bool = False) -> None:
    stations = ghcn.load_stations(refresh=refresh)
    inventory = ghcn.load_inventory(refresh=refresh)
    log.info("stations: %d US/territory stations, %d with inventory rows", len(stations), len(inventory))


def stage_match(refresh: bool = False) -> None:
    coords = pd.read_parquet(require(config.COORDS_FILE, "coords"))
    units = pd.read_parquet(require(config.UNITS_FILE, "visits"))
    match = station_match.match_parks_to_stations(coords, units)
    save_table(match, config.MATCH_FILE, csv_copy=True)


def stage_join(refresh: bool = False) -> None:
    visits = pd.read_parquet(require(config.VISITS_FILE, "visits"))
    units = pd.read_parquet(require(config.UNITS_FILE, "visits"))
    coords = pd.read_parquet(require(config.COORDS_FILE, "coords"))
    match = pd.read_parquet(require(config.MATCH_FILE, "match"))
    pm = integrate.build_park_month(visits, units, coords, match)
    save_table(pm, config.PARK_MONTH_FILE)
    report = integrate.quality_report(pm, units, coords, match)
    log.info("join: %d rows x %d columns -> %s", len(pm), pm.shape[1], config.PARK_MONTH_FILE)
    print(json.dumps(report, indent=2))


STAGE_FUNCS = {
    "visits": stage_visits,
    "coords": stage_coords,
    "stations": stage_stations,
    "match": stage_match,
    "join": stage_join,
}


def inspect_visits() -> None:
    """Print Main_Data.csv columns, sample rows, and the detected layout."""
    config.ensure_dirs()
    nps_visits.inspect(nps_visits.ensure_main_data())


def run(stages=("all",), refresh: bool = False) -> None:
    """Run the requested stages in pipeline order."""
    config.ensure_dirs()
    requested = set(STAGES) if "all" in stages else set(stages)
    unknown = requested - set(STAGES)
    if unknown:
        raise ValueError(f"Unknown stage(s) {sorted(unknown)}; choose from {STAGES} or 'all'")
    log.info("Data directory: %s", config.DATA_DIR)
    for name in STAGES:
        if name in requested:
            log.info("=== stage: %s ===", name)
            STAGE_FUNCS[name](refresh=refresh)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(prog="npsw", description="CIS 635 NPS x GHCN pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect-visits", help="show Main_Data.csv columns and detected layout")
    p_run = sub.add_parser("run", help="run pipeline stages")
    p_run.add_argument("stages", nargs="*", default=["all"], help=f"any of {', '.join(STAGES)}, or all")
    p_run.add_argument("--refresh", action="store_true", help="re-download source files")
    args = parser.parse_args(argv)

    if args.command == "inspect-visits":
        inspect_visits()
    else:
        run(args.stages or ["all"], refresh=args.refresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
