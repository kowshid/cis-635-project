"""Writers for synthetic GHCN-Daily / NPS files in the real on-disk formats."""
from __future__ import annotations

import calendar
import json
from pathlib import Path

import pandas as pd


def dly_line(station_id: str, year: int, month: int, element: str, values) -> str:
    """One .dly record. values[i] is an int, (int, qflag), or None (missing)."""
    parts = []
    for d in range(31):
        item = values[d] if d < len(values) else None
        if item is None:
            value, qflag = -9999, " "
        elif isinstance(item, tuple):
            value, qflag = item
        else:
            value, qflag = item, " "
        parts.append(f"{value:5d} {qflag} ")  # VALUE(5) MFLAG QFLAG SFLAG
    return f"{station_id:<11}{year:04d}{month:02d}{element:<4}" + "".join(parts)


def write_dly(path: Path, station_id: str, years, constants: dict[str, int]) -> None:
    """Full-month records with a constant raw value per element."""
    lines = []
    for year in years:
        for month in range(1, 13):
            ndays = calendar.monthrange(year, month)[1]
            for element, raw in constants.items():
                lines.append(dly_line(station_id, year, month, element, [raw] * ndays))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def station_line(sid, lat, lon, elev, state, name) -> str:
    return f"{sid:<11} {lat:8.4f} {lon:9.4f} {elev:6.1f} {state:<2} {name:<30} {'':<3} {'':<3} {'':<5}"


def inventory_line(sid, lat, lon, element, first, last) -> str:
    return f"{sid:<11} {lat:8.4f} {lon:9.4f} {element:<4} {first:4d} {last:4d}"


def build_fixture(data_dir: Path) -> None:
    """Three parks, three US stations (+1 Canadian), years 2010-2012."""
    nps = data_dir / "raw" / "nps"
    ghcn = data_dir / "raw" / "ghcn"
    nps.mkdir(parents=True, exist_ok=True)
    ghcn.mkdir(parents=True, exist_ok=True)

    # Same long layout as the real Main_Data.csv: UnitCode, Year, Month, Statistic, Value
    rows = []
    for unit in ("AAAA", "BBBB", "CCCC"):
        for year in (2010, 2011, 2012):
            for month in range(1, 13):
                visits = 0 if (unit == "AAAA" and month == 1) else 1000 + month
                rows.append({"UnitCode": unit, "Year": year, "Month": month, "Statistic": "TRV", "Value": visits})
                rows.append({"UnitCode": unit, "Year": year, "Month": month, "Statistic": "TNRV", "Value": 0})
    pd.DataFrame(rows).to_csv(nps / "Main_Data.csv", index=False)

    api = [
        {"parkCode": "aaaa", "fullName": "Alpha NP", "designation": "National Park",
         "states": "CO", "latitude": "40.0", "longitude": "-105.0"},
        {"parkCode": "bbbb", "fullName": "Beta NP", "designation": "National Park",
         "states": "AZ", "latitude": "36.0", "longitude": "-112.0"},
        {"parkCode": "cccc", "fullName": "Gamma NHS", "designation": "National Historic Site",
         "states": "XX", "latitude": "", "longitude": ""},
    ]
    (nps / "nps_api_parks.json").write_text(json.dumps(api))

    stations = [
        ("USC00000001", 40.05, -105.0, 1600.0, "CO", "ALPHA STATION"),
        ("USC00000002", 36.00, -112.1, 2100.0, "AZ", "BETA NEAR SPARSE"),
        ("USC00000003", 36.30, -112.0, 2000.0, "AZ", "BETA FAR GOOD"),
        ("CA000000001", 40.00, -105.0, 1500.0, "", "CANADA DECOY"),
    ]
    (ghcn / "ghcnd-stations.txt").write_text("\n".join(station_line(*s) for s in stations) + "\n")
    inv = [
        inventory_line(sid, lat, lon, el, 2000, 2026)
        for sid, lat, lon, *_ in stations
        for el in ("TMAX", "TMIN", "PRCP")
    ]
    (ghcn / "ghcnd-inventory.txt").write_text("\n".join(inv) + "\n")

    weather = {"TMAX": 250, "TMIN": 100, "PRCP": 20, "SNOW": 0, "SNWD": 0}
    write_dly(ghcn / "dly" / "USC00000001.dly", "USC00000001", (2010, 2011, 2012), weather)
    write_dly(ghcn / "dly" / "USC00000002.dly", "USC00000002", (2010,), weather)  # poor coverage
    write_dly(ghcn / "dly" / "USC00000003.dly", "USC00000003", (2010, 2011, 2012), weather)