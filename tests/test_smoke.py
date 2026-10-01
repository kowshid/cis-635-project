"""Run the whole pipeline offline on synthetic files in the real formats.

All source files are pre-placed in the data directory, so the download
helpers find them cached and never touch the network.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from helpers import build_fixture

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_end_to_end(tmp_path):
    data_dir = tmp_path / "data"
    build_fixture(data_dir)
    env = os.environ | {
        "NPSW_DATA_DIR": str(data_dir),
        "NPS_API_KEY": "",  # cache exists, so no key is needed
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "npsweather.pipeline", "run", "all"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr

    match = pd.read_parquet(data_dir / "interim" / "park_station_match.parquet").set_index("unit_code")
    assert match.loc["AAAA", "station_id"] == "USC00000001"
    assert match.loc["AAAA", "match_quality"] == "good"
    # nearest BBBB station has 1 of 3 years -> falls through to the 2nd candidate
    assert match.loc["BBBB", "station_id"] == "USC00000003"
    assert match.loc["BBBB", "candidate_rank"] == 2
    assert match.loc["CCCC", "match_quality"] == "no_coords"

    pm = pd.read_parquet(data_dir / "processed" / "park_month.parquet")
    assert len(pm) == 35 + 36 + 36  # AAAA starts at its first positive month (Feb 2010)
    assert pm.loc[pm.unit_code == "AAAA", "status"].value_counts()["seasonal_zero"] == 2
    assert pm.loc[pm.unit_code.isin(["AAAA", "BBBB"]), "weather_complete"].all()
    assert not pm.loc[pm.unit_code == "CCCC", "weather_complete"].any()
    row = pm[(pm.unit_code == "AAAA") & (pm.year == 2011) & (pm.month == 7)].iloc[0]
    assert row.tmax_mean_c == 25.0 and row.prcp_total_mm == 62.0 and row.mild_days == 31

    # names/designations come from the NPS API because Main_Data.csv has none
    assert pm.loc[pm.unit_code == "AAAA", "park_name"].iat[0] == "Alpha NP"
    assert pm.loc[pm.unit_code == "CCCC", "designation"].iat[0] == "National Historic Site"

    report = json.loads((data_dir / "reports" / "data_quality.json").read_text())
    assert report["match_quality"] == {"good": 2, "no_coords": 1}