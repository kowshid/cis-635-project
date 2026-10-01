import numpy as np
import pandas as pd
import pytest

from helpers import dly_line
from npsweather import ghcn, nps_visits, station_match, weather_monthly


# --------------------------------------------------------------------- GHCN
def test_parse_dly_scales_and_drops_missing_and_qc_failures(tmp_path):
    path = tmp_path / "X.dly"
    path.write_text(
        "\n".join(
            [
                dly_line("USW00000001", 2000, 2, "TMAX", [123, None, (50, "X")]),
                dly_line("USW00000001", 2000, 2, "PRCP", [254]),
                dly_line("USW00000001", 2000, 2, "WT01", [1]),  # ignored element
                dly_line("USW00000001", 1980, 2, "TMAX", [999]),  # outside years
            ]
        )
        + "\n"
    )
    daily = ghcn.parse_dly(path, 1990, 2025)
    assert list(daily.columns) == ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"]
    assert daily.loc["2000-02-01", "TMAX"] == pytest.approx(12.3)
    assert daily.loc["2000-02-01", "PRCP"] == pytest.approx(25.4)
    assert daily["TMAX"].notna().sum() == 1  # missing and QC-flagged days dropped


def test_parse_dly_empty_file(tmp_path):
    path = tmp_path / "empty.dly"
    path.write_text("")
    daily = ghcn.parse_dly(path, 1990, 2025)
    assert daily.empty and list(daily.columns) == ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"]


# ----------------------------------------------------------- monthly weather
def test_aggregate_monthly_counts_scaling_and_coverage_mask():
    idx = pd.date_range("2000-01-01", "2000-01-31", freq="D")
    jan = pd.DataFrame({"TMAX": 33.0, "TMIN": -1.0, "PRCP": 2.0}, index=idx)
    mar = pd.DataFrame({"TMAX": 10.0}, index=pd.date_range("2000-03-01", periods=10, freq="D"))
    out = weather_monthly.aggregate_monthly(pd.concat([jan, mar]), 2000, 2000).set_index("month")

    assert len(out) == 12
    j = out.loc[1]
    assert j["hot_days"] == 31 and j["freeze_days"] == 31 and j["wet_days"] == 31
    assert j["prcp_total_mm"] == pytest.approx(62.0)
    assert j["tavg_mean_c"] == pytest.approx(16.0)
    assert np.isnan(j["snow_total_mm"])  # SNOW never reported
    assert np.isnan(out.loc[2, "tmax_mean_c"])  # no data
    assert np.isnan(out.loc[3, "tmax_mean_c"])  # 10/31 days < coverage threshold
    assert out.loc[3, "n_tmax"] == 10


def test_coverage_uses_window():
    m = pd.DataFrame({"year": [2000] * 4, "month": [1, 2, 3, 4],
                      "tmax_mean_c": [1, 1, np.nan, 1], "tmin_mean_c": 0.0, "prcp_total_mm": 5.0})
    assert weather_monthly.coverage(m, pd.Timestamp("2000-01-01"), pd.Timestamp("2000-04-01")) == 0.75
    assert weather_monthly.coverage(m, pd.Timestamp("2000-01-01"), pd.Timestamp("2000-02-01")) == 1.0


# ----------------------------------------------------------------- visitation
def test_load_raw_visits_wide(tmp_path):
    path = tmp_path / "wide.csv"
    pd.DataFrame(
        {"UnitCode": ["acad", "ACAD"], "ParkName": ["Acadia NP"] * 2, "Year": [2000, 1970],
         "Month": [7, 7], "RecreationVisits": ["1,234", "5"]}
    ).to_csv(path, index=False)
    df = nps_visits.load_raw_visits(path)
    assert len(df) == 1  # 1970 is outside the study period
    row = df.iloc[0]
    assert (row.unit_code, row.year, row.month, row.rec_visits) == ("ACAD", 2000, 7, 1234)
    assert row.park_name == "Acadia NP"


def test_load_raw_visits_long_with_month_names(tmp_path):
    path = tmp_path / "long.csv"
    pd.DataFrame(
        {"UnitCode": ["YELL"] * 4, "Year": [2001] * 4,
         "Month": ["January", "January", "February", "February"],
         "Statistic": ["TRV", "TNRV", "TRV", "TNRV"], "Value": [100, 7, 200, 9]}
    ).to_csv(path, index=False)
    df = nps_visits.load_raw_visits(path)
    assert df[["month", "rec_visits"]].values.tolist() == [[1, 100], [2, 200]]


def test_load_raw_visits_reports_columns_on_failure(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Columns found"):
        nps_visits.load_raw_visits(path)


def test_clean_visits_status_labels():
    rows = []
    for year in (2000, 2001, 2002):
        for month in range(1, 13):
            if (year, month) == (2002, 3):
                continue  # absent row -> missing
            visits = 0 if month == 1 else 100
            if (year, month) == (2001, 7):
                visits = 0  # one-off zero -> isolated_zero
            rows.append({"unit_code": "TEST", "year": year, "month": month, "rec_visits": visits})
    panel, units = nps_visits.clean_visits(pd.DataFrame(rows))
    status = panel.set_index("date")["status"]

    assert panel["date"].min() == pd.Timestamp("2000-02-01")  # window starts at first positive
    assert status["2001-01-01"] == "seasonal_zero"
    assert status["2001-07-01"] == "isolated_zero"
    assert status["2002-03-01"] == "missing"
    assert status["2002-04-01"] == "positive"
    u = units.iloc[0]
    assert (u.n_seasonal_zero, u.n_isolated_zero, u.n_missing) == (2, 1, 1)


def test_shutdown_days_per_month():
    s = nps_visits.shutdown_days_per_month()
    assert s[pd.Timestamp("2013-10-01")] == 16
    assert s[pd.Timestamp("2018-12-01")] == 10
    assert s[pd.Timestamp("2019-01-01")] == 25


# ------------------------------------------------------------------ matching
def test_haversine_one_degree_latitude():
    assert station_match.haversine_km(0, 0, 1, 0) == pytest.approx(111.19, abs=0.05)


def test_inventory_overlap_takes_worst_element():
    pool = pd.DataFrame({"first_TMAX": [1990], "last_TMAX": [2025], "first_TMIN": [2000],
                         "last_TMIN": [2025], "first_PRCP": [1990], "last_PRCP": [2010]})
    share = station_match.inventory_overlap(pool, 1990, 2025)[0]
    assert share == pytest.approx(21 / 36)
