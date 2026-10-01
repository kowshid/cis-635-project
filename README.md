# CIS 635 — NPS monthly visitation × NOAA GHCN-Daily weather

Code for the CIS 635 KDD project. The pipeline links NPS monthly recreation
visits to weather from the nearest qualifying GHCN-Daily station. The output is one
park-month analysis table.

The same package runs locally (PyCharm / terminal) and in Colab. The notebook
in `notebooks/` is a thin driver for the Colab submission.

## Setup (macOS, uv)

```bash
brew install uv                      # once
cd cis635-nps-weather
uv sync                              # creates .venv (Python 3.12) and installs the package
cp .env.example .env && vim .env     # paste your NPS_API_KEY
uv run pytest                        # offline tests, ~2 s
```

The NPS API key is free from https://www.nps.gov/subjects/developer/get-started.htm.

**PyCharm:** Settings → Project → Python Interpreter → Add Local Interpreter →
*Select existing* → `.venv/bin/python`. For a run configuration, choose
*Module* `npsweather.pipeline` with parameters `run all`.

## Running

```bash
uv run npsw inspect-visits           # downloads Main_Data.csv, shows the detected column layout
uv run npsw run all                  # visits → coords → stations → match → join
uv run npsw run match join           # rerun selected stages
```

The first `match` run downloads one file per candidate station from NCEI. That is
roughly 400–600 files and 15–30 minutes. Later runs reuse the cache.

To share cached data with Colab, set `NPSW_DATA_DIR` in `.env` to a folder inside
Google Drive for desktop. The `.env.example` file shows an example path.

## Colab

1. Push this repo to GitHub. A public repo is simplest for grading.
2. Open `notebooks/01_data_pipeline.ipynb` in Colab and set `REPO_URL`.
3. Add `NPS_API_KEY` under Colab *Secrets* (the key icon) and enable notebook access.
4. Run all cells. Data is stored in `MyDrive/cis635/data`.

## Pipeline

| Stage | What it does | Main output |
|---|---|---|
| `visits` | Reads `Main_Data.csv`, builds a complete park-month panel per unit, labels every month (`positive`, `seasonal_zero`, `isolated_zero`, `missing`, `invalid`), and flags COVID and federal-shutdown months | `interim/visits_clean.parquet`, `interim/units.parquet` |
| `coords` | Gets park coordinates from the NPS API and applies `manual/park_coords_override.csv` | `interim/park_coords.parquet` |
| `stations` | Downloads the GHCN station list and inventory (US and territories) | `raw/ghcn/*.txt` |
| `match` | Walks each park's nearest eligible stations and keeps the first one with at least 85% monthly coverage | `interim/park_station_match.parquet` |
| `join` | Aggregates daily weather to months and joins it on (unit, year, month); writes a QA report | `processed/park_month.parquet`, `reports/data_quality.json` |

Every threshold and date range is in `src/npsweather/config.py`. That includes
the study period, the matching radii and coverage limits, the hot/freeze/wet-day
definitions, and the COVID and shutdown periods.

## Things to check after the first real run

- `reports/park_coords_missing.csv` lists units the NPS API could not locate.
  Add coordinates for the ones you want to keep to `manual/park_coords_override.csv`.
- In `interim/park_station_match.csv`, sort by `distance_km` and `coverage`.
  Large or linear units (trails, parkways, Alaska parks) are where a single
  station represents the park poorly. Decide whether to exclude them.
- If `npsw inspect-visits` reports a layout failure, adjust `FIELD_CANDIDATES`
  in `nps_visits.py` to match the real column names.

## Data sources

- NPS Visitor Use Statistics Data Package, 2025. NPS Social Science Program,
  NPS DataStore ref. 2317666. CC0 1.0.
- Menne et al. (2012), Global Historical Climatology Network – Daily (GHCN-Daily),
  NOAA NCEI, doi:10.7289/V5D21VHZ. U.S. Government public domain.
- NPS Data API (`/parks` endpoint) for park coordinates. Requires a free API key.
