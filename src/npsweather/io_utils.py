"""HTTP download helpers: retries, on-disk caching, atomic writes."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config

log = logging.getLogger(__name__)
_local = threading.local()  # one Session per thread (Sessions aren't thread-safe)


def get_session() -> requests.Session:
    """Return a per-thread requests.Session with retry/backoff configured."""
    session = getattr(_local, "session", None)
    if session is None:
        retry = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers["User-Agent"] = config.USER_AGENT
        _local.session = session
    return session


def download(
    url: str,
    dest: Path,
    *,
    overwrite: bool = False,
    headers: dict | None = None,
    params: dict | None = None,
) -> Path:
    """Download url to dest unless a non-empty dest already exists.

    Writes to a .part file first and renames on success, so an interrupted
    download never leaves a truncated file that looks valid.
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    log.debug("Downloading %s", url)
    with get_session().get(
        url, stream=True, timeout=config.HTTP_TIMEOUT, headers=headers, params=params
    ) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    tmp.replace(dest)
    return dest


def looks_like_html(path: Path, nbytes: int = 512) -> bool:
    """True if a 'data' file is actually an HTML page (e.g. a login/error page)."""
    with open(path, "rb") as fh:
        head = fh.read(nbytes).lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def save_table(df: pd.DataFrame, path: Path, *, csv_copy: bool = False) -> None:
    """Save a table as parquet; optionally also as CSV for eyeballing in PyCharm."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    if csv_copy:
        df.to_csv(path.with_suffix(".csv"), index=False)


def require(path: Path, stage: str) -> Path:
    """Fail with a helpful message if a previous stage's output is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run the '{stage}' stage first.")
    return path
