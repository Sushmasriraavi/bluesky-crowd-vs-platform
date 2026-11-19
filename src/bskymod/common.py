"""Shared helpers: config loading, paths, DuckDB connection, logging."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = p if p.is_absolute() else ROOT / p
    return cfg


def get_logger(name: str, logfile: Path | None = None) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    log.addHandler(h)
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


def connect(cfg: dict):
    import duckdb

    tmp = Path(cfg["duckdb"]["temp_dir"])
    tmp = tmp if tmp.is_absolute() else ROOT / tmp
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{cfg['duckdb']['memory_limit']}'")
    con.execute(f"SET threads={int(cfg['duckdb']['threads'])}")
    con.execute(f"SET temp_directory='{sql_str(str(tmp))}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET TimeZone='UTC'")  # all timestamps in UTC, independent of server locale
    return con


def sql_str(s: str) -> str:
    """Escape a value for use inside single quotes in SQL."""
    return s.replace("'", "''")


def qident(c: str) -> str:
    return '"' + c.replace('"', '""') + '"'
