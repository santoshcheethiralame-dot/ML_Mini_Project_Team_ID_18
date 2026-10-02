"""Loading and cleaning the Web Robots Kickstarter dump.

Only information that is known *at launch* is kept (goal, category, country,
currency, location, blurb, name, launch date, deadline). Everything that is
only known after launch (backers, pledged amount, staff pick, ...) is dropped
on purpose - using it would leak the outcome.
"""
from __future__ import annotations

import glob
import json
import os
import zipfile
from typing import Optional

import numpy as np
import pandas as pd

USECOLS = [
    "id", "name", "blurb", "goal", "state", "country", "currency",
    "deadline", "created_at", "launched_at", "static_usd_rate",
    "category", "location",
]
FINAL_STATES = {"successful": 1, "failed": 0}


def _read_one(handle, nrows: Optional[int] = None) -> pd.DataFrame:
    return pd.read_csv(handle, usecols=lambda c: c in USECOLS, nrows=nrows)


def _iter_csvs(path: str, max_files: Optional[int] = None):
    """Yield (label, DataFrame) for every CSV found in `path` (csv, zip or dir).

    `max_files` limits the number of *snapshot files* (zips / csvs), never the
    chunk CSVs inside a zip: each Web Robots snapshot is split into ~85 chunks,
    so counting chunks would silently read a fraction of one snapshot.
    """
    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(
            glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
            + glob.glob(os.path.join(path, "**", "*.zip"), recursive=True)
        )
    if not files:
        raise FileNotFoundError(
            f"No .csv or .zip files found under '{path}'. "
            "See data/README.md for download instructions."
        )
    if max_files is not None:
        files = files[:max_files]
    for f in files:
        if f.lower().endswith(".zip"):
            with zipfile.ZipFile(f) as z:
                for member in sorted(z.namelist()):
                    if member.lower().endswith(".csv"):
                        with z.open(member) as h:
                            yield f"{f}:{member}", _read_one(h)
        else:
            yield f, _read_one(f)


def load_raw(path: str, max_files: Optional[int] = None, verbose: bool = True):
    """Read all snapshots, keeping the *latest* row for every project id.

    The monthly Web Robots dumps are cumulative, so the same project shows up
    many times. Later snapshots are processed later (files are sorted by name,
    which embeds the timestamp), so keeping the last row keeps the final state.
    """
    acc: Optional[pd.DataFrame] = None
    n_rows_read = 0
    for label, df in _iter_csvs(path, max_files=max_files):
        n_rows_read += len(df)
        acc = df if acc is None else pd.concat([acc, df], ignore_index=True)
        acc = acc.drop_duplicates("id", keep="last")
        if verbose:
            print(f"  read {label}: {len(df):,} rows -> {len(acc):,} unique ids")
    return acc, n_rows_read


def _parse_json(s):
    if not isinstance(s, str):
        return {}
    try:
        d = json.loads(s)
        return d if isinstance(d, dict) else {}
    except (ValueError, TypeError):
        return {}


def clean(raw: pd.DataFrame, n_rows_read: int | None = None,
          launch_from: int | None = None, launch_to: int | None = None):
    """Turn the raw dump into a modelling table. Returns (df, stats).

    `launch_from` / `launch_to` are optional epoch-second bounds on the launch
    date. They exist because a single cumulative snapshot reaches back to 2009:
    without a bound the cohort spans ~17 years of platform drift. Default None
    keeps every finished project.
    """
    stats = {"n_rows_read": int(n_rows_read or len(raw)), "n_unique_ids": int(len(raw))}
    df = raw.copy()

    # --- keep only finished campaigns (successful / failed) -----------------
    df = df[df["state"].isin(FINAL_STATES)].copy()
    df["y"] = df["state"].map(FINAL_STATES).astype(int)
    stats["n_finished"] = int(len(df))

    # --- JSON columns -------------------------------------------------------
    cat = df["category"].map(_parse_json)
    slug = cat.map(lambda d: str(d.get("slug", "unknown")))
    df["parent_category"] = slug.str.split("/").str[0].str.strip().str.lower()
    df["category"] = cat.map(lambda d: d.get("name", "unknown")).astype(str)
    loc = df["location"].map(_parse_json)
    df["location_type"] = loc.map(lambda d: d.get("type", "unknown")).astype(str)
    df["location"] = loc.map(lambda d: d.get("name", "unknown")).astype(str)

    # --- goal in USD, durations --------------------------------------------
    df["usd_rate"] = pd.to_numeric(df["static_usd_rate"], errors="coerce").fillna(1.0)
    df["goal_usd"] = pd.to_numeric(df["goal"], errors="coerce") * df["usd_rate"]
    for c in ("launched_at", "deadline", "created_at"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["blurb"] = df["blurb"].fillna("").astype(str)
    df["name"] = df["name"].fillna("").astype(str)
    df["country"] = df["country"].fillna("unknown").astype(str)
    df["currency"] = df["currency"].fillna("unknown").astype(str)

    df = df.dropna(subset=["goal_usd", "launched_at", "deadline"])
    df = df[(df["goal_usd"] > 0) & (df["deadline"] > df["launched_at"])]

    # --- optional launch-window cohort filter --------------------------------
    if launch_from is not None:
        df = df[df["launched_at"] >= launch_from]
    if launch_to is not None:
        df = df[df["launched_at"] < launch_to]
    stats["n_after_window"] = int(len(df))

    # --- same de-duplication key as the paper -------------------------------
    before = len(df)
    df = df.drop_duplicates(["name", "blurb", "launched_at", "deadline"], keep="last")
    stats["n_dropped_name_dupes"] = int(before - len(df))

    keep = ["id", "name", "blurb", "goal_usd", "usd_rate", "currency", "country",
            "category", "parent_category", "location", "location_type",
            "launched_at", "deadline", "y"]
    df = df[keep].reset_index(drop=True)
    stats["n_final"] = int(len(df))
    stats["success_rate"] = float(df["y"].mean())
    stats["launch_min"] = pd.to_datetime(df["launched_at"].min(), unit="s").strftime("%Y-%m-%d")
    stats["launch_max"] = pd.to_datetime(df["launched_at"].max(), unit="s").strftime("%Y-%m-%d")
    stats["cohort_from"] = (pd.to_datetime(launch_from, unit="s").strftime("%Y-%m-%d")
                           if launch_from is not None else "all")
    stats["cohort_to"] = (pd.to_datetime(launch_to, unit="s").strftime("%Y-%m-%d")
                          if launch_to is not None else "all")
    stats["goal_usd_median"] = float(df["goal_usd"].median())
    stats["n_categories"] = int(df["category"].nunique())
    stats["n_parent_categories"] = int(df["parent_category"].nunique())
    stats["n_countries"] = int(df["country"].nunique())
    return df, stats


def split_data(df: pd.DataFrame, mode: str = "random", test_size: float = 0.3, seed: int = 42):
    """70/30 split. `temporal` trains on the earliest 70% of launches (harder, more realistic)."""
    if mode == "temporal":
        order = df.sort_values("launched_at").index.to_numpy()
        n_train = int(len(df) * (1 - test_size))
        tr, te = order[:n_train], order[n_train:]
        train, test = df.loc[tr], df.loc[te]
    else:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(df))
        n_train = int(len(df) * (1 - test_size))
        train, test = df.iloc[idx[:n_train]], df.iloc[idx[n_train:]]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def load_dataset(path: str, max_files: Optional[int] = None, cache: Optional[str] = None,
                 launch_from: int | None = None, launch_to: int | None = None):
    """Raw dump -> clean table, with an optional pickle cache."""
    if cache and os.path.exists(cache):
        print(f"Loading cached table {cache}")
        return pd.read_pickle(cache)
    raw, n_read = load_raw(path, max_files=max_files)
    df, stats = clean(raw, n_read, launch_from=launch_from, launch_to=launch_to)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        pd.to_pickle((df, stats), cache)
    return df, stats
