"""Sanity tests: leakage guards and an end-to-end smoke test on synthetic data."""
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.data import clean, load_raw, split_data
from src.features import SAMPLES, SmoothedTargetEncoder, FeatureBuilder
from src.models import evaluate, fit_model, make_lgbm
from src.synthetic import make_synthetic, write_synthetic_dump


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    d = tmp_path_factory.mktemp("dump")
    write_synthetic_dump(str(d), n=4000, seed=1)
    raw, n_read = load_raw(str(d), verbose=False)
    df, stats = clean(raw, n_read)
    return df, stats


def test_only_finished_and_deduped(tables):
    df, stats = tables
    assert set(df["y"].unique()) <= {0, 1}
    assert df["id"].is_unique
    assert stats["n_rows_read"] > stats["n_unique_ids"]  # cumulative snapshots were collapsed
    assert stats["n_final"] < stats["n_unique_ids"]       # canceled/live rows removed


def test_no_post_launch_columns(tables):
    df, _ = tables
    banned = {"backers_count", "pledged", "usd_pledged", "staff_pick", "spotlight", "state"}
    assert banned.isdisjoint(df.columns)


def test_oof_encoding_does_not_see_own_label(tables):
    """Run the real FeatureBuilder: a row with a never-seen level must get its fold's prior, not its own label.

    We give one training row a location that no other row has and label it 1. Out-of-fold, that level is
    unseen by the encoder that scores the row, so its target encoding must equal the mean label of the
    *other* folds. The full-train encoder (used for test rows) has seen the label, so it must differ.
    """
    from sklearn.model_selection import KFold

    df, _ = tables
    train, _ = split_data(df, "random")
    train = train.head(1500).copy().reset_index(drop=True)
    i = 7
    train.loc[i, "location"] = "A-LOCATION-NO-OTHER-ROW-HAS"
    train.loc[i, "y"] = 1
    y = train["y"].to_numpy()

    fb = FeatureBuilder(n_splits=5, seed=0)
    X = fb.fit_transform(train, y)

    folds = list(KFold(5, shuffle=True, random_state=0).split(train))
    tr_idx = next(tr for tr, va in folds if i in va)
    fold_prior = y[tr_idx].mean()
    oof_value = float(X.loc[i, "m_te_location"])
    assert oof_value == pytest.approx(fold_prior, rel=1e-4)

    full_value = float(fb.target_enc_["location"].transform(train.loc[[i], "location"])[0])
    assert full_value != pytest.approx(oof_value, rel=1e-4)   # the in-sample encoding has seen the label
    assert full_value > oof_value                              # ...and is pulled toward it (label is 1)


def test_feature_columns_match_between_train_and_test(tables):
    df, _ = tables
    train, test = split_data(df, "random")
    fb = FeatureBuilder()
    Xtr = fb.fit_transform(train, train["y"])
    Xte = fb.transform(test)
    assert list(Xtr.columns) == list(Xte.columns)
    assert not Xtr.isna().any().any() and not Xte.isna().any().any()
    # a single row (demo app path) works with only the needed blocks
    X1 = fb.transform(test.head(1), blocks=SAMPLES["S2_nb"])
    assert list(X1.columns) == fb.columns_for(SAMPLES["S2_nb"])


def test_model_beats_majority_baseline(tables):
    df, _ = tables
    train, test = split_data(df, "random")
    fb = FeatureBuilder()
    Xtr = fb.fit_transform(train, train["y"]); Xte = fb.transform(test)
    cols = fb.columns_for(SAMPLES["S2_nb"])
    m = fit_model(make_lgbm(0, n_estimators=80), Xtr[cols], train["y"].to_numpy())
    r = evaluate(m, Xte[cols], test["y"].to_numpy())
    assert r["AUC"] > 0.65


def _chunked_zip(path, frames):
    """A Web Robots snapshot is one zip holding many chunk CSVs, not one CSV."""
    with zipfile.ZipFile(path, "w") as z:
        for i, f in enumerate(frames):
            z.writestr(f"Kickstarter{i:03d}.csv", f.to_csv(index=False))


def test_max_files_counts_snapshots_not_chunks(tmp_path):
    """--max-files must read every chunk of the first N snapshots.

    Real snapshots are ~85 chunks; counting chunks instead of files silently
    trains on a fraction of one snapshot and still exits cleanly.
    """
    a, b = make_synthetic(60, seed=3), make_synthetic(40, seed=4)
    _chunked_zip(tmp_path / "Kickstarter_2026-01-01.zip",
                 [a.iloc[:20], a.iloc[20:40], a.iloc[40:]])
    _chunked_zip(tmp_path / "Kickstarter_2026-02-01.zip", [b])

    _, n_full = load_raw(str(tmp_path), verbose=False)
    _, n_one = load_raw(str(tmp_path), max_files=1, verbose=False)
    assert n_full == 100
    assert n_one == 60, "max_files=1 must read all 3 chunks of the first snapshot, not just the first chunk"


def test_launch_window_filter_bounds_the_cohort(tmp_path):
    """A cumulative snapshot reaches back years; --launch-from/--launch-to must bound it."""
    d = tmp_path / "dump"
    write_synthetic_dump(str(d), n=800, seed=5)
    raw, n = load_raw(str(d), verbose=False)

    full, s_full = clean(raw, n)
    cut = int(np.median(raw["launched_at"]))
    part, s_part = clean(raw, n, launch_from=cut)
    upper, _ = clean(raw, n, launch_to=cut)

    assert 0 < len(part) < len(full) and 0 < len(upper) < len(full)
    assert part["launched_at"].min() >= cut
    assert upper["launched_at"].max() < cut
    assert s_part["cohort_from"] != "all" and s_part["cohort_to"] == "all"
    assert s_full["cohort_from"] == "all" and s_full["cohort_to"] == "all"
    assert s_full["n_final"] == len(full)


def test_cache_is_rebuilt_when_the_data_changes(tmp_path):
    """A cached table must not be reused after the underlying files change (e.g. a completed re-download)."""
    from src.data import data_fingerprint, load_dataset

    d = tmp_path / "dump"
    write_synthetic_dump(str(d), n=600, seed=11)
    cache = str(tmp_path / "processed" / "table.pkl")

    df1, _ = load_dataset(str(d), cache=cache)
    df1_again, _ = load_dataset(str(d), cache=cache)           # unchanged data -> served from the cache
    assert len(df1_again) == len(df1)

    fp_before = data_fingerprint(str(d))
    extra = make_synthetic(300, seed=12)
    extra["id"] = extra["id"] + 10_000_000                     # new project ids
    _chunked_zip(d / "Kickstarter_9999-01-01.zip", [extra])    # a later snapshot appears
    assert data_fingerprint(str(d)) != fp_before

    df2, stats2 = load_dataset(str(d), cache=cache)
    assert len(df2) > len(df1), "stale cache was reused after the data changed"
    assert stats2["n_final"] == len(df2)


def test_cache_in_old_format_is_ignored(tmp_path):
    """Caches written before fingerprints existed were (df, stats) pairs; they must be rebuilt, not trusted."""
    from src.data import load_dataset

    d = tmp_path / "dump"
    write_synthetic_dump(str(d), n=400, seed=13)
    cache = str(tmp_path / "old.pkl")
    pd.to_pickle((pd.DataFrame({"id": [1]}), {"n_final": 1}), cache)
    df, stats = load_dataset(str(d), cache=cache)
    assert len(df) > 1 and stats["n_final"] == len(df)
