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


def test_oof_encoding_does_not_see_own_label():
    """A row whose category is unique must NOT get its own label back as its encoding."""
    cats = pd.Series(["a"] * 50 + ["b"] * 50 + ["rare"])
    y = np.array([1] * 50 + [0] * 50 + [1])
    insample = SmoothedTargetEncoder(smoothing=0.0).fit(cats, y).transform(cats)
    assert insample[-1] == 1.0  # in-sample encoding leaks the label...
    fb = FeatureBuilder(n_splits=5, seed=0)
    from sklearn.model_selection import KFold
    folds = list(KFold(5, shuffle=True, random_state=0).split(cats))
    oof = np.zeros(len(cats))
    for tr, va in folds:
        oof[va] = SmoothedTargetEncoder(smoothing=0.0).fit(cats.iloc[tr], y[tr]).transform(cats.iloc[va])
    # ...whereas out-of-fold falls back to the prior for the unseen level
    assert oof[-1] == pytest.approx(y[folds[[len(va) and (len(cats) - 1) in va for _, va in folds].index(True)][0]].mean())


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
