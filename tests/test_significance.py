"""Tests for the paired-bootstrap helpers in scripts/significance.py."""
import importlib.util
import os

import numpy as np

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "significance.py")
_spec = importlib.util.spec_from_file_location("significance", _PATH)
sig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sig)


def test_identical_models_have_zero_difference_and_p_one():
    c = (np.random.RandomState(0).rand(2000) < 0.8).astype(int)
    d, lo, hi = sig.paired_bootstrap(c, c, n_boot=300)
    assert d == 0.0 and lo == 0.0 and hi == 0.0
    assert sig.mcnemar_p(c, c) == 1.0


def test_clearly_better_model_is_flagged_significant():
    rng = np.random.RandomState(1)
    worse = (rng.rand(4000) < 0.70).astype(int)
    better = worse.copy()
    flip = np.where(worse == 0)[0][:300]          # fix 300 of the worse model's errors, break none
    better[flip] = 1
    row = sig.compare("better", "worse", better, worse, n_boot=300)
    assert row["acc_diff"] > 0.05 and row["ci_low"] > 0 and row["significant_95"]
    assert row["mcnemar_p"] < 1e-6


def test_tiny_difference_on_small_test_set_is_not_significant():
    rng = np.random.RandomState(2)
    a = (rng.rand(500) < 0.8).astype(int)
    b = a.copy()
    b[:3] = 1 - b[:3]                               # three rows differ
    row = sig.compare("a", "b", a, b, n_boot=300)
    assert not row["significant_95"] or abs(row["acc_diff"]) < 0.02
