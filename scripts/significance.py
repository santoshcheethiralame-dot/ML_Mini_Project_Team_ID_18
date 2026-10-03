"""Are the differences in the report bigger than test-set noise?

The ablation in the write-up reports LightGBM accuracy averaged over seeds (std ~0.001), but the seeds
only change the model, not the 14k test rows. The sampling error of accuracy on n test rows is about
sqrt(p(1-p)/n) ~ 0.3 points, so small gaps (e.g. +0.5) need a paired test, not a seed std.

This script refits the models on the same split as src/train.py and, for each comparison, resamples the
test rows with replacement (paired: both models are scored on the same resample) and reports the
accuracy difference with a 95% bootstrap confidence interval and an exact McNemar p-value.

Usage
-----
    python scripts/significance.py --data data/raw --launch-from 2024-07-01 --launch-to 2026-07-01
    python scripts/significance.py --synthetic 6000 --fast        # smoke test only, NOT results
Writes results/significance.json and results/significance.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_dataset, split_data  # noqa: E402
from src.features import SAMPLES, SAMPLE_LABELS, FeatureBuilder  # noqa: E402
from src.models import DEFAULT_LGBM, fit_model, get_models, make_lgbm  # noqa: E402
from src.train import _epoch  # noqa: E402


def paired_bootstrap(correct_a: np.ndarray, correct_b: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Accuracy(A) - Accuracy(B) with a 95% percentile CI, resampling test rows jointly."""
    correct_a, correct_b = np.asarray(correct_a, float), np.asarray(correct_b, float)
    n = len(correct_a)
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        diffs[i] = correct_a[idx].mean() - correct_b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(correct_a.mean() - correct_b.mean()), float(lo), float(hi)


def mcnemar_p(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Exact two-sided McNemar test on the rows where exactly one model is right."""
    a_only = int(np.sum((correct_a == 1) & (correct_b == 0)))
    b_only = int(np.sum((correct_a == 0) & (correct_b == 1)))
    if a_only + b_only == 0:
        return 1.0
    return float(binomtest(a_only, a_only + b_only, 0.5).pvalue)


def compare(name_a, name_b, ca, cb, n_boot):
    d, lo, hi = paired_bootstrap(ca, cb, n_boot)
    return {"A": name_a, "B": name_b, "acc_diff": d, "ci_low": lo, "ci_high": hi,
            "mcnemar_p": mcnemar_p(ca, cb), "significant_95": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--synthetic", type=int, default=0, help="N synthetic rows (smoke test only)")
    ap.add_argument("--launch-from", default=None)
    ap.add_argument("--launch-to", default=None)
    ap.add_argument("--split", choices=["random", "temporal"], default="random")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-models", action="store_true", help="only run the text-feature ablation")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    path = args.data
    if args.synthetic:
        from src.synthetic import write_synthetic_dump
        path = write_synthetic_dump("data/synthetic", args.synthetic)
        print("!! SYNTHETIC data - output is for testing the script only !!")

    df, _ = load_dataset(path, launch_from=_epoch(args.launch_from), launch_to=_epoch(args.launch_to))
    train, test = split_data(df, args.split, 0.3, args.seed)
    ytr, yte = train["y"].to_numpy(), test["y"].to_numpy()
    fb = FeatureBuilder(seed=args.seed)
    Xtr = fb.fit_transform(train, ytr)
    Xte = fb.transform(test)
    print(f"{len(train):,} train / {len(test):,} test rows; test SE of accuracy ~ "
          f"{np.sqrt(0.8 * 0.2 / len(test)) * 100:.2f} points")

    params = dict(DEFAULT_LGBM)
    if args.fast:
        params["n_estimators"] = 60

    # --- 1) text-feature ablation: every sample vs Metadata + NB (single fixed seed) -------------
    correct = {}
    for key, blocks in SAMPLES.items():
        cols = fb.columns_for(blocks)
        m = fit_model(make_lgbm(args.seed, **params), Xtr[cols], ytr)
        correct[key] = (m.predict(Xte[cols]) == yte).astype(int)
    rows = []
    base = "S2_nb"
    rows.append(compare(SAMPLE_LABELS["S2_nb"], SAMPLE_LABELS["S1_meta"], correct["S2_nb"], correct["S1_meta"], args.boot))
    for key in ("S3_nb_sent", "S4_nb_lda", "S5_nb_lsa"):
        rows.append(compare(SAMPLE_LABELS[key], SAMPLE_LABELS[base], correct[key], correct[base], args.boot))

    # --- 2) best model vs the others on Metadata + NB ----------------------------------------------
    if not args.skip_models:
        cols2 = fb.columns_for(SAMPLES["S2_nb"])
        preds = {}
        for name, model in get_models(args.seed, args.fast, params).items():
            fit_model(model, Xtr[cols2], ytr)
            preds[name] = (model.predict(Xte[cols2]) == yte).astype(int)
        for name in preds:
            if name != "G. Boosting":
                rows.append(compare("G. Boosting", name, preds["G. Boosting"], preds[name], args.boot))

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(table.round(4).to_string(index=False))

    os.makedirs(args.out, exist_ok=True)
    if args.synthetic:
        print("synthetic run: not writing results files")
        return
    table.to_csv(os.path.join(args.out, "significance.csv"), index=False)
    with open(os.path.join(args.out, "significance.json"), "w") as f:
        json.dump({"n_test": int(len(test)), "bootstrap_resamples": args.boot, "seed": args.seed,
                   "comparisons": rows}, f, indent=2)
    print(f"wrote {args.out}/significance.csv and .json")


if __name__ == "__main__":
    main()
