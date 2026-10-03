"""End-to-end experiment: data -> features -> models -> tables, figures, saved model.

Usage
-----
    python -m src.train --data data/raw                # real Web Robots dump
    python -m src.train --synthetic 20000 --fast       # pipeline smoke test (NOT results)
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

from .data import load_dataset, split_data
from .features import SAMPLES, SAMPLE_LABELS, FeatureBuilder
from .models import DEFAULT_LGBM, evaluate, fit_model, get_models, make_lgbm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*converge.*")

GREEN, DARK, GREY = "#05A862", "#0B3D2E", "#9AA5A0"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _epoch(s):
    """'YYYY-MM-DD' -> epoch seconds at 00:00 UTC (naive pandas timestamps are UTC). None passes through."""
    return int(pd.Timestamp(s).timestamp()) if s else None


# ------------------------------------------------------------------ experiments
def run_model_table(Xtr, ytr, Xte, yte, cols, seed, fast, lgbm_params):
    rows = []
    for name, model in get_models(seed, fast, lgbm_params).items():
        t = time.time()
        fit_model(model, Xtr[cols], ytr)
        m = evaluate(model, Xte[cols], yte)
        m.update(model=name, seconds=round(time.time() - t, 1))
        rows.append(m)
        log(f"    {name:12s} acc={m['A']:.3f} f1={m['F1']:.3f} auc={m['AUC']:.3f} ({m['seconds']}s)")
    return sorted(rows, key=lambda r: -r["A"])


def run_ablation(fb, Xtr, ytr, Xte, yte, seeds, fast, lgbm_params):
    res = {}
    for key, blocks in SAMPLES.items():
        cols = fb.columns_for(blocks)
        accs, f1s = [], []
        for s in range(seeds):
            params = dict(lgbm_params)
            if fast:
                params["n_estimators"] = 60
            m = fit_model(make_lgbm(s, **params), Xtr[cols], ytr)
            r = evaluate(m, Xte[cols], yte)
            accs.append(r["A"]); f1s.append(r["F1"])
        res[key] = {"label": SAMPLE_LABELS[key], "n_features": len(cols), "accs": accs, "f1s": f1s,
                    "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                    "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
        log(f"    {SAMPLE_LABELS[key]:28s} acc={res[key]['acc_mean']:.4f}±{res[key]['acc_std']:.4f} "
            f"f1={res[key]['f1_mean']:.4f}  ({len(cols)} feats)")
    return res


def tune_lgbm(Xtr, ytr, fast, seed):
    """Small randomized search (3-fold CV on a training subsample) - never touches the test set."""
    rng = np.random.RandomState(seed)
    n = min(len(Xtr), 15000 if fast else 40000)
    idx = rng.choice(len(Xtr), n, replace=False)
    grid = {"num_leaves": [31, 63, 127, 255], "learning_rate": [0.03, 0.05, 0.1],
            "n_estimators": [200, 400, 600], "min_child_samples": [10, 30, 60],
            "colsample_bytree": [0.3, 0.5, 0.8]}
    search = RandomizedSearchCV(make_lgbm(seed), grid, n_iter=4 if fast else 12, cv=3,
                                scoring="accuracy", random_state=seed, n_jobs=1)
    search.fit(Xtr.iloc[idx], ytr[idx])
    best = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, np.floating) else v)
            for k, v in search.best_params_.items()}
    log(f"    best params {best}  (cv acc {search.best_score_:.4f})")
    return {**DEFAULT_LGBM, **best}


# ---------------------------------------------------------------------- figures
def fig_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center",
                    color="white" if cm[i][j] > np.max(cm) / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Failed", "Successful"]); ax.set_yticks([0, 1], ["Failed", "Successful"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Gradient boosting - test set")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_importance(top, path):
    names = [n.replace("m_", "").replace("_", " ") for n, _ in top][::-1]
    vals = [v for _, v in top][::-1]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(names, vals, color=GREEN)
    ax.set_xlabel("Total gain"); ax.set_title("Top features (LightGBM, Metadata + NB)")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_models(t1, t2, path):
    names = [r["model"] for r in t2]
    a1 = {r["model"]: r["A"] for r in t1}; a2 = {r["model"]: r["A"] for r in t2}
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x - w / 2, [a1[n] for n in names], w, label="Metadata", color=GREY)
    ax.bar(x + w / 2, [a2[n] for n in names], w, label="Metadata + NB", color=GREEN)
    lo = min(list(a1.values()) + list(a2.values()))
    ax.set_ylim(max(0, lo - 0.05), 1.0); ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel("Test accuracy"); ax.legend(frameon=False); ax.set_title("Model comparison")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_ablation(abl, path):
    labels = [v["label"] for v in abl.values()][::-1]
    data = [v["accs"] for v in abl.values()][::-1]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bp = ax.boxplot(data, vert=False, patch_artist=True, tick_labels=labels)
    for b in bp["boxes"]:
        b.set(facecolor=GREEN, alpha=0.7)
    ax.set_xlabel("Test accuracy (LightGBM, across seeds)"); ax.set_title("Text-feature ablation")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_eda(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    pc = df.groupby("parent_category")["y"].agg(["mean", "size"]).sort_values("mean")
    pc = pc[pc["size"] >= 50]
    axes[0].barh(pc.index, pc["mean"], color=GREEN)
    axes[0].set_xlabel("Success rate"); axes[0].set_title("By parent category")
    bins = pd.qcut(df["goal_usd"], 10, duplicates="drop")
    g = df.groupby(bins, observed=True).agg(rate=("y", "mean"), goal=("goal_usd", "median"))
    axes[1].plot(g["goal"], g["rate"], marker="o", color=DARK)
    axes[1].set_xscale("log"); axes[1].set_xlabel("Funding goal (USD, decile median, log)")
    axes[1].set_ylabel("Success rate"); axes[1].set_title("By funding goal")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw", help="folder/zip/csv of the Web Robots dump")
    ap.add_argument("--synthetic", type=int, default=0, help="generate N synthetic rows instead (smoke test only)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--max-files", type=int, default=None, help="use only the first N snapshot files (dev)")
    ap.add_argument("--launch-from", default=None, help="cohort start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--launch-to", default=None, help="cohort end date YYYY-MM-DD (exclusive)")
    ap.add_argument("--split", choices=["random", "temporal"], default="random")
    ap.add_argument("--seeds", type=int, default=5, help="seeds for the text-feature ablation")
    ap.add_argument("--tune", action="store_true", help="random-search LightGBM hyper-parameters on train CV")
    ap.add_argument("--fast", action="store_true", help="small/quick models (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(args.models_dir, exist_ok=True)
    t0 = time.time()

    data_source = "real"
    path = args.data
    if args.synthetic:
        from .synthetic import write_synthetic_dump
        path = write_synthetic_dump("data/synthetic", args.synthetic)
        data_source = "synthetic"
        log("!! using SYNTHETIC data - results are for pipeline testing only !!")

    log("1/6 loading + cleaning data")
    launch_from, launch_to = _epoch(args.launch_from), _epoch(args.launch_to)
    win = f"{args.launch_from or 'min'}_{args.launch_to or 'max'}"
    cache = os.path.join("data", "processed",
                         f"table_{data_source}_{args.synthetic or 'all'}_{args.max_files or 'all'}_{win}.pkl")
    df, stats = load_dataset(path, max_files=args.max_files, cache=cache,
                             launch_from=launch_from, launch_to=launch_to)
    log(f"    {stats['n_final']:,} finished projects, success rate {stats['success_rate']:.3f}")
    log(f"    cohort: launched {stats['launch_min']} to {stats['launch_max']} "
        f"(filter {stats['cohort_from']} to {stats['cohort_to']})")

    train, test = split_data(df, args.split, 0.3, args.seed)
    ytr, yte = train["y"].to_numpy(), test["y"].to_numpy()
    stats.update(n_train=len(train), n_test=len(test), split=args.split,
                 train_success_rate=float(ytr.mean()), test_success_rate=float(yte.mean()),
                 baseline_acc=float(max(yte.mean(), 1 - yte.mean())))
    fig_eda(train, os.path.join(fig_dir, "eda.png"))

    log("2/6 building features (out-of-fold encodings)")
    fb = FeatureBuilder(seed=args.seed)
    Xtr = fb.fit_transform(train, ytr)
    Xte = fb.transform(test)
    log(f"    {Xtr.shape[1]} total features; sample 1 uses {len(fb.columns_for(SAMPLES['S1_meta']))}, "
        f"sample 2 uses {len(fb.columns_for(SAMPLES['S2_nb']))}")

    lgbm_params = dict(DEFAULT_LGBM)
    if args.tune:
        log("3/6 tuning LightGBM (train CV only)")
        lgbm_params = tune_lgbm(Xtr[fb.columns_for(SAMPLES['S2_nb'])], ytr, args.fast, args.seed)
    else:
        log("3/6 tuning skipped (using default LightGBM params; pass --tune to search)")

    log("4/6 model comparison, Sample 1 (metadata)")
    t1 = run_model_table(Xtr, ytr, Xte, yte, fb.columns_for(SAMPLES["S1_meta"]), args.seed, args.fast, lgbm_params)
    log("    model comparison, Sample 2 (metadata + NB)")
    cols2 = fb.columns_for(SAMPLES["S2_nb"])
    t2 = run_model_table(Xtr, ytr, Xte, yte, cols2, args.seed, args.fast, lgbm_params)

    log(f"5/6 text-feature ablation ({args.seeds} seeds)")
    abl = run_ablation(fb, Xtr, ytr, Xte, yte, args.seeds, args.fast, lgbm_params)

    log("6/6 final model, figures, saving")
    final_params = dict(lgbm_params)
    if args.fast:
        final_params["n_estimators"] = 60
    final = fit_model(make_lgbm(args.seed, **final_params), Xtr[cols2], ytr)
    pred = final.predict(Xte[cols2])
    cm = confusion_matrix(yte, pred, labels=[0, 1]).tolist()
    gain = pd.Series(final.booster_.feature_importance("gain"), index=cols2).sort_values(ascending=False)
    top = [(k, float(v)) for k, v in gain.head(15).items()]

    fig_confusion(cm, os.path.join(fig_dir, "confusion.png"))
    fig_importance(top, os.path.join(fig_dir, "importance.png"))
    fig_models(t1, t2, os.path.join(fig_dir, "models.png"))
    fig_ablation(abl, os.path.join(fig_dir, "ablation.png"))

    metrics = {
        "meta": {"data_source": data_source, "created": datetime.now().isoformat(timespec="seconds"),
                 "args": vars(args), "stats": stats, "n_features_total": int(Xtr.shape[1]),
                 "runtime_min": round((time.time() - t0) / 60, 1)},
        "table_sample1": t1, "table_sample2": t2, "ablation": abl,
        "confusion": cm, "top_features": top, "lgbm_params": lgbm_params,
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(t1).to_csv(os.path.join(args.out, "table_sample1.csv"), index=False)
    pd.DataFrame(t2).to_csv(os.path.join(args.out, "table_sample2.csv"), index=False)

    joblib.dump({"fb": fb, "model": final, "columns": cols2, "data_source": data_source,
                 "test_accuracy": float((pred == yte).mean())},
                os.path.join(args.models_dir, "bundle.joblib"), compress=3)
    log(f"done in {(time.time() - t0) / 60:.1f} min -> {args.out}/metrics.json, {args.models_dir}/bundle.joblib")


if __name__ == "__main__":
    main()
