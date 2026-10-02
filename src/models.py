"""Model zoo and evaluation (mirrors the models compared in the paper)."""
from __future__ import annotations

import inspect

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

DEFAULT_LGBM = dict(
    n_estimators=500, learning_rate=0.05, num_leaves=127, min_child_samples=30,
    subsample=0.75, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
)


class ThresholdRegressor(BaseEstimator, ClassifierMixin):
    """Linear probability model: regress y in {0,1}, predict 1 when output >= 0.5 (as in the paper)."""

    def __init__(self, reg=None):
        self.reg = reg

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array([0, 1])
        self.reg_ = self.reg.fit(X, y, sample_weight=sample_weight)
        return self

    def decision_function(self, X):
        return self.reg_.predict(X)

    def predict_proba(self, X):
        p = np.clip(self.reg_.predict(X), 0.0, 1.0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.reg_.predict(X) >= 0.5).astype(int)


def make_lgbm(seed=42, **overrides):
    params = {**DEFAULT_LGBM, **overrides}
    return LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **params)


def get_models(seed: int = 42, fast: bool = False, lgbm_params: dict | None = None) -> dict:
    lg = dict(lgbm_params or {})
    if fast:
        lg["n_estimators"] = 60
    rf_trees = 40 if fast else 200
    return {
        "G. Boosting": make_lgbm(seed, **lg),
        "R. Forest": RandomForestClassifier(n_estimators=rf_trees, max_depth=55, min_samples_leaf=10,
                                            max_features="sqrt", n_jobs=-1, random_state=seed),
        "Neural Net": make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(25,), batch_size=256, early_stopping=True,
            max_iter=10 if fast else 30, random_state=seed)),
        "SVM": make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual=False, max_iter=2000)),
        "Logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500)),
        "Ridge": make_pipeline(StandardScaler(), ThresholdRegressor(Ridge(alpha=1.0))),
        "OLS": make_pipeline(StandardScaler(), ThresholdRegressor(LinearRegression())),
        "Lasso": make_pipeline(StandardScaler(), ThresholdRegressor(Lasso(alpha=1e-4, max_iter=2000))),
    }


def fit_model(model, X, y):
    """Fit with balanced sample weights whenever the estimator supports them."""
    w = compute_sample_weight("balanced", y)
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if "sample_weight" in inspect.signature(est.fit).parameters:
        if hasattr(model, "steps"):
            model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
        else:
            model.fit(X, y, sample_weight=w)
    else:
        model.fit(X, y)
    return model


def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {
        "A": float(accuracy_score(y, pred)),
        "F1": float(f[1]),
        "P1": float(p[1]), "P0": float(p[0]),
        "R1": float(r[1]), "R0": float(r[0]),
        "AUC": float(roc_auc_score(y, _scores(model, X))),
    }
