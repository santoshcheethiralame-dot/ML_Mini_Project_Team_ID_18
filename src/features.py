"""Feature engineering for launch-time Kickstarter data.

Leakage control
---------------
* Target encodings and the Naive-Bayes text probability depend on the label.
  On the training set they are computed **out-of-fold** (5 folds): a row's
  value only uses labels from the other folds. Test/inference rows use
  encoders fitted on the full training set.
* Goal statistics per category level (mean log-goal) do not use the label, so
  they are fitted on the full training set. TF-IDF / LDA / LSA are likewise
  unsupervised and fitted on training text only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB

CAT_COLS = ["currency", "country", "category", "parent_category", "location", "location_type"]

# The five data samples of the paper (sentiment / LDA / embedding are the text extras)
SAMPLES = {
    "S1_meta": ["meta"],
    "S2_nb": ["meta", "nb"],
    "S3_nb_sent": ["meta", "nb", "sent"],
    "S4_nb_lda": ["meta", "nb", "lda"],
    "S5_nb_lsa": ["meta", "nb", "lsa"],
}
SAMPLE_LABELS = {
    "S1_meta": "Metadata",
    "S2_nb": "Metadata + NB",
    "S3_nb_sent": "Metadata + NB + Sentiment",
    "S4_nb_lda": "Metadata + NB + LDA",
    "S5_nb_lsa": "Metadata + NB + LSA",
}


class SmoothedTargetEncoder:
    """Replace each level by a shrunk mean of `values` (shrinks rare levels to the prior)."""

    def __init__(self, smoothing: float = 20.0):
        self.smoothing = smoothing

    def fit(self, cats: pd.Series, values: np.ndarray):
        values = np.asarray(values, dtype=float)
        self.prior_ = float(values.mean())
        g = pd.DataFrame({"c": cats.to_numpy(), "v": values}).groupby("c")["v"].agg(["sum", "count"])
        self.map_ = (g["sum"] + self.smoothing * self.prior_) / (g["count"] + self.smoothing)
        return self

    def transform(self, cats: pd.Series) -> np.ndarray:
        return cats.map(self.map_).fillna(self.prior_).to_numpy(dtype=float)


def _sentences(text: str):
    parts = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    return parts or [text]


@dataclass
class FeatureBuilder:
    n_splits: int = 5
    seed: int = 42
    lda_topics: int = 20
    lsa_dims: int = 50
    smoothing: float = 20.0
    min_cat_count: int = 50
    text_min_df: int = 10
    text_max_df: float = 0.35

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _launch_parts(df):
        dt = pd.to_datetime(df["launched_at"], unit="s")
        return dt.dt.month.to_numpy(), dt.dt.dayofweek.to_numpy()

    def _meta_static(self, df) -> pd.DataFrame:
        month, dow = self._launch_parts(df)
        goal = df["goal_usd"].to_numpy(dtype=float)
        return pd.DataFrame({
            "m_log_goal": np.log1p(goal),
            "m_duration_days": (df["deadline"].to_numpy() - df["launched_at"].to_numpy()) / 86400.0,
            "m_launch_month": month,
            "m_launch_dow": dow,
            "m_blurb_len": df["blurb"].str.len().to_numpy(),
            "m_name_len": df["name"].str.len().to_numpy(),
            "m_blurb_words": df["blurb"].str.split().str.len().fillna(0).to_numpy(),
            "m_name_words": df["name"].str.split().str.len().fillna(0).to_numpy(),
        })

    def _meta_goal_cols(self, df, log_goal) -> dict:
        out = {}
        for c in CAT_COLS:
            tg = self.goal_enc_[c].transform(df[c])
            out[f"m_goalmean_{c}"] = tg
            out[f"m_goaldiff_{c}"] = log_goal - tg
        return out

    def _onehots(self, df) -> dict:
        out = {}
        for col, tag in (("parent_category", "par"), ("category", "cat")):
            for i, lvl in enumerate(self.onehot_levels_[col]):
                safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(lvl))[:30]  # LightGBM rejects special chars
                out[f"m_oh_{tag}{i}_{safe}"] = (df[col].to_numpy() == lvl).astype(np.float32)
        return out

    # --------------------------------------------------------------------- fit
    def fit_transform(self, df: pd.DataFrame, y) -> pd.DataFrame:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa: F401  (fail early)

        df = df.reset_index(drop=True)
        y = np.asarray(y).astype(int)
        folds = list(KFold(self.n_splits, shuffle=True, random_state=self.seed).split(df))
        log_goal = np.log1p(df["goal_usd"].to_numpy(dtype=float))

        # category -> parent map and currency rates (used by the demo app)
        self.cat_parent_ = df.groupby("category")["parent_category"].agg(lambda s: s.mode().iat[0]).to_dict()
        self.currency_rate_ = df.groupby("currency")["usd_rate"].median().to_dict()
        self.levels_ = {c: sorted(df[c].unique().tolist()) for c in CAT_COLS}
        self.base_rate_ = float(y.mean())

        # ---- label-free statistics (fitted on all training rows) -------------
        self.goal_enc_ = {c: SmoothedTargetEncoder(self.smoothing).fit(df[c], log_goal) for c in CAT_COLS}
        self.onehot_levels_ = {}
        for c in ("parent_category", "category"):
            vc = df[c].value_counts()
            self.onehot_levels_[c] = sorted(vc[vc >= self.min_cat_count].index.tolist())

        # ---- meta block -----------------------------------------------------
        meta = self._meta_static(df)
        extra = self._meta_goal_cols(df, log_goal)

        # target encodings: out-of-fold on train, full-train encoder kept for test
        self.target_enc_ = {}
        for c in CAT_COLS:
            oof = np.zeros(len(df))
            for tr, va in folds:
                enc = SmoothedTargetEncoder(self.smoothing).fit(df.loc[tr, c], y[tr])
                oof[va] = enc.transform(df.loc[va, c])
            extra[f"m_te_{c}"] = oof
            self.target_enc_[c] = SmoothedTargetEncoder(self.smoothing).fit(df[c], y)
        extra.update(self._onehots(df))
        meta = pd.concat([meta, pd.DataFrame(extra)], axis=1)

        # ---- text vectorisers (unsupervised, train only) ---------------------
        text = df["blurb"].to_numpy()
        self.tfidf_ = TfidfVectorizer(stop_words="english", min_df=self.text_min_df,
                                      max_df=self.text_max_df, sublinear_tf=True)
        T = self.tfidf_.fit_transform(text)

        # NB probability, out-of-fold
        oof_nb = np.zeros(len(df))
        for tr, va in folds:
            nb = MultinomialNB(alpha=1.0).fit(T[tr], y[tr])
            oof_nb[va] = nb.predict_proba(T[va])[:, 1]
        self.nb_ = MultinomialNB(alpha=1.0).fit(T, y)
        nb_block = pd.DataFrame({"nb_prob": oof_nb})

        # sentiment (VADER)
        sent_block = self._sentiment(text)

        # LDA topics
        self.counter_ = CountVectorizer(stop_words="english", min_df=self.text_min_df, max_df=self.text_max_df)
        C = self.counter_.fit_transform(text)
        self.lda_ = LatentDirichletAllocation(n_components=self.lda_topics, learning_method="online",
                                              max_iter=8, batch_size=2048, random_state=self.seed, n_jobs=-1)
        lda_block = pd.DataFrame(self.lda_.fit_transform(C), columns=[f"lda_{i}" for i in range(self.lda_topics)])

        # LSA (dense semantic embedding; stands in for averaged Word2Vec)
        k = min(self.lsa_dims, max(2, T.shape[1] - 1))
        self.lsa_ = TruncatedSVD(n_components=k, random_state=self.seed)
        lsa_block = pd.DataFrame(self.lsa_.fit_transform(T), columns=[f"lsa_{i}" for i in range(k)])

        X = pd.concat([meta, nb_block, sent_block, lda_block, lsa_block], axis=1).astype(np.float32)
        self.columns_ = X.columns.tolist()
        return X

    def _sentiment(self, text) -> pd.DataFrame:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        if not hasattr(self, "_sia"):
            self._sia = SentimentIntensityAnalyzer()
        whole, mean_s = [], []
        for t in text:
            whole.append(self._sia.polarity_scores(t)["compound"])
            mean_s.append(np.mean([self._sia.polarity_scores(s)["compound"] for s in _sentences(t)]))
        return pd.DataFrame({"sent_compound": whole, "sent_mean_sentence": mean_s})

    def __getstate__(self):  # keep pickles small / picklable
        state = self.__dict__.copy()
        state.pop("_sia", None)
        return state

    # --------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame, blocks=("meta", "nb", "sent", "lda", "lsa")) -> pd.DataFrame:
        """Features for unseen rows (test set / demo). Only the requested blocks are computed."""
        df = df.reset_index(drop=True)
        log_goal = np.log1p(df["goal_usd"].to_numpy(dtype=float))
        parts = []
        if "meta" in blocks:
            meta = self._meta_static(df)
            extra = self._meta_goal_cols(df, log_goal)
            for c in CAT_COLS:
                extra[f"m_te_{c}"] = self.target_enc_[c].transform(df[c])
            extra.update(self._onehots(df))
            parts.append(pd.concat([meta, pd.DataFrame(extra)], axis=1))
        text = df["blurb"].to_numpy()
        if {"nb", "lsa"} & set(blocks):
            T = self.tfidf_.transform(text)
        if "nb" in blocks:
            parts.append(pd.DataFrame({"nb_prob": self.nb_.predict_proba(T)[:, 1]}))
        if "sent" in blocks:
            parts.append(self._sentiment(text))
        if "lda" in blocks:
            C = self.counter_.transform(text)
            parts.append(pd.DataFrame(self.lda_.transform(C), columns=[f"lda_{i}" for i in range(self.lda_topics)]))
        if "lsa" in blocks:
            parts.append(pd.DataFrame(self.lsa_.transform(T), columns=[f"lsa_{i}" for i in range(self.lsa_.n_components)]))
        X = pd.concat(parts, axis=1).astype(np.float32)
        # keep the exact training column order for the requested blocks
        return X[[c for c in self.columns_ if c in X.columns]]

    # ---------------------------------------------------------------- selection
    def columns_for(self, blocks) -> list[str]:
        prefix = {"meta": "m_", "nb": "nb_", "sent": "sent_", "lda": "lda_", "lsa": "lsa_"}
        return [c for c in self.columns_ if any(c.startswith(prefix[b]) for b in blocks)]
