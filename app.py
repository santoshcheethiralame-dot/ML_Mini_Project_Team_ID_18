"""Streamlit demo: predict whether a Kickstarter campaign will reach its goal.

Run:  streamlit run app.py      (after `python -m src.train ...` has created models/bundle.joblib)
"""
import datetime as dt
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import SAMPLES

st.set_page_config(page_title="Kickstarter Success Predictor", page_icon="🚀", layout="wide")


@st.cache_resource
def load_bundle(path="models/bundle.joblib"):
    return joblib.load(path)


if not os.path.exists("models/bundle.joblib"):
    st.error("No trained model found. Run `python -m src.train --data data/raw` first.")
    st.stop()

B = load_bundle()
fb, model, cols = B["fb"], B["model"], B["columns"]

st.title("🚀 Kickstarter Success Predictor")
st.caption("Predicts from launch-time information only (goal, category, country, blurb, dates). "
           "Model: gradient boosting on metadata + Naive-Bayes text feature.")
if B.get("data_source") == "synthetic":
    st.warning("This model was trained on SYNTHETIC test data - predictions are meaningless.")

# ---------------------------------------------------------------- inputs
left, right = st.columns([1, 1])
with left:
    name = st.text_input("Project name", "Pocket Dungeon: a tiny tabletop adventure")
    blurb = st.text_area("Short blurb (the one-liner under the title)",
                         "A handcrafted solo card game you can play anywhere, with original art by independent illustrators.",
                         height=100)
    cats = sorted(fb.cat_parent_)
    category = st.selectbox("Category", cats, index=0)
    parent = fb.cat_parent_[category]
    st.caption(f"Parent category: **{parent}**")
    currency = st.selectbox("Currency", sorted(fb.currency_rate_), index=sorted(fb.currency_rate_).index("USD") if "USD" in fb.currency_rate_ else 0)
    goal_local = st.number_input(f"Funding goal ({currency})", min_value=1.0, value=5000.0, step=500.0)
with right:
    country = st.selectbox("Country", fb.levels_["country"], index=fb.levels_["country"].index("US") if "US" in fb.levels_["country"] else 0)
    location = st.text_input("Creator city (optional, as shown on Kickstarter)", "")
    location_type = st.selectbox("Location type", fb.levels_["location_type"])
    launch = st.date_input("Launch date", dt.date.today() + dt.timedelta(days=14))
    duration = st.slider("Campaign length (days)", 1, 60, 30)

rate = fb.currency_rate_.get(currency, 1.0)
goal_usd = goal_local * rate


def build_row(goal_usd_value):
    launched = int(dt.datetime.combine(launch, dt.time(12)).timestamp())
    return pd.DataFrame([{
        "name": name, "blurb": blurb, "goal_usd": goal_usd_value, "usd_rate": rate, "currency": currency,
        "country": country, "category": category, "parent_category": parent,
        "location": location.strip() or "unknown", "location_type": location_type,
        "launched_at": launched, "deadline": launched + duration * 86400,
    }])


def predict(goal_value):
    X = fb.transform(build_row(goal_value), blocks=SAMPLES["S2_nb"])[cols]
    return float(model.predict_proba(X)[0, 1]), X


p, X = predict(goal_usd)
st.divider()
c1, c2 = st.columns([1, 2])
with c1:
    st.metric("Estimated chance of success", f"{p:.0%}")
    st.progress(min(max(p, 0.0), 1.0))
    verdict = "Likely to succeed" if p >= 0.5 else "Likely to fall short"
    st.subheader(verdict)
    st.caption(f"Goal in USD: ${goal_usd:,.0f}. The model was trained with class-balanced weights, "
               f"so 50% is the decision threshold (not the dataset's base rate).")
    st.caption(f"Text-only (blurb) signal from the Naive-Bayes feature: {X['nb_prob'].iloc[0]:.0%}")

with c2:
    st.markdown("**What-if: how does the funding goal change the prediction?**")
    mults = np.array([0.1, 0.25, 0.5, 1, 2, 4, 10])
    sweep = pd.DataFrame({"Goal (USD)": goal_usd * mults,
                          "P(success)": [predict(goal_usd * m)[0] for m in mults]}).set_index("Goal (USD)")
    sweep.index = [f"${v:,.0f}" for v in sweep.index]
    st.line_chart(sweep)

st.markdown("**Why this prediction? (top feature contributions, log-odds)**")
contrib = model.predict(X, pred_contrib=True)[0][:-1]
s = pd.Series(contrib, index=cols)
top = s.reindex(s.abs().sort_values(ascending=False).head(8).index).iloc[::-1]
top.index = [i.replace("m_", "").replace("_", " ") for i in top.index]
st.bar_chart(top)
st.caption("Positive bars push toward success, negative toward failure (LightGBM TreeSHAP values).")
