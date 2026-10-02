"""Streamlit demo: predict whether a Kickstarter campaign will reach its goal.

Run:  streamlit run app.py      (after `python -m src.train ...` has created models/bundle.joblib)

Design notes
------------
Layered on purpose. The default surface answers "will it work and why" in a few
seconds; everything methodological sits behind one expander so a viva can go as
deep as the examiner wants without burying the headline.

All custom visuals are built from `ks-`-prefixed HTML/CSS scoped to this file.
Nothing here targets Streamlit's internal DOM, so a Streamlit upgrade cannot
silently break the layout. Colours come from `.streamlit/config.toml` instead of
CSS overrides.
"""
import datetime as dt
import html
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import SAMPLES

st.set_page_config(
    page_title="Kickstarter Success Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
.ks-wrap{max-width:1180px}
.ks-eyebrow{font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;
  color:#8b949e;margin:0 0 .35rem}
.ks-h1{font-size:2.1rem;font-weight:700;line-height:1.15;margin:0 0 .4rem}
.ks-sub{color:#9aa4b2;font-size:.95rem;margin:0 0 1rem}
.ks-pills{display:flex;flex-wrap:wrap;gap:.4rem;margin:.6rem 0 1.2rem}
.ks-pill{background:#161b22;border:1px solid #2a313c;border-radius:999px;
  padding:.28rem .7rem;font-size:.76rem;color:#9aa4b2}
.ks-pill b{color:#e6edf3;font-weight:600}

.ks-card{background:#161b22;border:1px solid #2a313c;border-radius:14px;
  padding:1.1rem 1.2rem;margin-bottom:.9rem}
.ks-card-h{font-size:.74rem;letter-spacing:.12em;text-transform:uppercase;
  color:#8b949e;margin:0 0 .8rem}

.ks-hero{display:flex;align-items:baseline;gap:.7rem;flex-wrap:wrap}
.ks-big{font-size:3.4rem;font-weight:700;line-height:1}
.ks-verdict{font-size:1.05rem;font-weight:600}
.ks-good{color:#3ddc97}.ks-mid{color:#e3b341}.ks-bad{color:#f85149}

.ks-track{position:relative;height:12px;border-radius:999px;background:#21262d;
  margin:1.1rem 0 .35rem;overflow:visible}
.ks-fill{height:12px;border-radius:999px}
.ks-tick{position:absolute;top:-5px;width:2px;height:22px;background:#8b949e;opacity:.85}
.ks-legend{display:flex;gap:1.1rem;flex-wrap:wrap;font-size:.74rem;color:#8b949e;
  margin-top:.5rem}
.ks-legend i{display:inline-block;width:9px;height:9px;border-radius:2px;
  margin-right:.35rem;vertical-align:middle}

.ks-row{display:grid;grid-template-columns:132px 1fr 62px;align-items:center;
  gap:.7rem;padding:.24rem 0}
.ks-row-lab{font-size:.8rem;color:#c9d1d9;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap}
.ks-track-s{position:relative;height:9px;border-radius:999px;background:#21262d}
.ks-fill-s{height:9px;border-radius:999px}
.ks-val{font-size:.8rem;text-align:right;color:#c9d1d9;
  font-variant-numeric:tabular-nums}
.ks-row-cur .ks-row-lab{color:#3ddc97;font-weight:600}
.ks-row-cur .ks-track-s{box-shadow:0 0 0 1px #3ddc97}

.ks-axis{position:absolute;left:50%;top:-4px;width:1px;height:17px;background:#484f58}
.ks-pos{position:absolute;left:50%;background:#3ddc97;border-radius:0 4px 4px 0}
.ks-neg{position:absolute;right:50%;background:#f85149;border-radius:4px 0 0 4px}

.ks-facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));
  gap:.7rem;margin-top:.2rem}
.ks-fact{background:#0d1117;border:1px solid #2a313c;border-radius:10px;padding:.6rem .7rem}
.ks-fact span{display:block;font-size:.7rem;color:#8b949e;margin-bottom:.2rem}
.ks-fact b{font-size:.98rem;font-weight:600;font-variant-numeric:tabular-nums}
.ks-note{font-size:.8rem;color:#8b949e;margin:.7rem 0 0}
table.ks-tbl{width:100%;border-collapse:collapse;font-size:.83rem;margin-top:.4rem}
table.ks-tbl th{text-align:left;color:#8b949e;font-weight:600;font-size:.74rem;
  text-transform:uppercase;letter-spacing:.07em;padding:.4rem .5rem;
  border-bottom:1px solid #2a313c}
table.ks-tbl td{padding:.4rem .5rem;border-bottom:1px solid #1c2128;
  font-variant-numeric:tabular-nums}
table.ks-tbl td.pos{color:#3ddc97}table.ks-tbl td.neg{color:#f85149}
.ks-preset-hint{font-size:.76rem;color:#8b949e;margin:.15rem 0 .55rem}
</style>
"""

PRESETS = {
    "Tabletop game": dict(
        name="Pocket Dungeon: a tiny tabletop adventure",
        blurb="A handcrafted solo card game you can play anywhere, with original art "
              "by independent illustrators.",
        category="Tabletop Games", country="US", currency="USD",
        goal=5000.0, duration=30, location="Portland, OR", location_type="LocalAdmin",
        launch_offset=14,
    ),
    "Ambitious hardware": dict(
        name="AeroPress Pro: espresso, automated",
        blurb="A countertop machine with built-in grinder and scale. Precision brew, "
              "one button, dishwasher safe.",
        category="Gadgets", country="US", currency="USD",
        goal=120000.0, duration=60, location="Austin, TX", location_type="LocalAdmin",
        launch_offset=21,
    ),
    "Documentary film": dict(
        name="The Last Ferry Home",
        blurb="A short documentary about the final ferry route on a shrinking island, "
              "filmed over one winter.",
        category="Documentary", country="GB", currency="GBP",
        goal=30000.0, duration=30, location="Cornwall", location_type="County",
        launch_offset=10,
    ),
}

DEFAULTS = dict(
    name=PRESETS["Tabletop game"]["name"],
    blurb=PRESETS["Tabletop game"]["blurb"],
    category="Tabletop Games", country="US", currency="USD",
    goal=5000.0, duration=30, location="Portland, OR", location_type="LocalAdmin",
    launch_offset=14,
)


def _pct_class(p):
    return "ks-good" if p >= 0.5 else ("ks-mid" if p >= 0.35 else "ks-bad")


def _fill_class(p):
    return "#3ddc97" if p >= 0.5 else ("#e3b341" if p >= 0.35 else "#f85149")


@st.cache_resource
def load_bundle(path="models/bundle.joblib"):
    return joblib.load(path)


@st.cache_data
def load_meta(path="results/metrics.json"):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


st.markdown(CSS, unsafe_allow_html=True)

if not os.path.exists("models/bundle.joblib"):
    st.error("No trained model found. Run `python -m src.train --data data/raw` first.")
    st.stop()

B = load_bundle()
META = load_meta()
fb, model, cols = B["fb"], B["model"], B["columns"]
STATS = META.get("meta", {}).get("stats", {})
ABL = META.get("ablation", {})

def _wkey(key):
    return f"in_{key}"


def _apply(cfg):
    """Push a whole scenario into widget state, then the caller reruns."""
    for k, v in cfg.items():
        st.session_state[_wkey(k)] = (
            dt.date.today() + dt.timedelta(days=v) if k == "launch_offset" else v
        )


if "_ks_ready" not in st.session_state:
    _apply(DEFAULTS)
    st.session_state["_ks_ready"] = True

# ------------------------------------------------------------------ header
st.markdown(
    f'<p class="ks-eyebrow">UE24CS352A &middot; Machine Learning Mini-Project</p>'
    f'<h1 class="ks-h1">Will this Kickstarter reach its goal?</h1>'
    f'<p class="ks-sub">A prediction from launch-time information only &mdash; goal, category, '
    f'country, dates and the one-line blurb. No pledges, no comments, no outcomes.</p>',
    unsafe_allow_html=True,
)

if B.get("data_source") == "synthetic":
    st.error("This model was trained on SYNTHETIC data. Predictions are meaningless.")

if STATS:
    pills = [
        ("trained on", f"{STATS.get('n_final', 0):,} real projects"),
        ("cohort", f"{STATS.get('launch_min', '?')} to {STATS.get('launch_max', '?')}"),
        ("held-out accuracy", f"{B.get('test_accuracy', 0):.1%}"),
        ("majority baseline", f"{STATS.get('baseline_acc', 0):.1%}"),
        ("features", f"{len(cols)}"),
    ]
    st.markdown(
        '<div class="ks-pills">'
        + "".join(f'<span class="ks-pill">{k} <b>{html.escape(str(v))}</b></span>' for k, v in pills)
        + "</div>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ presets
st.markdown('<p class="ks-preset-hint">Start from an example</p>', unsafe_allow_html=True)
pc = st.columns([1, 1, 1, 1.2])
for i, (label, cfg) in enumerate(PRESETS.items()):
    if pc[i].button(label, key=f"preset_{i}", width="stretch"):
        _apply(cfg)
        st.rerun()
if pc[3].button("Reset", key="preset_reset", width="stretch"):
    _apply(DEFAULTS)
    st.rerun()

st.divider()

# ------------------------------------------------------------------ inputs
left, right = st.columns(2, gap="large")

with left:
    st.markdown('<p class="ks-card-h">The project</p>', unsafe_allow_html=True)
    name = st.text_input("Project name", key="in_name")
    blurb = st.text_area("One-line blurb", key="in_blurb", height=104)
    cats = sorted(fb.cat_parent_)
    category = st.selectbox("Category", cats, key="in_category")
    parent = fb.cat_parent_[category]

with right:
    st.markdown('<p class="ks-card-h">The campaign</p>', unsafe_allow_html=True)
    countries = sorted(fb.levels_["country"])
    country = st.selectbox("Country", countries, key="in_country")
    currencies = sorted(fb.currency_rate_)
    c1, c2 = st.columns(2)
    currency = c1.selectbox("Currency", currencies, key="in_currency")
    goal_local = c2.number_input(
        f"Goal ({currency})", min_value=1.0, key="in_goal", step=100.0
    )
    c3, c4 = st.columns(2)
    launch = c3.date_input(
        "Launch date",
        key="in_launch",
        min_value=dt.date.today(),
        max_value=dt.date.today() + dt.timedelta(days=365),
    )
    duration = c4.slider("Length (days)", 1, 60, key="in_duration")

    with st.expander("Location fields (usually leave blank)"):
        location = st.text_input("Creator location as shown on Kickstarter", key="in_location")
        location_type = st.selectbox("Location type", sorted(fb.levels_["location_type"]),
                                     key="in_location_type")

location = st.session_state.get("in_location", "")
location_type = st.session_state.get("in_location_type", "Country")

if launch < dt.date.today():
    st.warning("Launch date is in the past. Pick today or later — the model assumes the campaign "
               "has not started yet.")

rate = fb.currency_rate_.get(currency, 1.0)
goal_usd = goal_local * rate
deadline = launch + dt.timedelta(days=int(duration))


def build_row(goal_usd_value):
    launched = int(dt.datetime.combine(launch, dt.time(12)).timestamp())
    return pd.DataFrame([{
        "name": name, "blurb": blurb, "goal_usd": goal_usd_value, "usd_rate": rate,
        "currency": currency, "country": country, "category": category,
        "parent_category": parent, "location": location.strip() or "unknown",
        "location_type": location_type, "launched_at": launched,
        "deadline": launched + duration * 86400,
    }])


def predict(goal_value):
    X = fb.transform(build_row(goal_value), blocks=SAMPLES["S2_nb"])[cols]
    return float(model.predict_proba(X)[0, 1]), X


p, X = predict(goal_usd)

# ------------------------------------------------------------------ hero
st.divider()
hero_l, hero_r = st.columns([1, 1], gap="large")

verdict = "Likely to reach its goal" if p >= 0.5 else "More likely to fall short"
base = STATS.get("baseline_acc")

ticks = []
if base:
    ticks.append(f'<div class="ks-tick" style="left:{min(base, 1) * 100:.1f}%"></div>')
ticks.append('<div class="ks-tick" style="left:50%"></div>')

legend = '<span><i style="background:#8b949e"></i>50% decision threshold</span>'
if base:
    legend += (f'<span><i style="background:#8b949e"></i>'
               f'{base:.0%} of real projects succeed (base rate)</span>')

with hero_l:
    hero = [
        '<div class="ks-card">',
        '<p class="ks-card-h">Prediction</p>',
        '<div class="ks-hero">',
        f'<span class="ks-big {_pct_class(p)}">{p:.0%}</span>',
        f'<span class="ks-verdict {_pct_class(p)}">{verdict}</span>',
        "</div>",
        '<div class="ks-track">',
        f'<div class="ks-fill" style="width:{min(p, 1) * 100:.1f}%;'
        f'background:{_fill_class(p)}"></div>',
        "".join(ticks),
        "</div>",
        f'<div class="ks-legend">{legend}</div>',
        '<div class="ks-facts">',
        f'<div class="ks-fact"><span>Goal (USD)</span><b>${goal_usd:,.0f}</b></div>',
        f'<div class="ks-fact"><span>Closes</span><b>{deadline:%d %b %Y}</b></div>',
        f'<div class="ks-fact"><span>Blurb signal</span><b>{X["nb_prob"].iloc[0]:.0%}</b></div>',
        "</div>",
    ]
    if base:
        hero.append(
            f'<p class="ks-note">The model is trained with class-balanced weights, so 50% is '
            f'the decision threshold &mdash; not the {base:.0%} base rate of the data.</p>'
        )
    hero.append("</div>")
    st.markdown("".join(hero), unsafe_allow_html=True)

# --------------------------------------------------------- goal sensitivity
MULT = [0.1, 0.25, 0.5, 1, 2, 4, 10]
rows = []
for m in MULT:
    g = goal_usd * m
    rows.append((m, g, predict(g)[0]))

ladder = []
for m, g, gp in rows:
    cur = " ks-row-cur" if m == 1 else ""
    ladder.append(
        f'<div class="ks-row{cur}">'
        f'<div class="ks-row-lab">${g:,.0f}</div>'
        f'<div class="ks-track-s"><div class="ks-fill-s" style="width:{min(gp, 1) * 100:.1f}%;'
        f'background:{_fill_class(gp)}"></div></div>'
        f'<div class="ks-val">{gp:.0%}</div></div>'
    )

with hero_r:
    st.markdown(
        f'<div class="ks-card">'
        f'<p class="ks-card-h">How sensitive is this to the goal?</p>'
        f'{"".join(ladder)}'
        f'<p class="ks-note">Highlighted row is your current goal. The effect flattens once the '
        f'ask is large enough that money stops being the binding constraint.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------- why
contrib = model.predict(X, pred_contrib=True)[0][:-1]
s = pd.Series(contrib, index=cols)
top = s.reindex(s.abs().sort_values(ascending=False).head(8).index)
scale = float(s.abs().max()) or 1.0

bars = []
for fname, val in top.items():
    label = html.escape(fname.replace("m_", "").replace("_", " "))
    pct = min(abs(val) / scale * 50, 50)
    cls = "ks-pos" if val >= 0 else "ks-neg"
    bars.append(
        f'<div class="ks-row">'
        f'<div class="ks-row-lab">{label}</div>'
        f'<div class="ks-track-s"><div class="ks-axis"></div>'
        f'<div class="{cls}" style="width:{pct:.1f}%"></div></div>'
        f'<div class="ks-val">{val:+.2f}</div></div>'
    )

st.markdown(
    f'<div class="ks-card">'
    f'<p class="ks-card-h">Why this prediction &mdash; top 8 feature contributions</p>'
    f'{"".join(bars)}'
    f'<p class="ks-note">Green pushes toward success, red toward failure. Values are LightGBM '
    f'TreeSHAP contributions in log-odds, so they add up to the shift away from the base rate.</p>'
    f'</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------- depth (expander)
with st.expander("How this model works — and where it breaks"):
    st.markdown(
        "**What it uses.** Only what a backer sees on day zero: the funding goal and its "
        "currency, category and parent category, creator country and location type, campaign "
        "length, launch date and weekday, name and blurb lengths, and one text feature — an "
        "out-of-fold Naive-Bayes probability computed from the blurb.\n\n"
        "**Why it is honest.** Category, country and currency are high-cardinality, so they are "
        "encoded as (a) out-of-fold target means and (b) leave-one-out target differences. The "
        "Naive-Bayes feature is likewise fitted out-of-fold. Nothing is fitted on the test split, "
        "and projects are de-duplicated by name before the split."
    )

    if ABL:
        order = ["S1_meta", "S2_nb", "S3_nb_sent", "S4_nb_lda", "S5_nb_lsa"]
        label = {"S1_meta": "Metadata only", "S2_nb": "+ Naive-Bayes blurb",
                 "S3_nb_sent": "+ sentiment (VADER)", "S4_nb_lda": "+ LDA topics",
                 "S5_nb_lsa": "+ LSA embeddings"}
        a2 = ABL["S2_nb"]["acc_mean"]
        trows = []
        for k in order:
            if k not in ABL:
                continue
            a = ABL[k]["acc_mean"]
            d = 100 * (a - a2)
            if k == "S2_nb":
                cell = '<td>—</td><td>—</td>'
            else:
                cls = "pos" if d > 0.05 else ("neg" if d < -0.05 else "")
                cell = f'<td class="{cls}">{d:+.2f}</td><td class="">{a * 100:.2f}%</td>'
            trows.append(f"<tr><td>{label[k]}</td>{cell}<td>{a * 100:.2f}%</td></tr>")
        st.markdown(
            '<p class="ks-card-h" style="margin-top:.6rem">Text-feature ablation '
            '(mean accuracy over 5 seeds)</p>'
            '<table class="ks-tbl"><tr><th>Feature set</th><th>Δ vs NB</th>'
            '<th>Accuracy</th></tr>' + "".join(trows) + "</table>"
            '<p class="ks-note">The blurb supports one well-regularised text feature; sentiment '
            'adds nothing measurable on a one-liner and the topic / embedding blocks each add '
            'about half a point.</p>',
            unsafe_allow_html=True,
        )

    if STATS:
        st.markdown(
            f'<table class="ks-tbl"><tr><th>Cohort</th><th>Value</th></tr>'
            f'<tr><td>Projects after cleaning</td><td>{STATS.get("n_final", 0):,}</td></tr>'
            f'<tr><td>Rows read from 8 snapshots</td><td>{STATS.get("n_rows_read", 0):,}</td></tr>'
            f'<tr><td>Train / test split</td><td>{STATS.get("n_train", 0):,} / '
            f'{STATS.get("n_test", 0):,}</td></tr>'
            f'<tr><td>Success rate</td><td>{STATS.get("success_rate", 0):.1%}</td></tr>'
            f'<tr><td>Median goal</td><td>${STATS.get("goal_usd_median", 0):,.0f}</td></tr>'
            f'<tr><td>Distinct categories</td><td>{STATS.get("n_categories", 0)}</td></tr>'
            f'<tr><td>Distinct countries</td><td>{STATS.get("n_countries", 0)}</td></tr>'
            f'</table>',
            unsafe_allow_html=True,
        )

    st.markdown(
        "**Where it breaks.** Launch-time metadata only — no campaign story, no images or video, "
        "no creator track record, no comment velocity. A well-told project with a modest goal can "
        "under-score; a slick video with an ambitious goal can over-score. The split is random, "
        "so it does not test whether the model survives a change in the platform's era — a "
        "temporal split would be the stricter test."
    )

st.markdown(
    '<p class="ks-note" style="margin-top:1.4rem">Example projects are illustrative, not real '
    'campaigns. Trained on the Web Robots Kickstarter dump, launches '
    f'{STATS.get("launch_min", "?")} to {STATS.get("launch_max", "?")}.</p>',
    unsafe_allow_html=True,
)