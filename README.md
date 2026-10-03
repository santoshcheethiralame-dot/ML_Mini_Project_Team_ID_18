# Predicting Kickstarter Project Success

UE24CS352A Machine Learning mini-project. Predicts whether a Kickstarter campaign will reach its funding goal
using **only information available at launch** (goal, category, country, currency, location, dates, name and
one-line blurb). Replicates and extends *Crowdfunding: Predicting Kickstarter Project Success*
(Khosla, Reinecke, Wittenbrink, Stanford).

Team: _Cheethirala Sai Santosh (PES1UG24CS127)_, _Chhavi Siddarth Wadhwa (PES1UG24CS132)_
&nbsp;|&nbsp; Team ID: _18_ &nbsp;|&nbsp; Problem #: _56_

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` is **pinned** to the exact versions the reported run used (numpy 2.5.3, pandas 3.0.6,
scikit-learn 1.9.1, LightGBM 4.7.0). Keep it that way: with loose floors a re-install can silently change the
numbers in the write-up, and your write-up is only defensible if the code reproduces them.

1. Download the data into `data/raw/` (see [`data/README.md`](data/README.md) - the snapshots are split into
   ~85 chunk CSVs each, so check the chunk count before trusting a download).
2. Run the whole experiment (features, 8 models, ablation, figures, saved model):

```bash
python -m src.train --data data/raw --launch-from 2024-07-01 --launch-to 2026-07-01
#   add --tune for a hyper-parameter search, --split temporal for a stricter test
#   --max-files N limits the number of snapshot FILES read (not the chunks inside one)
```

3. Generate the write-up PDF and the slide deck from the real numbers:

```bash
python scripts/make_report.py --authors "Name1 (SRN1), Name2 (SRN2)" --problem-id 56 --team-id 18  # -> results/writeup.pdf
python scripts/make_slides.py --authors "Name1 (SRN1), Name2 (SRN2)" --problem-id 56 --team-id 18   # -> results/slides.pptx
```

4. Launch the live demo:

```bash
streamlit run app.py        # -> http://localhost:8501
```

### What the demo shows

`app.py` is layered on purpose. The default surface answers *will it work, and why* in a few
seconds; the methodology sits behind one expander so a viva can go as deep as the examiner wants
without burying the headline.

| Layer | What it gives you |
|---|---|
| Header | Provenance pills: rows read, cohort dates, held-out accuracy, majority baseline, feature count |
| Presets | Three scenarios (tabletop game, ambitious hardware, documentary film) scoring **56% / 6% / 25%** on the same model - click through them instead of typing an example live |
| Inputs | Split into *the project* and *the campaign*; location fields hidden behind an expander; launch date cannot be in the past |
| Prediction | One card: probability, verdict, and a gauge marking both the 50% decision threshold **and** the 71% base rate, so the class-balancing is visible rather than asserted |
| Sensitivity | A seven-rung goal ladder from 0.1x to 10x the ask, showing the effect flattening once money stops being the binding constraint |
| Explanation | Top 8 TreeSHAP contributions as signed bars from a centre axis, so you can see which features pushed the number |
| Expander | Feature groups, the leakage-safety argument, the 5-seed text ablation, cohort table, and an explicit *where it breaks* section |

Dark mode and colours come from `.streamlit/config.toml`. Every custom visual is built from
`ks-`-prefixed HTML/CSS scoped to `app.py` - nothing targets Streamlit's internal DOM, so a
Streamlit upgrade cannot silently break the layout.

`tests/test_app.py` runs the app headlessly and asserts it renders, produces a number, and that
the number responds to the goal and the presets. If you change the UI, run it.

Smoke test without the dataset (**synthetic data, numbers are meaningless**):
`python -m src.train --synthetic 12000 --fast --seeds 2`

Tests: `pytest -q`

## What the pipeline does

| Step | File | Notes |
|---|---|---|
| Load + clean | `src/data.py` | reads Web Robots zips/CSVs, keeps latest row per project id, keeps successful/failed only, converts goal to USD, de-duplicates on (name, blurb, launch, deadline), 70/30 split |
| Features | `src/features.py` | metadata, **out-of-fold** target encodings, **out-of-fold** Naive-Bayes blurb probability, optional sentiment / LDA / LSA |
| Models | `src/models.py` | LightGBM, random forest, MLP, linear SVM, logistic, ridge / OLS / lasso (thresholded at 0.5) |
| Experiments | `src/train.py` | Table for Sample 1 (metadata) and Sample 2 (+NB), 5-sample text ablation over seeds, figures, `metrics.json`, saved model |
| Demo | `app.py` | Streamlit app with probability, goal what-if curve and TreeSHAP feature contributions |
| Report / slides | `scripts/` | built from `results/metrics.json`, so numbers always match the code |

Outputs land in `results/` (`metrics.json`, `table_sample*.csv`, `figures/`, `writeup.pdf`, `slides.pptx`) and
`models/bundle.joblib`.

## Results from the real run

8 complete snapshots (`2025-10-13` … `2026-06-11`), cohort = projects launched 2024-07-01 to 2026-06-04.
2,144,137 rows read -> 231,080 unique ids -> 205,063 finished -> **47,765** after the launch-window filter.
70.9% succeeded, so the **majority-class baseline is 70.71%** and every accuracy below must be read against it.

| | Accuracy | F1 | AUC |
|---|---|---|---|
| Majority-class baseline | 70.71% | - | - |
| Gradient boosting, metadata only | 80.96% | 0.862 | 0.878 |
| **Gradient boosting, metadata + NB** | **81.65%** | **0.867** | **0.887** |

Text-feature ablation (LightGBM, mean of 5 seeds): metadata 80.93% -> +NB 81.77% -> +sentiment 81.74% ->
+LDA 82.30% -> +LSA 82.33%. So the Naive-Bayes blurb probability is worth about +0.8 points, VADER sentiment adds
nothing measurable, and the LDA / LSA blocks add roughly +0.5 each. Gradient boosting on metadata + NB is
**+10.9 points over the majority-class baseline**; the paper reports about 83% on its own cohort.

Confusion matrix (gradient boosting, test set): recall 0.848 on successful projects but 0.740 on failed ones -
missed campaigns are the harder error.

### Are the differences real? (paired significance)

`scripts/significance.py` refits the models on the same split and scores them on the same 14,330 test
rows, using a paired bootstrap (2,000 resamples, 95% CI) and an exact McNemar test. Point estimates
are for one model seed (42), so they differ slightly from the 5-seed means above.

| Comparison | Acc. diff | 95% CI | McNemar p |
|---|---|---|---|
| Metadata + NB vs Metadata | +0.68 pts | [0.22, 1.14] | 0.004 |
| + Sentiment vs + NB | +0.10 pts | [-0.29, 0.45] | 0.63 |
| + LDA vs + NB | +0.79 pts | [0.37, 1.19] | 0.0002 |
| + LSA vs + NB | +0.68 pts | [0.17, 1.14] | 0.005 |
| G. Boosting vs Logistic (runner-up) | +1.99 pts | [1.49, 2.51] | 4e-14 |

Reading: the NB, LDA and LSA gains are statistically real but small (under 1 point); sentiment adds
nothing detectable; gradient boosting's 2-2.5 point lead over every other model is well outside test
noise. The intervals capture test-sampling noise only, not variation from re-splitting or retraining.
Full output: `results/significance.csv` / `.json`.

## Design decisions worth knowing for the review

- **No leakage.** Label-dependent features (target encodings, NB probability) are computed out-of-fold on train;
  test rows use encoders fitted on all of train. `tests/test_pipeline.py` checks this and that no post-launch
  columns (backers, pledged, staff pick) are used.
- **Duplicates removed before splitting**, so the same project cannot be in both train and test.
- **Bounded cohort.** A single cumulative snapshot reaches back to 2009, so an unfiltered run mixes ~17 years of
  platform drift into the launch-date features. We restrict to launches in the 24 months before the newest
  snapshot (2024-07 to 2026-06), which also guarantees every campaign has finished and so no label is still
  settling. `--launch-from` / `--launch-to` control this and default to *no* filter.
- **Snapshot completeness.** A half-downloaded snapshot can still be a valid zip, so partial files would load
  silently. Check the chunk count (~86 for 2026) before using a download; `data/README.md` has the details.
- **Class balance.** 70.9% of our cohort succeeded, so a model that always answers "successful" already scores
  70.71%. We use balanced sample weights and report F1 and per-class precision/recall alongside accuracy, plus
  the majority-class baseline.
- **Honest tuning, and it did not help.** `--tune` runs a random search with 3-fold CV on a training subsample;
  the test set is only used for final reporting. On our data the search picked a *smaller* model
  (31 leaves / 200 trees vs the default 127 / 500) and **lost** 1.4 accuracy points (80.22% vs 81.65%), with
  slightly better AUC (0.891 vs 0.887) - better ranking, worse operating point. We therefore report the untuned
  defaults and keep this as a caveat: a 3-fold CV search on accuracy over a 71/29 split is a noisy objective.
- **Cohort is not the paper's.** The snapshots we could obtain are Oct 2025 - Jun 2026 (there is no April 2019
  snapshot in the archive), so our cohort is 2024-2026 launches, not the paper's 2019-2021 window. Absolute
  accuracies are not directly comparable; the comparison that matters is against our own majority-class baseline.

### Differences from the paper

| Paper | Here | Why |
|---|---|---|
| Stanford CoreNLP sentiment | VADER | no Java server needed |
| Pretrained Google-News Word2Vec (3.6 GB) | 50-d LSA (TruncatedSVD of TF-IDF) | no large download; still a dense semantic embedding |
| Keras SELU network | scikit-learn MLP (ReLU, 25 units) | fewer dependencies |
| Lasso with a large penalty | small penalty on standardised features | the paper's lasso underperforms mostly because of over-regularisation |
| Smoothing: levels < 1000 projects fall back to the mean | Bayesian smoothing (pseudo-count 20) | keeps information from smaller categories |

## Repository layout

```
app.py                 Streamlit demo
src/                   data.py, features.py, models.py, train.py, synthetic.py
scripts/               make_report.py, make_slides.py
tests/                 leakage + pipeline tests
data/README.md         how to get the data
results/               generated tables, figures, PDF, slides
```

## Submission checklist

- [ ] Private GitHub repo shared with the faculty and TAs (Settings > Collaborators)
- [ ] `README.md` has team names and problem number filled in
- [x] Real run completed: `results/metrics.json` shows `"data_source": "real"`
- [x] `results/writeup.pdf` regenerated from the real run and within the page limit (2 pages, `--max-pages 1`
      for a one-page requirement)
- [x] Slides regenerated (12) and `streamlit run app.py` serves the demo
- [ ] Slides rehearsed: leakage explanation (slide 5), metrics vs the majority-class baseline, and why the
      extra text features add so little. **State it precisely**: sentiment -0.03, LDA +0.53, LSA +0.56,
      NB +0.84 - the blurb supports one well-regularised text feature, the extras refine it. Do not claim the
      text features add nothing; the report and slide generators now word this from the real numbers.
- [x] `requirements.txt` pinned to the reported versions
- [x] One-page fallback PDF built (`results/writeup_1page.pdf`, 7pt) in case the faculty want one page
