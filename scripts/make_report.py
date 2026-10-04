"""Build the 2-page PDF write-up from results/metrics.json (numbers are never typed by hand).

    python scripts/make_report.py --authors "Name1 (SRN1), Name2 (SRN2)" --problem-id 7
"""
import argparse
import io
import json
import math
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

DARK, GREEN = colors.HexColor("#0B3D2E"), colors.HexColor("#05A862")
PAPER_ACC = 0.83  # Khosla et al. best accuracy


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def significance_note(path):
    """One sentence from results/significance.json (paired bootstrap + McNemar); '' if the file is absent."""
    if not os.path.exists(path):
        return ""
    sig = json.load(open(path))
    comps = sig["comparisons"]

    def fmt(r):
        return f"{100 * r['acc_diff']:+.2f} [{100 * r['ci_low']:.2f}, {100 * r['ci_high']:.2f}]"

    def label(r):
        return "NB" if r["B"] == "Metadata" else r["A"].split(" + ")[-1]

    abl = [r for r in comps if r["B"] in ("Metadata", "Metadata + NB")]
    mods = [r for r in comps if r["A"] == "G. Boosting"]
    parts = "; ".join(f"{label(r)} {fmt(r)}" + ("" if r["significant_95"] else " (not significant)") for r in abl)
    out = (f"<b>Significance</b> (paired bootstrap on the same {sig['n_test']:,} test rows; accuracy change in points "
           f"with 95% CI): {parts}.")
    if mods:
        lo, hi = (100 * min(r["acc_diff"] for r in mods), 100 * max(r["acc_diff"] for r in mods))
        pmax = max(r["mcnemar_p"] for r in mods)
        out += (f" Gradient boosting leads each of the other {len(mods)} models by {lo:.1f} to {hi:.1f} points "
                f"(McNemar p &lt; {10 ** math.ceil(math.log10(pmax)):.0e}).")
    return out


def build_story(M, fs, authors, problem_id, fig_dir, team_id="", show_sig=True):
    body = ParagraphStyle("b", fontName="Helvetica", fontSize=fs, leading=fs * 1.28, alignment=TA_JUSTIFY, spaceAfter=3)
    h = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=fs + 1.5, textColor=DARK, spaceBefore=5, spaceAfter=2)
    title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=fs + 5, leading=(fs + 5) * 1.2, textColor=DARK, spaceAfter=3)
    sub = ParagraphStyle("s", fontName="Helvetica", fontSize=fs, leading=fs * 1.25, textColor=colors.HexColor("#444444"), spaceAfter=4)
    warn = ParagraphStyle("w", parent=body, textColor=colors.red, fontName="Helvetica-Bold")
    bullet = ParagraphStyle("bl", parent=body, leftIndent=10, bulletIndent=0, spaceAfter=1.5)

    st, meta = M["meta"]["stats"], M["meta"]
    t2, t1, abl = M["table_sample2"], M["table_sample1"], M["ablation"]
    best = t2[0]
    a_s1, a_s2 = abl["S1_meta"]["acc_mean"], abl["S2_nb"]["acc_mean"]
    extras = {k: abl[k]["acc_mean"] - a_s2 for k in ("S3_nb_sent", "S4_nb_lda", "S5_nb_lsa")}
    cm = M["confusion"]
    r0 = cm[0][0] / max(1, sum(cm[0]))
    r1 = cm[1][1] / max(1, sum(cm[1]))

    S = []
    S.append(Paragraph("Predicting Kickstarter Project Success from Launch-Time Information", title))
    tid = f"Team #{team_id} &nbsp;|&nbsp; " if team_id else ""
    pid = f"Problem #{problem_id} &nbsp;|&nbsp; " if problem_id else ""
    S.append(Paragraph(f"UE24CS352A Machine Learning Mini-Project &nbsp;|&nbsp; {tid}{pid}{authors}", sub))
    if meta["data_source"] != "real":
        S.append(Paragraph("DRAFT GENERATED FROM SYNTHETIC SMOKE-TEST DATA - NOT REAL RESULTS. "
                           "Re-run the pipeline on the real dataset before submitting.", warn))

    S.append(Paragraph("1. Problem statement", h))
    S.append(Paragraph(
        "Given only information available <i>when a Kickstarter campaign launches</i> (funding goal, category, country, "
        "currency, creator location, campaign length, launch date, and the project name and short blurb), predict whether "
        "the campaign will reach its funding goal. This is binary classification. We follow Khosla, Reinecke and "
        f"Wittenbrink (Stanford, <i>Crowdfunding: Predicting Kickstarter Project Success</i>), who report about {pct(PAPER_ACC, 0)} "
        "accuracy, and compare against their results.", body))

    S.append(Paragraph("2. Dataset", h))
    S.append(Paragraph(
        f"We use the public Web Robots Kickstarter scrape (monthly cumulative snapshots). Reading {st['n_rows_read']:,} rows gave "
        f"{st['n_unique_ids']:,} unique project ids (latest snapshot kept per id). We keep finished campaigns only "
        f"(successful or failed; live and canceled removed), drop rows with missing goal or dates, and de-duplicate on "
        f"(name, blurb, launch date, deadline) as in the paper, leaving <b>{st['n_final']:,} projects</b> launched "
        f"{st['launch_min']} to {st['launch_max']} ({pct(st['success_rate'])} succeeded; majority-class accuracy "
        f"{pct(st['baseline_acc'])}). Goals are converted to USD (median ${st['goal_usd_median']:,.0f}). The split is a "
        f"{st['split']} 70/30 split: {st['n_train']:,} train / {st['n_test']:,} test. Post-launch fields (backers, pledged amount, "
        "staff pick) are never used, since they would leak the outcome.", body))

    S.append(Paragraph("3. Approach", h))
    S.append(Paragraph(
        "<b>Features.</b> (a) Metadata: log goal, campaign length, launch month and weekday, name/blurb lengths, one-hot parent "
        "category and category; (b) <b>out-of-fold target encodings</b> (5 folds, smoothed) of currency, country, category, parent "
        "category, location and location type, plus the mean goal per level and the project's goal relative to it; (c) text: "
        "TF-IDF unigrams of the blurb fed to Multinomial Naive Bayes whose <b>out-of-fold</b> success probability becomes a feature, "
        "and optional VADER sentiment, 20-topic LDA, and 50-dimensional LSA embedding.", body))
    S.append(Paragraph(
        "<b>Models.</b> Gradient boosting (LightGBM), random forest, one-hidden-layer neural net, linear SVM, logistic regression, "
        "and ridge / OLS / lasso linear probability models thresholded at 0.5. All use balanced sample weights where supported. "
        "<b>Protocol.</b> Fit on train only; report accuracy, F1, per-class precision/recall and ROC-AUC on the untouched test set; "
        f"text-feature ablation over {abl['S1_meta']['accs'].__len__()} seeds. "
        "<b>Differences from the paper:</b> VADER instead of Stanford CoreNLP, LSA instead of pretrained Word2Vec, and scikit-learn's "
        "ReLU MLP instead of a SELU network.", body))

    S.append(Paragraph("4. Implementation overview", h))
    S.append(Paragraph(
        "Python (pandas, scikit-learn, LightGBM). <font face='Courier'>src/data.py</font> loads and cleans the dump; "
        "<font face='Courier'>src/features.py</font> builds leakage-safe features (a <font face='Courier'>FeatureBuilder</font> that "
        "is fitted on train and reused unchanged for test and the demo); <font face='Courier'>src/models.py</font> defines the "
        "model zoo and metrics; <font face='Courier'>src/train.py</font> runs all experiments and writes tables, figures and the saved model; "
        "<font face='Courier'>app.py</font> is a Streamlit demo (probability, goal what-if curve, per-prediction feature contributions); "
        "<font face='Courier'>tests/</font> checks leakage guards and the pipeline end to end.", body))

    S.append(Paragraph("5. Results (test set)", h))
    hdr = ["Model", "Acc", "F1", "P1", "P0", "R1", "R0", "AUC"]
    rows = [hdr] + [[r["model"], *(f"{r[k]:.3f}" for k in ("A", "F1", "P1", "P0", "R1", "R0", "AUC"))] for r in t2]
    tbl = Table(rows, hAlign="LEFT", colWidths=[1.05 * inch] + [0.55 * inch] * 7)
    tbl.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", fs - 0.5), ("FONT", (0, 1), (-1, -1), "Helvetica", fs - 0.5),
        ("BACKGROUND", (0, 0), (-1, 0), DARK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EAF6F0")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, DARK),
    ]))
    S.append(Paragraph(f"<b>Table 1.</b> Models on Metadata + NB features (Sample 2). Class 1 = successful.", body))
    S.append(tbl)
    S.append(Spacer(1, 3))
    abl_txt = "; ".join(f"{abl[k]['label']} {pct(abl[k]['acc_mean'])}" for k in abl)
    S.append(Paragraph(f"<b>Text-feature ablation</b> (LightGBM accuracy, mean over seeds): {abl_txt}.", body))
    sig_txt = significance_note(os.path.join(os.path.dirname(fig_dir), "significance.json"))
    if show_sig and sig_txt:
        S.append(Paragraph(sig_txt, body))
    figs = []
    for fn, w in (("importance.png", 3.35), ("confusion.png", 2.9)):
        p = os.path.join(fig_dir, fn)
        if os.path.exists(p):
            from reportlab.lib.utils import ImageReader
            iw, ih = ImageReader(p).getSize()
            figs.append(Image(p, width=w * inch, height=w * inch * ih / iw))
    if figs:
        S.append(Table([figs], hAlign="CENTER"))

    S.append(Paragraph("6. Conclusions", h))
    lift = 100 * (a_s2 - a_s1)
    sent_x, lda_x, lsa_x = (100 * extras[k] for k in ("S3_nb_sent", "S4_nb_lda", "S5_nb_lsa"))
    biggest_text_change = max(abs(sent_x), abs(lda_x), abs(lsa_x))
    gap = 100 * (best["A"] - t2[min(3, len(t2) - 1)]["A"])
    bullets = [
        f"<b>{best['model']}</b> is the best model: {pct(best['A'])} accuracy, F1 {best['F1']:.3f}, AUC {best['AUC']:.3f}, "
        f"{100 * (best['A'] - st['baseline_acc']):+.1f} points over the majority-class baseline "
        f"(paper: about {pct(PAPER_ACC, 0)} accuracy).",
        f"The out-of-fold Naive-Bayes blurb probability lifts accuracy from {pct(a_s1)} to {pct(a_s2)} ({lift:+.2f} points). "
        f"Measured against metadata + NB, VADER sentiment moves it {sent_x:+.2f} points, LDA {lda_x:+.2f} and LSA {lsa_x:+.2f}. "
        f"A one-line blurb supports one well-regularised text feature; the largest change from any further text block is "
        f"{biggest_text_change:.2f} points, so the extras refine that feature rather than drive the result.",
        (f"The top four models are within {gap:.1f} points of each other, so feature engineering matters more than model choice."
         if gap < 1.0 else f"The best model leads the fourth-best by {gap:.1f} points, so model choice matters here."),
        (f"For gradient boosting (confusion matrix above), recall on failed projects is {r0:.2f} vs {r1:.2f} on successful ones: "
         + ("failures are harder to catch than successes. " if r0 + 0.03 < r1 else
            "successes are harder to catch than failures. " if r1 + 0.03 < r0 else "errors are roughly balanced. ")
         + f"{cm[1][0]:,} successes were predicted as failures and {cm[0][1]:,} failures as successes."),
        "<b>Limitations / future work:</b> launch-time metadata and a one-line blurb are limited signals; the long project story, "
        "images and creator history are not used. A temporal split (<font face='Courier'>--split temporal</font>) gives a more "
        "realistic estimate under platform drift. Transformer sentence embeddings are the natural next step.",
    ]
    for b in bullets:
        S.append(Paragraph(b, bullet, bulletText="\u2022"))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/metrics.json")
    ap.add_argument("--out", default="results/writeup.pdf")
    ap.add_argument("--authors", default="Team members (SRNs)")
    ap.add_argument("--problem-id", default="")
    ap.add_argument("--team-id", default="")
    ap.add_argument("--max-pages", type=int, default=2)
    args = ap.parse_args()
    M = json.load(open(args.metrics))
    fig_dir = os.path.join(os.path.dirname(args.metrics), "figures")

    for fs in (9.5, 9, 8.5, 8, 7.5, 7):
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=0.65 * inch, rightMargin=0.65 * inch,
                                topMargin=0.55 * inch, bottomMargin=0.55 * inch,
                                title="Predicting Kickstarter Project Success", author=args.authors)
        doc.build(build_story(M, fs, args.authors, args.problem_id, fig_dir, args.team_id,
                                 show_sig=args.max_pages > 1))
        if doc.page <= args.max_pages:
            break
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(buf.getvalue())
    print(f"wrote {args.out}: {doc.page} page(s), body font {fs}pt")
    if doc.page > args.max_pages:
        print("WARNING: could not fit within the page limit")
    if M["meta"]["data_source"] != "real":
        print("WARNING: metrics come from SYNTHETIC data; the PDF is watermarked as a draft")


if __name__ == "__main__":
    main()
