"""Smoke tests for the Streamlit demo.

The demo is the thing an examiner actually looks at, so a silent breakage here
costs more than anywhere else in the project. These tests execute app.py the way
Streamlit does and assert the three things that must never regress: it renders,
it produces a number, and that number responds to the inputs.
"""
import os
import re

import pytest

pytest.importorskip("streamlit.testing")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
BUNDLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "models", "bundle.joblib")

pytestmark = pytest.mark.skipif(
    not os.path.exists(BUNDLE),
    reason="needs models/bundle.joblib - run `python -m src.train --data data/raw` first",
)

BIG = re.compile(r'ks-big\s+\S+">(\d+)%')


def _pct(at):
    match = BIG.search(" ".join(m.value for m in at.markdown))
    assert match, "prediction card did not render a percentage"
    return int(match.group(1))


def _run():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, f"app raised: {[str(e.value) for e in at.exception]}"
    return at


def test_renders_prediction_and_explanation():
    at = _run()
    assert 0 <= _pct(at) <= 100
    blob = " ".join(m.value for m in at.markdown)
    assert "top 8 feature contributions" in blob
    assert "How sensitive is this to the goal?" in blob
    # the depth layer must be present, even though it is collapsed
    assert any("How this model works" in e.label for e in at.expander)


def test_lower_goal_raises_predicted_success():
    at = _run()
    before = _pct(at)
    at.number_input(key="in_goal").set_value(500.0).run()
    assert not at.exception
    assert _pct(at) > before, "halving the goal 10x should not lower the prediction"


def test_presets_change_the_verdict():
    at = _run()
    default = _pct(at)
    [b for b in at.button if b.label == "Ambitious hardware"][0].click().run()
    assert not at.exception
    assert _pct(at) < default, r"a $120k gadget should not score like a $5k board game"


def test_no_synthetic_model_banner_on_real_bundle():
    """The app must not claim SYNTHETIC when the bundle was trained on real data."""
    at = _run()
    assert not [e for e in at.error if "SYNTHETIC" in str(e.value)]