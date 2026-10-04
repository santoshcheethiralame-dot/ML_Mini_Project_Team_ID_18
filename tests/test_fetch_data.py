"""Tests for the snapshot fetcher's verification logic.

The whole point of scripts/fetch_data.py is that a truncated download gets caught. These
tests build genuinely truncated-but-valid zips and assert they are rejected, because that is
precisely the failure mode a real partial download produces.
"""
import importlib.util
import os
import zipfile

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_ROOT, "scripts", "fetch_data.py")
_spec = importlib.util.spec_from_file_location("fetch_data", _PATH)
fd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fd)


def _zip_with(path, n_chunks, name="Kickstarter"):
    with zipfile.ZipFile(path, "w") as z:
        for i in range(n_chunks):
            z.writestr(f"{name}{i:03d}.csv", "id,name,blurb\n")
    return path


def test_manifest_matches_the_reported_cohort():
    assert len(fd.REPORTED) == 8
    for stamp, chunks, mb in fd.REPORTED:
        assert stamp.endswith("Z") and "T" in stamp
        assert chunks is not None and chunks >= 83
        assert mb > 300
    stamps = [s for s, _, _ in fd.REPORTED]
    assert stamps == sorted(stamps), "manifest should be oldest first"
    assert len(set(stamps)) == 8


def test_complete_snapshot_verifies(tmp_path):
    p = _zip_with(tmp_path / "Kickstarter_x.zip", 85)
    ok, reason = fd.verify(str(p), 85, os.path.getsize(p) / 1024 / 1024)
    assert ok, reason


def test_truncated_but_valid_zip_is_rejected(tmp_path):
    """The real failure mode: 63 chunks instead of 86, still a perfectly valid zip."""
    p = _zip_with(tmp_path / "Kickstarter_y.zip", 63)
    with zipfile.ZipFile(p) as z:  # it really is a valid archive
        assert len(z.namelist()) == 63
    ok, reason = fd.verify(str(p), 86, 358.0)
    assert not ok
    assert "truncated" in reason and "63" in reason and "86" in reason


def test_wrong_size_is_rejected_even_with_right_chunk_count(tmp_path):
    p = _zip_with(tmp_path / "Kickstarter_z.zip", 86)
    ok, reason = fd.verify(str(p), 86, 358.0)
    assert not ok
    assert "size" in reason


def test_missing_file_is_not_a_pass(tmp_path):
    ok, reason = fd.verify(str(tmp_path / "nope.zip"), 86, 358.0)
    assert not ok and reason == "not downloaded"


def test_corrupt_file_is_rejected(tmp_path):
    p = tmp_path / "Kickstarter_bad.zip"
    p.write_bytes(b"not a zip at all")
    ok, reason = fd.verify(str(p), 86, 358.0)
    assert not ok and "valid zip" in reason


def test_extras_are_labelled_and_never_silent():
    for stamp, chunks, _mb, note in fd.EXTRAS:
        assert note, f"{stamp} must carry a note explaining why it is an extra"
    partials = [s for s, c, _, n in fd.EXTRAS if "PARTIAL" in n]
    assert len(partials) == 3, "the three known-partial snapshots must stay flagged"
    for stamp in partials:
        assert stamp not in {s for s, _, _ in fd.REPORTED}


@pytest.mark.parametrize("stamp", [s for s, _, _ in fd.REPORTED])
def test_url_pattern_is_consistent(stamp):
    assert fd.url_for(stamp) == f"{fd.BASE}/Kickstarter_{stamp}.zip"
    assert fd.path_for(stamp).endswith(f"Kickstarter_{stamp}.zip")
