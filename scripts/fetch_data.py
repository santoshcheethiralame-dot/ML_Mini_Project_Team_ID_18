"""Fetch and verify the 8 Web Robots Kickstarter snapshots behind the reported results.

Usage
-----
    python scripts/fetch_data.py              # download + verify the reported 8
    python scripts/fetch_data.py --check      # verify what is already in data/raw, download nothing
    python scripts/fetch_data.py --extras     # additionally fetch later snapshots (NOT used in the report)

Why this script exists
----------------------
`src/data.py` cannot tell a complete snapshot from a truncated one. A partial download is
still a structurally valid ZIP, so `zipfile` opens it without complaint and the loader reads
whatever chunks are inside - silently training on a fraction of a month. The three snapshots
we excluded (2026-07 / 08 / 09) were exactly that: valid zips with 37, 79 and 63 chunks
instead of ~86.

So every download is verified against the chunk count of the copy we actually trained on
before it is moved into place, and a mismatch is a hard error, not a warning.

Only the stdlib is used (urllib) so this does not add a dependency.
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile

BASE = "https://s3.amazonaws.com/weruns/forfun/Kickstarter"
RAW = os.path.join("data", "raw")

# The 8 snapshots behind results/metrics.json. `chunks` and `mb` are what the copies we
# trained on actually contained - verified locally, not guessed.
REPORTED = [
    ("2025-10-13T07_42_31_884Z", 83, 342.1),
    ("2025-12-18T03_20_24_296Z", 84, 347.8),
    ("2026-01-12T09_37_51_016Z", 84, 349.2),
    ("2026-02-12T03_20_22_018Z", 85, 350.9),
    ("2026-03-12T03_20_26_556Z", 85, 352.5),
    ("2026-04-13T11_08_31_431Z", 86, 354.2),
    ("2026-05-12T07_26_57_886Z", 86, 354.7),
    ("2026-06-11T03_20_11_324Z", 86, 357.6),
]

# Reachable, but NOT part of the reported cohort. Downloading these changes the numbers in
# the write-up unless you re-run training, and the last three are known-partial downloads.
EXTRAS = [
    ("2025-11-12T12_09_07_111Z", None, 347.0, "not used in the reported run"),
    ("2026-07-24T08_26_14_228Z", 37, 142.0, "PARTIAL - known truncated download"),
    ("2026-08-12T08_12_02_805Z", 79, 322.0, "PARTIAL - known truncated download"),
    ("2026-09-10T03_20_48_478Z", 63, 256.0, "PARTIAL - known truncated download"),
]

SIZE_TOLERANCE = 0.02  # +/- 2%


def url_for(stamp: str) -> str:
    return f"{BASE}/Kickstarter_{stamp}.zip"


def path_for(stamp: str) -> str:
    return os.path.join(RAW, f"Kickstarter_{stamp}.zip")


def inspect(path: str) -> tuple[int, float]:
    """(csv chunk count, size in MB) for a zip on disk."""
    size_mb = os.path.getsize(path) / 1024 / 1024
    with zipfile.ZipFile(path) as z:
        chunks = sum(1 for m in z.namelist() if m.lower().endswith(".csv"))
    return chunks, size_mb


def verify(path: str, expect_chunks: int | None, expect_mb: float) -> tuple[bool, str]:
    """True/False plus a human-readable reason. Missing file counts as not verified."""
    if not os.path.exists(path):
        return False, "not downloaded"
    try:
        chunks, size_mb = inspect(path)
    except zipfile.BadZipFile:
        return False, "not a valid zip"
    if expect_chunks is not None and chunks != expect_chunks:
        return False, f"truncated: {chunks} csv chunks, expected {expect_chunks}"
    if abs(size_mb - expect_mb) / expect_mb > SIZE_TOLERANCE:
        return False, f"size {size_mb:.0f} MB differs from expected {expect_mb:.0f} MB"
    return True, f"{chunks} chunks, {size_mb:.0f} MB"


def download(stamp: str, dest: str) -> bool:
    import urllib.request

    tmp = dest + ".part"
    url = url_for(stamp)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            with open(tmp, "wb") as fh:
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    fh.write(block)
                    done += len(block)
                    pct = f"{done * 100 / total:5.1f}%" if total else "  ?  "
                    print(f"\r    {pct}  {done / 1024 / 1024:6.0f} MB", end="", flush=True)
        print()
        return True
    except Exception as exc:  # noqa: BLE001 - report and move on
        print(f"\r    download failed: {exc}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def fetch(stamps, check_only: bool) -> int:
    os.makedirs(RAW, exist_ok=True)
    failures = []
    for stamp, expect_chunks, expect_mb in stamps:
        dest = path_for(stamp)
        ok, reason = verify(dest, expect_chunks, expect_mb)
        if ok:
            print(f"  OK       {stamp}  ({reason})")
            continue
        if reason != "not downloaded":
            print(f"  REJECT   {stamp}  ({reason})")
        if check_only:
            failures.append((stamp, reason))
            continue
        if reason != "not downloaded":
            os.remove(dest)
        print(f"  fetching {stamp}")
        if not download(stamp, dest):
            failures.append((stamp, "download failed"))
            continue
        ok, reason = verify(dest, expect_chunks, expect_mb)
        if ok:
            print(f"  VERIFIED {stamp}  ({reason})")
        else:
            print(f"  FAILED   {stamp}  ({reason}) - removing so it cannot be used")
            os.remove(dest)
            failures.append((stamp, reason))
    return len(failures)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify existing files only, download nothing")
    ap.add_argument("--extras", action="store_true",
                    help="also fetch snapshots outside the reported cohort (changes results)")
    args = ap.parse_args()

    print("Reported cohort (8 snapshots behind results/metrics.json)")
    targets = list(REPORTED)
    bad = fetch(targets, args.check)

    if args.extras:
        print("\nExtras - NOT part of the reported run")
        for stamp, chunks, mb, note in EXTRAS:
            print(f"  note: {stamp} is {note}")
        bad += fetch([(s, c, m) for s, c, m, _ in EXTRAS], args.check)

    if bad:
        print(f"\n{bad} snapshot(s) failed verification. Do not train until this is fixed -")
        print("a truncated snapshot is a valid zip and the loader will not complain.")
        return 1
    print(f"\nAll {len(targets)} reported snapshots verified. Next:")
    print("  python -m src.train --data data/raw --launch-from 2024-07-01 --launch-to 2026-07-01")
    return 0


if __name__ == "__main__":
    sys.exit(main())
