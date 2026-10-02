# Data

The project uses the public **Web Robots Kickstarter dataset** (monthly scrapes of every Kickstarter project):
<https://webrobots.io/kickstarter-datasets/>

## What a snapshot actually is

Each monthly `.zip` is **not one CSV** - it holds the whole dump split into ~85 chunk CSVs
(`Kickstarter.csv`, `Kickstarter001.csv`, ... `Kickstarter085.csv`), roughly 20 MB of text each.
A 2026 snapshot is ~86 chunks / ~375 MB zipped and holds ~272,000 rows / ~211,000 unique project ids.

The snapshots are **cumulative**: every snapshot contains every earlier project. The loader therefore keeps the
**latest** row per project id, so the newest snapshot alone reproduces the same table as all of them combined.
Earlier snapshots only add projects that later snapshots dropped.

### Check a snapshot is complete before using it

A partial download can still be a *valid* zip (a readable central directory), so Python will not complain and
`zipfile` will happily list the members it has. Compare the chunk count against its neighbours:

```python
import zipfile
with zipfile.ZipFile(path) as z:
    print(len(z.namelist()))        # 86 = complete for 2026; ~37 = partial
```

The three files we had to discard were `2026-07` (37 chunks, 43%), `2026-08` (79, 92%) and `2026-09` (63, 73%) -
each one's size ratio matched its chunk ratio, which is the signature of a partial download.

## Snapshots used for the reported results

| | |
|---|---|
| Snapshots | `2025-10-13`, `2025-12-18`, `2026-01-12`, `2026-02-12`, `2026-03-12`, `2026-04-13`, `2026-05-12`, `2026-06-11` (8 files) |
| Rows read | 2,144,137 |
| Unique project ids | 231,080 |
| Finished (successful/failed) | 205,063 |
| **Cohort after launch-window filter** | **47,765** (launched 2024-07-01 to 2026-06-04) |

`2025-11` is absent from the archive we pulled; it is harmless because the snapshots are cumulative.

Two things worth knowing about the archive:

- **There is no April 2019 snapshot.** The archive jumps from `2015-10` straight to `2019-05`, so the
  "April 2019 to April 2021" window quoted in earlier revisions of this file could not be fetched as written.
- Snapshots only exist back to 2015-10 (and a single 2015-10 file before that), **not** to 2009 as the project
  page suggests. A snapshot nevertheless contains projects launched from 2009 onwards.

## Download

1. Open the page above and download the monthly archives (`Kickstarter_YYYY-MM-DDT....zip`).
2. Put the `.zip` files, unmodified, in `data/raw/`. Sub-folders are fine.

```
data/raw/
  Kickstarter_2025-10-13T07_42_31_884Z.zip
  Kickstarter_2025-12-18T03_20_24_296Z.zip
  ...
```

`data/raw/` is git-ignored (the files are large).

## Restricting the cohort

Because one cumulative snapshot reaches back to 2009, an unfiltered run mixes ~17 years of platform drift into
the launch-date features. Bound the cohort by launch date:

```bash
python -m src.train --data data/raw --launch-from 2024-07-01 --launch-to 2026-07-01
```

`--launch-to` is exclusive. The 24-month window we report on (2024-07 to 2026-06) also guarantees every campaign
has finished - Kickstarter campaigns run 15-60 days - so no label is still settling, and a `--split temporal`
test stays meaningful. Omit both flags to keep every finished project.

## Development

`--max-files N` limits the number of **snapshot files** read (the first N, oldest first). It deliberately does
*not* count the chunk CSVs inside a zip - counting chunks would silently read a fraction of a single snapshot
while still exiting cleanly.