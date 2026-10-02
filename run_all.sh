#!/usr/bin/env bash
# Full real-data run. Usage: ./run_all.sh "Name1 (SRN1), Name2 (SRN2)" 56 18
set -euo pipefail
AUTHORS="${1:-Team members}"
PID="${2:-}"
TID="${3:-}"
python -m src.train --data data/raw --launch-from 2024-07-01 --launch-to 2026-07-01
python scripts/make_report.py --authors "$AUTHORS" --problem-id "$PID" --team-id "$TID"
python scripts/make_slides.py --authors "$AUTHORS" --problem-id "$PID" --team-id "$TID"
echo "Now run: streamlit run app.py"
