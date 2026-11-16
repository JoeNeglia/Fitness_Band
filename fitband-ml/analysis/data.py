"""
data.py — the only way the modelling scripts load windows.csv.

Session 5 is the final test set. To make sure no development script can see
it by accident, load_windows() removes session 5 by default and checks
(assert) that none of its rows are returned. Only final.py may ask for it
with include_final=True, and that request is refused from any other script.
"""

import sys
from pathlib import Path

import pandas as pd

import config

# Counts how many session-5 rows have been handed out in this Python run.
# run_all.py checks that this is still 0 at the end (the leakage check).
final_rows_returned = 0


def load_windows(include_final=False):
    global final_rows_returned
    if not config.WINDOWS_CSV.exists():
        sys.exit("data/processed/windows.csv not found. Run: python analysis/label.py")

    windows = pd.read_csv(config.WINDOWS_CSV)
    windows["transition"] = windows["transition"].astype(bool)

    if include_final:
        # Only the final test script may see session 5.
        caller = Path(sys.argv[0]).name
        assert caller == "final.py", f"include_final=True is only allowed in final.py, not {caller}"
        final_rows_returned += int((windows["session"] == config.FINAL_SESSION).sum())
        return windows

    development = windows[windows["session"] != config.FINAL_SESSION].reset_index(drop=True)
    assert config.FINAL_SESSION not in set(development["session"]), "session 5 leaked into development data"
    return development
