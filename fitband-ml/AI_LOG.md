# AI log

Bugs and wrong assumptions made while building this project with an AI
assistant (Claude), and how each one was caught. Newest first.

| # | What went wrong | How it was caught | Fix |
|---|---|---|---|
| 9 | A sentence in the generated findings said the running k-means clusters "mostly separated sessions", but their adjusted Rand index against session was only 0.08-0.20. | Read the generated sentence against the numbers beside it. | Reworded to say only what the numbers show, pointing to the crosstab. |
| 8 | In a dry-run test of final.py (sessions 1-3 → session 4 standing in for session 5), the test script changed `FINAL_SESSION` *before* loading the data. The loader then removed session 4 and returned session 5. | The assertion in the test script (`"session5" not in sessions`) stopped the run before anything was computed. | Load the development data first, then relabel. The dry run then passed. Session 5 was never used for any computation. |
| 7 | `final.py` wrote its tables with `DataFrame.to_markdown`, which needs the `tabulate` package. It isn't installed, so the one-time run would have crashed *after* computing the session-5 results and *before* saving them. | Code review before the dry run. | Replaced with plain-text tables (`to_string`), then confirmed with the dry run. |
| 6 | The tuning loop turned the inputs into NumPy arrays for fitting, but predicted on DataFrames. scikit-learn printed "X has feature names, but … was fitted without feature names". | The warnings in the Stage 2 output. | Kept DataFrames throughout. Reran: zero warnings, identical RMSE values (7.93 / 8.30). |
| 5 | The assistant's earlier per-stage analysis of sessions 4-5 assumed 2-minute pace steps. The protocol was 100-s steps (3 per 5-minute block). | The band's step counter: walking cadence in session 4 jumps at 100 s and 200 s. | label.py uses 100-s steps. Earlier per-stage numbers from chat are superseded by `side_experiment_pace_steps.csv`. |
| 4 | Two recordings saved a band file with only a header (0 rows) while the strap recorded normally. | Row counts printed by collect.py and checked afterwards. | Power-cycling the band fixed it. The empty files were deleted before analysis. |
| 3 | The first test recording's first ~14 s had no skin contact. The FFT read ~150 BPM there, inflating its RMSE to 45 BPM. | Compared the FFT against the strap per window: the error was concentrated where the band's `status` was 0. | label.py drops windows where `status == 0` for more than 10% of rows (FFT RMSE on that test fell to 2.5). |
| 2 | The HRM 200 sends a packet every 0.5 s and repeats the previous RR values, so beats were counted twice. | Strap rows were ~2 per second, not the expected ~1, and the RR list repeated between consecutive packets. | label.py skips a packet whose RR text equals the previous packet's. The strap check (mean \|hr_truth − hr_truth_rr\|) is about 1 BPM. |
| 1 | collect.py assumed the strap would connect without pairing. The HRM 200 requires an encrypted (paired) connection ("Encryption is insufficient"). | The first real test run failed with that error. | Retry-with-reconnect loop and a pairing note in the data collection guide. |

## Checks that confirmed the code against known numbers

- FFT baseline RMSE on sessions 1-3: 27.85 BPM overall and 2.01 BPM at rest, matching the ~28 / ~2 BPM measured before the pipeline existed.
- Window counts: 49-50 for every (session, activity) block.
- Leakage: `data.final_rows_returned == 0` after run_all.py. No script except final.py contains `include_final=True`, and `load_windows(include_final=True)` called from anywhere else raises an AssertionError (tested).
- Side experiment vs the dry run of final.py on the same split: identical Stage 1 and Stage 2 results.
