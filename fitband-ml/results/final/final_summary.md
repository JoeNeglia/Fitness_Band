# Final test: train sessions 1-4, test session 5 (run once)

Run at 2026-11-15 20:26. Headline model: gbm_fft_rest. Stage 1 classifier: random_forest.

Chosen settings (tuned inside sessions 1-4): {'ridge': {'pca_components': 8}, 'gbm': {'max_depth': 2, 'n_estimators': 200}}

## Stage 1

Macro-F1 on session 5: 1.000

```
              predicted resting  predicted typing  predicted walking  predicted running
true resting                 49                 0                  0                  0
true typing                   0                50                  0                  0
true walking                  0                 0                 50                  0
true running                  0                 0                  0                 49
```

## Stage 2 (RMSE, BPM)

```
                                           method               column  rmse_all  coverage_pct  rmse_resting  rmse_typing  rmse_walking  rmse_running
                    Baseline: FFT on every window                  fft     34.24        100.00          1.70        25.02         35.22         53.21
     Baseline: activity mean (predicted activity)        activity_mean     11.14        100.00          1.52         3.27         15.77         15.33
  Baseline: activity mean (true activity, ORACLE) activity_mean_oracle     11.14        100.00          1.52         3.27         15.77         15.33
                      Baseline: band's device_bpm               device     80.25         97.47        130.07        68.76         34.10         51.97
         Model: scaler -> PCA -> ridge (readable)                ridge     11.05        100.00          2.02         4.03         13.78         16.71
              Model: gradient boosting (flexible)                  gbm      9.93        100.00          4.73         2.79         13.91         13.07
                       Model + FFT at rest: ridge       ridge_fft_rest     11.04        100.00          1.70         4.03         13.78         16.71
Model + FFT at rest: gradient boosting (HEADLINE)         gbm_fft_rest      9.69        100.00          1.70         2.79         13.91         13.07
```

## Slices (gbm_fft_rest)

```
                            variable            level          kind  windows  blocks  rmse  ci_low  ci_high
                          heart rate           >= 100 pre-specified       57       2 11.81    9.04    19.52
                          heart rate            < 100 pre-specified      141       4  8.68    2.32    15.53
                             session         session5 pre-specified      198       4  9.69    2.32    13.50
                           pace step                0 pre-specified       34       2 16.01   12.87    18.63
                           pace step                1 pre-specified       34       2 12.63    8.21    15.86
                           pace step                2 pre-specified       31       2 11.23    9.40    12.71
                acc_mag_std quartile         Q1 (low) pre-specified       50       2  1.68    0.01     1.70
                acc_mag_std quartile               Q2 pre-specified       49       1  2.82     NaN      NaN
                acc_mag_std quartile               Q3 pre-specified       49       1 14.01     NaN      NaN
                acc_mag_std quartile        Q4 (high) pre-specified       50       2 12.99    8.19    13.07
                   ppg_mean quartile         Q1 (low) pre-specified       50       3 10.07    3.25    15.87
                   ppg_mean quartile               Q2 pre-specified       49       3  9.97    2.31    17.61
                   ppg_mean quartile               Q3 pre-specified       49       4 11.60    2.31    14.98
                   ppg_mean quartile        Q4 (high) pre-specified       50       3  6.39    1.39    15.71
                         label noise |hr - hr_rr| > 5 pre-specified        5       2  8.35    5.36     8.94
                         label noise            clean pre-specified      193       4  9.72    2.27    13.66
                  activity (context)          resting       context       49       1  1.70     NaN      NaN
                  activity (context)           typing       context       50       1  2.79     NaN      NaN
                  activity (context)          walking       context       50       1 13.91     NaN      NaN
                  activity (context)          running       context       49       1 13.07     NaN      NaN
transition (<30 s after pace change)              yes   exploratory       20       2 12.51   10.40    14.31
```

Session 5 has only 4 blocks (one per activity), so the block-bootstrap intervals are wide.

## Proxy check (within-block correlation)

```
       method  within_block_correlation
 gbm_fft_rest                     0.433
          gbm                     0.431
        ridge                     0.089
          fft                    -0.172
activity_mean                     0.000
```
