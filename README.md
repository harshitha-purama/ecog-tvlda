# ECoG hand-pose: grid decimation + continuous glove decoding

Builds on Gruenwald et al. (2019), *Front. Neurosci.* 13:901 (TVLDA).
Data: `ECoG_Handpose.mat` (one subject, 60-electrode 6x10 high-density grid, 5 mm spacing,
1200 Hz, 90 rock-paper-scissors trials, 30 per gesture, 5-channel data glove).

## Files
| File | What it does |
|---|---|
| `exp0_reproduce.py` | Closest reproduction of the paper's full-grid result, incl. a nested-CV check |
| `exp0b_10mm.py` | Same improvements applied to the simulated 10 mm grids |
| `ecog_pipeline.py` | Preprocessing (CAR, 50 Hz notch cascade, AR(10) whitening, 50-300 Hz log-power in 50 ms bins) and TVLDA with PCA + one-vs-one min-max, following the paper |
| `exp1_decimation.py` | Simulated coarser grids (5/10/15/20 mm, every offset), 2x2 pooled "large contacts", random electrode subsets |
| `exp1b_lmp_fusion.py` | Adds low-frequency LMP features to high-gamma |
| `exp2_glove.py` | Continuous ridge-regression decoding of the 5 glove channels (causal 0-500 ms lags, blocked 5-fold CV) |
| `make_figures.py` | Produces `figures/fig1_decimation.png`, `figures/fig2_glove.png` from `results/` |

## Data
The recording is **not** in this repository (123 MB, above GitHub's 100 MB file limit, and it is patient data).


## Setup
`pip install -r requirements.txt`


## Results (10 repeats of stratified 10-fold CV; NF = best of 1-3 PCA components, as in the paper)

**Reproduction of the paper (60 electrodes, 20 x 10-fold CV, paper's NF rule):**

| Setting | Accuracy |
|---|---|
| Paper's settings as written | 93.8 +- 1.1% |
| + trial window centred 0.3 s later (on the actual movement) | 97.2 +- 0.7% |
| + trial 0 excluded (glove shows atypical execution) | **97.8 +- 1.0%** |
| Same, with window and NF chosen by nested CV (no test-data tuning) | 96.7 +- 1.6% |
| Paper, mean of its 3 high-density RPS subjects | 99.0% |

With 30 trials per gesture, one trial is worth 3.3 points per class (the paper's "quantization margin"),
so 97.8% is within one trial-step of the paper's 99.0%. The remaining errors are ~2 of 89 trials; one of them
(trial 69) is misclassified in every run although the glove shows a normal gesture, so it was kept.
Applying the same two changes to the 10 mm grids raises them from 85.0% to **87.6%** (paper's standard-grid subjects: 86.9%).

**Experiment 1, grid density:**

| Layout | Electrodes | Accuracy |
|---|---|---|
| 5 mm (full grid) | 60 | 96.0% |
| 10 mm (standard-grid spacing), mean of 4 offsets | 15 | 85.0% (82.1-87.8) |
| 10 mm, 2x2-pooled larger contacts | 15 | 85.1% |
| 15 mm | 6-8 | 80.4% (67.2-88.7) |
| 20 mm | 4-6 | 76.4% (68.8-83.9) |

Random subsets: 4 el. 72%, 8 el. 83%, 15 el. 86%, 30 el. 93%, 60 el. 96%.
Thinning to standard spacing reproduces the paper's ~87% standard-grid figure *within one subject*,
supporting density (not just subject differences) as the cause of the gap.
LMP fusion (exp1b) did **not** help: concatenating LMP with high-gamma lowered accuracy.

**Experiment 2, continuous finger decoding (held-out Pearson r):**

| | thumb | index | middle | ring | little | mean |
|---|---|---|---|---|---|---|
| 60 el. high-gamma | 0.74 | 0.63 | 0.62 | 0.81 | 0.75 | **0.71** |
| 15 el. (10 mm) high-gamma | 0.64 | 0.44 | 0.44 | 0.64 | 0.58 | 0.55 |
| Oracle gesture template (knows true cues) | 0.85 | 0.97 | 0.97 | 0.98 | 0.93 | 0.94 |

## Caveats
- Single subject, single session, 90 trials: a case study, not a general result.
- Decimated electrodes are still 1.5 mm contacts; real standard grids use ~3 mm contacts (2x2 pooling is a rough proxy).
- Glove movements here are stereotyped per gesture (index/middle r = 0.99; "open hand" barely differs from rest),
  so the regression largely tracks *which gesture* is made, not independent finger control.
  The oracle row shows how much of the glove signal a perfect gesture recogniser would explain.
- Whitening coefficients are estimated from the first 60 s of the recording (label-free).
