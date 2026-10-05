"""
Experiment 1: how much does high-density coverage buy?
Simulate coarser grids from the 6x10 / 5 mm grid in the same subject and
classify rock/paper/scissors with TVLDA. Features are recomputed per layout
(CAR over the remaining electrodes only, as a real coarser grid would see).
"""
import os
os.makedirs('results', exist_ok=True); os.makedirs('figures', exist_ok=True)
import json
import numpy as np
from ecog_pipeline import *

REPS = 10
NFS = (1, 2, 3)

e, cue, glove = load("ECoG_Handpose.mat")
onsets, labels = cue_onsets(cue)
shift = grand_onset_shift(highgamma_logpower(e), onsets)


def evaluate(ecog_subset, extra_lmp=False):
    F = highgamma_logpower(ecog_subset)
    if extra_lmp:
        L = lowfreq_lmp(ecog_subset)
        L = (L - L.mean(1, keepdims=True)) / L.std(1, keepdims=True)
        F = np.vstack([F, L[:, :F.shape[1]]])
    X = epoch(F, onsets, shift)
    res = {nf: cv_accuracy(X, labels, n_comp=nf, reps=REPS) for nf in NFS if nf <= X.shape[2]}
    best = max(res, key=lambda k: res[k].mean())
    return {"nf1": 100 * res[1].mean(), "best": 100 * res[best].mean(),
            "best_sd": 100 * res[best].std(), "best_nf": best}


def spaced_subsets(step):
    """All offsets of a grid keeping every `step`-th row and column."""
    out = []
    for r0 in range(step):
        for c0 in range(step):
            rows = list(range(r0, 10, step)); cols = list(range(c0, 6, step))
            if len(cols) < 2 or len(rows) < 2:
                continue
            out.append([c * 10 + r for c in cols for r in rows])
    return out


results = {}

# 1) spacing curve: 5, 10, 15, 20 mm, averaged over every possible offset
for step in (1, 2, 3, 4):
    per = [evaluate(e[s]) for s in spaced_subsets(step)]
    results[f"spacing_{5 * step}mm"] = {
        "n_electrodes": len(spaced_subsets(step)[0]),
        "n_offsets": len(per),
        "best_mean": float(np.mean([p["best"] for p in per])),
        "best_min": float(np.min([p["best"] for p in per])),
        "best_max": float(np.max([p["best"] for p in per])),
        "nf1_mean": float(np.mean([p["nf1"] for p in per])),
        "per_offset": per,
    }
    print(f"{5 * step:>2} mm: {results[f'spacing_{5 * step}mm']['best_mean']:.1f}% "
          f"(range {results[f'spacing_{5 * step}mm']['best_min']:.1f}-"
          f"{results[f'spacing_{5 * step}mm']['best_max']:.1f}, "
          f"{results[f'spacing_{5 * step}mm']['n_electrodes']} electrodes, {len(per)} offsets)")

# 2) larger-contact approximation: average each 2x2 block into one virtual
#    electrode -> 15 virtual contacts at 10 mm spacing
blocks = []
for c0 in (0, 2, 4):
    for r0 in range(0, 10, 2):
        blocks.append([c * 10 + r for c in (c0, c0 + 1) for r in (r0, r0 + 1)])
e_pooled = np.stack([e[b].mean(0) for b in blocks])
results["pooled_2x2_10mm"] = evaluate(e_pooled)
print("pooled 2x2 (15 virtual, 10 mm):", results["pooled_2x2_10mm"])

# 3) random electrode subsets of size N (layout-free curve)
rng = np.random.default_rng(0)
curve = {}
for n in (4, 8, 15, 30, 60):
    draws = 1 if n == 60 else 8
    accs = [evaluate(e[np.sort(rng.choice(60, n, replace=False))])["best"] for _ in range(draws)]
    curve[n] = (float(np.mean(accs)), float(np.std(accs)))
    print(f"random {n:>2} electrodes: {curve[n][0]:.1f} +- {curve[n][1]:.1f}")
results["random_subsets"] = curve

# 4) can extra low-frequency features recover what decimation lost?
per = [evaluate(e[s], extra_lmp=True) for s in spaced_subsets(2)]
results["spacing_10mm_plus_LMP"] = {"best_mean": float(np.mean([p["best"] for p in per])),
                                     "per_offset": per}
print("10 mm + LMP:", results["spacing_10mm_plus_LMP"]["best_mean"])
results["full_60_plus_LMP"] = evaluate(e, extra_lmp=True)
print("60 ch + LMP:", results["full_60_plus_LMP"])

json.dump(results, open("results/results_decimation.json", "w"), indent=1, default=float)
