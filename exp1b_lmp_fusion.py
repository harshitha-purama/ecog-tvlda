"""Exp 1b: does adding baseline-corrected low-frequency LMP features to
high-gamma (feature-level fusion) recover accuracy on coarse grids?"""
import numpy as np
from ecog_pipeline import *

e, cue, _ = load("ECoG_Handpose.mat")
on, lab = cue_onsets(cue)
sh = grand_onset_shift(highgamma_logpower(e), on)
z = lambda A: (A - A.mean((0, 1))) / A.std((0, 1))

def lmp_epochs(sub):
    L = epoch(lowfreq_lmp(e[sub]), on, sh)
    return L - L[:, :5].mean(1, keepdims=True)        # pre-cue baseline per trial

def best(X):
    return max(100 * cv_accuracy(X, lab, n_comp=k, reps=10).mean() for k in (1, 2, 3))

subsets = {"60ch": list(range(60))}
for r0 in (0, 1):
    for c0 in (0, 1):
        subsets[f"10mm_r{r0}c{c0}"] = [c * 10 + r for c in range(c0, 6, 2) for r in range(r0, 10, 2)]
for name, sub in subsets.items():
    H = epoch(highgamma_logpower(e[sub]), on, sh)
    L = lmp_epochs(sub)
    print(f"{name:<12} LMP alone {best(L):5.1f}   HG+LMP {best(np.concatenate([z(H), z(L)], 2)):5.1f}")
