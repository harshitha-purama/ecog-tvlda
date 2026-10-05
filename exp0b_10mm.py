"""Apply the same two changes (later window, drop trial 0) to the simulated 10 mm grids."""
import numpy as np
from ecog_pipeline import *
e, cue, _ = load("ECoG_Handpose.mat"); on, lab = cue_onsets(cue)
sh = grand_onset_shift(highgamma_logpower(e), on); keep = np.arange(len(lab)) != 0
r = []
for r0 in (0, 1):
    for c0 in (0, 1):
        sub = [c * 10 + rw for c in range(c0, 6, 2) for rw in range(r0, 10, 2)]
        X = epoch(highgamma_logpower(e[sub]), on, sh + 0.3)[keep]
        r.append(max(100 * cv_accuracy(X, lab[keep], n_comp=k, reps=10).mean() for k in (1, 2, 3)))
print("10 mm, later window + drop trial 0:", np.round(r, 1), "mean %.1f" % np.mean(r))
