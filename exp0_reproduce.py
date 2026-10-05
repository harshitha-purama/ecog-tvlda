"""
Reproduce the paper's full-grid RPS accuracy as closely as possible.
Paper settings plus two documented changes:
  (a) trial window centred later (+0.3 s), on the actual movement rather than
      the first rise of high-gamma;
  (b) trial 0 excluded: the glove shows the hand was not at rest before the cue
      (all fingers extend after the 'open hand' cue), i.e. atypical execution.
A nested-CV estimate chooses the window shift and the number of PCA components
on training folds only, so it involves no tuning on test data.

Usage: python exp0_reproduce.py config 0|1|2     (20 x 10-fold CV, NF 1..6)
       python exp0_reproduce.py nested
       python exp0_reproduce.py summary
"""
import os
os.makedirs('results', exist_ok=True); os.makedirs('figures', exist_ok=True)
import json, os, sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from ecog_pipeline import *

e, cue, _ = load("ECoG_Handpose.mat")
on, lab = cue_onsets(cue)
if not os.path.exists("hg_full.npy"):
    np.save("hg_full.npy", highgamma_logpower(e))   # paper band 50-300 Hz, whitened
F = np.load("hg_full.npy")
sh0 = grand_onset_shift(F, on)
keep = np.arange(len(lab)) != 0
allt = np.ones_like(keep)
CONFIGS = [("paper settings", 0.0, allt),
           ("+ later window", 0.3, allt),
           ("+ later window + drop trial 0", 0.3, keep)]


def representative(X, y, reps=20, nfs=range(1, 7)):
    """Paper's rule: smallest NF within the quantization margin (1/trials per class) of the best."""
    acc = {k: cv_accuracy(X, y, n_comp=k, reps=reps) for k in nfs}
    best = max(a.mean() for a in acc.values())
    qm = 1 / np.bincount(y)[1:].min()
    k = min(k for k, a in acc.items() if a.mean() >= best - qm)
    return {"acc": 100 * acc[k].mean(), "sd": 100 * acc[k].std(), "NF": k,
            "best_any_NF": 100 * best, "per_NF": {n: 100 * a.mean() for n, a in acc.items()}}


mode = sys.argv[1]
if mode == "config":
    name, d, m = CONFIGS[int(sys.argv[2])]
    r = representative(epoch(F, on, sh0 + d)[m], lab[m])
    print(f"{name:<32} {r['acc']:.1f} +- {r['sd']:.1f}%  NF={r['NF']}  (best any NF {r['best_any_NF']:.1f})")
    json.dump(r, open(f"repro_config{sys.argv[2]}.json", "w"), indent=1, default=float)

elif mode == "nested":
    shifts, nfs = (0.0, 0.1, 0.2, 0.3, 0.4), (1, 2, 3)
    Xs = {d: epoch(F, on, sh0 + d)[keep] for d in shifts}
    y = lab[keep]
    outer, chosen = [], []
    for r in range(3):
        pred = np.empty_like(y)
        for tr, te in StratifiedKFold(10, shuffle=True, random_state=100 + r).split(Xs[0.0], y):
            scores = {}
            for d in shifts:
                for k in nfs:
                    p = np.empty(len(tr), int)
                    for itr, ite in StratifiedKFold(5, shuffle=True, random_state=r).split(tr, y[tr]):
                        p[ite] = TVLDAMulti(n_comp=k).fit(Xs[d][tr][itr], y[tr][itr]).predict(Xs[d][tr][ite])
                    scores[(d, k)] = np.mean(p == y[tr])
            d, k = max(scores, key=scores.get)
            chosen.append((d, k))
            pred[te] = TVLDAMulti(n_comp=k).fit(Xs[d][tr], y[tr]).predict(Xs[d][te])
        outer.append(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))
    r = {"acc": 100 * np.mean(outer), "sd": 100 * np.std(outer), "chosen": chosen}
    print("nested CV (no test-data tuning)   %.1f +- %.1f%%" % (r["acc"], r["sd"]))
    json.dump(r, open("repro_nested.json", "w"), indent=1, default=float)

elif mode == "summary":
    out = {CONFIGS[i][0]: json.load(open(f"repro_config{i}.json")) for i in range(3)}
    out["nested CV (drop trial 0)"] = json.load(open("repro_nested.json"))
    json.dump(out, open("results/results_reproduction.json", "w"), indent=1)
    for k, v in out.items():
        print(f"{k:<32} {v['acc']:.1f} +- {v['sd']:.1f}%")
