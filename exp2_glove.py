"""
Experiment 2: continuous decoding of finger flexion from ECoG.
Ridge regression from causally lagged features (past 0-500 ms) to the five
data-glove channels, evaluated with contiguous-block cross-validation.
The decoder never sees cues, so it runs continuously (no triggers).
"""
import os
os.makedirs('results', exist_ok=True); os.makedirs('figures', exist_ok=True)
import json
import numpy as np
from sklearn.linear_model import RidgeCV
from ecog_pipeline import *

FINGERS = ["thumb", "index", "middle", "ring", "little"]
LAGS = 11                  # 0..10 bins = 0..500 ms of past features
FOLDS = 5
GAP = FEAT_FS              # 1 s discarded at each test-block edge
ALPHAS = np.logspace(0, 5, 11)

e, cue, glove = load("ECoG_Handpose.mat")
nb = e.shape[1] // BIN
G = glove[:, :nb * BIN].reshape(5, nb, BIN).mean(2).T          # (T, 5) at 20 Hz
cue_b = cue[:nb * BIN].reshape(nb, BIN)[:, 0]


def lagged(F):
    """(C, T) -> (T, C*LAGS), row t holds features from t-LAGS+1..t (causal)."""
    C, T = F.shape
    out = np.zeros((T, C * LAGS))
    for l in range(LAGS):
        out[l:, l * C:(l + 1) * C] = F[:, :T - l].T
    return out


def features(ecog_subset, lmp=False):
    F = highgamma_logpower(ecog_subset)
    if lmp:
        L = lowfreq_lmp(ecog_subset)[:, :F.shape[1]]
        F = np.vstack([F, L])
    return lagged(F)


def smooth_pred(p, k=5):
    """Causal moving average of the decoder output (250 ms)."""
    out = np.copy(p)
    for t in range(len(p)):
        out[t] = p[max(0, t - k + 1):t + 1].mean(0)
    return out


def blocked_cv(Z, Y):
    T = len(Y)
    edges = np.linspace(LAGS, T, FOLDS + 1).astype(int)
    rs, preds = [], np.full_like(Y, np.nan)
    for f in range(FOLDS):
        te = np.arange(edges[f], edges[f + 1])
        tr_mask = np.ones(T, bool)
        tr_mask[:LAGS] = False
        tr_mask[max(0, te[0] - GAP):min(T, te[-1] + GAP + 1)] = False
        mu, sd = Z[tr_mask].mean(0), Z[tr_mask].std(0) + 1e-9
        m = RidgeCV(alphas=ALPHAS).fit((Z[tr_mask] - mu) / sd, Y[tr_mask])
        p = smooth_pred(m.predict((Z[te] - mu) / sd))
        preds[te] = p
        rs.append([np.corrcoef(p[:, j], Y[te, j])[0, 1] for j in range(Y.shape[1])])
    return np.array(rs).mean(0), preds


def oracle_template():
    """Upper-bound reference: knows the true cue onsets and gestures and
    predicts each gesture's average glove trajectory. If the ECoG decoder
    approaches this, it is mostly recognising gestures rather than tracking
    finger-specific variation within a gesture."""
    on, lab = cue_onsets(cue)
    onb = on // BIN
    win = 5 * FEAT_FS
    templ = {c: np.mean([G[o:o + win] for o, l in zip(onb, lab) if l == c and o + win <= len(G)], 0)
             for c in np.unique(lab)}
    rest = G[cue_b == 0].mean(0)
    P = np.tile(rest, (len(G), 1))
    for o, l in zip(onb, lab):
        n = min(win, len(G) - o)
        P[o:o + n] = templ[l][:n]
    return np.array([np.corrcoef(P[LAGS:, j], G[LAGS:, j])[0, 1] for j in range(5)])


results = {}

r, preds_full = blocked_cv(features(e), G)
results["hg_60ch"] = r.tolist()
np.save("results/glove_pred_60ch.npy", preds_full); np.save("results/glove_true.npy", G)
print("HG, 60 ch          ", np.round(r, 2), "mean %.2f" % r.mean())

r, _ = blocked_cv(features(e, lmp=True), G)
results["hg_lmp_60ch"] = r.tolist()
print("HG+LMP, 60 ch      ", np.round(r, 2), "mean %.2f" % r.mean())

for name, lmp in (("hg_10mm", False), ("hg_lmp_10mm", True)):
    rr = []
    for r0 in (0, 1):
        for c0 in (0, 1):
            sub = [c * 10 + rw for c in range(c0, 6, 2) for rw in range(r0, 10, 2)]
            rr.append(blocked_cv(features(e[sub], lmp=lmp), G)[0])
    results[name] = np.mean(rr, 0).tolist()
    print(f"{name:<19}", np.round(np.mean(rr, 0), 2), "mean %.2f" % np.mean(rr))

o = oracle_template()
results["oracle_template"] = o.tolist()
print("oracle template    ", np.round(o, 2), "mean %.2f" % o.mean())

json.dump(results, open("results/results_glove.json", "w"), indent=1)
