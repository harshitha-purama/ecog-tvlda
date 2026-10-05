"""
Preprocessing and TVLDA, following Gruenwald et al. (2019), Front. Neurosci. 13:901.

Data: ECoG_Handpose.mat, variable y (67 x N), fs = 1200 Hz
  row 0      time
  rows 1-60  ECoG electrodes 1..60 (6 x 10 high-density grid, 5 mm spacing)
  row 61     cue (0 relax, 1 fist/rock, 2 peace/scissors, 3 open/paper)
  rows 62-66 data glove (thumb, index, middle, ring, little)
"""
import numpy as np
import scipy.io as sio
import scipy.signal as ss
from scipy.linalg import solve_toeplitz

FS = 1200
LINE = 50            # Japan (Hokkaido) mains; confirmed from spectrum
BIN = 60             # 50 ms variance window at 1200 Hz -> 20 Hz feature rate
FEAT_FS = FS // BIN


def load(path):
    y = sio.loadmat(path)["y"]
    return y[1:61].astype(np.float64), y[61].astype(int), y[62:67].astype(np.float64)


# ---------------------------------------------------------------- grid layout
def grid_position(ch):
    """Electrode index 0..59 -> (row, col) on the 10-row x 6-column grid.
    Per data_description.pdf, electrodes 1-10 form the first column and
    51-60 the last."""
    return ch % 10, ch // 10


def decimated_subset(row_offset, col_offset):
    """Every other row and column -> 10 mm spacing, 5 x 3 = 15 electrodes."""
    return [ch for ch in range(60)
            if grid_position(ch)[0] % 2 == row_offset
            and grid_position(ch)[1] % 2 == col_offset]


# ---------------------------------------------------------------- preprocessing
def whiten(x, order=10):
    """Time-domain AR whitening per channel (paper eq. 1-2), coefficients from
    Yule-Walker on the full recording (label-free, so no label leakage)."""
    out = np.empty_like(x)
    for c in range(x.shape[0]):
        xc = x[c] - x[c].mean()
        seg = xc[:FS * 60]
        r = np.array([seg[:len(seg) - k] @ seg[k:] for k in range(order + 1)])
        a = solve_toeplitz(r[:order], -r[1:order + 1])
        out[c] = ss.lfilter(np.r_[1.0, a], [1.0], xc)
    return out


def highgamma_logpower(ecog, band=(50, 300), whitening=True, notch=True):
    """CAR -> notch cascade -> (whitening) -> bandpass -> 50 ms log variance.
    All filters are causal (lfilter/sosfilt), as in an online system.
    Returns (n_channels, n_bins)."""
    x = ecog - ecog.mean(0, keepdims=True)                     # common average
    x = ss.sosfilt(ss.butter(2, 1.0, "highpass", fs=FS, output="sos"), x, axis=1)  # remove DC drift
    if notch:
        for h in range(1, 7):
            f0 = LINE * h
            sos = ss.butter(3, [f0 - 2.5, f0 + 2.5], "bandstop", fs=FS, output="sos")  # 6th order
            x = ss.sosfilt(sos, x, axis=1)
    if whitening:
        x = whiten(x)
    sos = ss.butter(3, band, "bandpass", fs=FS, output="sos")  # 6th order
    x = ss.sosfilt(sos, x, axis=1)
    nb = x.shape[1] // BIN
    v = x[:, :nb * BIN].reshape(x.shape[0], nb, BIN).var(axis=2)
    return np.log(v + 1e-12)


def lowfreq_lmp(ecog, cutoff=4.0):
    """Local motor potential: CAR'd raw signal low-passed and averaged per bin."""
    x = ecog - ecog.mean(0, keepdims=True)
    x = ss.sosfilt(ss.butter(2, 0.1, "highpass", fs=FS, output="sos"), x, axis=1)
    x = ss.sosfilt(ss.butter(4, cutoff, "lowpass", fs=FS, output="sos"), x, axis=1)
    nb = x.shape[1] // BIN
    return x[:, :nb * BIN].reshape(x.shape[0], nb, BIN).mean(axis=2)


def cue_onsets(cue):
    idx = np.where((np.diff(cue) != 0) & (cue[1:] > 0))[0] + 1
    return idx, cue[idx]


def epoch(feat, onsets_samples, shift_s, half_s=0.75):
    """Cut trials of +-half_s around (cue + shift_s). Returns (trials, NS, NCh)."""
    centre = np.round((onsets_samples / FS + shift_s) * FEAT_FS).astype(int)
    h = int(round(half_s * FEAT_FS))
    return np.stack([feat[:, c - h:c + h].T for c in centre])


def grand_onset_shift(feat, onsets_samples):
    """Latency of grand-average high-gamma onset after the cue (label-free).
    The paper shifts this onset to the centre of the trial window."""
    centre = np.round(onsets_samples / FS * FEAT_FS).astype(int)
    ga = np.mean([feat[:, c - 20:c + 50].mean(0) for c in centre], axis=0)
    base = ga[:20].mean()
    ga = ga - base
    thr = 0.5 * ga[20:].max()
    return (np.argmax(ga[20:] > thr)) / FEAT_FS


# ---------------------------------------------------------------- TVLDA
def _smooth_time(a, k=2):
    """Bidirectional moving average of +-k samples along axis 0."""
    out = np.empty_like(a)
    n = a.shape[0]
    for t in range(n):
        out[t] = a[max(0, t - k):min(n, t + k + 1)].mean(0)
    return out


def _tv_stats(XA, XB, smooth=0, offdiag=1.0, ridge=1e-6):
    """Per-time-sample LDA weights w[n] and offsets d[n] (eq. 10-14)."""
    muA, muB = XA.mean(0), XB.mean(0)                    # (NS, NF)
    def covs(X, mu):
        D = X - mu
        return np.einsum("tni,tnj->nij", D, D) / (X.shape[0] - 1)
    S = 0.5 * (covs(XA, muA) + covs(XB, muB))            # (NS, NF, NF)
    if smooth:
        muA, muB, S = _smooth_time(muA, smooth), _smooth_time(muB, smooth), _smooth_time(S, smooth)
    if offdiag < 1.0:
        diag = np.einsum("nii->ni", S)
        S = offdiag * S
        idx = np.arange(S.shape[1])
        S[:, idx, idx] = diag
    nf = S.shape[1]
    S = S + ridge * np.trace(S, axis1=1, axis2=2)[:, None, None] / nf * np.eye(nf)
    dmu = muB - muA
    W = np.stack([np.linalg.solve(S[n], dmu[n]) for n in range(len(dmu))])  # (NS, NF)
    d = 0.5 * np.einsum("ni,ni->n", W, muA + muB)
    return W, d


class TVLDABinary:
    """Binary TVLDA with intrinsic PCA feature reduction (section 2.7)."""
    def __init__(self, n_comp=1, smooth=2, offdiag=1.0):
        self.n_comp, self.smooth, self.offdiag = n_comp, smooth, offdiag

    def fit(self, XA, XB):
        Wy, _ = _tv_stats(XA, XB, smooth=self.smooth, offdiag=self.offdiag)
        _, _, Vt = np.linalg.svd(Wy, full_matrices=False)
        self.P = Vt[:self.n_comp].T                       # (NCh, NF)
        self.W, self.d = _tv_stats(XA @ self.P, XB @ self.P, smooth=0)
        return self

    def score(self, X):
        """Negative -> class A, positive -> class B (eq. 31)."""
        Xp = X @ self.P
        return np.einsum("tni,ni->t", Xp, self.W) - self.d.sum()


class TVLDAMulti:
    """One-vs-one TVLDA with the paper's min-max decision rule (eq. 37)."""
    def __init__(self, **kw):
        self.kw = kw

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = {}
        for i, a in enumerate(self.classes):
            for b in self.classes[i + 1:]:
                self.models[(a, b)] = TVLDABinary(**self.kw).fit(X[y == a], X[y == b])
        return self

    def predict(self, X):
        C = len(self.classes)
        Z = np.zeros((len(X), C, C))                      # Z[:, p, q]: low -> favours p over q
        for (a, b), m in self.models.items():
            ia, ib = np.searchsorted(self.classes, [a, b])
            s = m.score(X)
            Z[:, ia, ib], Z[:, ib, ia] = s, -s
        Z[:, np.arange(C), np.arange(C)] = -np.inf
        return self.classes[np.argmin(Z.max(axis=2), axis=1)]


def cv_accuracy(X, y, n_comp=1, reps=10, folds=10, seed=0, **kw):
    """Repeated stratified k-fold; returns per-repetition balanced accuracy."""
    from sklearn.model_selection import StratifiedKFold
    accs = []
    for r in range(reps):
        pred = np.empty_like(y)
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(X, y):
            pred[te] = TVLDAMulti(n_comp=n_comp, **kw).fit(X[tr], y[tr]).predict(X[te])
        accs.append(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))
    return np.array(accs)
