import os
os.makedirs('results', exist_ok=True); os.makedirs('figures', exist_ok=True)
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open("results/results_decimation.json")); g = json.load(open("results/results_glove.json"))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
sp = [5, 10, 15, 20]
m = [d[f"spacing_{s}mm"]["best_mean"] for s in sp]
lo = [d[f"spacing_{s}mm"]["best_min"] for s in sp]; hi = [d[f"spacing_{s}mm"]["best_max"] for s in sp]
ax[0].fill_between(sp, lo, hi, alpha=.2, color="C0", label="range over grid offsets")
ax[0].plot(sp, m, "o-", color="C0", label="mean over offsets")
ax[0].axhline(99.0, ls="--", c="C2", lw=1); ax[0].text(20, 99.6, "paper: HD subjects 99.0%", ha="right", fontsize=8, c="C2")
ax[0].axhline(86.9, ls="--", c="C3", lw=1); ax[0].text(20, 87.5, "paper: standard-grid subjects 86.9%", ha="right", fontsize=8, c="C3")
ax[0].axhline(100/3, ls=":", c="gray", lw=1); ax[0].text(5.2, 35, "chance", fontsize=8, c="gray")
for s, n in zip(sp, [60, 15, "6-8", "4-6"]):
    ax[0].annotate(f"{n} el.", (s, lo[sp.index(s)]), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8)
ax[0].set(xlabel="Electrode spacing (mm)", ylabel="RPS accuracy (%)", title="Simulated coarser grids, same subject",
          ylim=(30, 102), xticks=sp); ax[0].legend(fontsize=8, loc="lower right")
rs = d["random_subsets"]; n = sorted(int(k) for k in rs)
ax[1].errorbar(n, [rs[str(k)][0] for k in n], [rs[str(k)][1] for k in n], fmt="o-", capsize=3)
ax[1].axhline(100/3, ls=":", c="gray", lw=1)
ax[1].set(xscale="log", xticks=n, xticklabels=n, xlabel="Number of electrodes (random subsets)",
          ylabel="RPS accuracy (%)", title="Accuracy vs electrode count", ylim=(30, 102))
plt.tight_layout(); plt.savefig("figures/fig1_decimation.png", dpi=150)

fing = ["thumb", "index", "middle", "ring", "little"]
fig = plt.figure(figsize=(11, 6.5))
ax = fig.add_subplot(2, 1, 1); x = np.arange(5); w = .26
for i, (k, lab) in enumerate([("hg_60ch", "ECoG, 60 el. (5 mm)"), ("hg_10mm", "ECoG, 15 el. (10 mm)"),
                              ("oracle_template", "oracle: true cue + gesture template")]):
    ax.bar(x + (i - 1) * w, g[k], w, label=lab)
ax.set(xticks=x, xticklabels=fing, ylabel="Pearson r (held-out)", ylim=(0, 1.15), title="Continuous finger-flexion decoding")
ax.legend(fontsize=8, ncol=3, loc="upper center")
P = np.load("results/glove_pred_60ch.npy"); T = np.load("results/glove_true.npy")
t0, t1 = 3000, 3000 + 60 * 20; tt = np.arange(t1 - t0) / 20
for j, f in enumerate([3, 1]):
    a = fig.add_subplot(2, 2, 3 + j)
    a.plot(tt, T[t0:t1, f], c="k", lw=1, label="glove"); a.plot(tt, P[t0:t1, f], c="C1", lw=1, label="decoded")
    a.set(title=f"{fing[f]} finger, 60 s held-out", xlabel="Time (s)", ylabel="Flexion (norm.)")
    if j == 0: a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/fig2_glove.png", dpi=150)
