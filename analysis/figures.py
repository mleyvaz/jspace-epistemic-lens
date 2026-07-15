# -*- coding: utf-8 -*-
"""Figura 1 (masas log + IC bootstrap) y scores (T,I,F)/H calibrados contra la condicion control."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"
df = pd.read_csv("data/jspace_battery.csv")
LAYERS = sorted(df.layer.unique())
CLEAN = [l for l in LAYERS if l >= 22]
rng = np.random.default_rng(7)

def boot_ci(a, n=4000):
    a = np.asarray(a)
    means = a[rng.integers(0, len(a), (n, len(a)))].mean(axis=1)
    return a.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

# ---------- Figura 1: masas log crudas ----------
colors = {"T": "tab:blue", "I": "tab:orange", "F": "tab:green"}
titles = {"kable": "First-person false belief (KaBLE-style)",
          "conflict": "Conflicting evidence",
          "control": "Factual control"}
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, cond in zip(axes, ["conflict", "kable", "control"]):
    sub = df[df.cond == cond]
    for comp, c in colors.items():
        mu, lo, hi = [], [], []
        for L in LAYERS:
            m, l, h = boot_ci(sub[sub.layer == L][f"log_{comp}"].values)
            mu.append(m); lo.append(l); hi.append(h)
        ax.plot(LAYERS, mu, marker="o", ms=3.5, color=c, label=comp)
        ax.fill_between(LAYERS, lo, hi, color=c, alpha=0.18)
    ax.axvspan(17.5, 21.5, color="gray", alpha=0.13)
    ax.set_title(f"{titles[cond]} (n=20)", fontsize=10)
    ax.set_xlabel("layer"); ax.grid(alpha=0.3)
axes[0].set_ylabel(r"$\log_{10}$ lexicon probability mass (mean $\pm$ 95% CI)")
axes[0].legend(fontsize=9, title="lexicon", title_fontsize=8)
plt.tight_layout()
plt.savefig(OUT + "/fig1_logmass.png", dpi=220)
plt.close()

# ---------- Scores (T,I,F) y H calibrados contra control ----------
def score(x, base):
    mu, sd = base.mean(), max(base.std(ddof=1), 0.5)
    return np.maximum(0.0, np.tanh((x - mu) / sd / 2))

rows = []
for cond in ["conflict", "kable"]:
    for L in CLEAN:
        base = {k: df[(df.cond == "control") & (df.layer == L)][f"log_{k}"].values
                for k in "TIF"}
        sub = df[(df.cond == cond) & (df.layer == L)]
        T = score(sub.log_T.values, base["T"])
        I = score(sub.log_I.values, base["I"])
        F = score(sub.log_F.values, base["F"])
        H = np.maximum(0.0, T + I + F - 1)
        rows.append(dict(cond=cond, layer=L, T=T.mean(), I=I.mean(), F=F.mean(),
                         H=H.mean(), H_pos=(H > 0).mean()))
sc = pd.DataFrame(rows)
sc.to_csv("data/jspace_scores_vs_control.csv", index=False)
print("SCORES MEDIOS (calibrados contra condicion control, n=20):")
print(sc.round(2).to_string(index=False))

# ---------- Figura 2: H por capa ----------
fig, ax = plt.subplots(figsize=(6.5, 4))
for cond, c, lbl in [("conflict", "tab:red", "Conflicting evidence"),
                     ("kable", "tab:purple", "False belief (KaBLE)")]:
    s = sc[sc.cond == cond]
    ax.plot(s.layer, s.H, marker="o", color=c, label=lbl)
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("layer"); ax.set_ylabel(r"mean hyper-truth $H=\max(0,\,T+I+F-1)$")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + "/fig2_hypertruth.png", dpi=220)
print("\nfiguras guardadas en", OUT)
