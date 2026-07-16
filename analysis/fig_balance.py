# -*- coding: utf-8 -*-
"""Figura del balance neutrosofico B = log10 F - log10 T por capa y condicion."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "."
old = pd.read_csv(BASE + "/data/jspace_battery.csv")
new = pd.read_csv(BASE + "/data/jspace_controls.csv")
for df in (old, new):
    df["balance"] = df.log_F - df.log_T
allc = pd.concat([old, new])
LAYERS = sorted(allc.layer.unique())
rng = np.random.default_rng(7)

def boot_ci(a, n=4000):
    a = np.asarray(a)
    means = a[rng.integers(0, len(a), (n, len(a)))].mean(axis=1)
    return a.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

series = [("conflict", "tab:red", "Conflicting evidence"),
          ("agree", "tab:green", "Agreeing evidence (template control)"),
          ("kable", "tab:purple", "False first-person belief"),
          ("kable_true", "tab:gray", "True first-person belief (control)")]

fig, ax = plt.subplots(figsize=(8.5, 5))
for cond, c, lbl in series:
    sub = allc[allc.cond == cond]
    mu, lo, hi = [], [], []
    for L in LAYERS:
        m, l, h = boot_ci(sub[sub.layer == L].balance.values)
        mu.append(m); lo.append(l); hi.append(h)
    ax.plot(LAYERS, mu, marker="o", ms=4, color=c, label=lbl)
    ax.fill_between(LAYERS, lo, hi, color=c, alpha=0.15)
ax.axhline(0, color="black", lw=0.9)
ax.axvspan(17.5, 21.5, color="gray", alpha=0.13)
ax.text(24.5, 0.12, "F dominates", fontsize=9, color="dimgray")
ax.text(24.5, -0.35, "T dominates", fontsize=9, color="dimgray")
ax.set_xlabel("layer")
ax.set_ylabel(r"neutrosophic verdict balance  $B=\log_{10}F-\log_{10}T$")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(BASE + "/figures/fig3_balance.png", dpi=220)
print("fig3 guardada")
