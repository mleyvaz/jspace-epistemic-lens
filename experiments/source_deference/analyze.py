# -*- coding: utf-8 -*-
"""Analysis of the source-deference battery (full run).

Computes, from source_deference_full.csv:
  - overall truth-tracking (all facts are TRUE => truth-tracking => mean B < 0)
  - R  : recency index (nuisance; balanced design cancels it when order-averaged)
  - affirm-credence per source type: how negative B is when type X vouches for
    the true claim (more negative = more internally credited)
  - deny-pull per source type: how positive B is when type X denies the true
    claim (more positive = more followed into falsehood)
  - G  : geographic discount = affirm_credence(GS) - affirm_credence(NE)
         (>0 => Global-South evidence internally discounted)
  - D_prestige : prestige-over-truth = deny_pull(NE) - deny_pull(GS)
         (>0 => the elite is followed into falsehood more than a Southern source)
All indices are order-averaged, so recency (which is balanced across conditions)
cancels out.

B = log10(F) - log10(T), averaged over clean layers 22-30, per stimulus.
Run from the repo root or from this folder with the CSV alongside.
"""
import io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "source_deference_full.csv")
if not os.path.exists(CSV):
    CSV = "source_deference_full.csv"
df = pd.read_csv(CSV)
if "B" not in df.columns:
    df["B"] = df.log_F - df.log_T

CLEAN = list(range(22, 31))
d = df[df.layer.isin(CLEAN)]
# per-stimulus mean balance
ps = d.groupby(["id", "lang", "pair", "aff_type", "den_type", "aff_first"]).B.mean().reset_index()

def boot_ci(a, n=4000, seed=7):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    if len(a) == 0:
        return (np.nan, np.nan, np.nan)
    means = a[rng.integers(0, len(a), (n, len(a)))].mean(axis=1)
    return a.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

def report(a, b, label, alt="two-sided"):
    ma, la, ha = boot_ci(a); mb, lb, hb = boot_ci(b)
    if len(a) and len(b):
        u, p = st.mannwhitneyu(a, b, alternative=alt)
    else:
        p = np.nan
    print(f"  {label}: {ma:+.3f} [{la:+.2f},{ha:+.2f}]  vs  {mb:+.3f} [{lb:+.2f},{hb:+.2f}]"
          f"   diff={ma-mb:+.3f}  p={p:.1e}")
    return ma - mb

def run(sub, tag):
    print("\n" + "=" * 74)
    print(f"CONDITION SET: {tag}  (n stimuli = {len(sub)})")
    print("=" * 74)
    print(f"  overall mean B = {sub.B.mean():+.3f}   (negative => truth-tracking dominates)")

    # recency
    bl = sub[sub.aff_first == True].B    # denier last
    bf = sub[sub.aff_first == False].B   # denier first
    print("\n RECENCY (nuisance):")
    report(bl, bf, "B(denier last) vs B(denier first)")

    # affirm-credence and deny-pull per type
    print("\n AFFIRM-CREDENCE per source type (more NEGATIVE = truth more credited):")
    ac = {}
    for X in ("NE", "GS", "WK"):
        vals = sub[sub.aff_type == X].B
        if len(vals):
            m, lo, hi = boot_ci(vals)
            ac[X] = vals
            print(f"   {X}: {m:+.3f} [{lo:+.2f},{hi:+.2f}]  (n={len(vals)})")

    print("\n DENY-PULL per source type (more POSITIVE = followed into falsehood):")
    dp = {}
    for X in ("NE", "GS", "WK"):
        vals = sub[sub.den_type == X].B
        if len(vals):
            m, lo, hi = boot_ci(vals)
            dp[X] = vals
            print(f"   {X}: {m:+.3f} [{lo:+.2f},{hi:+.2f}]  (n={len(vals)})")

    print("\n KEY INDICES:")
    if "GS" in ac and "NE" in ac:
        G = report(ac["GS"], ac["NE"], "G  geographic discount  B(GS affirms)-B(NE affirms)", alt="greater")
        print("     G>0 => Global-South-affirmed truth is internally LESS credited than North-elite.")
    if "NE" in dp and "GS" in dp:
        D = report(dp["NE"], dp["GS"], "D_prestige  B(NE denies)-B(GS denies)", alt="greater")
        print("     D>0 => the elite is followed into falsehood MORE than a Southern source (authority>truth).")

for lang, tag in [(None, "ALL"), ("en", "ENGLISH"), ("es", "SPANISH")]:
    sub = ps if lang is None else ps[ps.lang == lang]
    if len(sub):
        run(sub, tag)

# ---- figure: credence ladder ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (col, title, ylab) in zip(axes, [
            ("aff_type", "Affirm-credence (lower = truth more credited)", "mean B when source affirms"),
            ("den_type", "Deny-pull (higher = followed into falsehood)", "mean B when source denies")]):
        for j, lang in enumerate(["en", "es"]):
            s = ps[ps.lang == lang]
            types = ["NE", "GS", "WK"]
            means = [s[s[col] == X].B.mean() for X in types]
            ax.bar(np.arange(3) + (j - 0.5) * 0.35, means, width=0.33,
                   label={"en": "English", "es": "Spanish"}[lang])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(3)); ax.set_xticklabels(["North-elite", "Global South", "Weak/blog"])
        ax.set_title(title, fontsize=9); ax.set_ylabel(ylab, fontsize=9); ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(HERE, "fig_source_credence.png")
    plt.savefig(out, dpi=200)
    print("\nfigure saved:", out)
except Exception as e:
    print("\n(figure skipped:", e, ")")
