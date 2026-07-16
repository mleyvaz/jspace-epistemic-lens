# -*- coding: utf-8 -*-
"""Analisis de rescate: el BALANCE del veredicto (log_F - log_T) vs la evidencia,
con template identico (conflict vs agree)."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
from scipy import stats as st

BASE = "data"
new = pd.read_csv(BASE + "/jspace_controls.csv")
old = pd.read_csv(BASE + "/jspace_battery.csv")
CLEAN = list(range(22, 31))

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*a.std(ddof=1)**2 + (nb-1)*b.std(ddof=1)**2) / (na+nb-2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan

for df in (new, old):
    df["balance"] = df.log_F - df.log_T   # >0: F domina; <0: T domina

print("BALANCE DEL VEREDICTO (log F - log T) — mismo template, evidencia opuesta")
print("=" * 88)
surv = 0
for L in CLEAN:
    a = old[(old.cond == "conflict") & (old.layer == L)].balance.values
    b = new[(new.cond == "agree") & (new.layer == L)].balance.values
    u, p = st.mannwhitneyu(a, b, alternative="greater")
    d = cohens_d(a, b)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    if p < 0.01:
        surv += 1
    print(f"  capa {L:>2}: conflicto={a.mean():+6.2f}  acuerdo={b.mean():+6.2f}  "
          f"delta={a.mean()-b.mean():5.2f}  d={d:5.2f}  p={p:.1e} {sig}")
print(f"\n  capas con p<0.01: {surv}/{len(CLEAN)}")

# clasificacion por item: ¿el signo del balance predice la condicion?
print("\nCLASIFICACION POR ITEM (balance medio capas 22-26):")
for cond, df, expect in [("conflict", old, "F>T"), ("agree", new, "T>F")]:
    sub = df[(df.cond == cond) & (df.layer.isin([22, 23, 24, 25, 26]))]
    bal = sub.groupby("item").balance.mean()
    ok = (bal > 0).sum() if expect == "F>T" else (bal < 0).sum()
    print(f"  {cond:<9} esperado {expect}: {ok}/20 items correctos")

# ¿y KaBLE? balance mito vs verdad (buscando cualquier senal epistemtica residual)
print("\nKaBLE — balance F-T mito vs verdad (residual):")
for L in [22, 24, 26, 28, 30]:
    a = old[(old.cond == "kable") & (old.layer == L)].balance.values
    b = new[(new.cond == "kable_true") & (new.layer == L)].balance.values
    u, p = st.mannwhitneyu(a, b, alternative="greater")
    print(f"  capa {L:>2}: mito={a.mean():+6.2f}  verdad={b.mean():+6.2f}  "
          f"delta={a.mean()-b.mean():+5.2f}  p={p:.1e}")
