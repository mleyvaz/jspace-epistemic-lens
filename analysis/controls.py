# -*- coding: utf-8 -*-
"""Controles de confundido C1 (agree) y C2 (kable_true) vs condiciones originales."""
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

def contrast(dfa, ca, dfb, cb, col, label):
    print(f"\n--- {label} ---")
    survives = 0
    for L in CLEAN:
        a = dfa[(dfa.cond == ca) & (dfa.layer == L)][col].values
        b = dfb[(dfb.cond == cb) & (dfb.layer == L)][col].values
        u, p = st.mannwhitneyu(a, b, alternative="greater")
        d = cohens_d(a, b)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        flag = ""
        if p < 0.01 and (a.mean() - b.mean()) >= 1.5:
            survives += 1
            flag = "  <-- sobrevive"
        print(f"  capa {L:>2}: {ca}={a.mean():6.2f}  {cb}={b.mean():6.2f}  "
              f"delta={a.mean()-b.mean():5.2f}  d={d:5.2f}  p={p:.1e} {sig}{flag}")
    print(f"  CAPAS QUE SOBREVIVEN (delta>=1.5 y p<0.01): {survives}/{len(CLEAN)}")
    return survives

print("=" * 86)
print("CONTROL 1: F conflicto vs F agree (mismo template, fuentes concuerdan)")
print("=" * 86)
s1 = contrast(old, "conflict", new, "agree", "log_F", "F: conflicto vs acuerdo")

print()
print("=" * 86)
print("CONTROL 1b: ¿el template 'agree' quedo limpio? (agree vs control factual)")
print("=" * 86)
contrast(new, "agree", old, "control", "log_F", "F: agree vs control factual")

print()
print("=" * 86)
print("CONTROL 2: I KaBLE-mito vs I KaBLE-verdad (mismo template, creencia verdadera)")
print("=" * 86)
s2 = contrast(old, "kable", new, "kable_true", "log_I", "I: mito vs verdad")

print()
print("=" * 86)
print("EXTRA: T en agree (¿el acuerdo produce veredicto T interno?)")
print("=" * 86)
for L in [22, 24, 26, 28, 30]:
    a = new[(new.cond == "agree") & (new.layer == L)].log_T.values
    c = old[(old.cond == "control") & (old.layer == L)].log_T.values
    f = new[(new.cond == "agree") & (new.layer == L)].log_F.values
    print(f"  capa {L:>2}: T_agree={a.mean():6.2f}  F_agree={f.mean():6.2f}  "
          f"T_control={c.mean():6.2f}")

print("\nRESUMEN MEDIAS (capas limpias 22-30):")
allc = pd.concat([old, new])
piv = allc[allc.layer.isin(CLEAN)].groupby("cond")[["log_T", "log_I", "log_F"]].mean().round(2)
print(piv.to_string())
