# -*- coding: utf-8 -*-
"""Estadistica de la bateria J-space: contrastes por capa vs control + tamanos de efecto."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
from scipy import stats as st

df = pd.read_csv("data/jspace_battery.csv")
LAYERS_CLEAN = [22, 23, 24, 25, 26, 27, 28, 29, 30]  # banda 18-21 excluida

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*a.std(ddof=1)**2 + (nb-1)*b.std(ddof=1)**2) / (na+nb-2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan

print("=" * 90)
print("CONTRASTES POR CAPA (condicion vs control), Mann-Whitney U + Cohen's d, capas limpias 22-30")
print("=" * 90)
tests = [("conflict", "log_F", "F: conflicto vs control"),
         ("conflict", "log_T", "T: conflicto vs control"),
         ("kable",    "log_I", "I: KaBLE vs control"),
         ("kable",    "log_F", "F: KaBLE vs control")]
for cond, col, label in tests:
    print(f"\n--- {label} ---")
    for L in LAYERS_CLEAN:
        a = df[(df.cond == cond) & (df.layer == L)][col].values
        b = df[(df.cond == "control") & (df.layer == L)][col].values
        u, p = st.mannwhitneyu(a, b, alternative="greater")
        d = cohens_d(a, b)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  capa {L:>2}: media={a.mean():6.2f} vs control={b.mean():6.2f}  "
              f"delta={a.mean()-b.mean():5.2f}  d={d:5.2f}  p={p:.2e} {sig}")

print("\n" + "=" * 90)
print("COLAPSO DEL VEREDICTO (conflicto): masa F pico vs salida, por item")
print("=" * 90)
sub = df[df.cond == "conflict"]
peaks, outs, drops = [], [], []
for i in sub.item.unique():
    it = sub[sub.item == i]
    peak = it[it.layer.isin([22, 23, 24, 25, 26])].log_F.max()
    out = it[it.layer == 30].log_F.iloc[0]
    peaks.append(peak); outs.append(out); drops.append(peak - out)
drops = np.array(drops)
w, pw = st.wilcoxon(drops, alternative="greater")
print(f"  caida log10 (pico capas 22-26 -> capa 30): media={drops.mean():.2f} ordenes, "
      f"mediana={np.median(drops):.2f}, min={drops.min():.2f}, max={drops.max():.2f}")
print(f"  items con caida >1 orden: {(drops > 1).sum()}/20 | >2 ordenes: {(drops > 2).sum()}/20")
print(f"  Wilcoxon (caida>0): W={w:.0f}, p={pw:.2e}")

print("\n" + "=" * 90)
print("COEXISTENCIA T+F (conflicto, capas 22-26): items donde ambas masas superan la del control")
print("=" * 90)
for L in [22, 23, 24, 25, 26]:
    ctlT = df[(df.cond == "control") & (df.layer == L)].log_T
    ctlF = df[(df.cond == "control") & (df.layer == L)].log_F
    thrT, thrF = ctlT.mean() + 2*ctlT.std(ddof=1), ctlF.mean() + 2*ctlF.std(ddof=1)
    it = df[(df.cond == "conflict") & (df.layer == L)]
    both = ((it.log_T > thrT) & (it.log_F > thrF)).sum()
    print(f"  capa {L}: {both}/20 items con T y F simultaneamente > media control + 2sd")

print("\n" + "=" * 90)
print("FIRMA KaBLE: dominancia de I sobre F por item (capas 22-30)")
print("=" * 90)
sub = df[(df.cond == "kable") & (df.layer.isin(LAYERS_CLEAN))]
dom = sub.groupby("item").apply(lambda g: (g.log_I - g.log_F).mean(), include_groups=False)
print(f"  items con I>F en promedio: {(dom > 0).sum()}/20 | delta medio: {dom.mean():.2f} ordenes")

# medias por condicion/capa para el paper
print("\nMEDIAS POR CONDICION Y CAPA (log10 masa):")
piv = df[df.layer.isin(LAYERS_CLEAN)].groupby(["cond", "layer"])[["log_T", "log_I", "log_F", "ent"]].mean().round(2)
print(piv.to_string())
