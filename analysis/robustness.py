# -*- coding: utf-8 -*-
"""Apendice C: robustez del squashing (tanh rectificada vs min-max vs rank/ECDF)."""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd

df = pd.read_csv("data/jspace_battery.csv")
CLEAN = [l for l in sorted(df.layer.unique()) if l >= 22]

def s_tanh(x, base):
    mu, sd = base.mean(), max(base.std(ddof=1), 0.5)
    return np.maximum(0.0, np.tanh((x - mu) / sd / 2))

def s_minmax(x, base):
    # 0 = maximo del control; 1 = tres ordenes de magnitud por encima
    return np.clip((np.asarray(x) - base.max()) / 3.0, 0, 1)

def s_rank(x, base):
    # ECDF contra control: fraccion de items control estrictamente por debajo
    return np.array([(base < v).mean() for v in np.asarray(x)])

METHODS = {"rect-tanh (paper)": s_tanh, "min-max": s_minmax, "rank/ECDF": s_rank}

print(f"{'metodo':<20} {'confl F':>8} {'kable I':>8} {'confl %H>0':>11} {'kable %H>0':>11} {'orden OK':>9}")
for name, fn in METHODS.items():
    cF, kI, cH, kH, order_ok = [], [], [], [], []
    for L in CLEAN:
        base = {k: df[(df.cond == "control") & (df.layer == L)][f"log_{k}"].values for k in "TIF"}
        scores = {}
        for cond in ["conflict", "kable"]:
            sub = df[(df.cond == cond) & (df.layer == L)]
            T = fn(sub.log_T.values, base["T"])
            I = fn(sub.log_I.values, base["I"])
            F = fn(sub.log_F.values, base["F"])
            H = np.maximum(0.0, T + I + F - 1)
            scores[cond] = dict(T=T.mean(), I=I.mean(), F=F.mean(), Hpos=(H > 0).mean())
        cF.append(scores["conflict"]["F"]); cH.append(scores["conflict"]["Hpos"])
        kI.append(scores["kable"]["I"]);    kH.append(scores["kable"]["Hpos"])
        # invariantes cualitativos: F domina en conflicto; I domina en KaBLE
        order_ok.append(scores["conflict"]["F"] > scores["conflict"]["I"]
                        and scores["kable"]["I"] > scores["kable"]["F"])
    print(f"{name:<20} {np.mean(cF):>8.2f} {np.mean(kI):>8.2f} {np.mean(cH)*100:>10.0f}% "
          f"{np.mean(kH)*100:>10.0f}% {sum(order_ok)}/{len(order_ok)}")
