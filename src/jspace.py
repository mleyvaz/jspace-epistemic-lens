# -*- coding: utf-8 -*-
"""Lexicon projection of Jacobian-lens readouts into epistemic triplets (T, I, F).

Companion code for "Verdict Collapse and Sustained Indeterminacy: Reading
Epistemic States of Language Models with the Jacobian Lens".

Requires the `jlens` package from https://github.com/anthropics/jacobian-lens
and a (pre-fitted) JacobianLens for your model.

Typical usage in a notebook (see notebooks/colab_jspace_battery.ipynb):

    import jspace
    lexids = jspace.build_lexicons(tokenizer)
    jl, _, _ = lens.apply(model, prompt, layers=range(18, 31), positions=[-2])
    for layer, logits in jl.items():
        masses, entropy = jspace.log_masses(logits[0], lexids)
"""
import math

import torch

# Judgment-only lexicons. High-frequency function words (not, no, yes) are
# deliberately excluded: their polysemy saturates the measure everywhere,
# including factual controls (see paper, Section 3.2).
LEXICONS = {
    "T": ["true", "correct", "accurate", "valid", "confirmed", "truth",
          "factual", "正确"],
    "F": ["false", "incorrect", "wrong", "myth", "invalid", "inaccurate",
          "impossible", "mistaken", "错误", "是错误的"],
    "I": ["depends", "maybe", "unclear", "uncertain", "possibly", "perhaps",
          "disputed", "unknown", "debated", "ambiguous", "可能"],
}


def build_lexicons(tokenizer, lexicons=None):
    """Expand word lists to single-token id sets (case and leading-space variants)."""
    lexicons = lexicons or LEXICONS
    out = {}
    for key, words in lexicons.items():
        ids = set()
        for w in words:
            variants = {w, w.capitalize(), w.upper(),
                        " " + w, " " + w.capitalize(), " " + w.upper()}
            for v in variants:
                t = tokenizer.encode(v, add_special_tokens=False)
                if len(t) == 1:
                    ids.add(t[0])
        out[key] = torch.tensor(sorted(ids))
    return out


def log_masses(logits, lexids):
    """log10 probability mass per lexicon + normalized entropy of the lens distribution."""
    q = torch.softmax(logits.float(), dim=-1)
    entropy = -(q * (q + 1e-12).log()).sum().item() / math.log(q.numel())
    masses = {k: math.log10(q[ids].sum().item() + 1e-12) for k, ids in lexids.items()}
    return masses, entropy


def score(x, baseline_values, sd_floor=0.5):
    """Rectified tanh of the z-score in log space against a control baseline.

    Do NOT renormalize masses over the union of lexicons: that forces
    T + I + F = 1 by construction and destroys the paraconsistent signal
    (see paper, Section 3.2). Each component is calibrated independently.
    """
    n = len(baseline_values)
    mu = sum(baseline_values) / n
    var = sum((v - mu) ** 2 for v in baseline_values) / n
    sd = max(var ** 0.5, sd_floor)
    return max(0.0, math.tanh((x - mu) / sd / 2))


def hyper_truth(t, i, f):
    """H = max(0, T + I + F - 1): positive only when components jointly exceed
    what any single-valued truth assignment could produce."""
    return max(0.0, t + i + f - 1)
