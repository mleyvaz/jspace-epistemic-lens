# jspace-epistemic-lens

Code, stimuli, and data for **"Verdict Collapse and Sustained Indeterminacy:
Reading Epistemic States of Language Models with the Jacobian Lens"**
(preprint, 2026).

We project readouts of Anthropic's [Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html)
([code](https://github.com/anthropics/jacobian-lens)) onto small lexicons of
support (*T*), refutation (*F*), and hedging (*I*), calibrated against matched
factual controls, and track the resulting epistemic triplet across layers.

![Figure 1](figures/fig1_logmass.png)

## Findings (Qwen3.5-4B, official pre-fitted lens, n=20 per condition)

- **Verdict collapse.** Under conflicting evidence, refutation mass surges
  3.3–4.7 orders of magnitude above control at layers 22–26 (Cohen's *d* =
  1.5–3.3, all *p* < 1e-4), then falls by a median of **2.4 orders of
  magnitude** before the output layer (19/20 items; Wilcoxon *p* = 9.5e-7).
  The output emits hedged modality instead. The verdict is suppressed, not
  erased: residual refutation mass at the last layer still sits ~2.4 orders
  above control.
- **Sustained indeterminacy.** Under first-person false beliefs (KaBLE-style),
  hedging mass dominates refutation in **20/20 items** (mean gap 3.1 orders)
  and persists undiminished to the output. No suppressed verdict — the
  intermediate layers never commit.
- **Paraconsistent coexistence.** In 50–80% of conflict items, support and
  refutation masses are *simultaneously* elevated >2 SD above control at the
  same layer and position — a state with no classical description, summarized
  by the hyper-truth index H = max(0, T+I+F−1).

## Repository layout

```
src/jspace.py        lexicon projection: build_lexicons, log_masses, score, hyper_truth
stimuli/stimuli.json 60 prompts (20 conflict / 20 false-belief / 20 control)
notebooks/           end-to-end Colab notebook (T4 GPU, ~25 min)
data/                battery results (per item x layer log masses) + calibrated scores
analysis/            statistics.py (contrasts), figures.py, robustness.py
figures/             paper figures
paper/               LaTeX source and compiled PDF
```

## Reproduce

**Battery (GPU):** open `notebooks/colab_jspace_battery.ipynb` in Google Colab
(T4), run all cells. Downloads Qwen3.5-4B and the official pre-fitted lens
from `neuronpedia/jacobian-lens`, runs the 60 stimuli, and writes
`jspace_battery.csv`.

**Analysis (CPU, from the released CSV):**

```bash
pip install pandas numpy scipy matplotlib
python analysis/statistics.py   # per-layer contrasts, collapse and coexistence stats
python analysis/figures.py      # Figures 1-2 + calibrated scores
python analysis/robustness.py   # squashing-choice robustness (paper Appendix C)
```

Run from the repository root.

## Method in one paragraph

At each layer ℓ the lens gives a distribution `q` over the vocabulary for a
chosen position (we use the penultimate token). For each lexicon X we compute
the log mass `m_X = log10 Σ_{t∈L_X} q(t)`. Raw log masses are the primary
evidence. Calibrated scores are rectified-tanh z-scores against the control
condition, computed **independently per component** — we never renormalize
over the lexicon union, since that would force T+I+F=1 by construction and
destroy the paraconsistent signal we are testing for. See `src/jspace.py`.

## Caveats

Correlational readouts of dispositions, not causal claims about beliefs; a
single 4B model; small English-centric lexicons (with single-token Chinese
additions); one position per prompt. See the paper's Limitations section.

## Citation

```bibtex
@article{leyva2026verdict,
  title  = {Verdict Collapse and Sustained Indeterminacy: Reading Epistemic
            States of Language Models with the Jacobian Lens},
  author = {Leyva-V{\'a}zquez, Maikel and Matheu P{\'e}rez, Alexis and
            Smarandache, Florentin},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/mleyvaz/jspace-epistemic-lens}
}
```

## License

MIT (this repository). The Jacobian lens code and pre-fitted lenses are by
Anthropic (Apache 2.0) and are **not** bundled here; the notebook downloads
them at run time. Model weights are subject to their own licenses.
