# MaxEntLab

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22752925.svg)](https://doi.org/10.5281/zenodo.22752925)

A Maximum Entropy constraint-based modeling tool. Fit constraint weights to tableaux, inspect the predicted distribution, and adjust the weights manually. It works in the browser or from the command line.

* Fits Maximum Entropy models by L-BFGS-B with an analytic gradient
* Handles **hidden structure**: candidates that share an overt output are
  marginalized over automatically
* Handles categorical and variable data
* Frequencies are always normalized per tableau
* L1, L2 or no regularization, with adjustable strength
* Interactive tableau: sort, search, filter, and edit weights by hand to see
  the predicted distribution change

```
maxentlab/
  maxent_solver.py      the model
  app.py                Shiny for Python front end
  run_model.py          command-line driver
  requirements.txt
  sample_input_files/
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+, numpy, scipy, pandas and shiny.

## Quick start

Launch the app:

```bash
shiny run --reload app.py
```

Open the URL it prints (`http://127.0.0.1:8000` by default), upload a file from
`sample_input_files/`, and click **Generate grammar**.

Or fit a file from the command line:

```bash
python run_model.py sample_input_files/input_with_hidden_structure_variable.txt
```

## Input format

Tab-separated (or comma-separated `.csv`), UTF-8:

| column          |                                                                                           |
| --------------- | ----------------------------------------------------------------------------------------- |
| `input`         | the underlying form                                      |
| `output`        | the surface candidates                                                               |
| `hidden`        | *optional* — the hidden structure. Include this column to model hidden structure |
| `probability`   | observed frequency: a probability, a percentage or a raw count                            |
| everything else | one column per constraint, holding violation counts                                       |

```
input   output  hidden   probability  Iamb  Trochee  *StressHi
kata    KAta    (KAta)   0.7          1     0        0
kata    kaTA    (kaTA)   0.3          0     1        0
kata    KAta    (KA)ta   0.7          0     1        0
kata    kaTA    ka(TA)   0.3          0     1        0
```

Header names are matched case-insensitively.

**Frequencies are always normalized within a tableau.** `0.7`/`0.3`, `70`/`30`
and `7000`/`3000` all describe the same distribution and give the same model.

**With hidden structure,** frequency belongs to the overt output rather than to
an individual parse, so either convention works: repeat the output's frequency
on each of its parse rows (as above), or split it between them. Both give the
same result.

## The model

For each candidate *i* in the tableau of input *x*:

```
score(i)      = exp(-violations(i) · w)
P(i | x)      = score(i) / Z(x),        Z(x) = sum of scores in that tableau
P(output | x) = sum of P(i | x) over the parses of that output
```

Weights are fitted by minimizing the negative log-likelihood of the observed
distribution over **overt outputs**, plus a regularization term:

```
NLL = -sum_x sum_y  q(y | x) · log P(y | x)  +  prior
```

where `q` is the observed distribution. With
hidden structure, the gradient weights each parse by its posterior share
`q(y|x) · P(parse | y, x)`.

The prior is `Σ (w − μ)² / 2σ²` for L2 and `α · Σ |w − μ|` for L1. Optimization
is L-BFGS-B; weights are constrained to be non-negative unless you allow
negative weights.

Each tableau contributes equally to the likelihood; the model is not weighted
by the token frequency of a tableau.

## Using the app

**Fitting.** Choose a file, pick a prior and its strength, click **Generate
grammar**.

* **Prior**: L2 (Gaussian), L1, or None. A large σ (the default, 1000) is
  effectively unregularized. For L1, a larger α is a stronger prior.
* **Allow negative weights**: off by default, so weights stay ≥ 0.

**Reading the fit.** The status panel reports the number of tableaux and
constraints, whether hidden structure was found, the log-likelihood, and the
mean absolute error.

**Exploring the model.** The weights appear in the *Constraint weights* panel
and again as the top row of the tableau. Double-click any weight in the panel, modify it,
and press Enter: harmony, predicted probabilities and error all recompute
immediately, so you can test the model by hand. **Reset** restores the learned
weights.

**Navigating the tableau.** Click a column header to sort by it. The search box
filters by input, output or parse as you type. The column filter takes a
column, an operator (`contains`, `=`, `≠`, `>`, `≥`, `<`, `≤`) and a value;
`Error > 0.1` to find poor fits, for example. **Clear** resets both.

**Downloading.** **Download output** saves the full tableau (not just the
filtered view) as tab-separated text: a header row, a row of weights under
their constraint columns, then the data.

## Output columns

| column            |                                                                |
| ----------------- | -------------------------------------------------------------- |
| `Harmony`         | `-(violations · weights)`; higher is more harmonic            |
| `Predicted_parse` | *hidden structure only*; probability of that individual parse |
| `Predicted`       | probability of the overt output, summed over its parses        |
| `Observed`        | observed probability of the overt output                       |
| `Error`           | `Observed - Predicted`                                         |
| `MAE_per_tableau` | mean absolute error within that tableau                        |
| `MAE_overall`     | mean absolute error across the data                            |

With hidden structure, error is measured once per overt output rather than once
per parse, so a tableau is not penalized for spreading an output's probability
over several parses.

## Using it from Python

Everything the app does can be scripted, which is the easiest way to fit many
files at once. Run from a clone of this repository, so that `maxent_solver.py`
is on the import path:

```python
from maxent_solver import (read_input, optimize_weights,
                           predict_probabilities, write_output)

data = read_input("my_tableau.txt")
fit = optimize_weights(data, reg="L2", sigma=1.44)
preds = predict_probabilities(data, fit["weights"], fit["constraint_names"])
write_output(preds, fit["weights"], fit["constraint_names"], "predictions.txt")
```

`optimize_weights` returns the weights, the log-likelihood, the constraint
names, whether hidden structure was used, and the optimizer's status. It also
takes `alpha`, `mu`, `init_weights`, `lower_bound` and `upper_bound`.

## Online version

MaxEntLab runs in the browser:

**[maxentlab.alinirheche.org](https://maxentlab.alinirheche.org)**

It takes a few seconds to start the first time.

## Citation

Nirheche, Ali. 2026. *MaxEntLab: A Maximum Entropy constraint-based modeling tool.*
Amherst, MA: University of Massachusetts Amherst. https://doi.org/10.5281/zenodo.22752925

This DOI always resolves to the latest version. To cite the exact version you used,
use its version-specific DOI from the [Zenodo record](https://doi.org/10.5281/zenodo.22752925)
(version 1.0.0: https://doi.org/10.5281/zenodo.22752926).
