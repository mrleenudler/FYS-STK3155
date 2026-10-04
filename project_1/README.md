# FYS-STK3155 Project 1: Regression analysis, resampling methods and gradient descent

OLS, Ridge and Lasso regression on noisy samples of Runge's function
f(x) = 1/(1 + 25x²), x ∈ [−1, 1], with bootstrap, k-fold cross-validation and
gradient descent (plain, momentum, AdaGrad, RMSProp, Adam, SGD).

Author: andreand. The report is `report/main.pdf`.

## Files

| File | Content |
|---|---|
| `common.py` | Data generation and split, design matrix, scaling, OLS and Ridge solvers, bootstrap, error measures, scikit-learn wrapper for our own estimator |
| `optim.py` | Cost functions, analytic and JAX gradients, update rules, full-batch gradient descent and SGD |
| `main.py` | One function per project part (`part_a` … `part_i`); writes figures and logged output |
| `tests.py` | Unit tests of `common.py` and `optim.py` |
| `results/` | Logged terminal output of each part (`part_a.txt` … `part_i.txt`); the numbers in the report come from these files |
| `figures/` | All figures (PDF) |
| `report/` | LaTeX source of the report (`main.tex` and the section files) and the compiled PDF |

## Running

Requires Python 3.11 or newer with NumPy, scikit-learn, Matplotlib and JAX. With [uv](https://docs.astral.sh/uv/):

```
uv run tests.py          # unit tests, a few seconds
uv run main.py           # the fast parts b, c, d, e, h: about 4-5 min
uv run main.py f i       # selected parts, here the slow ones
uv run main.py all       # everything: about 30 min
```

Without uv, replace `uv run` with `python`.

Running `main.py` without arguments skips the slow parts and says so; they are
run by naming them. Run times measured on a 2-core 2.8 GHz machine:

| Part | a | b | c | d | e | f | g | h | i |
|---|---|---|---|---|---|---|---|---|---|
| Time (s) | 206 | 36 | 49 | 113 | 56 | 578 | 322 | 14 | 529 |

Every experiment is identified by a seed that fixes both the data and the
train/test split, so all runs are reproducible. Figures go to `figures/` and
the terminal output is also written to `results/`. The only number that
changes between runs is the timing of automatic differentiation in part e.

## Use of AI tools

The code was written with Claude (Opus 5.5, claude.ai, September 2026) at level 4
(substantial) in the course's classification. Each file has an `LLM-assisted`
note in its module docstring, and each function a short tag. See the appendix
of the report for the full declaration.
