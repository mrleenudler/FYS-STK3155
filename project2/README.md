# FYS-STK3155 Project 2: Classification and regression with our own neural network

A feed-forward neural network written from scratch, with backpropagation, a
flexible number of layers and nodes, sigmoid, ReLU, Leaky ReLU, linear and
softmax activations, the mean squared error and the cross entropy as cost
functions, and optional L1 and L2 penalties. It is trained with the gradient
descent code from Project 1 (plain gradient descent, and stochastic gradient
descent with RMSprop or Adam), and used for

- regression on noisy samples of Runge's function f(x) = 1/(1 + 25x²), the data
  sets of Project 1, compared with OLS, Ridge and Lasso from Project 1;
- classification of the MNIST handwritten digits, compared with logistic
  regression (the network without hidden layers) and with scikit-learn.

Author: andreand. The report is `report/main.pdf`.

## Files

| File | Content |
|---|---|
| `Code/nn.py` | Activation and cost functions with derivatives, the `NeuralNetwork` class (feed-forward pass, backpropagation, gradient by JAX for testing) and the training loop |
| `Code/optim.py` | Update rules from Project 1 (plain, momentum, AdaGrad, RMSprop, Adam), unchanged |
| `Code/common.py` | Data, design matrix, scaling and OLS from Project 1, unchanged |
| `Code/main.py` | One function per project part (`part_b` … `part_e`); writes the figures and the logged output |
| `Code/tests.py` | The backpropagation against `jax.grad`, and a network without hidden layers against the closed-form OLS solution |
| `Code/results/` | Logged terminal output of each part (`part_b.txt` … `part_e.txt`); the numbers in the report come from these files. The `*.npz` files are cached hyperparameter grids |
| `Code/figures/` | All figures (PDF) |
| `report/` | LaTeX source of the report (`main.tex` and the section files) and the compiled PDF |
| `requirements.txt` | The Python packages needed |

## Installation

Python 3.13 with NumPy, Matplotlib, scikit-learn and JAX (the same environment as
Project 1). For example, from this folder:

```
python -m venv .venv
.venv\Scripts\activate          (Windows; on Linux and macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

With [uv](https://docs.astral.sh/uv/) the same is `uv venv` followed by
`uv pip install -r requirements.txt`.

The MNIST data are not in the repository. Download `mnist.npz` (11 MB) from
<https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz> and put it in
this folder, next to `Code/`.

## Running

From the `Code` folder:

```
python tests.py          # the two tests, under 30 s
python main.py           # parts b, c and d: 3-6 min
python main.py e         # part e (MNIST): 4-8 min
python main.py all       # all parts
```

Parts can also be named one by one, for instance `python main.py b c`. The
hyperparameter grids of parts c, d and e are cached in `Code/results/*.npz`.
Delete a file to compute its grid again; this adds 3-7 min (c), 4-8 min (d)
and 8-17 min (e). The run times were measured on two different 2-core cloud
machines; the lower numbers are from the faster one.

Every experiment is identified by seeds that fix the data, the splits, the
initial weights and the order of the minibatches, so all runs are reproducible.
Only the timings in the logs change from run to run.

## Use of AI tools

The code was written with Claude (Opus 5.5, claude.ai, October 2026) at level 4
(substantial) in the course's classification; the backpropagation in
`nn.py` (`NeuralNetwork.compute_gradient`) is our own code from the week 42
exercises. Each file has an `LLM-assisted` note in its module docstring, and
the larger functions a short tag that refers to it. See the appendix of the
report for the full declaration.
