"""
FYS-STK3155 Project 2: regression and classification with our own
feed-forward neural network (nn.py).

How to run (from the Code folder, with the Python environment of the README):

    python main.py           parts b, c and d (3 to 6 minutes, depending on the machine)
    python main.py b e       only the named parts
    python main.py all       all parts (part e, MNIST, takes 4 to 8 minutes more)

Each part has its own function below (part_b, part_c, part_d, part_e), which
calls smaller functions, one for each experiment. Every part
- prints its results and also saves them in results/part_X.txt (the numbers
  in the report come from these files), and
- saves its figures as PDF in figures/.

The hyperparameter grids of parts c, d and e take several minutes, so they are
saved in results/*.npz and read from there the next time. Delete a file to
compute its grid again.

Data for the regression (parts b-d) are those of Project 1: n = 100 points of
Runge's function with noise sigma = 0.1, split 80/20 into training and test
points (common.split_data). The hyperparameters are chosen on a validation set
taken from the 80 training points (64 training / 16 validation); the test
points are only used for the final comparisons with OLS and Project 1.

LLM-assisted
------------
Tool: Claude (Opus 5.5, claude.ai, October 2026)
Role: Wrote the file (level 4): the experiments of each part, the logging and
the figures, following the project text and the choices discussed with
andreand (Python scripts instead of notebooks, the Project 1 data and
benchmark, validation split for the hyperparameters, the two illustrations).
Verification: tests.py, and the checks printed by each part.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from sklearn.model_selection import train_test_split

from common import design_matrix, fit_ols, mse, runge, split_data
from nn import NeuralNetwork, accuracy, one_hot, relu, train

# ---------------------------------------------------------------------------
# Settings shared by all parts
# ---------------------------------------------------------------------------

SEED = 42            # the data set of Project 1 we use as our main example
N_MAIN = 100         # number of data points
SIGMA_MAIN = 0.1     # standard deviation of the noise
DEGREE_OLS = 12      # lowest test MSE on the seed-42 split in Project 1 (part a)

FIGURE_DIR = Path("figures")
RESULT_DIR = Path("results")

# Colours and markers in a fixed order (blue, orange, aqua, yellow), as in Project 1
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GREY = "#777777"
MARKERS = ["o", "s", "^", "D"]

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
    "legend.frameon": False,
})

# Names of the activations as they are written in the logs and figures
ACT_LABEL = {"sigmoid": "Sigmoid", "relu": "ReLU", "leaky_relu": "Leaky ReLU"}


# ---------------------------------------------------------------------------
# Logging: print a line and keep a copy, which is saved in results/
# ---------------------------------------------------------------------------

log_lines: list[str] = []      # the lines of the part that is running now


def log(text: str = "") -> None:
    """
    Print text and keep it for the result file of the part.

    LLM-assisted: Claude, see module docstring.
    """
    print(text)
    log_lines.append(text)


def save_log(filename: str) -> None:
    """
    Write the kept lines to results/filename and start again with an empty list.

    LLM-assisted: Claude, see module docstring.
    """
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / filename).write_text("\n".join(log_lines) + "\n")
    log_lines.clear()


def save_figure(fig: plt.Figure, name: str) -> None:
    """
    Save a figure as figures/name.pdf and close it.

    LLM-assisted: Claude, see module docstring.
    """
    FIGURE_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}.pdf")
    plt.close(fig)


def save_figure_tight(fig: plt.Figure, name: str) -> None:
    """
    Save a figure as figures/name.pdf, cut tightly around its content, and close
    it. Used for the figures with a colour bar, where tight_layout does not work.

    LLM-assisted: Claude, see module docstring.
    """
    FIGURE_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# The Runge data and small helpers for the regression parts
# ---------------------------------------------------------------------------

def runge_data(seed: int = SEED) -> dict[str, np.ndarray]:
    """
    The Project 1 data for one seed:
    - x_train_full, y_train_full: the 80 training points of Project 1
    - x_test, y_test:             the 20 test points of Project 1
    - x_train, y_train:           64 of the training points, to train on while
                                  choosing hyperparameters
    - x_val, y_val:               the other 16 training points (validation)
    The arrays are the raw values, not standardised.

    LLM-assisted: Claude, see module docstring.
    """
    x_train_full, x_test, y_train_full, y_test = split_data(N_MAIN, SIGMA_MAIN, seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2,
                                                      random_state=seed)
    return {"x_train": x_train, "x_val": x_val, "x_train_full": x_train_full, "x_test": x_test,
            "y_train": y_train, "y_val": y_val, "y_train_full": y_train_full, "y_test": y_test}


def standardise(x: np.ndarray, x_fit: np.ndarray) -> np.ndarray:
    """
    Standardise x with the mean and standard deviation of x_fit (the points the
    network is trained on), and return it as a column vector of shape (N, 1),
    which is the input format of the network (one row per sample).

    LLM-assisted: Claude, see module docstring.
    """
    return ((x - x_fit.mean()) / x_fit.std()).reshape(-1, 1)


def regression_network(hidden: list[int], activation: str = "sigmoid", seed: int = SEED) -> NeuralNetwork:
    """
    Network for the Runge data: one input (x), the given hidden layers, all with
    the given activation, and one linear output node, with the MSE as cost.
    Example: hidden = [50, 50] is two hidden layers of 50 nodes.

    LLM-assisted: Claude, see module docstring.
    """
    layer_sizes = hidden + [1]
    activations = [activation] * len(hidden) + ["linear"]
    return NeuralNetwork(1, layer_sizes, activations, "mse", seed=seed)


def network_mse(net: NeuralNetwork, X: np.ndarray, y: np.ndarray) -> float:
    """
    MSE of the network on the inputs X (column vector) and targets y.

    LLM-assisted: Claude, see module docstring.
    """
    return mse(y, net.predict(X).ravel())


def ols_mse(x_fit: np.ndarray, y_fit: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray,
            degree: int = DEGREE_OLS) -> float:
    """
    MSE of OLS with scaling from Project 1 (common.fit_ols), fitted on
    (x_fit, y_fit) and evaluated on (x_eval, y_eval).

    LLM-assisted: Claude, see module docstring.
    """
    scaler, theta = fit_ols(x_fit, y_fit, degree)
    return mse(y_eval, scaler.predict(design_matrix(x_eval, degree), theta))


def label(hidden: list[int]) -> str:
    """
    Hidden layers as text for the logs, e.g. [50, 50] -> "50 x 50".

    LLM-assisted: Claude, see module docstring.
    """
    if not hidden:
        return "none"
    return " x ".join(str(h) for h in hidden)


# ---------------------------------------------------------------------------
# Part b: the network against OLS on the Runge data; learning rates
# ---------------------------------------------------------------------------

EPOCHS_B = 1000
BATCH_B = 16                           # minibatch size for SGD
GAMMAS_B = np.logspace(-4, 0, 9)       # learning rates 1e-4, 3.2e-4, ..., 1
# The three methods: (name in the report, update rule in optim.py, minibatch size).
# Minibatch size None means plain gradient descent with all the data in every step.
METHODS_B = [("GD", "plain", None), ("SGD + RMSprop", "rmsprop", BATCH_B), ("SGD + Adam", "adam", BATCH_B)]
ARCHITECTURES_B = [[50], [100], [50, 50], [100, 100]]
SEEDS_B = range(42, 52)                # ten data sets for the repeated comparison


def b_learning_rates(data: dict) -> dict[str, float]:
    """
    Validation MSE after EPOCHS_B epochs for every learning rate in GAMMAS_B and
    every method, with one hidden layer of 50 sigmoid nodes. Returns the best
    learning rate of each method, e.g. {"plain": 0.1, "rmsprop": 0.001, ...}.

    LLM-assisted: Claude, see module docstring.
    """
    X_tr = standardise(data["x_train"], data["x_train"])
    X_val = standardise(data["x_val"], data["x_train"])
    y_tr = data["y_train"].reshape(-1, 1)
    ols_val = ols_mse(data["x_train"], data["y_train"], data["x_val"], data["y_val"])

    log(f"\nLearning rates: one hidden layer of 50 sigmoid nodes, linear output, MSE,")
    log(f"{EPOCHS_B} epochs, minibatch {BATCH_B} for SGD; trained on 64 points, validation MSE on 16.")
    log(f"OLS degree {DEGREE_OLS} fitted on the same 64 points: validation MSE {ols_val:.3e}")
    log("gamma      " + "".join(f"{name:>16}" for name, _, _ in METHODS_B))

    # val[i, j]: validation MSE of method i with learning rate j (NaN if it diverged)
    val = np.full((len(METHODS_B), len(GAMMAS_B)), np.nan)
    for j, gamma in enumerate(GAMMAS_B):
        for i, (_, method, batch) in enumerate(METHODS_B):
            net = regression_network([50])
            result = train(net, X_tr, y_tr, method, gamma, EPOCHS_B, batch_size=batch, seed=SEED)
            if not result["diverged"]:
                val[i, j] = network_mse(net, X_val, data["y_val"])
        # one row of the table: this learning rate, all methods
        row = ""
        for v in val[:, j]:
            if np.isfinite(v):
                row += f"{v:16.3e}"
            else:
                row += f"{'diverged':>16}"
        log(f"{gamma:8.1e}   " + row)

    # The best learning rate of each method
    best = {}
    for i, (name, method, _) in enumerate(METHODS_B):
        j = int(np.nanargmin(val[i]))          # index of the lowest validation MSE (NaN ignored)
        best[method] = float(GAMMAS_B[j])
        log(f"  best {name}: gamma = {GAMMAS_B[j]:.1e}, validation MSE {val[i, j]:.3e}")

    # Figure: validation MSE against the learning rate, one curve per method
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for i, (name, _, _) in enumerate(METHODS_B):
        ax.plot(GAMMAS_B, val[i], marker=MARKERS[i], color=COLOURS[i], label=name)
    ax.axhline(ols_val, color=GREY, ls="--", lw=1, label=f"OLS, degree {DEGREE_OLS}")
    y_top = 0.5
    # Values above the top of the plot (GD at gamma = 1) are written as numbers at the top
    for i, (name, _, _) in enumerate(METHODS_B):
        for j in np.flatnonzero(val[i] > y_top):
            ax.annotate(f"{val[i, j]:.0e}", (GAMMAS_B[j], y_top), xytext=(0, -12), textcoords="offset points",
                        ha="center", fontsize=7, color=COLOURS[i], annotation_clip=False)
    ax.set(xscale="log", yscale="log", xlabel=r"Learning rate $\gamma$", ylabel="Validation MSE",
           ylim=(3e-3, y_top))
    ax.legend()
    save_figure(fig, "b_learning_rates")
    return best


def b_training_curves(data: dict, best: dict[str, float]) -> None:
    """
    Training and validation MSE after every epoch for each method at its best
    learning rate (one hidden layer of 50 sigmoid nodes).

    LLM-assisted: Claude, see module docstring.
    """
    X_tr = standardise(data["x_train"], data["x_train"])
    X_val = standardise(data["x_val"], data["x_train"])
    y_tr = data["y_train"].reshape(-1, 1)

    # train() calls this function after every epoch and keeps what it returns
    def record(net: NeuralNetwork) -> tuple[float, float]:
        return network_mse(net, X_tr, data["y_train"]), network_mse(net, X_val, data["y_val"])

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    log("\nTraining curves at the best learning rate (MSE after 10 / 100 / 1000 epochs, train | validation)")
    for i, (name, method, batch) in enumerate(METHODS_B):
        net = regression_network([50])
        result = train(net, X_tr, y_tr, method, best[method], EPOCHS_B, batch_size=batch, seed=SEED, record=record)
        hist = np.array(result["history"])     # row e-1: (training MSE, validation MSE) after epoch e
        epochs = np.arange(1, len(hist) + 1)
        ax.plot(epochs, hist[:, 0], color=COLOURS[i], lw=1.2, label=f"{name}, training")
        ax.plot(epochs, hist[:, 1], color=COLOURS[i], lw=1.2, ls="--", label=f"{name}, validation")
        log(f"  {name:14s} " + "  ".join(f"{hist[e - 1, 0]:.2e} | {hist[e - 1, 1]:.2e}" for e in (10, 100, 1000)))
    ax.axhline(SIGMA_MAIN**2, color=GREY, ls=":", lw=1, label=r"$\sigma^2$")
    ax.set(xscale="log", yscale="log", xlabel="Epoch", ylabel="MSE")
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save_figure(fig, "b_training_curves")


def b_architectures(data: dict, best: dict[str, float]) -> dict[str, NeuralNetwork]:
    """
    One and two hidden layers of 50 and 100 sigmoid nodes, each method at its
    best learning rate, trained on all 80 training points; test MSE against OLS
    of degree DEGREE_OLS on the same split. Returns the networks trained with
    RMSprop, for the figure of the fits.

    LLM-assisted: Claude, see module docstring.
    """
    X_tr = standardise(data["x_train_full"], data["x_train_full"])
    X_te = standardise(data["x_test"], data["x_train_full"])
    y_tr = data["y_train_full"].reshape(-1, 1)
    ols_test = ols_mse(data["x_train_full"], data["y_train_full"], data["x_test"], data["y_test"])

    log(f"\nTest MSE, trained on all 80 training points (seed {SEED}), {EPOCHS_B} epochs")
    log(f"  OLS degree {DEGREE_OLS}: {ols_test:.3e}")
    log("hidden     parameters" + "".join(f"{name:>16}" for name, _, _ in METHODS_B) + "   time (s)")
    networks = {}
    for hidden in ARCHITECTURES_B:
        row = []
        t0 = time.perf_counter()
        for _, method, batch in METHODS_B:
            net = regression_network(hidden)
            result = train(net, X_tr, y_tr, method, best[method], EPOCHS_B, batch_size=batch, seed=SEED)
            if result["diverged"]:
                row.append(np.nan)
            else:
                row.append(network_mse(net, X_te, data["y_test"]))
            if method == "rmsprop":
                networks[label(hidden)] = net
        n_par = sum(W.size + b.size for W, b in net.layers)     # number of weights and biases
        log(f"{label(hidden):10s} {n_par:10d}" + "".join(f"{v:16.3e}" for v in row)
            + f"   {time.perf_counter() - t0:8.1f}")
    return networks


def b_fits(data: dict, networks: dict[str, NeuralNetwork]) -> None:
    """
    Figure: the data, Runge's function, OLS of degree DEGREE_OLS and the networks
    (RMSprop) with one hidden layer of 50 and two of 100 nodes, over [-1, 1].

    LLM-assisted: Claude, see module docstring.
    """
    x = np.linspace(-1, 1, 400)
    X_plot = standardise(x, data["x_train_full"])
    scaler, theta = fit_ols(data["x_train_full"], data["y_train_full"], DEGREE_OLS)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.scatter(data["x_train_full"], data["y_train_full"], s=10, color=GREY, alpha=0.6, label="Training data")
    ax.scatter(data["x_test"], data["y_test"], s=14, facecolors="none", edgecolors="k", lw=0.8, label="Test data")
    ax.plot(x, runge(x), color="k", lw=1, label="Runge's function")
    ax.plot(x, scaler.predict(design_matrix(x, DEGREE_OLS), theta), color=COLOURS[0], label=f"OLS, degree {DEGREE_OLS}")
    for i, key in enumerate(["50", "100 x 100"]):
        ax.plot(x, networks[key].predict(X_plot).ravel(), color=COLOURS[i + 1],
                ls=["-", "--"][i], label=f"Network, hidden {key}")
    ax.set(xlabel="$x$", ylabel="$y$", ylim=(-0.4, 1.3))
    ax.legend(fontsize=8)
    save_figure(fig, "b_fits")


def b_repetitions(best: dict[str, float]) -> None:
    """
    One test MSE on 20 points says little (the seed-42 test set has no points
    near the peak), so the comparison of OLS (degree DEGREE_OLS) and the network
    (one hidden layer of 50 sigmoid nodes, SGD + RMSprop at the learning rate
    chosen above) is repeated for the ten data sets SEEDS_B, with new data and a
    new split each time.

    LLM-assisted: Claude, see module docstring.
    """
    ols, net_mse = [], []
    for seed in SEEDS_B:
        data = runge_data(seed)
        X_tr = standardise(data["x_train_full"], data["x_train_full"])
        X_te = standardise(data["x_test"], data["x_train_full"])
        net = regression_network([50], seed=seed)
        train(net, X_tr, data["y_train_full"].reshape(-1, 1), "rmsprop", best["rmsprop"], EPOCHS_B,
              batch_size=BATCH_B, seed=seed)
        ols.append(ols_mse(data["x_train_full"], data["y_train_full"], data["x_test"], data["y_test"]))
        net_mse.append(network_mse(net, X_te, data["y_test"]))
    ols, net_mse = np.array(ols), np.array(net_mse)
    log(f"\nRepeated over seeds {SEEDS_B[0]}-{SEEDS_B[-1]} (new data and split each time), test MSE")
    log(f"  OLS degree {DEGREE_OLS}:        median {np.median(ols):.3e}, "
        f"quartiles {np.percentile(ols, 25):.3e} - {np.percentile(ols, 75):.3e}")
    log(f"  Network 50, RMSprop:  median {np.median(net_mse):.3e}, "
        f"quartiles {np.percentile(net_mse, 25):.3e} - {np.percentile(net_mse, 75):.3e}")
    log(f"  Network better than OLS in {np.sum(net_mse < ols)} of {len(ols)} repetitions")


def b_checks(data: dict, best: dict[str, float]) -> None:
    """
    Two checks used in the discussion of part b:
    1. What is in each data set (seed 42): how many points lie near the peak
       (|x| < 0.2), the mean squared noise (y - f(x))^2 and the variance of y.
       This explains why the validation MSE can be lower than the training MSE.
    2. What happens with more epochs: each method at its best learning rate,
       one hidden layer of 50 sigmoid nodes, validation MSE after 1000, 3000
       and 10000 epochs (one run of 10000 epochs, read off along the way).

    LLM-assisted: Claude, see module docstring.
    """
    log("\nWhat is in the data sets (seed 42)")
    for name in ["train", "val", "test"]:
        x, y = data[f"x_{name}"], data[f"y_{name}"]
        n_peak = int(np.sum(np.abs(x) < 0.2))
        noise = np.mean((y - runge(x)) ** 2)
        log(f"  {name:5s}: {n_peak:2d} of {len(x):2d} points with |x| < 0.2, mean squared noise {noise:.4f}, "
            f"variance of y {np.var(y):.4f}")
    log("  (the variance of y is the MSE of the best constant prediction, the mean of y)")

    X_tr = standardise(data["x_train"], data["x_train"])
    X_val = standardise(data["x_val"], data["x_train"])
    y_tr = data["y_train"].reshape(-1, 1)

    def record(net: NeuralNetwork) -> float:
        return network_mse(net, X_val, data["y_val"])

    log("\nMore epochs: validation MSE after 1000 / 3000 / 10000 epochs, 50 sigmoid nodes, best learning rate")
    for name, method, batch in METHODS_B:
        net = regression_network([50])
        hist = train(net, X_tr, y_tr, method, best[method], 10000, batch_size=batch, seed=SEED, record=record)["history"]
        log(f"  {name:14s} gamma = {best[method]:.1e}: " + "  ".join(f"{hist[e - 1]:.2e}" for e in (1000, 3000, 10000)))


def b_hinges(data: dict) -> None:
    """
    Illustration of the features a hidden layer makes: one hidden layer of ten
    ReLU nodes, trained on the 80 training points. Node j adds the hinge
    w_j * max(0, a_j x + c_j) to the output, with its kink at x = -c_j / a_j,
    and the output is the output bias plus the sum of the hinges.
    Three panels: each hinge, the running sum, and the final curve.

    LLM-assisted: Claude, see module docstring.
    """
    n_nodes = 10
    m, s = data["x_train_full"].mean(), data["x_train_full"].std()   # needed below for the kinks
    X_tr = standardise(data["x_train_full"], data["x_train_full"])
    net = regression_network([n_nodes], "relu", seed=7)
    train(net, X_tr, data["y_train_full"].reshape(-1, 1), "adam", 3e-3, 4000, batch_size=BATCH_B, seed=7)
    (W1, b1), (W2, b2) = net.layers

    x = np.linspace(-1, 1, 801)
    X_plot = standardise(x, data["x_train_full"])
    hinges = relu(X_plot @ W1 + b1) * W2.ravel()       # column j: the contribution of node j
    output = b2[0] + hinges.sum(axis=1)                 # the output of the network
    assert np.allclose(output, net.predict(X_plot).ravel()), "the hinges do not add up to the network output"
    kinks = m + s * (-b1 / W1.ravel())                  # where a_j x + c_j = 0, in the original x
    active = np.abs(hinges).max(axis=0) > 1e-9          # nodes that are not zero on the whole interval
    order = np.argsort(np.where(active, kinks, np.inf)) # active nodes from left to right, inactive last
    n_active = int(np.sum(active))
    log(f"\nIllustration: {n_nodes} ReLU nodes (seed 7, Adam, gamma = 3e-3, 4000 epochs, all 80 training points)")
    log(f"  active nodes on [-1, 1]: {n_active} of {n_nodes}; kinks at x = "
        + ", ".join(f"{k:.3f}" for k in np.sort(kinks[active])))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    # Panel 1: the hinge of each active node, with its kink
    for j in order[:n_active]:
        ax[0].plot(x, hinges[:, j], color=COLOURS[0], lw=1.2, alpha=0.8)
        if -1 <= kinks[j] <= 1:
            ax[0].plot(kinks[j], 0, "o", color=COLOURS[1], ms=5, mec="k", mew=0.5)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].plot([], [], "o", color=COLOURS[1], mec="k", mew=0.5, label="kink, $x = -c_j/a_j$")
    ax[0].set(title=f"1. Each of the {n_active} active nodes: $w_j\\,\\max(0,\\,a_j x + c_j)$",
              xlabel="$x$", ylabel="Contribution to the output")
    ax[0].legend(fontsize=8)
    # Panel 2: start with the output bias and add the nodes one at a time, from left to right
    blues = plt.get_cmap("Blues")
    running_sum = np.full_like(x, b2[0])
    ax[1].plot(x, running_sum, color=blues(0.25), lw=1, label="bias $b$ only")
    for k, j in enumerate(order[:n_active], start=1):
        running_sum = running_sum + hinges[:, j]
        last = k == n_active
        ax[1].plot(x, running_sum, color=blues(0.3 + 0.7 * k / n_active), lw=2.2 if last else 1,
                   label="all nodes" if last else None)
    ax[1].set(title="2. Adding the nodes one at a time (left to right)", xlabel="$x$", ylabel="Running sum")
    ax[1].legend(fontsize=8, loc="upper left")
    # Panel 3: the result against the data
    ax[2].scatter(data["x_train_full"], data["y_train_full"], s=10, color=GREY, alpha=0.6, label="training data")
    ax[2].plot(x, runge(x), color="k", lw=1, label="Runge's function")
    ax[2].plot(x, output, color=COLOURS[0], lw=2, label=f"network, {n_nodes} ReLU nodes")
    for j in order[:n_active]:
        if -1 <= kinks[j] <= 1:
            ax[2].axvline(kinks[j], color=COLOURS[1], lw=0.6, alpha=0.6)
    ax[2].plot([], [], color=COLOURS[1], lw=0.6, label="kinks")
    ax[2].set(title="3. The sum: a piecewise linear curve", xlabel="$x$", ylabel="$y$", ylim=(-0.3, 1.2))
    ax[2].legend(fontsize=8)
    save_figure(fig, "illustration_relu_hinges")


def part_b() -> None:
    """
    Part b: learning rates for plain GD, SGD + RMSprop and SGD + Adam; one and
    two hidden layers of 50 and 100 sigmoid nodes against OLS; repetition over
    data sets; checks for the discussion; the hinge illustration.

    LLM-assisted: Claude, see module docstring.
    """
    log("=" * 72)
    log("Part b) Feed-forward network against OLS for the Runge function")
    log("=" * 72)
    log(f"Data: n = {N_MAIN}, sigma = {SIGMA_MAIN}, seed {SEED}: 64 training / 16 validation / 20 test points.")
    log("Input: x only (one feature), standardised with training statistics. Output: one linear node.")
    log("Weights ~ N(0, 1/n_in) for sigmoid layers (lecture notes Sec. 8.10/8.13), biases 0.01.")
    data = runge_data(SEED)
    best = b_learning_rates(data)
    b_training_curves(data, best)
    networks = b_architectures(data, best)
    b_fits(data, networks)
    b_repetitions(best)
    b_checks(data, best)
    b_hinges(data)
    log(f"\nFigures saved in {FIGURE_DIR}/")
    save_log("part_b.txt")


# ---------------------------------------------------------------------------
# Part c: activation functions, depth and width
# ---------------------------------------------------------------------------

ACTIVATIONS_C = ["sigmoid", "relu", "leaky_relu"]
LAYERS_C = [1, 2, 3, 5]                 # numbers of hidden layers
NODES_C = [5, 10, 25, 50, 100]          # numbers of nodes in each hidden layer
SEEDS_C = [42, 43, 44]                  # three data sets; we report the median over them
GAMMA_C = 1e-3                          # Adam at the learning rate chosen in part b


def c_grid() -> dict[str, np.ndarray]:
    """
    Training and validation MSE for every activation, number of hidden layers
    and number of nodes per layer (the same number in every hidden layer), with
    SGD + Adam (gamma = GAMMA_C, minibatch BATCH_B, EPOCHS_B epochs), median
    over the data sets SEEDS_C (new data, split and initial weights each time).

    Returns {"train": ..., "val": ..., "ols_val": ...}, where train[a, l, n] is
    the median training MSE of activation a, LAYERS_C[l] layers and NODES_C[n]
    nodes. The result is saved in results/part_c_grid.npz, since the grid takes
    a few minutes, and read from there the next time.

    LLM-assisted: Claude, see module docstring.
    """
    cache = RESULT_DIR / "part_c_grid.npz"
    if cache.exists():
        log(f"(grid read from {cache}; delete the file to recompute)")
        return dict(np.load(cache))

    shape = (len(ACTIVATIONS_C), len(LAYERS_C), len(NODES_C), len(SEEDS_C))
    train_mse = np.full(shape, np.nan)
    val_mse = np.full(shape, np.nan)
    for i_seed, seed in enumerate(SEEDS_C):
        data = runge_data(seed)
        X_tr = standardise(data["x_train"], data["x_train"])
        X_val = standardise(data["x_val"], data["x_train"])
        y_tr = data["y_train"].reshape(-1, 1)
        for i_act, act in enumerate(ACTIVATIONS_C):
            for i_layers, n_layers in enumerate(LAYERS_C):
                for i_nodes, n_nodes in enumerate(NODES_C):
                    net = regression_network([n_nodes] * n_layers, act, seed=seed)
                    result = train(net, X_tr, y_tr, "adam", GAMMA_C, EPOCHS_B, batch_size=BATCH_B, seed=seed)
                    if not result["diverged"]:
                        train_mse[i_act, i_layers, i_nodes, i_seed] = network_mse(net, X_tr, data["y_train"])
                        val_mse[i_act, i_layers, i_nodes, i_seed] = network_mse(net, X_val, data["y_val"])

    # OLS of degree DEGREE_OLS on the same validation sets, as a reference
    ols_val = []
    for seed in SEEDS_C:
        data = runge_data(seed)
        ols_val.append(ols_mse(data["x_train"], data["y_train"], data["x_val"], data["y_val"]))

    # Median over the data sets (the last axis)
    out = {"train": np.median(train_mse, axis=3), "val": np.median(val_mse, axis=3), "ols_val": np.median(ols_val)}
    RESULT_DIR.mkdir(exist_ok=True)
    np.savez(cache, **out)
    return out


def c_heatmaps(grid: dict[str, np.ndarray]) -> None:
    """
    Log the grids of part c, and draw heat maps of the median validation MSE
    against layers and nodes, one panel per activation, on a common logarithmic
    colour scale.

    LLM-assisted: Claude, see module docstring.
    """
    val = grid["val"]
    log(f"\nMedian validation MSE over seeds {SEEDS_C[0]}-{SEEDS_C[-1]} (rows: hidden layers, columns: nodes per layer)")
    log(f"OLS degree {DEGREE_OLS} on the same data: median validation MSE {float(grid['ols_val']):.3e}")
    for a, act in enumerate(ACTIVATIONS_C):
        log(f"  {ACT_LABEL[act]}")
        log("  layers " + "".join(f"{n:>11d}" for n in NODES_C))
        for l, n_layers in enumerate(LAYERS_C):
            log(f"  {n_layers:6d} " + "".join(f"{v:11.3e}" for v in val[a, l]))
        # row and column of the lowest value in this activation's grid
        l_best, n_best = np.unravel_index(np.nanargmin(val[a]), val[a].shape)
        log(f"  best: {LAYERS_C[l_best]} layer(s) of {NODES_C[n_best]} nodes, {val[a, l_best, n_best]:.3e}")

    log(f"\nMedian training MSE over seeds {SEEDS_C[0]}-{SEEDS_C[-1]} (same runs)")
    for a, act in enumerate(ACTIVATIONS_C):
        log(f"  {ACT_LABEL[act]}")
        log("  layers " + "".join(f"{n:>11d}" for n in NODES_C))
        for l, n_layers in enumerate(LAYERS_C):
            log(f"  {n_layers:6d} " + "".join(f"{v:11.3e}" for v in grid["train"][a, l]))

    # Heat maps. LogNorm: the colour scale is logarithmic in the MSE, the same for all panels.
    norm = LogNorm(vmin=np.nanmin(val), vmax=np.nanmax(val))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for a, (act, ax) in enumerate(zip(ACTIVATIONS_C, axes)):
        im = ax.imshow(val[a], cmap="Blues", norm=norm, aspect="auto")   # light = low MSE, dark = high
        for l in range(len(LAYERS_C)):
            for n in range(len(NODES_C)):
                v = val[a, l, n]
                # the MSE in units of 1e-3 in each cell; white text on dark cells
                ax.text(n, l, f"{v * 1e3:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if norm(v) > 0.6 else "black")
        ax.set(xticks=range(len(NODES_C)), xticklabels=NODES_C, yticks=range(len(LAYERS_C)), yticklabels=LAYERS_C,
               xlabel="Nodes per hidden layer", title=ACT_LABEL[act])
        ax.grid(False)
    axes[0].set_ylabel("Hidden layers")
    fig.colorbar(im, ax=axes, label=r"Validation MSE (numbers: $\times 10^{-3}$)", shrink=0.9)
    save_figure_tight(fig, "c_heatmaps")


def c_overfitting(grid: dict[str, np.ndarray]) -> None:
    """
    Training and validation MSE against the number of hidden layers (50 nodes
    per layer), for each activation: does the gap between them grow with the
    number of parameters?

    LLM-assisted: Claude, see module docstring.
    """
    n50 = NODES_C.index(50)                    # the column with 50 nodes per layer
    log("\nTraining | validation MSE against depth, 50 nodes per layer (medians)")
    log("layers  parameters" + "".join(f"{ACT_LABEL[a]:>24s}" for a in ACTIVATIONS_C))
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for a, act in enumerate(ACTIVATIONS_C):
        ax.plot(LAYERS_C, grid["train"][a, :, n50], marker=MARKERS[a], color=COLOURS[a], ls="--", lw=1.2,
                label=f"{ACT_LABEL[act]}, training")
        ax.plot(LAYERS_C, grid["val"][a, :, n50], marker=MARKERS[a], color=COLOURS[a],
                label=f"{ACT_LABEL[act]}, validation")
    for l, n_layers in enumerate(LAYERS_C):
        n_par = sum(W.size + b.size for W, b in regression_network([50] * n_layers).layers)
        row = ""
        for a in range(len(ACTIVATIONS_C)):
            row += f"   {grid['train'][a, l, n50]:.2e} | {grid['val'][a, l, n50]:.2e}"
        log(f"{n_layers:6d}  {n_par:10d}" + row)
    ax.axhline(SIGMA_MAIN**2, color=GREY, ls=":", lw=1, label=r"$\sigma^2$")
    ax.set(xticks=LAYERS_C, xlabel="Hidden layers (50 nodes each)", ylabel="MSE (median over seeds)", yscale="log")
    ax.legend(fontsize=7, ncol=2)
    save_figure(fig, "c_overfitting")


def c_gradient_norms() -> None:
    """
    Typical size of the weight gradient in every layer of a network with five
    hidden layers of 50 nodes, at initialisation and after 100 epochs, for each
    activation (the vanishing gradient of lecture notes Sec. 8.9, Fig. 8.14).
    The size is the root mean square over the weights of the layer, not the
    norm as in Fig. 8.14: the first and the last layer have 50 weights and the
    others 2500, and the norm grows with the number of weights.

    LLM-assisted: Claude, see module docstring.
    """
    data = runge_data(SEED)
    X_tr = standardise(data["x_train"], data["x_train"])
    y_tr = data["y_train"].reshape(-1, 1)
    log("\nRMS of dC/dW over the weights of each layer, five hidden layers of 50 nodes (layer 1 = first hidden, 6 = output)")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for a, act in enumerate(ACTIVATIONS_C):
        net = regression_network([50] * 5, act, seed=SEED)
        for ax, when in zip(axes, ["at initialisation", "after 100 epochs"]):
            if when == "after 100 epochs":
                # the same network, now trained for 100 epochs
                train(net, X_tr, y_tr, "adam", GAMMA_C, 100, batch_size=BATCH_B, seed=SEED)
            # root mean square of the weight gradient of each layer
            norms = [np.sqrt(np.mean(dW**2)) for dW, _ in net.compute_gradient(X_tr, y_tr)]
            ax.plot(range(1, 7), norms, marker=MARKERS[a], color=COLOURS[a], label=ACT_LABEL[act])
            ax.set(title=when, xlabel="Layer", yscale="log")
            log(f"  {ACT_LABEL[act]:11s} {when:18s}: " + "  ".join(f"{v:.1e}" for v in norms)
                + f"   (first / output: {norms[0] / norms[-1]:.1e})")
    axes[0].set_ylabel(r"RMS of $\partial C / \partial W^l$ per weight")
    axes[0].legend()
    save_figure(fig, "c_gradient_norms")


def c_checks() -> None:
    """
    Two checks used in the discussion of part c:
    1. How precise is a validation MSE on 16 points? The MSE is a mean of 16
       squared errors, so its standard error is std(squared errors) / sqrt(16).
       Shown for OLS of degree DEGREE_OLS and for the ReLU network with two
       hidden layers of 10 nodes, seeds SEEDS_C.
    2. Does the sigmoid network overfit if it is trained longer? Five hidden
       layers of 50 sigmoid nodes, 5000 epochs instead of 1000, median over
       SEEDS_C.

    LLM-assisted: Claude, see module docstring.
    """
    log(f"\nStandard error of a validation MSE on 16 points: std(squared errors) / sqrt(16)")
    for seed in SEEDS_C:
        data = runge_data(seed)
        # squared errors of OLS on the 16 validation points
        scaler, theta = fit_ols(data["x_train"], data["y_train"], DEGREE_OLS)
        err_ols = (data["y_val"] - scaler.predict(design_matrix(data["x_val"], DEGREE_OLS), theta)) ** 2
        # squared errors of the network on the same points
        X_tr = standardise(data["x_train"], data["x_train"])
        X_val = standardise(data["x_val"], data["x_train"])
        net = regression_network([10, 10], "relu", seed=seed)
        train(net, X_tr, data["y_train"].reshape(-1, 1), "adam", GAMMA_C, EPOCHS_B, batch_size=BATCH_B, seed=seed)
        err_net = (data["y_val"] - net.predict(X_val).ravel()) ** 2
        parts = []
        for name, err in [(f"OLS degree {DEGREE_OLS}", err_ols), ("ReLU 2 x 10", err_net)]:
            se = err.std(ddof=1) / np.sqrt(len(err))
            parts.append(f"{name}: {err.mean():.2e} +- {se:.1e} ({100 * se / err.mean():.0f} %)")
        log(f"  seed {seed}:  " + "   ".join(parts))

    log("\nSigmoid, five hidden layers of 50 nodes, trained longer (median over seeds, train | validation MSE)")
    results = {1000: [], 5000: []}         # epochs -> list of (training MSE, validation MSE), one per seed
    for seed in SEEDS_C:
        data = runge_data(seed)
        X_tr = standardise(data["x_train"], data["x_train"])
        X_val = standardise(data["x_val"], data["x_train"])
        net = regression_network([50] * 5, "sigmoid", seed=seed)

        def record(n: NeuralNetwork) -> tuple[float, float]:
            return network_mse(n, X_tr, data["y_train"]), network_mse(n, X_val, data["y_val"])

        hist = train(net, X_tr, data["y_train"].reshape(-1, 1), "adam", GAMMA_C, 5000, batch_size=BATCH_B,
                     seed=seed, record=record)["history"]
        for epochs in results:
            results[epochs].append(hist[epochs - 1])
    for epochs, values in results.items():
        values = np.array(values)
        log(f"  {epochs:5d} epochs: {np.median(values[:, 0]):.2e} | {np.median(values[:, 1]):.2e}")


def part_c() -> None:
    """
    Part c: sigmoid, ReLU and Leaky ReLU in the hidden layers, as functions of
    the number of hidden layers and nodes; overfitting; gradient sizes per
    layer; checks for the discussion.

    LLM-assisted: Claude, see module docstring.
    """
    log("=" * 72)
    log("Part c) Activation functions, depth and width")
    log("=" * 72)
    log(f"Runge data as in part b (64 training / 16 validation points), SGD + Adam, gamma = {GAMMA_C}, "
        f"minibatch {BATCH_B}, {EPOCHS_B} epochs.")
    log("Weights ~ N(0, 1/n_in) for sigmoid and N(0, 2/n_in) for ReLU and Leaky ReLU (He), biases 0.01.")
    grid = c_grid()
    c_heatmaps(grid)
    c_overfitting(grid)
    c_gradient_norms()
    c_checks()
    log(f"\nFigures saved in {FIGURE_DIR}/")
    save_log("part_c.txt")


# ---------------------------------------------------------------------------
# Part d: L2 and L1 penalties, against Ridge and Lasso of Project 1
# ---------------------------------------------------------------------------

HIDDEN_D = [50, 50, 50]      # the ReLU network of part c that overfits: training MSE below sigma^2
ACTIVATION_D = "relu"
LAMBDAS_D = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
GAMMAS_D = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
SEEDS_D_TUNE = [42, 43, 44]  # validation sets for choosing lambda and gamma
SEEDS_D_TEST = range(42, 62) # the 20 data sets of Project 1, part i
# Project 1, part i (results/part_i.txt): median test MSE over the same 20 data sets, models chosen by 5-fold CV
PROJECT1_TEST = {"OLS": 1.5455e-02, "Ridge": 1.3345e-02, "Lasso": 1.3937e-02}


def d_grid() -> dict[str, np.ndarray]:
    """
    Median validation MSE over SEEDS_D_TUNE for both penalties ("l2", "l1"),
    every lambda and every learning rate, SGD + Adam, ReLU network HIDDEN_D.
    The penalty is lam * sum W^2 or lam * sum |W| over the weights of all layers
    (not the biases). Returns {"l2": grid, "l1": grid}, where grid[i, j] belongs
    to LAMBDAS_D[i] and GAMMAS_D[j]. Saved in results/part_d_grid.npz.

    LLM-assisted: Claude, see module docstring.
    """
    cache = RESULT_DIR / "part_d_grid.npz"
    if cache.exists():
        log(f"(grid read from {cache}; delete the file to recompute)")
        return dict(np.load(cache))

    out = {}
    for penalty in ["l2", "l1"]:
        val = np.full((len(LAMBDAS_D), len(GAMMAS_D), len(SEEDS_D_TUNE)), np.nan)
        for i_seed, seed in enumerate(SEEDS_D_TUNE):
            data = runge_data(seed)
            X_tr = standardise(data["x_train"], data["x_train"])
            X_val = standardise(data["x_val"], data["x_train"])
            for i, lam in enumerate(LAMBDAS_D):
                for j, gamma in enumerate(GAMMAS_D):
                    net = regression_network(HIDDEN_D, ACTIVATION_D, seed=seed)
                    result = train(net, X_tr, data["y_train"].reshape(-1, 1), "adam", gamma, EPOCHS_B,
                                   batch_size=BATCH_B, seed=seed, lam=lam, penalty=penalty)
                    if not result["diverged"]:
                        val[i, j, i_seed] = network_mse(net, X_val, data["y_val"])
        out[penalty] = np.median(val, axis=2)          # median over the data sets
    RESULT_DIR.mkdir(exist_ok=True)
    np.savez(cache, **out)
    return out


def d_heatmaps(grid: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    """
    Log the grids and draw heat maps of the median validation MSE against
    lambda and gamma for L2 and L1, on a common colour scale. Returns the best
    (lambda, gamma) for each penalty, e.g. {"l2": (0.0, 1e-4), "l1": (1e-6, 1e-4)}.

    LLM-assisted: Claude, see module docstring.
    """
    best = {}
    vmin = min(np.nanmin(grid["l2"]), np.nanmin(grid["l1"]))
    vmax = max(np.nanmax(grid["l2"]), np.nanmax(grid["l1"]))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for penalty, ax, name in zip(["l2", "l1"], axes, ["$L_2$", "$L_1$"]):
        val = grid[penalty]
        log(f"\n{penalty.upper()}: median validation MSE over seeds {SEEDS_D_TUNE[0]}-{SEEDS_D_TUNE[-1]} "
            f"(rows lambda, columns gamma)")
        log("  lambda  " + "".join(f"{g:>11.0e}" for g in GAMMAS_D))
        for i, lam in enumerate(LAMBDAS_D):
            log(f"  {lam:7.0e} " + "".join(f"{v:11.3e}" for v in val[i]))
        # row (lambda) and column (gamma) of the lowest validation MSE
        i, j = np.unravel_index(np.nanargmin(val), val.shape)
        best[penalty] = (LAMBDAS_D[i], GAMMAS_D[j])
        log(f"  best: lambda = {LAMBDAS_D[i]:.0e}, gamma = {GAMMAS_D[j]:.0e}, validation MSE {val[i, j]:.3e}"
            f"  (lambda = 0 at its best gamma: {np.nanmin(val[0]):.3e})")

        im = ax.imshow(val, cmap="Blues", norm=norm, aspect="auto")   # light = low MSE, dark = high
        for r in range(len(LAMBDAS_D)):
            for c in range(len(GAMMAS_D)):
                ax.text(c, r, f"{val[r, c] * 1e3:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if norm(val[r, c]) > 0.6 else "black")
        # orange square around the best cell
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=COLOURS[1], lw=2))
        ax.set(xticks=range(len(GAMMAS_D)), xticklabels=[f"{g:.0e}" for g in GAMMAS_D],
               yticks=range(len(LAMBDAS_D)), yticklabels=["0"] + [f"{lam:.0e}" for lam in LAMBDAS_D[1:]],
               xlabel=r"Learning rate $\gamma$", title=f"{name} penalty")
        ax.grid(False)
    axes[0].set_ylabel(r"$\lambda$")
    fig.colorbar(im, ax=axes, label=r"Validation MSE (numbers: $\times 10^{-3}$)", shrink=0.9)
    save_figure_tight(fig, "d_heatmaps")
    return best


def d_test(grid: dict[str, np.ndarray], best: dict[str, tuple[float, float]]) -> None:
    """
    The network without penalty and with the chosen L2 and L1 penalties on the
    20 data sets of Project 1 (seeds 42-61, 80/20 split): test MSE, against the
    median test MSE of OLS, Ridge and Lasso from Project 1, part i.

    LLM-assisted: Claude, see module docstring.
    """
    gamma0 = GAMMAS_D[int(np.nanargmin(grid["l2"][0]))]   # best learning rate without penalty (row lambda = 0)
    # name -> (lambda, gamma, penalty)
    configs = {"no penalty": (0.0, gamma0, None), "L2": (*best["l2"], "l2"), "L1": (*best["l1"], "l1")}
    test = {name: [] for name in configs}                  # name -> one test MSE per data set
    for seed in SEEDS_D_TEST:
        data = runge_data(seed)
        X_tr = standardise(data["x_train_full"], data["x_train_full"])
        X_te = standardise(data["x_test"], data["x_train_full"])
        for name, (lam, gamma, penalty) in configs.items():
            net = regression_network(HIDDEN_D, ACTIVATION_D, seed=seed)
            train(net, X_tr, data["y_train_full"].reshape(-1, 1), "adam", gamma, EPOCHS_B, batch_size=BATCH_B,
                  seed=seed, lam=lam, penalty=penalty)
            test[name].append(network_mse(net, X_te, data["y_test"]))

    log(f"\nTest MSE on the 20 data sets of Project 1 (seeds {SEEDS_D_TEST[0]}-{SEEDS_D_TEST[-1]}), trained on all 80 points")
    log("method                         lambda   gamma     median   25th-75th percentile")
    for name, (lam, gamma, _) in configs.items():
        t = np.array(test[name])
        log(f"network {ACTIVATION_D} 3x50, {name:10s} {lam:7.0e} {gamma:7.0e}  {np.median(t):.3e}   "
            f"{np.percentile(t, 25):.3e} - {np.percentile(t, 75):.3e}")
    for name, v in PROJECT1_TEST.items():
        log(f"Project 1, {name:5s} (5-fold CV over degree and lambda)          {v:.3e}")

    # The test MSE the true function f itself would get: the mean squared noise
    noise = []
    for seed in SEEDS_D_TEST:
        data = runge_data(seed)
        noise.append(np.mean((data["y_test"] - runge(data["x_test"])) ** 2))
    log(f"Mean squared noise in the 20 test sets (the error of the true function): median {np.median(noise):.3e}")


def part_d() -> None:
    """
    Part d: L2 and L1 penalties on the weights; lambda x gamma grids on
    validation data; the chosen models on the 20 data sets of Project 1 against
    Ridge and Lasso.

    LLM-assisted: Claude, see module docstring.
    """
    log("=" * 72)
    log("Part d) L2 and L1 penalties")
    log("=" * 72)
    log(f"Network: {len(HIDDEN_D)} hidden layers of {HIDDEN_D[0]} {ACTIVATION_D} nodes (overfits in part c), linear output, "
        f"MSE + lambda * sum W^2 (L2) or lambda * sum |W| (L1) over all weights, biases not penalised.")
    log(f"SGD + Adam, minibatch {BATCH_B}, {EPOCHS_B} epochs. L1 gradient: lambda * sign(W), sign(0) = 0.")
    grid = d_grid()
    best = d_heatmaps(grid)
    d_test(grid, best)
    log(f"\nFigures saved in {FIGURE_DIR}/")
    save_log("part_d.txt")


# ---------------------------------------------------------------------------
# Part e: classification of MNIST
# ---------------------------------------------------------------------------

MNIST_FILE = Path(__file__).resolve().parent.parent / "mnist.npz"   # mnist.npz next to the Code folder, see README
EPOCHS_E = 20
BATCH_E = 128
SEED_E = 2026
GAMMAS_E = [1e-4, 3e-4, 1e-3, 3e-3]
LAMBDAS_E = [0.0, 1e-5, 1e-4, 1e-3]
HIDDEN_E = [100]                               # network for the gamma x lambda grid
ARCHITECTURES_E = [[50], [100], [200], [100, 100]]
ACTIVATIONS_E = ["sigmoid", "relu", "leaky_relu"]
SEEDS_E_SPREAD = [2026, 2027, 2028]            # seeds for the run-to-run check


def mnist_data() -> dict[str, np.ndarray]:
    """
    MNIST with the canonical split. 10 000 of the 60 000 training images,
    drawn at random (seed SEED_E), are held out as validation set for all
    choices of hyperparameters; the network is trained on the other 50 000.
    The 10 000 test images are used once, for the final models. Each image is
    flattened to 784 pixel values, which are divided by 255 to lie in [0, 1].

    LLM-assisted: Claude, see module docstring.
    """
    with np.load(MNIST_FILE) as data:
        x = data["x_train"].reshape(60000, -1) / 255.0
        y = data["y_train"].astype(int)
        x_test = data["x_test"].reshape(10000, -1) / 255.0
        y_test = data["y_test"].astype(int)
    order = np.random.default_rng(SEED_E).permutation(60000)     # the 60 000 indices in random order
    train_idx, val_idx = order[:50000], order[50000:]
    return {"x_train": x[train_idx], "y_train": y[train_idx], "x_val": x[val_idx], "y_val": y[val_idx],
            "x_test": x_test, "y_test": y_test}


def mnist_network(hidden: list[int], activation: str = "relu", seed: int = SEED_E) -> NeuralNetwork:
    """
    784 inputs, the given hidden layers, a softmax output layer of 10 nodes and
    the cross entropy. hidden = [] gives logistic regression (no hidden layer).

    LLM-assisted: Claude, see module docstring.
    """
    layer_sizes = hidden + [10]
    activations = [activation] * len(hidden) + ["softmax"]
    return NeuralNetwork(784, layer_sizes, activations, "cross_entropy", seed=seed)


def train_mnist(net: NeuralNetwork, data: dict, gamma: float, lam: float = 0.0, record=None,
                seed: int = SEED_E) -> dict:
    """
    SGD + Adam on the 50 000 training images, EPOCHS_E epochs of minibatches of
    BATCH_E, with the L2 penalty lam * sum W^2 if lam > 0. The seed sets the
    order of the minibatches.

    LLM-assisted: Claude, see module docstring.
    """
    if lam > 0:
        penalty = "l2"
    else:
        penalty = None
    return train(net, data["x_train"], one_hot(data["y_train"], 10), "adam", gamma, EPOCHS_E, batch_size=BATCH_E,
                 seed=seed, lam=lam, penalty=penalty, record=record)


def e_grids(data: dict) -> dict[str, np.ndarray]:
    """
    Validation accuracy
    1. against gamma and lambda for one hidden layer of 100 ReLU nodes,
    2. against activation and hidden layers at the best (gamma, lambda) of 1,
    3. of logistic regression against gamma.
    Saved in results/part_e_grid.npz, since this takes several minutes.

    LLM-assisted: Claude, see module docstring.
    """
    cache = RESULT_DIR / "part_e_grid.npz"
    if cache.exists():
        log(f"(grids read from {cache}; delete the file to recompute)")
        return dict(np.load(cache))

    # 1. learning rate and penalty
    grid = np.full((len(LAMBDAS_E), len(GAMMAS_E)), np.nan)
    for i, lam in enumerate(LAMBDAS_E):
        for j, gamma in enumerate(GAMMAS_E):
            net = mnist_network(HIDDEN_E, "relu")
            if not train_mnist(net, data, gamma, lam)["diverged"]:
                grid[i, j] = accuracy(net.predict(data["x_val"]), data["y_val"])
    i, j = np.unravel_index(np.nanargmax(grid), grid.shape)    # the highest accuracy
    lam_best, gamma_best = LAMBDAS_E[i], GAMMAS_E[j]

    # 2. activation and hidden layers
    arch = np.full((len(ACTIVATIONS_E), len(ARCHITECTURES_E)), np.nan)
    for a, act in enumerate(ACTIVATIONS_E):
        for m, hidden in enumerate(ARCHITECTURES_E):
            net = mnist_network(hidden, act)
            if not train_mnist(net, data, gamma_best, lam_best)["diverged"]:
                arch[a, m] = accuracy(net.predict(data["x_val"]), data["y_val"])

    # 3. logistic regression
    logreg = np.full(len(GAMMAS_E), np.nan)
    for j, gamma in enumerate(GAMMAS_E):
        net = mnist_network([])
        if not train_mnist(net, data, gamma)["diverged"]:
            logreg[j] = accuracy(net.predict(data["x_val"]), data["y_val"])

    out = {"grid": grid, "arch": arch, "logreg": logreg, "best": np.array([lam_best, gamma_best])}
    RESULT_DIR.mkdir(exist_ok=True)
    np.savez(cache, **out)
    return out


def e_report_grids(grids: dict) -> tuple[list[int], str, float, float, float]:
    """
    Log the grids, draw the gamma x lambda heat map, and return the chosen
    network (hidden layers, activation, gamma, lambda) and the learning rate of
    logistic regression.

    LLM-assisted: Claude, see module docstring.
    """
    grid, arch, logreg = grids["grid"], grids["arch"], grids["logreg"]
    lam_best, gamma_best = grids["best"]
    log(f"\nValidation accuracy, one hidden layer of {HIDDEN_E[0]} ReLU nodes (rows lambda, columns gamma)")
    log("  lambda  " + "".join(f"{g:>10.0e}" for g in GAMMAS_E))
    for i, lam in enumerate(LAMBDAS_E):
        log(f"  {lam:7.0e} " + "".join(f"{v:10.4f}" for v in grid[i]))
    log(f"  best: lambda = {lam_best:.0e}, gamma = {gamma_best:.0e}")

    log(f"\nValidation accuracy at gamma = {gamma_best:.0e}, lambda = {lam_best:.0e} (rows activation, columns hidden layers)")
    log("  activation  " + "".join(f"{label(h):>11s}" for h in ARCHITECTURES_E))
    for a, act in enumerate(ACTIVATIONS_E):
        log(f"  {ACT_LABEL[act]:10s}  " + "".join(f"{v:11.4f}" for v in arch[a]))
    a, m = np.unravel_index(np.nanargmax(arch), arch.shape)          # the best activation and hidden layers
    log(f"  best: {ACT_LABEL[ACTIVATIONS_E[a]]}, hidden layers {label(ARCHITECTURES_E[m])}")
    # standard error of an accuracy p on n images: sqrt(p (1 - p) / n), lecture notes Eq. 2.24
    n_val = 10000
    se = np.sqrt(arch[a, m] * (1 - arch[a, m]) / n_val)
    log(f"  standard error of a validation accuracy near {arch[a, m]:.3f}: {se:.4f} (Eq. 2.24, n = {n_val})")

    log("\nLogistic regression (no hidden layer), validation accuracy against gamma")
    log("  " + "  ".join(f"{g:.0e}: {v:.4f}" for g, v in zip(GAMMAS_E, logreg)))
    gamma_lr = GAMMAS_E[int(np.nanargmax(logreg))]

    # Heat map of the gamma x lambda grid; dark = high accuracy
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    im = ax.imshow(grid, cmap="Blues", aspect="auto")
    for i in range(len(LAMBDAS_E)):
        for j in range(len(GAMMAS_E)):
            # position of the value between the lowest and highest, for the text colour
            frac = (grid[i, j] - np.nanmin(grid)) / (np.nanmax(grid) - np.nanmin(grid))
            ax.text(j, i, f"{100 * grid[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if frac > 0.6 else "black")
    i, j = LAMBDAS_E.index(lam_best), GAMMAS_E.index(gamma_best)
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=COLOURS[1], lw=2))
    ax.set(xticks=range(len(GAMMAS_E)), xticklabels=[f"{g:.0e}" for g in GAMMAS_E], yticks=range(len(LAMBDAS_E)),
           yticklabels=["0"] + [f"{lam:.0e}" for lam in LAMBDAS_E[1:]], xlabel=r"Learning rate $\gamma$",
           ylabel=r"$\lambda$ ($L_2$)", title=f"Validation accuracy (%), {HIDDEN_E[0]} ReLU nodes")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Validation accuracy")
    save_figure_tight(fig, "e_heatmap")
    return ARCHITECTURES_E[m], ACTIVATIONS_E[a], float(gamma_best), float(lam_best), gamma_lr


def e_seed_spread(data: dict, hidden: list[int], act: str, gamma: float, lam: float) -> None:
    """
    How much does the validation accuracy of the chosen network change if only
    the random parts of the training change (the initial weights and the order
    of the minibatches)? Same data and hyperparameters, seeds SEEDS_E_SPREAD.
    The accuracy is recorded after every epoch, to see how much it jumps from
    one epoch to the next with a constant learning rate.

    LLM-assisted: Claude, see module docstring.
    """
    def record(net: NeuralNetwork) -> float:
        return accuracy(net.predict(data["x_val"]), data["y_val"])

    log(f"\nSame network ({label(hidden)} {ACT_LABEL[act]}), other seeds for initialisation and minibatch order")
    log("  validation accuracy after epoch 19 and 20, and lowest / highest over epochs 11-20")
    for seed in SEEDS_E_SPREAD:
        net = mnist_network(hidden, act, seed=seed)
        hist = np.array(train_mnist(net, data, gamma, lam, record=record, seed=seed)["history"])
        # hist[e - 1] is the accuracy after epoch e
        log(f"  seed {seed}:  epoch 19: {hist[18]:.4f}  epoch 20: {hist[19]:.4f}   "
            f"epochs 11-20: {hist[10:].min():.4f} - {hist[10:].max():.4f}")


def confusion_matrix_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> np.ndarray:
    """
    K x K confusion matrix: element (i, j) is the number of samples of true
    class i that were predicted as class j (rows true, columns predicted).

    LLM-assisted: Claude, see module docstring.
    """
    cm = np.zeros((k, k), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm


def e_final(data: dict, hidden: list[int], act: str, gamma: float, lam: float,
            gamma_lr: float) -> dict[str, NeuralNetwork]:
    """
    Train the chosen network and logistic regression (recording the validation
    accuracy after every epoch), and evaluate them once on the test set:
    accuracy with standard error, the confusion matrix of the network (checked
    against scikit-learn), precision and recall for each digit, the most
    frequent confusions, and scikit-learn's LogisticRegression and
    MLPClassifier (same hidden layer) as references. Returns the two trained
    models (for the weight figure).

    LLM-assisted: Claude, see module docstring.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    from sklearn.neural_network import MLPClassifier

    def record(net: NeuralNetwork) -> float:
        return accuracy(net.predict(data["x_val"]), data["y_val"])

    # name -> (network, learning rate, lambda)
    models = {"logistic regression": (mnist_network([]), gamma_lr, 0.0),
              f"network {label(hidden)} {ACT_LABEL[act]}": (mnist_network(hidden, act), gamma, lam)}
    curves, preds = {}, {}           # validation accuracy per epoch, and predicted digits on the test set
    for name, (net, g, l) in models.items():
        t0 = time.perf_counter()
        curves[name] = train_mnist(net, data, g, l, record=record)["history"]
        preds[name] = np.argmax(net.predict(data["x_test"]), axis=1)
        log(f"  trained {name} (gamma = {g:.0e}, lambda = {l:.0e}) in {time.perf_counter() - t0:.0f} s")

    # The scikit-learn references are trained first, so that all the test results come together in the log
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)     # they stop at the iteration limit, as we ask
        t0 = time.perf_counter()
        sk_lr = LogisticRegression(max_iter=200).fit(data["x_train"], data["y_train"])
        time_sk_lr = time.perf_counter() - t0
        t0 = time.perf_counter()
        if act == "sigmoid":
            sk_activation = "logistic"
        else:
            sk_activation = "relu"
        sk_mlp = MLPClassifier(hidden_layer_sizes=tuple(hidden), activation=sk_activation,
                               solver="adam", learning_rate_init=gamma, batch_size=BATCH_E, max_iter=EPOCHS_E,
                               alpha=0.0, random_state=SEED_E).fit(data["x_train"], data["y_train"])
        time_sk_mlp = time.perf_counter() - t0

    n = len(data["y_test"])
    most_frequent = np.bincount(data["y_train"]).argmax()      # the most common digit in the training data
    log(f"\nTest set ({n} images), used once")
    log(f"  {'most frequent digit':32s} accuracy {np.mean(data['y_test'] == most_frequent):.4f}")
    for name, p in preds.items():
        acc = np.mean(p == data["y_test"])
        log(f"  {name:32s} accuracy {acc:.4f} +- {np.sqrt(acc * (1 - acc) / n):.4f}")
    log(f"  {'scikit-learn LogisticRegression':32s} accuracy {sk_lr.score(data['x_test'], data['y_test']):.4f}")
    log(f"  {'scikit-learn MLPClassifier':32s} accuracy {sk_mlp.score(data['x_test'], data['y_test']):.4f}")
    log(f"  (most frequent digit in the training data: {most_frequent}; scikit-learn LogisticRegression: lbfgs, "
        f"200 iterations, default L2 penalty C = 1, {time_sk_lr:.0f} s; MLPClassifier: same hidden layer, Adam, "
        f"learning rate, minibatch size and {EPOCHS_E} epochs as our network, no penalty, {time_sk_mlp:.0f} s)")

    # Confusion matrix of the network, and precision and recall for each digit
    name = list(preds)[1]                       # the network (the second model)
    cm = confusion_matrix_k(data["y_test"], preds[name])
    log(f"\nConfusion matrix of the {name} on the test set (rows true digit, columns predicted)")
    log("  agrees with scikit-learn: " + str(np.array_equal(cm, sk_confusion_matrix(data["y_test"], preds[name]))))
    log("       " + "".join(f"{j:6d}" for j in range(10)))
    for i in range(10):
        log(f"  {i:3d}  " + "".join(f"{cm[i, j]:6d}" for j in range(10)))
    recall = np.diag(cm) / cm.sum(axis=1)       # correct / all images of the digit (row sum)
    precision = np.diag(cm) / cm.sum(axis=0)    # correct / all images predicted as the digit (column sum)
    log("  digit      " + "".join(f"{j:7d}" for j in range(10)))
    log("  precision  " + "".join(f"{v:7.3f}" for v in precision))
    log("  recall     " + "".join(f"{v:7.3f}" for v in recall))
    # The six largest elements off the diagonal: the most frequent confusions
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(off, axis=None)[::-1][:6]
    log("  most frequent confusions (true -> predicted): " + ", ".join(
        f"{i} -> {j}: {off[i, j]}" for i, j in zip(*np.unravel_index(order, off.shape))))
    cm_lr = confusion_matrix_k(data["y_test"], preds["logistic regression"])
    log("  errors per true digit, logistic regression: " + " ".join(f"{v:4d}" for v in cm_lr.sum(axis=1) - np.diag(cm_lr)))
    log("  errors per true digit, network:             " + " ".join(f"{v:4d}" for v in cm.sum(axis=1) - np.diag(cm)))

    # Figure: the confusion matrix without the diagonal, and precision and recall
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    im = ax[0].imshow(off, cmap="Blues")
    for i in range(10):
        for j in range(10):
            if i != j and off[i, j] > 0:
                ax[0].text(j, i, str(off[i, j]), ha="center", va="center", fontsize=7,
                           color="white" if off[i, j] > 0.6 * off.max() else "black")
    ax[0].set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="True digit",
              title=f"Test errors, {name} (diagonal removed)")
    ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], shrink=0.85, label="Number of test images")
    w = 0.38                                    # width of the bars
    ax[1].bar(np.arange(10) - w / 2, precision, w, color=COLOURS[0], label="Precision")
    ax[1].bar(np.arange(10) + w / 2, recall, w, color=COLOURS[1], label="Recall")
    ax[1].set(xticks=range(10), xlabel="Digit", ylim=(min(precision.min(), recall.min()) - 0.02, 1.0),
              title="Per-class precision and recall")
    ax[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=2, fontsize=8)
    save_figure(fig, "e_confusion")

    # Figure: validation accuracy after every epoch
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for (name_c, c), colour, marker in zip(curves.items(), COLOURS, MARKERS):
        ax.plot(range(1, len(c) + 1), c, marker=marker, color=colour, label=name_c)
    ax.set(xlabel="Epoch", ylabel="Validation accuracy", xticks=range(0, EPOCHS_E + 1, 5))
    ax.legend()
    save_figure(fig, "e_training_curves")

    trained = {}
    for name_m, (net, _, _) in models.items():
        trained[name_m] = net
    return trained


def e_weights(models: dict[str, NeuralNetwork]) -> None:
    """
    Illustration of what logistic regression and the hidden layer learn: the
    weights from the 784 pixels shown as 28 x 28 images. Logistic regression
    has one column of weights per digit (a template); the network has one
    column per hidden node. 20 of the hidden nodes are chosen at random.

    LLM-assisted: Claude, see module docstring.
    """
    W_logreg = models["logistic regression"].layers[0][0]               # (784, 10): one template per digit
    network = [net for name, net in models.items() if name != "logistic regression"][0]
    W_hidden = network.layers[0][0]                                       # (784, nodes): one feature per node
    pick = np.random.default_rng(0).choice(W_hidden.shape[1], 20, replace=False)

    def show(ax: plt.Axes, w: np.ndarray) -> None:
        """One weight column as a 28 x 28 image, red positive and blue negative, on its own colour scale."""
        v = np.abs(w).max()
        ax.imshow(w.reshape(28, 28), cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig = plt.figure(figsize=(12, 4.6))
    grid = fig.add_gridspec(3, 10, hspace=0.35, wspace=0.08)
    for k in range(10):                           # top row: the templates of logistic regression
        ax = fig.add_subplot(grid[0, k])
        show(ax, W_logreg[:, k])
        ax.set_title(f"digit {k}", fontsize=8)
        if k == 0:
            ax.set_ylabel("logistic\nregression", fontsize=8)
    for n, j in enumerate(pick):                  # two lower rows: 20 hidden nodes
        ax = fig.add_subplot(grid[1 + n // 10, n % 10])
        show(ax, W_hidden[:, j])
        if n % 10 == 0:
            ax.set_ylabel("hidden\nnodes", fontsize=8)
    fig.suptitle("Weights as 28x28 images (red positive, blue negative). Top: the ten class templates of logistic "
                 f"regression. Bottom: 20 of the {W_hidden.shape[1]} hidden nodes of the network", fontsize=9)
    save_figure_tight(fig, "illustration_mnist_weights")


def part_e() -> None:
    """
    Part e: MNIST with softmax output and cross entropy; logistic regression as
    the baseline; hyperparameters chosen on a validation set; one evaluation on
    the test set with confusion matrix; the weight illustration.

    LLM-assisted: Claude, see module docstring.
    """
    log("=" * 72)
    log("Part e) Classification of MNIST")
    log("=" * 72)
    log("60 000 training images split at random (seed 2026) into 50 000 training / 10 000 validation; "
        "10 000 test images used once at the end.")
    log(f"Pixels scaled to [0, 1]; softmax output with cross entropy; SGD + Adam, minibatch {BATCH_E}, {EPOCHS_E} epochs; "
        "He initialisation for ReLU, N(0, 1/n_in) otherwise.")
    data = mnist_data()
    grids = e_grids(data)
    hidden, act, gamma, lam, gamma_lr = e_report_grids(grids)
    e_seed_spread(data, hidden, act, gamma, lam)
    models = e_final(data, hidden, act, gamma, lam, gamma_lr)
    e_weights(models)
    log(f"\nFigures saved in {FIGURE_DIR}/")
    save_log("part_e.txt")


# ---------------------------------------------------------------------------
# Running the parts from the command line
# ---------------------------------------------------------------------------

PARTS = {"b": part_b, "c": part_c, "d": part_d, "e": part_e}
SLOW_PARTS = {"e": "4 to 8 min with the grids cached in results/, 12 to 25 min without"}


def main() -> None:
    """
    Run the parts named on the command line, all parts with "all", or, without
    arguments, all parts except the slow ones in SLOW_PARTS.

    LLM-assisted: Claude, see module docstring.
    """
    args = [a.lower() for a in sys.argv[1:]]        # e.g. ["b", "e"] for "python main.py b e"
    if args == ["all"]:
        selected = list(PARTS)
    elif args:
        unknown = [a for a in args if a not in PARTS]
        if unknown:
            sys.exit(f"Unknown part(s): {', '.join(unknown)}. Choose from {', '.join(PARTS)} or 'all'.")
        selected = args
    else:
        selected = [p for p in PARTS if p not in SLOW_PARTS]
        for part, note in SLOW_PARTS.items():
            print(f"(part {part} is not run by default: {note}; run it with 'python main.py {part}')")
    for part in selected:
        t0 = time.perf_counter()
        PARTS[part]()
        print(f"(part {part}: {time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
