"""
FYS-STK3155 Project 1: Regression analysis, resampling methods and gradient descent.

    uv run main.py           the fast parts (b, c, d, e, h), about 4-5 minutes
    uv run main.py a f       the named parts
    uv run main.py all       all parts, about 30 minutes

The slow parts (a, f, g, i) are skipped unless named; SLOW_PARTS below gives
their approximate run times. Figures are saved as PDF in figures/, and the
terminal output is also written to results/ (the numbers in the report).

Every experiment is identified by a seed that fixes both the data (x values
and noise) and the train/test split. Repeated experiments use the
deterministic seeds 42, 43, ..., 241 (parts a and b) or 42, ..., 61 (parts c,
d and i); seed 42 is the single split that is also reported on its own.

LLM-assisted
------------
Tool: Claude (Opus 5.5, claude.ai, September 2026)
Role: Wrote the file (level 4) from the choices agreed with the author:
x ~ U[-1, 1], 80/20 split, standardised columns and centred y with training
statistics, OLS via the pseudoinverse, Ridge with the 1/n convention,
repetitions with fixed seeds, medians with interquartile bands, and the
experiments of each part. Every function below carries a short LLM-assisted
tag that refers to this note.
Verification: tests.py, and checks printed by each part (for example that the
first repetition reproduces the single split exactly, that lambda = 0 is OLS,
and that gradient descent reaches the closed-form solutions).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from common import (PolynomialRegression, Scaler, bootstrap_bias_variance, design_matrix, fit_ols,
                    fit_ridge, lasso_fit_sklearn, make_data, mse, ols_fit, r2, ridge_fit, ridge_fit_svd,
                    runge, split_data)
import time
import jax
import jax.numpy as jnp
from optim import METHODS, cost, gradient_analytic, gradient_descent, gradient_jax, hessian, optimiser_step, sgd

SEED = 42
SEEDS = range(42, 242)
DEGREES_MAIN = range(1, 16)
DEGREES_EXTENDED = range(1, 26)
N_MAIN = 100
SIGMA_MAIN = 0.1
METRICS = ["mse_train", "mse_test", "r2_train", "r2_test"]

FIGURE_DIR = Path("figures")
RESULT_DIR = Path("results")

# Categorical colours in fixed order (blue, orange, aqua, yellow)
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GREY = "#777777"

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Log:
    """
    Prints to the terminal and keeps a copy for the results folder.

    LLM-assisted: Claude, see module docstring.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def save(self, filename: str) -> None:
        RESULT_DIR.mkdir(exist_ok=True)
        (RESULT_DIR / filename).write_text("\n".join(self.lines) + "\n")


def save_figure(fig: plt.Figure, name: str) -> None:
    """
    Save a figure as PDF in the figure folder and close it.

    LLM-assisted: Claude, see module docstring.
    """
    FIGURE_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}.pdf")
    plt.close(fig)


def ols_errors(n: int, noise_std: float, seed: int, degrees: range) -> dict[str, np.ndarray]:
    """
    One experiment: generate and split the data with the given seed, fit scaled
    OLS for every degree, and return training/test MSE and R2 per degree.

    LLM-assisted: Claude, see module docstring.
    """
    x_train, x_test, y_train, y_test = split_data(n, noise_std, seed)
    errors = {key: np.empty(len(degrees)) for key in METRICS}
    for i, degree in enumerate(degrees):
        scaler, theta = fit_ols(x_train, y_train, degree)
        y_fit = scaler.predict(design_matrix(x_train, degree), theta)
        y_pred = scaler.predict(design_matrix(x_test, degree), theta)
        errors["mse_train"][i] = mse(y_train, y_fit)
        errors["mse_test"][i] = mse(y_test, y_pred)
        errors["r2_train"][i] = r2(y_train, y_fit)
        errors["r2_test"][i] = r2(y_test, y_pred)
    return errors


def repeated_ols(n: int, noise_std: float, degrees: range) -> dict[str, np.ndarray]:
    """
    Repeat the experiment for every seed in SEEDS. Each repetition draws new
    data and a new split. Returns arrays of shape (repetitions, degrees).

    LLM-assisted: Claude, see module docstring.
    """
    runs = [ols_errors(n, noise_std, seed, degrees) for seed in SEEDS]
    return {key: np.array([run[key] for run in runs]) for key in METRICS}


def quartiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    25th percentile, median and 75th percentile over the repetitions (axis 0).

    LLM-assisted: Claude, see module docstring.
    """
    q25, q50, q75 = np.percentile(values, [25, 50, 75], axis=0)
    return q25, q50, q75


def plot_median_band(ax: plt.Axes, degrees: range, values: np.ndarray, colour: str,
                     label: str, marker: str = "o", band_alpha: float = 0.18) -> None:
    """
    Plot the median over repetitions as a line and the interquartile range
    as a shaded band.

    LLM-assisted: Claude, see module docstring.
    """
    q25, q50, q75 = quartiles(values)
    ax.fill_between(degrees, q25, q75, color=colour, alpha=band_alpha, lw=0)
    ax.plot(degrees, q50, f"{marker}-", color=colour, label=label)


def best_degree(values: np.ndarray, degrees: range) -> tuple[int, float]:
    """
    Degree with the lowest median over repetitions, and that median.

    LLM-assisted: Claude, see module docstring.
    """
    median = np.median(values, axis=0)
    k = int(np.argmin(median))
    return degrees[k], float(median[k])


def mark_best_degree(ax: plt.Axes, values: np.ndarray, degrees: range, colour: str) -> None:
    """
    Mark the degree with the lowest median as an enlarged open marker.

    LLM-assisted: Claude, see module docstring.
    """
    degree, value = best_degree(values, degrees)
    ax.plot(degree, value, "o", ms=11, mfc="none", mec=colour, mew=1.8)


# ---------------------------------------------------------------------------
# Part a) OLS
# ---------------------------------------------------------------------------

def a_errors_against_degree(log: Log) -> None:
    """
    MSE and R2 against degree for the main setup: one split (seed 42) and the
    median with interquartile range over 200 repetitions.

    LLM-assisted: Claude, see module docstring.
    """
    degrees = DEGREES_MAIN
    single = ols_errors(N_MAIN, SIGMA_MAIN, SEED, degrees)
    rep = repeated_ols(N_MAIN, SIGMA_MAIN, degrees)
    n_train = int(round(N_MAIN * 0.8))
    assert np.allclose(rep["mse_test"][0], single["mse_test"]), "First repetition must equal the single split"

    log(f"\nMain setup: n = {N_MAIN} ({n_train} training / {N_MAIN - n_train} test points), "
        f"sigma = {SIGMA_MAIN}, sigma^2 = {SIGMA_MAIN**2:.1e}")

    log(f"\nOne split (seed {SEED})")
    log(f"{'degree':>6} {'MSE train':>11} {'MSE test':>11} {'R2 train':>9} {'R2 test':>9}")
    for i, degree in enumerate(degrees):
        log(f"{degree:>6} {single['mse_train'][i]:>11.3e} {single['mse_test'][i]:>11.3e} "
            f"{single['r2_train'][i]:>9.4f} {single['r2_test'][i]:>9.4f}")
    k = int(np.argmin(single["mse_test"]))
    log(f"Lowest test MSE at degree {degrees[k]}: {single['mse_test'][k]:.3e}")

    log(f"\n{len(SEEDS)} repetitions (seeds {SEEDS[0]}-{SEEDS[-1]}): new data and new split each time")
    log(f"{'degree':>6} {'MSE train':>11} {'MSE test':>11} {'test q25':>10} {'test q75':>10} "
        f"{'MSE test':>11} {'R2 train':>9} {'R2 test':>9}")
    log(f"{'':>6} {'median':>11} {'median':>11} {'':>10} {'':>10} {'mean':>11} {'median':>9} {'median':>9}")
    q25, q50, q75 = quartiles(rep["mse_test"])
    for i, degree in enumerate(degrees):
        log(f"{degree:>6} {np.median(rep['mse_train'][:, i]):>11.3e} {q50[i]:>11.3e} "
            f"{q25[i]:>10.3e} {q75[i]:>10.3e} {np.mean(rep['mse_test'][:, i]):>11.3e} "
            f"{np.median(rep['r2_train'][:, i]):>9.4f} {np.median(rep['r2_test'][:, i]):>9.4f}")
    degree, value = best_degree(rep["mse_test"], degrees)
    log(f"Lowest median test MSE at degree {degree}: {value:.3e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, key_train, key_test in [(ax1, "mse_train", "mse_test"), (ax2, "r2_train", "r2_test")]:
        plot_median_band(ax, degrees, rep[key_train], COLOURS[0], "Training, median")
        plot_median_band(ax, degrees, rep[key_test], COLOURS[1], "Test, median", marker="s")
        ax.plot(degrees, single[key_train], "--", color=COLOURS[0], lw=1, label=f"Training, seed {SEED}")
        ax.plot(degrees, single[key_test], "--", color=COLOURS[1], lw=1, label=f"Test, seed {SEED}")
        ax.set_xlabel("Polynomial degree")
    ax1.axhline(SIGMA_MAIN**2, color=GREY, ls=":", lw=1, label=r"$\sigma^2$")
    ax1.set_yscale("log")
    ax1.set_ylabel("MSE")
    ax2.set_ylabel(r"$R^2$")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGURE_DIR / "a_mse_r2_degree.pdf")
    plt.close(fig)


def a_check_against_sklearn(log: Log) -> None:
    """
    Compare the own OLS coefficients with scikit-learn on the same scaled data.

    LLM-assisted: Claude, see module docstring.
    """
    degree = 5
    x_train, _, y_train, _ = split_data(N_MAIN, SIGMA_MAIN, SEED)
    scaler, theta_own = fit_ols(x_train, y_train, degree)
    X_train_s = scaler.transform_X(design_matrix(x_train, degree))
    theta_skl = LinearRegression(fit_intercept=False).fit(X_train_s, scaler.transform_y(y_train)).coef_
    log(f"\nCheck against scikit-learn, degree {degree} (seed {SEED}):")
    log(f"  own:          {np.array2string(theta_own, precision=5)}")
    log(f"  scikit-learn: {np.array2string(theta_skl, precision=5)}")
    log(f"  max |difference| = {np.max(np.abs(theta_own - theta_skl)):.1e}")


def a_scaling(log: Log) -> None:
    """
    Scaled against unscaled OLS on the same data: condition numbers,
    difference in predictions, and size of the coefficients.

    LLM-assisted: Claude, see module docstring.
    """
    x_train, x_test, y_train, y_test = split_data(N_MAIN, SIGMA_MAIN, SEED)
    log(f"\nScaled vs unscaled OLS (seed {SEED}). Unscaled: intercept column kept, no centering.")
    log(f"{'degree':>6} {'cond(X) scaled':>15} {'cond(X) unscaled':>17} {'max |pred diff|':>16} "
        f"{'max |theta| std':>16} {'max |theta| orig':>17}")
    for degree in [5, 10, 15, 20, 25]:
        scaler, theta = fit_ols(x_train, y_train, degree)
        X_train = design_matrix(x_train, degree)
        pred_scaled = scaler.predict(design_matrix(x_test, degree), theta)

        X_train_u = design_matrix(x_train, degree, intercept=True)
        theta_u = ols_fit(X_train_u, y_train)
        pred_unscaled = design_matrix(x_test, degree, intercept=True) @ theta_u

        theta_0, theta_orig = scaler.original_coefficients(theta)
        log(f"{degree:>6} {np.linalg.cond(scaler.transform_X(X_train)):>15.2e} "
            f"{np.linalg.cond(X_train_u):>17.2e} "
            f"{np.max(np.abs(pred_scaled - pred_unscaled)):>16.2e} "
            f"{np.max(np.abs(theta)):>16.2e} "
            f"{max(abs(theta_0), np.max(np.abs(theta_orig))):>17.2e}")


def a_coefficients(log: Log) -> None:
    """
    Coefficients against degree (seed 42): a map of every theta_j on the
    standardised columns (degrees 1-15), and the largest |theta_j| on
    standardised columns and in the original polynomial (degrees 1-25).
    Also logs the largest odd and even coefficients against the size of the
    odd and even parts of the fitted curve, (p(x) -/+ p(-x)) / 2.

    LLM-assisted: Claude, see module docstring.
    """
    degrees = DEGREES_MAIN
    x_train, _, y_train, _ = split_data(N_MAIN, SIGMA_MAIN, SEED)
    theta_grid = np.full((len(degrees), max(degrees)), np.nan)
    for i, degree in enumerate(degrees):
        _, theta = fit_ols(x_train, y_train, degree)
        theta_grid[i, :degree] = theta

    x_grid = np.linspace(-1, 1, 2001)
    log(f"\nOdd and even coefficients against odd and even parts of the fit (seed {SEED})")
    log(f"{'degree':>6} {'max|theta| odd':>15} {'max|theta| even':>16} {'max|odd part|':>14} {'max|even part|':>15}")
    for degree in [5, 9, 15]:
        scaler, theta = fit_ols(x_train, y_train, degree)
        plus = scaler.predict(design_matrix(x_grid, degree), theta)
        minus = scaler.predict(design_matrix(-x_grid, degree), theta)
        powers = np.arange(1, degree + 1)
        log(f"{degree:>6} {np.max(np.abs(theta[powers % 2 == 1])):>15.2f} "
            f"{np.max(np.abs(theta[powers % 2 == 0])):>16.2f} "
            f"{np.max(np.abs(plus - minus) / 2):>14.3f} {np.max(np.abs(plus + minus) / 2):>15.3f}")

    largest_std, largest_orig = [], []
    for degree in DEGREES_EXTENDED:
        scaler, theta = fit_ols(x_train, y_train, degree)
        theta_0, theta_orig = scaler.original_coefficients(theta)
        largest_std.append(np.max(np.abs(theta)))
        largest_orig.append(max(abs(theta_0), np.max(np.abs(theta_orig))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    limit = np.nanmax(np.abs(theta_grid))
    image = ax1.imshow(theta_grid.T, origin="lower", cmap="RdBu_r", aspect="auto",
                       norm=SymLogNorm(linthresh=1e-2, vmin=-limit, vmax=limit),
                       extent=[0.5, max(degrees) + 0.5, 0.5, max(degrees) + 0.5])
    ax1.grid(False)
    ax1.set_xticks(list(degrees))
    ax1.set_yticks(list(degrees))
    ax1.set_xlabel("Polynomial degree of the model")
    ax1.set_ylabel(r"Coefficient $\theta_j$ (power $x^j$)")
    fig.colorbar(image, ax=ax1, label=r"$\theta_j$, standardised columns (symlog)")

    ax2.semilogy(DEGREES_EXTENDED, largest_std, "o-", color=COLOURS[0], label="Standardised columns")
    ax2.semilogy(DEGREES_EXTENDED, largest_orig, "s-", color=COLOURS[1], label="Original polynomial")
    ax2.axvline(max(degrees) + 0.5, color=GREY, ls=":", lw=1)
    ax2.set_xlabel("Polynomial degree of the model")
    ax2.set_ylabel(r"Largest $|\theta_j|$")
    ax2.legend()
    save_figure(fig, "a_theta_degree")


def a_fits(log: Log) -> None:
    """
    The training data (seed 42), Runge's function and fits of selected degrees.

    LLM-assisted: Claude, see module docstring.
    """
    x_train, _, y_train, _ = split_data(N_MAIN, SIGMA_MAIN, SEED)
    x_plot = np.linspace(-1, 1, 400)
    log(f"\nFits (seed {SEED}): training points span [{x_train.min():.3f}, {x_train.max():.3f}]")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(x_train, y_train, "o", color="#9a9a9a", ms=3, label="Training data")
    ax.plot(x_plot, runge(x_plot), color="#222222", lw=1.2, ls="--", label="Runge function")
    for colour, degree in zip(COLOURS[:3], [3, 8, 15]):
        scaler, theta = fit_ols(x_train, y_train, degree)
        y_plot = scaler.predict(design_matrix(x_plot, degree), theta)
        log(f"  degree {degree:>2}: prediction at x = -1: {y_plot[0]:.3f}, at x = 1: {y_plot[-1]:.3f} "
            f"(f = {runge(1.0):.3f})")
        ax.plot(x_plot, y_plot, color=colour, label=f"Degree {degree}")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_ylim(-0.4, 1.4)
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save_figure(fig, "a_fits")


def a_dependence_on_n(log: Log) -> None:
    """
    Median test MSE against degree (1-25) for several data set sizes.

    LLM-assisted: Claude, see module docstring.
    """
    degrees = DEGREES_EXTENDED
    n_values = [30, 50, 100, 1000]
    runs = {n: repeated_ols(n, SIGMA_MAIN, degrees) for n in n_values}

    log(f"\nDependence on n (sigma = {SIGMA_MAIN}): median test MSE over {len(SEEDS)} repetitions")
    log(f"{'degree':>6} " + " ".join(f"{f'n={n}':>10}" for n in n_values))
    medians = {n: np.median(runs[n]["mse_test"], axis=0) for n in n_values}
    for i, degree in enumerate(degrees):
        log(f"{degree:>6} " + " ".join(f"{medians[n][i]:>10.3e}" for n in n_values))
    for n in n_values:
        degree, value = best_degree(runs[n]["mse_test"], degrees)
        log(f"  n = {n:>4}: lowest median test MSE at degree {degree} ({value:.3e})")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for colour, n in zip(COLOURS, n_values):
        plot_median_band(ax, degrees, runs[n]["mse_test"], colour, f"$n = {n}$", band_alpha=0.12)
        mark_best_degree(ax, runs[n]["mse_test"], degrees, colour)
    ax.axhline(SIGMA_MAIN**2, color=GREY, ls=":", lw=1, label=r"$\sigma^2$")
    ax.set_yscale("log")
    ax.set_ylim(5e-3, 1.0)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Test MSE")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save_figure(fig, "a_mse_n")


def a_dependence_on_sigma(log: Log) -> None:
    """
    Median training and test MSE against degree (1-25) for several noise levels.

    LLM-assisted: Claude, see module docstring.
    """
    degrees = DEGREES_EXTENDED
    sigma_values = [0.01, 0.1, 0.5]
    runs = {s: repeated_ols(N_MAIN, s, degrees) for s in sigma_values}

    log(f"\nDependence on sigma (n = {N_MAIN}): median training / test MSE over {len(SEEDS)} repetitions")
    log(f"{'degree':>6} " + " ".join(f"{f'sigma={s}':>23}" for s in sigma_values))
    for i, degree in enumerate(degrees):
        log(f"{degree:>6} " + " ".join(
            f"{np.median(runs[s]['mse_train'][:, i]):>11.3e} {np.median(runs[s]['mse_test'][:, i]):>11.3e}"
            for s in sigma_values))
    for s in sigma_values:
        degree, value = best_degree(runs[s]["mse_test"], degrees)
        log(f"  sigma = {s:>4}: lowest median test MSE at degree {degree} ({value:.3e}), sigma^2 = {s**2:.1e}")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for colour, s in zip(COLOURS, sigma_values):
        plot_median_band(ax, degrees, runs[s]["mse_test"], colour, rf"Test, $\sigma = {s}$", band_alpha=0.12)
        mark_best_degree(ax, runs[s]["mse_test"], degrees, colour)
        ax.plot(degrees, np.median(runs[s]["mse_train"], axis=0), "--", color=colour, lw=1.2,
                label=rf"Training, $\sigma = {s}$")
        ax.axhline(s**2, color=colour, ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_ylim(top=1.0)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("MSE (median)")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save_figure(fig, "a_mse_sigma")


def extrapolation_distance(x_train: np.ndarray, x_test: np.ndarray) -> float:
    """
    How far the test points reach outside the interval covered by the
    training points (0 if every test point lies inside it).

    LLM-assisted: Claude, see module docstring.
    """
    return float(max(0.0, x_train.min() - x_test.min(), x_test.max() - x_train.max()))


def a_extrapolation(log: Log) -> None:
    """
    Diagnostic for the gap between mean and median test MSE: are the extreme
    runs the ones where test points lie outside the training interval?
    For every repetition (n = 100, sigma = 0.1) the test MSE is computed with
    all test points and with only the test points inside the training interval.

    LLM-assisted: Claude, see module docstring.
    """
    degrees = [5, 10, 15, 20]
    distance = np.empty(len(SEEDS))
    mse_all = np.empty((len(SEEDS), len(degrees)))
    mse_inside = np.empty((len(SEEDS), len(degrees)))
    for r, seed in enumerate(SEEDS):
        x_train, x_test, y_train, y_test = split_data(N_MAIN, SIGMA_MAIN, seed)
        distance[r] = extrapolation_distance(x_train, x_test)
        inside = (x_test >= x_train.min()) & (x_test <= x_train.max())
        for i, degree in enumerate(degrees):
            scaler, theta = fit_ols(x_train, y_train, degree)
            y_pred = scaler.predict(design_matrix(x_test, degree), theta)
            mse_all[r, i] = mse(y_test, y_pred)
            mse_inside[r, i] = mse(y_test[inside], y_pred[inside])

    outside = distance > 0
    log(f"\nExtrapolation diagnostic (n = {N_MAIN}, sigma = {SIGMA_MAIN}, {len(SEEDS)} repetitions)")
    log(f"Repetitions with at least one test point outside the training interval: "
        f"{outside.sum()} of {len(SEEDS)}")
    log(f"Largest distance outside: {distance.max():.3f}, median distance when outside: "
        f"{np.median(distance[outside]):.3f}")
    log(f"\n{'':>6} {'all test points':>25} {'only inside training range':>28} {'median test MSE':>25}")
    log(f"{'degree':>6} {'median':>12} {'mean':>12} {'median':>14} {'mean':>13} "
        f"{'extrapolating':>14} {'not extrap.':>11}")
    for i, degree in enumerate(degrees):
        log(f"{degree:>6} {np.median(mse_all[:, i]):>12.3e} {np.mean(mse_all[:, i]):>12.3e} "
            f"{np.median(mse_inside[:, i]):>14.3e} {np.mean(mse_inside[:, i]):>13.3e} "
            f"{np.median(mse_all[outside, i]):>14.3e} {np.median(mse_all[~outside, i]):>11.3e}")
    for i, degree in enumerate(degrees):
        worst = np.argsort(mse_all[:, i])[-10:]
        log(f"  degree {degree:>2}: {outside[worst].sum()} of the 10 worst repetitions extrapolate")

    # Worst repetitions at degree 20 among those that do NOT extrapolate:
    # where is the worst test point, and how large is the gap in the training data there?
    degree = 20
    i = degrees.index(degree)
    candidates = [r for r in np.argsort(mse_all[:, i])[::-1] if not outside[r]][:5]
    spacings = []
    log(f"\nDegree {degree}, the 5 worst repetitions without extrapolation:")
    log(f"{'seed':>6} {'test MSE':>10} {'x worst':>8} {'gap there':>10} {'pred - f':>9} {'noise':>7}")
    for r in candidates:
        seed = SEEDS[r]
        x_train, x_test, y_train, y_test = split_data(N_MAIN, SIGMA_MAIN, seed)
        scaler, theta = fit_ols(x_train, y_train, degree)
        y_pred = scaler.predict(design_matrix(x_test, degree), theta)
        k = int(np.argmax((y_test - y_pred) ** 2))
        x_sorted = np.sort(x_train)
        j = int(np.searchsorted(x_sorted, x_test[k]))
        log(f"{seed:>6} {mse_all[r, i]:>10.3f} {x_test[k]:>8.3f} {x_sorted[j] - x_sorted[j - 1]:>10.3f} "
            f"{y_pred[k] - runge(x_test[k]):>9.2f} {y_test[k] - runge(x_test[k]):>7.3f}")
    for seed in SEEDS:
        x_train = split_data(N_MAIN, SIGMA_MAIN, seed)[0]
        spacings.append(np.median(np.diff(np.sort(x_train))))
    log(f"Median spacing between neighbouring training points: {np.median(spacings):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, degree, colour in zip(axes, [10, 20], COLOURS[:2]):
        i = degrees.index(degree)
        ax.plot(distance[~outside], mse_all[~outside, i], "o", color=GREY, ms=4, alpha=0.6,
                label="All test points inside")
        ax.plot(distance[outside], mse_all[outside, i], "o", color=colour, ms=4,
                label="Test point(s) outside")
        ax.axhline(SIGMA_MAIN**2, color=GREY, ls=":", lw=1, label=r"$\sigma^2$")
        ax.set_yscale("log")
        ax.set_xlabel("Distance outside the training interval")
        ax.set_title(f"Degree {degree}", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Test MSE (one repetition)")
    save_figure(fig, "a_extrapolation")


def a_single_split_representativeness(log: Log) -> None:
    """
    How representative is the single split (seed 42)? Where the test points
    lie, the spread of y in the test and training sets, and the test MSE
    compared with the expected MSE on new data. The latter is computed from
    the known Runge function as mean((y_pred - f)^2) over a fine grid on
    [-1, 1], plus sigma^2.

    LLM-assisted: Claude, see module docstring.
    """
    x_train, x_test, y_train, y_test = split_data(N_MAIN, SIGMA_MAIN, SEED)
    log(f"\nRepresentativeness of the single split (seed {SEED})")
    log(f"  test points with |x| < 0.2:     {np.sum(np.abs(x_test) < 0.2)} of {len(x_test)}")
    log(f"  training points with |x| < 0.2: {np.sum(np.abs(x_train) < 0.2)} of {len(x_train)}")
    log(f"  largest y: test {y_test.max():.3f}, training {y_train.max():.3f}")
    log(f"  Var(y): test {np.var(y_test):.3f}, training {np.var(y_train):.3f}")
    log(f"  mean squared noise in the test points: {np.mean((y_test - runge(x_test)) ** 2):.2e} "
        f"(sigma^2 = {SIGMA_MAIN**2:.1e})")
    x_grid = np.linspace(-1, 1, 20001)
    log(f"{'degree':>8} {'test MSE':>10} {'expected MSE on new data':>26}")
    for degree in [5, 10, 15]:
        scaler, theta = fit_ols(x_train, y_train, degree)
        test_mse = mse(y_test, scaler.predict(design_matrix(x_test, degree), theta))
        new_data = np.mean((scaler.predict(design_matrix(x_grid, degree), theta) - runge(x_grid)) ** 2)
        log(f"{degree:>8} {test_mse:>10.4f} {new_data + SIGMA_MAIN**2:>26.4f}")


def part_a() -> None:
    """
    Part a) OLS for the Runge function.

    LLM-assisted: Claude, see module docstring.
    """
    FIGURE_DIR.mkdir(exist_ok=True)
    log = Log()
    log("=" * 72)
    log("Part a) OLS for the Runge function")
    log("=" * 72)
    a_errors_against_degree(log)
    a_single_split_representativeness(log)
    a_check_against_sklearn(log)
    a_scaling(log)
    a_coefficients(log)
    a_fits(log)
    a_dependence_on_n(log)
    a_dependence_on_sigma(log)
    a_extrapolation(log)
    log(f"\nFigures saved in {FIGURE_DIR}/")
    log.save("part_a.txt")


# ---------------------------------------------------------------------------
# Part b) Ridge
# ---------------------------------------------------------------------------

LAMBDAS_GRID = np.concatenate([[0.0], np.logspace(-6, 1, 15)])  # 0 = OLS, then 1e-6 ... 10
LAMBDAS_SHOWN = [0.0, 1e-6, 1e-4, 1e-2]                         # the values drawn as curves


def lambda_label(lam: float) -> str:
    """
    Label for a penalty value; lambda = 0 is shown as OLS.

    LLM-assisted: Claude, see module docstring.
    """
    return "OLS" if lam == 0 else rf"Ridge $\lambda = 10^{{{int(round(np.log10(lam)))}}}$"


def text_label(lam: float) -> str:
    """
    Plain-text label for the terminal output.

    LLM-assisted: Claude, see module docstring.
    """
    return "OLS" if lam == 0 else f"lambda={lam:.0e}"


def ridge_errors(n: int, noise_std: float, seed: int, degrees: range,
                 lambdas: np.ndarray) -> dict[str, np.ndarray]:
    """
    One experiment: generate and split the data with the given seed, and fit
    every degree with every lambda (lambda = 0 is OLS). The design matrix and
    the scaling are built once per degree. Returns arrays of shape
    (degrees, lambdas) with training/test MSE and R2.

    LLM-assisted: Claude, see module docstring.
    """
    x_train, x_test, y_train, y_test = split_data(n, noise_std, seed)
    errors = {key: np.empty((len(degrees), len(lambdas))) for key in METRICS}
    for i, degree in enumerate(degrees):
        X_train = design_matrix(x_train, degree)
        scaler = Scaler().fit(X_train, y_train)
        X_train_s = scaler.transform_X(X_train)
        y_train_c = scaler.transform_y(y_train)
        X_test = design_matrix(x_test, degree)
        for j, lam in enumerate(lambdas):
            theta = ols_fit(X_train_s, y_train_c) if lam == 0 else ridge_fit(X_train_s, y_train_c, lam)
            y_fit = scaler.predict(X_train, theta)
            y_pred = scaler.predict(X_test, theta)
            errors["mse_train"][i, j] = mse(y_train, y_fit)
            errors["mse_test"][i, j] = mse(y_test, y_pred)
            errors["r2_train"][i, j] = r2(y_train, y_fit)
            errors["r2_test"][i, j] = r2(y_test, y_pred)
    return errors


def repeated_ridge(n: int, noise_std: float, degrees: range, lambdas: np.ndarray) -> dict[str, np.ndarray]:
    """
    ridge_errors for every seed in SEEDS. Arrays of shape (repetitions, degrees, lambdas).

    LLM-assisted: Claude, see module docstring.
    """
    runs = [ridge_errors(n, noise_std, seed, degrees, lambdas) for seed in SEEDS]
    return {key: np.array([run[key] for run in runs]) for key in METRICS}


def b_errors(log: Log, runs: dict[int, dict[str, np.ndarray]]) -> None:
    """
    Tables and figures of test MSE and R2 against degree and lambda for
    n = 100 and n = 30 (median and mean over the repetitions), and the
    heat maps of the median test MSE over degree and lambda.

    LLM-assisted: Claude, see module docstring.
    """
    degrees = DEGREES_EXTENDED
    lambdas = LAMBDAS_GRID
    shown = [int(np.where(lambdas == lam)[0][0]) for lam in LAMBDAS_SHOWN]

    for n, rep in runs.items():
        median = np.median(rep["mse_test"], axis=0)
        mean = np.mean(rep["mse_test"], axis=0)
        log(f"\nn = {n}, sigma = {SIGMA_MAIN}: test MSE over {len(SEEDS)} repetitions, median / mean")
        log(f"{'degree':>6} " + " ".join(f"{text_label(lambdas[j]):>21}" for j in shown))
        for i, degree in enumerate(degrees):
            if degree in (1, 5, 10, 15, 20, 25) or n == 100 and degree in (12, 14):
                log(f"{degree:>6} " + " ".join(f"{median[i, j]:>10.3e} {mean[i, j]:>10.3e}" for j in shown))
        i_best, j_best = np.unravel_index(np.argmin(median), median.shape)
        i_ols = int(np.argmin(median[:, 0]))
        log(f"  lowest median test MSE, OLS:   degree {degrees[i_ols]}, {median[i_ols, 0]:.3e}")
        log(f"  lowest median test MSE, any:   degree {degrees[i_best]}, lambda = {lambdas[j_best]:.0e}, "
            f"{median[i_best, j_best]:.3e}")
        j_last = len(lambdas) - 1
        log(f"  largest lambda ({lambdas[j_last]:.0e}): median test MSE at degree 1 / 5 / 15 / 25: "
            + " / ".join(f"{median[degrees.index(d), j_last]:.3e}" for d in (1, 5, 15, 25)))
        for j in shown:
            k = int(np.argmin(median[:, j]))
            spread = median[:, j].max() / median[:, j].min()
            log(f"  {text_label(lambdas[j]):>12}: best degree {degrees[k]:>2} "
                f"({median[k, j]:.3e}); largest / smallest median over degrees 1-25: {spread:.3g}")

    # Figure: test MSE and R2 against degree, n = 100
    rep = runs[N_MAIN]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
    for colour, j in zip(COLOURS, shown):
        plot_median_band(ax1, degrees, rep["mse_test"][:, :, j], colour, lambda_label(lambdas[j]), band_alpha=0.12)
        mark_best_degree(ax1, rep["mse_test"][:, :, j], degrees, colour)
        ax2.plot(degrees, np.median(rep["r2_test"][:, :, j], axis=0), "o-", color=colour,
                 label=lambda_label(lambdas[j]))
    ax1.axhline(SIGMA_MAIN**2, color=GREY, ls=":", lw=1, label=r"$\sigma^2$")
    ax1.set_yscale("log")
    ax1.set_ylim(5e-3, 0.2)
    ax1.set_ylabel("Test MSE")
    ax2.set_ylim(-0.1, 1)
    ax2.set_ylabel(r"Test $R^2$ (median)")
    for ax in (ax1, ax2):
        ax.set_xlabel("Polynomial degree")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGURE_DIR / "b_mse_r2_degree.pdf")
    plt.close(fig)

    # Figure: heat maps of the median test MSE, n = 100 and n = 30
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    norm = LogNorm(vmin=SIGMA_MAIN**2, vmax=0.1)
    for ax, (n, rep_n) in zip(axes, runs.items()):
        median = np.median(rep_n["mse_test"], axis=0)
        image = ax.imshow(np.clip(median.T, None, 0.1), origin="lower", aspect="auto", cmap="Blues", norm=norm,
                          extent=[0.5, max(degrees) + 0.5, -0.5, len(lambdas) - 0.5])
        i_best, j_best = np.unravel_index(np.argmin(median), median.shape)
        ax.plot(degrees[i_best], j_best, "o", ms=12, mfc="none", mec=COLOURS[1], mew=2)
        ax.axhline(0.5, color="white", lw=2)
        ax.grid(False)
        ax.set_title(f"$n = {n}$", fontsize=10)
        ax.set_xlabel("Polynomial degree")
    axes[0].set_yticks(range(len(lambdas)))
    axes[0].set_yticklabels(["OLS"] + [f"$10^{{{e:g}}}$" for e in np.log10(lambdas[1:])])
    axes[0].set_ylabel(r"$\lambda$")
    fig.colorbar(image, ax=axes, label="Median test MSE (clipped at 0.1)")
    fig.savefig(FIGURE_DIR / "b_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def b_svd_shrinkage(log: Log) -> None:
    """
    The shrinkage of the singular-value directions for degree 15 (seed 42):
    sigma_i^2 against n*lambda, the shrinkage factor sigma_i^2 / (sigma_i^2 + n lambda),
    how much of the data lies in each pattern u_i, and the component of the
    solution along each v_i for OLS and Ridge.

    LLM-assisted: Claude, see module docstring.
    """
    degree = 15
    x_train, _, y_train, _ = split_data(N_MAIN, SIGMA_MAIN, SEED)
    X_train = design_matrix(x_train, degree)
    scaler = Scaler().fit(X_train, y_train)
    X_s, y_c = scaler.transform_X(X_train), scaler.transform_y(y_train)
    n = X_s.shape[0]
    U, s, Vt = np.linalg.svd(X_s, full_matrices=False)
    uy = U.T @ y_c
    index = np.arange(1, degree + 1)
    ridge_lams = [lam for lam in LAMBDAS_SHOWN if lam > 0]

    log(f"\nSVD of the standardised design matrix, degree {degree}, seed {SEED} (n_train = {n})")
    log(f"{'i':>3} {'sigma_i':>10} {'sigma_i^2':>10} {'|u_i^T y|':>10} {'OLS |coef|':>11} "
        + " ".join(f"{f'factor {lam:.0e}':>13}" for lam in ridge_lams))
    for i in range(degree):
        log(f"{i + 1:>3} {s[i]:>10.3e} {s[i]**2:>10.3e} {abs(uy[i]):>10.3e} {abs(uy[i]) / s[i]:>11.3e} "
            + " ".join(f"{s[i]**2 / (s[i]**2 + n * lam):>13.3f}" for lam in ridge_lams))
    log(f"  noise level for comparison: sigma = {SIGMA_MAIN}")
    for lam in ridge_lams:
        df = np.sum(s**2 / (s**2 + n * lam))
        log(f"  lambda = {lam:.0e}: n*lambda = {n * lam:.1e}, directions with sigma_i^2 < n*lambda: "
            f"{np.sum(s**2 < n * lam)} of {degree}, sum of shrinkage factors = {df:.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    ax = axes[0, 0]
    ax.semilogy(index, s**2, "o-", color="#222222", label=r"$\sigma_i^2$")
    for colour, lam in zip(COLOURS[1:], ridge_lams):
        ax.axhline(n * lam, color=colour, ls="--", lw=1.2, label=rf"$n\lambda$, $\lambda = 10^{{{int(np.log10(lam))}}}$")
    ax.set_ylabel(r"$\sigma_i^2$ (curvature along $v_i$, up to $2/n$)")
    ax.set_title(r"1. $\sigma_i^2$ against the added $n\lambda$", fontsize=10, loc="left")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for colour, lam in zip(COLOURS[1:], ridge_lams):
        ax.plot(index, s**2 / (s**2 + n * lam), "o-", color=colour, label=rf"$\lambda = 10^{{{int(np.log10(lam))}}}$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(r"$\sigma_i^2 / (\sigma_i^2 + n\lambda)$")
    ax.set_title("2. Shrinkage factor", fontsize=10, loc="left")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.semilogy(index, np.abs(uy), "o-", color="#222222", label=r"$|u_i^T y|$")
    ax.axhline(SIGMA_MAIN, color=GREY, ls=":", lw=1.2, label=rf"noise level $\sigma = {SIGMA_MAIN}$")
    ax.set_ylabel(r"$|u_i^T y|$")
    ax.set_title(r"3. How much of the data lies in pattern $u_i$", fontsize=10, loc="left")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.semilogy(index, np.abs(uy) / s, "o-", color=COLOURS[0], label="OLS")
    for colour, lam in zip(COLOURS[1:], ridge_lams):
        ax.semilogy(index, np.abs(uy) * s / (s**2 + n * lam), "o-", color=colour,
                    label=rf"Ridge $\lambda = 10^{{{int(np.log10(lam))}}}$")
    ax.set_ylabel(r"$|$component of $\hat\theta$ along $v_i|$")
    ax.set_title(r"4. The solution along each $v_i$", fontsize=10, loc="left")
    ax.legend(fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel("Index $i$ (steepest to flattest direction)")
        ax.set_xticks(index)
    save_figure(fig, "b_svd_shrinkage")


def b_check_ridge(log: Log) -> None:
    """
    Validation of the Ridge code (degree 15, seed 42): against scikit-learn's
    Ridge with alpha = n * lambda, and against the SVD form of the solution.

    LLM-assisted: Claude, see module docstring.
    """
    degree = 15
    x_train, _, y_train, _ = split_data(N_MAIN, SIGMA_MAIN, SEED)
    X_train = design_matrix(x_train, degree)
    scaler = Scaler().fit(X_train, y_train)
    X_s, y_c = scaler.transform_X(X_train), scaler.transform_y(y_train)
    n = X_s.shape[0]
    log(f"\nCheck of the Ridge code, degree {degree} (seed {SEED})")
    log(f"{'lambda':>8} {'max |own - scikit-learn|':>26} {'max |own - SVD form|':>22}")
    for lam in LAMBDAS_SHOWN[1:]:
        own = ridge_fit(X_s, y_c, lam)
        skl = Ridge(alpha=n * lam, fit_intercept=False).fit(X_s, y_c).coef_
        log(f"{lam:>8.0e} {np.max(np.abs(own - skl)):>26.1e} {np.max(np.abs(own - ridge_fit_svd(X_s, y_c, lam))):>22.1e}")
    log(f"  lambda = 0 reproduces the OLS test MSE of part a exactly (checked in part_b)")


def b_coefficients(log: Log) -> None:
    """
    Largest |theta_j| on the standardised columns against degree (seed 42)
    for OLS and Ridge.

    LLM-assisted: Claude, see module docstring.
    """
    x_train, _, y_train, _ = split_data(N_MAIN, SIGMA_MAIN, SEED)
    largest = {lam: [np.max(np.abs(fit_ridge(x_train, y_train, degree, lam)[1])) for degree in DEGREES_EXTENDED]
               for lam in LAMBDAS_SHOWN}
    log(f"\nLargest |theta_j| on standardised columns (seed {SEED})")
    log(f"{'degree':>6} " + " ".join(f"{('OLS' if lam == 0 else f'lam={lam:.0e}'):>11}" for lam in LAMBDAS_SHOWN))
    for i, degree in enumerate(DEGREES_EXTENDED):
        if degree in (5, 10, 15, 20, 25):
            log(f"{degree:>6} " + " ".join(f"{largest[lam][i]:>11.3e}" for lam in LAMBDAS_SHOWN))

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for colour, lam in zip(COLOURS, LAMBDAS_SHOWN):
        ax.semilogy(DEGREES_EXTENDED, largest[lam], "o-", color=colour, label=lambda_label(lam))
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(r"Largest $|\theta_j|$, standardised columns")
    ax.legend(fontsize=8)
    save_figure(fig, "b_theta_degree")


def part_b() -> None:
    """
    Part b) Ridge regression for the Runge function.

    LLM-assisted: Claude, see module docstring.
    """
    FIGURE_DIR.mkdir(exist_ok=True)
    log = Log()
    log("=" * 72)
    log("Part b) Ridge regression for the Runge function")
    log("=" * 72)
    log(f"Cost: (1/n)||y - X theta||^2 + lambda ||theta||^2, standardised columns, centred y.")
    log(f"lambda grid: 0 (OLS) and " + ", ".join(f"{lam:.1e}" for lam in LAMBDAS_GRID[1:]))

    runs = {n: repeated_ridge(n, SIGMA_MAIN, DEGREES_EXTENDED, LAMBDAS_GRID) for n in (N_MAIN, 30)}
    ols_check = ols_errors(N_MAIN, SIGMA_MAIN, SEED, DEGREES_EXTENDED)["mse_test"]
    assert np.allclose(runs[N_MAIN]["mse_test"][0, :, 0], ols_check), "lambda = 0 must reproduce OLS"

    b_check_ridge(log)
    b_errors(log, runs)
    b_svd_shrinkage(log)
    b_coefficients(log)
    log(f"\nFigures saved in {FIGURE_DIR}/")
    log.save("part_b.txt")


# ---------------------------------------------------------------------------
# Part c) Bias-variance with the bootstrap
# ---------------------------------------------------------------------------

SEEDS_RESAMPLING = range(42, 62)   # 20 data sets for the resampling studies
N_BOOT = 100
DEGREES_C = range(1, 17)


def expected_new_data_mse(x_train: np.ndarray, y_train: np.ndarray, degree: int, lam: float = 0.0) -> float:
    """
    MSE the fitted model would get on new data: mean of (prediction - f)^2
    over a fine grid on [-1, 1], plus sigma^2. Possible only because f is known.

    LLM-assisted: Claude, see module docstring.
    """
    x_grid = np.linspace(-1, 1, 2001)
    scaler, theta = fit_ridge(x_train, y_train, degree, lam)
    return float(np.mean((scaler.predict(design_matrix(x_grid, degree), theta) - runge(x_grid)) ** 2) + SIGMA_MAIN**2)


def c_bootstrap(n: int, degrees: range) -> dict[str, np.ndarray]:
    """
    Bootstrap bias-variance decomposition for every data set in
    SEEDS_RESAMPLING (80/20 split, N_BOOT resamples of the training set).
    Also the squared bias measured against the true f instead of y_test.
    Arrays of shape (data sets, degrees).

    LLM-assisted: Claude, see module docstring.
    """
    keys = ["error", "bias2", "variance", "bias2_true"]
    out = {key: np.empty((len(SEEDS_RESAMPLING), len(degrees))) for key in keys}
    for r, seed in enumerate(SEEDS_RESAMPLING):
        x_train, x_test, y_train, y_test = split_data(n, SIGMA_MAIN, seed)
        rng = np.random.default_rng(seed)
        for i, degree in enumerate(degrees):
            result = bootstrap_bias_variance(x_train, y_train, x_test, y_test, degree, N_BOOT, rng)
            out["error"][r, i] = result["error"]
            out["bias2"][r, i] = result["bias2"]
            out["variance"][r, i] = result["variance"]
            out["bias2_true"][r, i] = np.mean((runge(x_test) - result["mean_prediction"]) ** 2)
    return out


def part_c() -> None:
    """
    Part c) Bias-variance trade-off with the bootstrap (OLS).

    LLM-assisted: Claude, see module docstring.
    """
    FIGURE_DIR.mkdir(exist_ok=True)
    log = Log()
    log("=" * 72)
    log("Part c) Bias-variance trade-off with the bootstrap (OLS)")
    log("=" * 72)
    log(f"{len(SEEDS_RESAMPLING)} data sets (seeds {SEEDS_RESAMPLING[0]}-{SEEDS_RESAMPLING[-1]}), 80/20 split, "
        f"{N_BOOT} bootstrap resamples of the training set, sigma = {SIGMA_MAIN}")
    log("Values are medians over the data sets. error = bias^2 + variance holds exactly for each data set.")
    log("bias^2 is measured against y_test (contains sigma^2); bias^2 (f) is measured against the true f.")

    n_values = [50, 100, 400]
    runs = {n: c_bootstrap(n, DEGREES_C) for n in n_values}
    identity = max(np.max(np.abs(r["error"] - r["bias2"] - r["variance"]) / r["error"]) for r in runs.values())
    log(f"Largest relative |error - bias^2 - variance| / error over all runs: {identity:.1e}")

    for n in n_values:
        med = {key: np.median(value, axis=0) for key, value in runs[n].items()}
        log(f"\nn = {n} ({int(0.8 * n)} training points)")
        log(f"{'degree':>6} {'error':>10} {'bias^2':>10} {'variance':>10} {'bias^2 (f)':>11} {'bias^2 - bias^2(f)':>19}")
        for i, degree in enumerate(DEGREES_C):
            log(f"{degree:>6} {med['error'][i]:>10.3e} {med['bias2'][i]:>10.3e} {med['variance'][i]:>10.3e} "
                f"{med['bias2_true'][i]:>11.3e} {med['bias2'][i] - med['bias2_true'][i]:>19.3e}")
        k = int(np.argmin(med["error"]))
        cross = next((d for d, b, v in zip(DEGREES_C, med["bias2"], med["variance"]) if v > b), None)
        log(f"  lowest median error at degree {DEGREES_C[k]} ({med['error'][k]:.3e}); "
            f"variance first exceeds bias^2 at degree {cross}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for ax, n in zip(axes, n_values):
        med = {key: np.median(value, axis=0) for key, value in runs[n].items()}
        ax.semilogy(DEGREES_C, med["error"], "o-", color="#222222", label="Error")
        ax.semilogy(DEGREES_C, med["bias2"], "s-", color=COLOURS[0], label=r"Bias$^2$ (vs. $y$)")
        ax.semilogy(DEGREES_C, med["bias2_true"], "--", color=COLOURS[0], lw=1.2, label=r"Bias$^2$ (vs. $f$)")
        ax.semilogy(DEGREES_C, med["variance"], "^-", color=COLOURS[1], label="Variance")
        ax.axhline(SIGMA_MAIN**2, color=GREY, ls=":", lw=1, label=r"$\sigma^2$")
        ax.set_ylim(3e-4, 1.0)
        ax.set_title(f"$n = {n}$", fontsize=10)
        ax.set_xlabel("Polynomial degree")
    axes[0].set_ylabel("Median over 20 data sets")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGURE_DIR / "c_bias_variance.pdf")
    plt.close(fig)

    # Training and test error against complexity (Hastie et al., Fig. 2.11), n = 50
    n = 50
    rep = repeated_ols(n, SIGMA_MAIN, DEGREES_EXTENDED)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    plot_median_band(ax, DEGREES_EXTENDED, rep["mse_train"], COLOURS[0], "Training")
    plot_median_band(ax, DEGREES_EXTENDED, rep["mse_test"], COLOURS[1], "Test", marker="s")
    ax.axhline(SIGMA_MAIN**2, color=GREY, ls=":", lw=1, label=r"$\sigma^2$")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.0)
    ax.set_xlabel("Polynomial degree (model complexity)")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=8)
    save_figure(fig, "c_train_test")
    med_train = np.median(rep["mse_train"], axis=0)
    med_test = np.median(rep["mse_test"], axis=0)
    log(f"\nTraining and test MSE against degree, n = {n}, {len(SEEDS)} repetitions (median)")
    for degree in (1, 5, 8, 10, 15, 20, 25):
        i = degree - 1
        log(f"  degree {degree:>2}: training {med_train[i]:.3e}, test {med_test[i]:.3e}")

    log(f"\nFigures saved in {FIGURE_DIR}/")
    log.save("part_c.txt")


# ---------------------------------------------------------------------------
# Part d) Cross-validation
# ---------------------------------------------------------------------------

LAMBDAS_CV = np.logspace(-8, 0, 17)


def cv_pipeline(degree: int, lam: float):
    """
    Pipeline with all preprocessing inside: design matrix, standardisation
    (fitted on the training folds only) and our own OLS/Ridge.

    LLM-assisted: Claude, see module docstring.
    """
    return make_pipeline(FunctionTransformer(lambda x, d=degree: design_matrix(x.ravel(), d)),
                         StandardScaler(), PolynomialRegression(lam))


def cv_mse(x: np.ndarray, y: np.ndarray, degree: int, lam: float, k: int, seed: int) -> float:
    """
    k-fold cross-validated MSE with scikit-learn's KFold and cross_val_score.

    LLM-assisted: Claude, see module docstring.
    """
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    scores = cross_val_score(cv_pipeline(degree, lam), x.reshape(-1, 1), y, cv=kfold,
                             scoring="neg_mean_squared_error")
    return float(-np.mean(scores))


def part_d() -> None:
    """
    Part d) Cross-validation for OLS and Ridge, compared with the bootstrap.

    LLM-assisted: Claude, see module docstring.
    """
    FIGURE_DIR.mkdir(exist_ok=True)
    log = Log()
    log("=" * 72)
    log("Part d) Cross-validation")
    log("=" * 72)
    n = N_MAIN
    degrees = DEGREES_C
    log(f"n = {n}, sigma = {SIGMA_MAIN}, {len(SEEDS_RESAMPLING)} data sets. CV uses all n points; "
        f"the bootstrap and the reference use the 80/20 split of part c.")
    log("Preprocessing (design matrix, standardisation) is refitted inside every training fold (Pipeline).")

    cv = {k: np.empty((len(SEEDS_RESAMPLING), len(degrees))) for k in (5, 10)}
    boot = c_bootstrap(n, degrees)["error"]
    reference = np.empty((len(SEEDS_RESAMPLING), len(degrees)))
    for r, seed in enumerate(SEEDS_RESAMPLING):
        x, y = make_data(n, SIGMA_MAIN, np.random.default_rng(seed))
        x_train, _, y_train, _ = split_data(n, SIGMA_MAIN, seed)
        for i, degree in enumerate(degrees):
            for k in (5, 10):
                cv[k][r, i] = cv_mse(x, y, degree, 0.0, k, seed)
            reference[r, i] = expected_new_data_mse(x_train, y_train, degree)

    med = {"CV k=5": np.median(cv[5], axis=0), "CV k=10": np.median(cv[10], axis=0),
           "bootstrap": np.median(boot, axis=0), "new data": np.median(reference, axis=0)}
    log(f"\nOLS, median over data sets")
    log(f"{'degree':>6} " + " ".join(f"{key:>11}" for key in med))
    for i, degree in enumerate(degrees):
        log(f"{degree:>6} " + " ".join(f"{value[i]:>11.3e}" for value in med.values()))
    for key, value in med.items():
        k = int(np.argmin(value))
        log(f"  {key:>10}: lowest at degree {degrees[k]} ({value[k]:.3e})")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    styles = {"CV k=5": (COLOURS[0], "o-"), "CV k=10": (COLOURS[2], "s-"),
              "bootstrap": (COLOURS[1], "^-"), "new data": ("#222222", "--")}
    labels = {"CV k=5": "CV, $k = 5$", "CV k=10": "CV, $k = 10$", "bootstrap": "Bootstrap test error",
              "new data": "Expected MSE on new data"}
    for key, value in med.items():
        colour, style = styles[key]
        ax.semilogy(degrees, value, style, color=colour, label=labels[key])
        mark_best_degree(ax, value[None, :], degrees, colour)
    ax.axhline(SIGMA_MAIN**2, color=GREY, ls=":", lw=1, label=r"$\sigma^2$")
    ax.set_ylim(5e-3, 1.0)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("MSE (median over 20 data sets)")
    ax.legend(fontsize=8)
    save_figure(fig, "d_cv_vs_bootstrap")

    # Ridge: CV (k = 5) over degree and lambda
    cv_ridge = np.empty((len(SEEDS_RESAMPLING), len(degrees), len(LAMBDAS_CV)))
    for r, seed in enumerate(SEEDS_RESAMPLING):
        x, y = make_data(n, SIGMA_MAIN, np.random.default_rng(seed))
        for i, degree in enumerate(degrees):
            for j, lam in enumerate(LAMBDAS_CV):
                cv_ridge[r, i, j] = cv_mse(x, y, degree, lam, 5, seed)
    med_ridge = np.median(cv_ridge, axis=0)
    i_best, j_best = np.unravel_index(np.argmin(med_ridge), med_ridge.shape)
    log(f"\nRidge, CV with k = 5, median over data sets")
    log(f"  lowest: degree {degrees[i_best]}, lambda = {LAMBDAS_CV[j_best]:.0e}, CV MSE {med_ridge[i_best, j_best]:.3e}")
    log(f"  OLS (k = 5) lowest: degree {degrees[int(np.argmin(med['CV k=5']))]}, {np.min(med['CV k=5']):.3e}")
    for degree in (5, 10, 15):
        i = degrees.index(degree)
        j = int(np.argmin(med_ridge[i]))
        log(f"  degree {degree:>2}: best lambda {LAMBDAS_CV[j]:.0e} ({med_ridge[i, j]:.3e}); OLS {med['CV k=5'][i]:.3e}")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    image = ax.imshow(np.clip(med_ridge.T, None, 0.1), origin="lower", aspect="auto", cmap="Blues",
                      norm=LogNorm(vmin=SIGMA_MAIN**2, vmax=0.1),
                      extent=[0.5, max(degrees) + 0.5, -0.5, len(LAMBDAS_CV) - 0.5])
    ax.plot(degrees[i_best], j_best, "o", ms=12, mfc="none", mec=COLOURS[1], mew=2)
    ax.grid(False)
    ax.set_yticks(range(0, len(LAMBDAS_CV), 2))
    ax.set_yticklabels([f"$10^{{{e:g}}}$" for e in np.log10(LAMBDAS_CV[::2])])
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(r"$\lambda$")
    fig.colorbar(image, ax=ax, label="Median CV MSE, $k = 5$ (clipped at 0.1)")
    save_figure(fig, "d_cv_ridge")

    log(f"\nFigures saved in {FIGURE_DIR}/")
    log.save("part_d.txt")


# ---------------------------------------------------------------------------
# Parts e-h) Gradient methods: common problem
# ---------------------------------------------------------------------------

DEGREE_GD = 6
LAMBDA_GD = 1e-2
METHOD_LABELS = {"plain": "Plain GD", "momentum": r"Momentum ($\beta = 0.9$)", "adagrad": "AdaGrad",
                 "rmsprop": r"RMSProp ($\rho = 0.99$)", "adam": "Adam"}
METHOD_COLOURS = {"plain": "#222222", "momentum": COLOURS[0], "adagrad": COLOURS[2],
                  "rmsprop": COLOURS[3], "adam": COLOURS[1]}


def gd_problem(degree: int = DEGREE_GD) -> dict:
    """
    The fixed regression problem used for the gradient methods: seed 42,
    n = 100, standardised design matrix of the given degree, centred y.

    LLM-assisted: Claude, see module docstring.
    """
    x_train, x_test, y_train, y_test = split_data(N_MAIN, SIGMA_MAIN, SEED)
    X_train = design_matrix(x_train, degree)
    scaler = Scaler().fit(X_train, y_train)
    return {"X": scaler.transform_X(X_train), "y": scaler.transform_y(y_train), "scaler": scaler,
            "X_test": design_matrix(x_test, degree), "y_test": y_test}


def iteration_prediction(eigenvalues: np.ndarray, gamma: float, distance0: float, tol: float) -> float:
    """
    Predicted number of plain-GD iterations to reduce the distance to the
    minimum from distance0 to tol: the slowest mode shrinks by the factor
    max_i |1 - gamma mu_i| per step, mu_i the Hessian eigenvalues (course notes, Sec. 4.5).

    LLM-assisted: Claude, see module docstring.
    """
    rate = np.max(np.abs(1 - gamma * eigenvalues))
    return np.inf if rate >= 1 else float(np.log(distance0 / tol) / -np.log(rate))


# ---------------------------------------------------------------------------
# Part e) Gradient descent with a fixed learning rate
# ---------------------------------------------------------------------------

def part_e() -> None:
    """
    Part e) Plain gradient descent for OLS and Ridge; analytic and JAX gradients.

    LLM-assisted: Claude, see module docstring.
    """
    FIGURE_DIR.mkdir(exist_ok=True)
    log = Log()
    log("=" * 72)
    log("Part e) Gradient descent with a fixed learning rate")
    log("=" * 72)
    prob = gd_problem()
    X, y = prob["X"], prob["y"]
    p = X.shape[1]
    log(f"Degree {DEGREE_GD}, seed {SEED}, {X.shape[0]} training points, Ridge lambda = {LAMBDA_GD}")

    # Gradient check
    theta_random = np.random.default_rng(0).normal(size=p)
    log("\nAnalytic gradient vs. jax.grad, max |difference|")
    for name, lam in [("OLS", 0.0), ("Ridge", LAMBDA_GD)]:
        for label, theta in [("random theta", theta_random), ("theta = 0", np.zeros(p))]:
            diff = np.max(np.abs(gradient_analytic(theta, X, y, lam) - gradient_jax(theta, X, y, lam)))
            log(f"  {name:>5}, {label:<12}: {diff:.1e}")

    # Cost of AD relative to one cost evaluation, on a larger synthetic problem
    # (at our size, 80 x 6, the timing is dominated by call overhead)
    rng_t = np.random.default_rng(1)
    Xj = jnp.asarray(rng_t.normal(size=(100_000, 50)))
    yj = jnp.asarray(rng_t.normal(size=100_000))
    tj = jnp.asarray(rng_t.normal(size=50))
    cost_jit = jax.jit(cost)
    grad_jit = jax.jit(jax.grad(cost))
    cost_jit(tj, Xj, yj).block_until_ready()
    grad_jit(tj, Xj, yj).block_until_ready()
    reps, blocks = 100, 5
    t_cost, t_grad = np.inf, np.inf
    for _ in range(blocks):   # best of 5 blocks for each, alternating, to reduce timing noise
        start = time.perf_counter()
        for _ in range(reps):
            cost_jit(tj, Xj, yj).block_until_ready()
        t_cost = min(t_cost, (time.perf_counter() - start) / reps)
        start = time.perf_counter()
        for _ in range(reps):
            grad_jit(tj, Xj, yj).block_until_ready()
        t_grad = min(t_grad, (time.perf_counter() - start) / reps)
    log(f"\nTiming on a 100000 x 50 problem (jit, best of {blocks} blocks of {reps} calls): cost {t_cost * 1e3:.2f} ms, "
        f"gradient (reverse mode) {t_grad * 1e3:.2f} ms, ratio {t_grad / t_cost:.2f}")

    results = {}
    for name, lam in [("OLS", 0.0), ("Ridge", LAMBDA_GD)]:
        theta_hat = ols_fit(X, y) if lam == 0 else ridge_fit(X, y, lam)
        eig = np.linalg.eigvalsh(hessian(X, lam))
        gamma_max, gamma_opt = 2 / eig.max(), 2 / (eig.max() + eig.min())
        tol = 1e-6
        d0 = float(np.linalg.norm(theta_hat))
        log(f"\n{name}: Hessian eigenvalues {eig.min():.3e} ... {eig.max():.3f}, kappa = {eig.max() / eig.min():.3e}")
        log(f"  gamma_max = 2/mu_max = {gamma_max:.4f}, gamma* = 2/(mu_max + mu_min) = {gamma_opt:.4f}")
        log(f"  stop when ||theta - theta_hat|| < {tol:g}; start theta = 0, ||theta_hat|| = {d0:.3f}")
        log(f"  {'gamma/gamma_max':>15} {'iterations':>11} {'predicted':>10} {'status':>10}")
        fractions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99, gamma_opt / gamma_max, 1.01, 1.05]
        scan = []
        for frac in fractions:
            gamma = frac * gamma_max
            run = gradient_descent(lambda t: gradient_analytic(t, X, y, lam), np.zeros(p), "plain", gamma, 400_000,
                                   converged=lambda t: np.linalg.norm(t - theta_hat) < tol)
            predicted = iteration_prediction(eig, gamma, d0, tol)
            status = "diverged" if run["diverged"] else ("converged" if run["converged"] else "max iter")
            scan.append((frac, run["iterations"], predicted, status))
            label = f"{frac:.3f}" + (" (gamma*)" if abs(frac - gamma_opt / gamma_max) < 1e-12 else "")
            log(f"  {label:>15} {run['iterations']:>11} {predicted:>10.0f} {status:>10}")
        results[name] = {"scan": scan, "eig": eig, "theta_hat": theta_hat, "gamma_max": gamma_max, "lam": lam}

        # Converged solution vs closed form, and analytic vs JAX gradient in GD
        gamma = 0.99 * gamma_max
        run_a = gradient_descent(lambda t: gradient_analytic(t, X, y, lam), np.zeros(p), "plain", gamma, 400_000,
                                 converged=lambda t: np.linalg.norm(t - theta_hat) < tol)
        test_gd = mse(prob["y_test"], prob["scaler"].predict(prob["X_test"], run_a["theta"]))
        test_cf = mse(prob["y_test"], prob["scaler"].predict(prob["X_test"], theta_hat))
        log(f"  at 0.99 gamma_max: test MSE GD {test_gd:.6e}, closed form {test_cf:.6e}")
        run_j = gradient_descent(lambda t: gradient_jax(t, X, y, lam), np.zeros(p), "plain", gamma, 2000)
        run_a2 = gradient_descent(lambda t: gradient_analytic(t, X, y, lam), np.zeros(p), "plain", gamma, 2000)
        log(f"  2000 GD iterations with JAX vs analytic gradient: max |theta difference| = "
            f"{np.max(np.abs(run_j['theta'] - run_a2['theta'])):.1e}")

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9))
    for colour, name in zip(COLOURS, results):
        scan = results[name]["scan"]
        ok = [(f, it) for f, it, _, s in scan if s == "converged"]
        ax1.semilogy([f for f, _ in ok], [it for _, it in ok], "o", color=colour, label=f"{name}, measured")
        grid = np.linspace(0.04, 0.995, 300)
        pred = [iteration_prediction(results[name]["eig"], g * results[name]["gamma_max"],
                                     np.linalg.norm(results[name]["theta_hat"]), 1e-6) for g in grid]
        ax1.semilogy(grid, pred, "-", color=colour, lw=1, label=f"{name}, predicted")
    ax1.axvline(1.0, color=GREY, ls=":", lw=1)
    ax1.set_xlabel(r"$\gamma / \gamma_{\max}$,  $\gamma_{\max} = 2/\mu_{\max}$")
    ax1.set_ylabel(r"Iterations to $\|\theta - \hat\theta\| < 10^{-6}$")
    ax1.legend(fontsize=8)
    for colour, (name, frac) in zip(COLOURS, [("OLS", 0.5), ("OLS", 0.99), ("OLS", 1.01), ("Ridge", 0.99)]):
        r_ = results[name]
        lam = r_["lam"]
        run = gradient_descent(lambda t: gradient_analytic(t, X, y, lam), np.zeros(p), "plain",
                               frac * r_["gamma_max"], 20000,
                               record=lambda t: np.linalg.norm(t - r_["theta_hat"]))
        ax2.semilogy(np.arange(1, len(run["history"]) + 1), run["history"], color=colour,
                     label=rf"{name}, $\gamma = {frac}\,\gamma_{{\max}}$")
    ax2.set_ylim(1e-7, 1e3)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$\|\theta - \hat\theta\|$")
    ax2.legend(fontsize=8)
    save_figure(fig, "e_gd")
    log(f"\nFigures saved in {FIGURE_DIR}/")
    log.save("part_e.txt")


# ---------------------------------------------------------------------------
# Part f) Momentum and adaptive learning rates
# ---------------------------------------------------------------------------

GAMMAS_F = np.logspace(-4, 0, 17)
MAX_ITER_F = 100_000
TOL_F = 1e-8


def part_f() -> None:
    """
    Part f) Plain GD, momentum, AdaGrad, RMSProp and Adam for OLS and Ridge:
    iterations to reach an excess cost below TOL_F, scanned over the
    (initial) learning rate.

    LLM-assisted: Claude, see module docstring.
    """
    FIGURE_DIR.mkdir(exist_ok=True)
    log = Log()
    log("=" * 72)
    log("Part f) Momentum, AdaGrad, RMSProp and Adam")
    log("=" * 72)
    prob = gd_problem()
    X, y = prob["X"], prob["y"]
    p = X.shape[1]
    np_cost = lambda t, lam: float(np.mean((y - X @ t) ** 2) + lam * np.sum(t**2))   # same as cost(), in NumPy (faster here)
    log(f"Degree {DEGREE_GD}, seed {SEED}, start theta = 0. Criterion: C(theta) - C(theta_hat) < {TOL_F:g}, "
        f"at most {MAX_ITER_F} iterations.")
    log("Parameters: momentum beta = 0.9, RMSProp rho = 0.99, Adam beta1 = 0.9, beta2 = 0.999, eps = 1e-8")

    scans, curves = {}, {}
    for name, lam in [("OLS", 0.0), ("Ridge", LAMBDA_GD)]:
        theta_hat = ols_fit(X, y) if lam == 0 else ridge_fit(X, y, lam)
        c_hat = float(cost(theta_hat, X, y, lam))
        scans[name] = {}
        log(f"\n{name}: iterations (- = not reached, div = diverged)")
        log(f"{'gamma':>9} " + " ".join(f"{m:>9}" for m in METHODS))
        table = {m: [] for m in METHODS}
        for gamma in GAMMAS_F:
            row = []
            for m in METHODS:
                run = gradient_descent(lambda t: gradient_analytic(t, X, y, lam), np.zeros(p), m, gamma, MAX_ITER_F,
                                       converged=lambda t: np_cost(t, lam) - c_hat < TOL_F)
                value = run["iterations"] if run["converged"] else (-1 if run["diverged"] else 0)
                table[m].append(value)
                row.append("div" if value == -1 else ("-" if value == 0 else str(value)))
            log(f"{gamma:>9.2e} " + " ".join(f"{v:>9}" for v in row))
        scans[name] = table
        log(f"  best learning rate per method ({name}):")
        best = {}
        for m in METHODS:
            ok = [(it, g) for it, g in zip(table[m], GAMMAS_F) if it > 0]
            if ok:
                it, g = min(ok)
                usable = [g for it_, g in zip(table[m], GAMMAS_F) if it_ > 0]
                best[m] = g
                log(f"    {m:>9}: {it:>6} iterations at gamma = {g:.2e}; converges for gamma in "
                    f"[{min(usable):.1e}, {max(usable):.1e}] ({len(usable)} of {len(GAMMAS_F)} values)")
            else:
                log(f"    {m:>9}: never reached the criterion")
        if name == "OLS":
            for m in METHODS:
                g = best.get(m, 1e-3)   # RMSProp never converges; shown at gamma = 1e-3
                run = gradient_descent(lambda t: gradient_analytic(t, X, y, lam), np.zeros(p), m, g,
                                       MAX_ITER_F, record=lambda t: np_cost(t, lam) - c_hat)
                curves[m] = (g, run["history"])

    theta_hat = ols_fit(X, y)
    c_hat = float(cost(theta_hat, X, y))
    log("\nRMSProp, OLS: excess cost and step over the last 1000 iterations")
    for gamma in (1e-4, 1e-3, 1e-2):
        thetas = []
        run = gradient_descent(lambda t: gradient_analytic(t, X, y), np.zeros(p), "rmsprop", gamma, MAX_ITER_F,
                               record=lambda t: (thetas.append(t), np_cost(t, 0.0) - c_hat)[1])
        tail = np.array(run["history"][-1000:])
        steps = np.abs(np.diff(np.array(thetas[-1001:]), axis=0))
        dist = np.abs(np.array(thetas[-1000:]) - theta_hat)
        log(f"  gamma = {gamma:.0e}: excess cost min {tail.min():.2e}, max {tail.max():.2e}; "
            f"|step| per coefficient min {steps.min():.2e}, max {steps.max():.2e}; "
            f"|theta_j - theta_hat_j| min {dist.min():.2e}, max {dist.max():.2e}")
    g_adam, hist = curves["adam"]
    hist = np.array(hist)
    log(f"\nAdam, OLS, gamma = {g_adam:.2e}: largest excess cost in windows of 1000 iterations")
    log("  " + ", ".join(f"{k}-{k + 1000}: {hist[k:k + 1000].max():.1e}" for k in (1000, 2000, 5000, 10000, MAX_ITER_F - 1000)))
    # Near the minimum Adam is momentum GD with learning rate gamma / sqrt(r_hat_j) per coefficient. Track the
    # largest eigenvalue of gamma D H (D = diag of 1 / sqrt(r_hat)) against the stability limit of that iteration,
    # 2 (1 + beta1) / (1 - beta1), found as for plain GD from the multiplier per step along each eigenvector.
    beta1 = 0.9
    limit = 2 * (1 + beta1) / (1 - beta1)
    H = hessian(X)
    theta, state = np.zeros(p), {}
    crossed = burst = None
    reached = int(np.argmax(hist < TOL_F)) + 1
    for t in range(1, 3001):
        theta = optimiser_step("adam", theta, gradient_analytic(theta, X, y), state, t, g_adam)
        r_hat = state["r"] / (1 - 0.999**t)
        top = np.max(np.real(np.linalg.eigvals(g_adam * np.diag(1 / (np.sqrt(r_hat) + 1e-8)) @ H)))
        if t > reached and crossed is None and top > limit:
            crossed = t
        if t > reached and burst is None and hist[t - 1] > 1e-4:
            burst = t
            break
    run = gradient_descent(lambda t: gradient_analytic(t, X, y), np.zeros(p), "adam", 1e-3, MAX_ITER_F,
                           record=lambda t: np_cost(t, 0.0) - c_hat)
    log(f"  Adam with gamma = 1e-3: largest excess cost over the last 1000 iterations {max(run['history'][-1000:]):.1e}")
    log(f"  criterion met at iteration {reached}; effective step passes the stability limit "
        f"2(1+beta1)/(1-beta1) = {limit:.0f} at iteration {crossed}; first excess cost > 1e-4 at iteration {burst}; "
        f"excess cost at iteration {crossed}: {hist[crossed - 1]:.1e}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    for ax, name in zip(axes[:2], scans):
        for m in METHODS:
            its = np.array(scans[name][m], dtype=float)
            mask = its > 0
            ax.loglog(GAMMAS_F[mask], its[mask], "o-", color=METHOD_COLOURS[m], label=METHOD_LABELS[m], ms=4)
        ax.set_ylim(10, MAX_ITER_F * 1.5)
        ax.axhline(MAX_ITER_F, color=GREY, ls=":", lw=1)
        ax.set_xlabel(r"(Initial) learning rate $\gamma$")
        ax.set_ylabel(f"Iterations to excess cost $< 10^{{{int(np.log10(TOL_F))}}}$")
        ax.set_title(name if name == "OLS" else rf"Ridge, $\lambda = {LAMBDA_GD}$", fontsize=10)
    axes[0].legend(fontsize=8)
    ax = axes[2]
    for m in ["adam", "plain", "momentum", "adagrad", "rmsprop"]:   # Adam first, so its oscillations lie underneath
        g, hist = curves[m]
        ax.loglog(np.arange(1, len(hist) + 1), np.maximum(hist, 1e-16), color=METHOD_COLOURS[m],
                  lw=0.6 if m == "adam" else 1.5, label=rf"{METHOD_LABELS[m].split(' (')[0]}, $\gamma = {g:.1e}$")
    ax.axhline(TOL_F, color=GREY, ls=":", lw=1)
    ax.set_ylim(1e-12, 1)
    ax.set_xlim(1, 3e4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$C(\theta) - C(\hat\theta)$, OLS")
    ax.set_title("OLS, best learning rate (RMSProp: $10^{-3}$)", fontsize=10)
    ax.legend(fontsize=7)
    save_figure(fig, "f_methods")
    log(f"\nFigures saved in {FIGURE_DIR}/")
    log.save("part_f.txt")


# ---------------------------------------------------------------------------
# Part g) Lasso by gradient descent
# ---------------------------------------------------------------------------

LAMBDAS_LASSO = [1e-4, 1e-3, 1e-2]


def part_g() -> None:
    """
    Part g) Lasso with our gradient methods (JAX subgradient), compared with
    scikit-learn, and a comparison of OLS, Ridge and Lasso.

    LLM-assisted: Claude, see module docstring.
    """
    FIGURE_DIR.mkdir(exist_ok=True)
    log = Log()
    log("=" * 72)
    log("Part g) Lasso by gradient descent")
    log("=" * 72)
    prob = gd_problem()
    X, y = prob["X"], prob["y"]
    n, p = X.shape
    log(f"jax.grad(jnp.abs) at 0.0: {float(jax.grad(jnp.abs)(0.0))}, at -0.0: {float(jax.grad(jnp.abs)(-0.0))}; "
        f"analytic sign(0) = {np.sign(0.0)}")
    log("Subdifferential of |t| at 0 is [-1, 1], so both 1 and 0 are valid subgradients.")
    gamma_plain = 0.99 * 2 / np.linalg.eigvalsh(hessian(X)).max()
    n_iter = 100_000
    log(f"Degree {DEGREE_GD}; plain GD gamma = {gamma_plain:.4f} (0.99 gamma_max of the data term) and Adam gamma = 1e-3, "
        f"{n_iter} iterations each, JAX gradient. scikit-learn Lasso with alpha = lambda/2.")

    rows = []
    gd_thetas = {}
    for lam in LAMBDAS_LASSO:
        theta_skl = lasso_fit_sklearn(X, y, lam)
        c_skl = float(cost(theta_skl, X, y, lam, "lasso"))
        grad = lambda t, lam=lam: gradient_jax(t, X, y, lam, "lasso")
        run_plain = gradient_descent(grad, np.zeros(p), "plain", gamma_plain, n_iter)
        run_adam = gradient_descent(grad, np.zeros(p), "adam", 1e-3, n_iter)
        gd_thetas[lam] = (run_plain["theta"], run_adam["theta"])
        residual = y - X @ theta_skl
        zero = theta_skl == 0
        kkt = np.abs((2 / n) * X.T @ residual)[zero]
        log(f"\nlambda = {lam:g}")
        log(f"  {'':>12} " + " ".join(f"theta_{j + 1:<6}" for j in range(p)) + f" {'cost - C_skl':>13}")
        for label, theta in [("scikit-learn", theta_skl), ("plain GD", run_plain["theta"]), ("Adam", run_adam["theta"])]:
            log(f"  {label:>12} " + " ".join(f"{v:>+8.4f}" for v in theta)
                + f" {float(cost(theta, X, y, lam, 'lasso')) - c_skl:>13.2e}")
        log(f"  zero coefficients in scikit-learn: {int(zero.sum())}; KKT |(2/n) x_j^T r| for these: "
            + (", ".join(f"{v:.2e}" for v in kkt) if zero.any() else "-") + f" (must be <= lambda = {lam:g})")
        log(f"  smallest |theta_j| from GD: plain {np.min(np.abs(run_plain['theta'])):.1e}, "
            f"Adam {np.min(np.abs(run_adam['theta'])):.1e}")
        for label, theta in [("scikit-learn", theta_skl), ("plain GD", run_plain["theta"]), ("Adam", run_adam["theta"])]:
            rows.append((lam, label, mse(prob["y_test"], prob["scaler"].predict(prob["X_test"], theta))))

    # Fixed-step subgradient descent does not settle: dependence on the learning rate
    lam = 1e-2
    theta_skl = lasso_fit_sklearn(X, y, lam)
    c_skl = float(cost(theta_skl, X, y, lam, "lasso"))
    eigval, eigvec = np.linalg.eigh(hessian(X))
    v_max = eigvec[:, -1]
    log(f"\nPlain subgradient GD for lambda = {lam:g} (analytic subgradient, sign(0) = 0), {n_iter} iterations.")
    log("  Steepest mode: multiplier 1 - gamma mu_max per step (mu: Hessian eigenvalues), and the component of theta - theta_skl along its "
        "eigenvector in the last two iterations.")
    for frac in (0.99, 0.9, 0.5, 0.1):
        gamma = frac * gamma_plain / 0.99
        thetas = []
        run = gradient_descent(lambda t: gradient_analytic(t, X, y, lam, "lasso"), np.zeros(p), "plain", gamma, n_iter,
                               record=lambda t: (thetas.append(t), float(cost(t, X, y, lam, "lasso")) - c_skl)[1])
        tail = np.array(run["history"][-1000:])
        modes = [float((t - theta_skl) @ v_max) for t in thetas[-2:]]
        flips = int(np.sum(np.sign(thetas[-1]) != np.sign(thetas[-2])))
        log(f"  gamma = {frac:.2f} gamma_max: excess cost over the last 1000 iterations "
            f"{tail.min():.1e} ... {tail.max():.1e}; theta_5, theta_6 = {run['theta'][4]:+.4f}, {run['theta'][5]:+.4f}; "
            f"multiplier {1 - gamma * eigval[-1]:+.2f}, steepest mode {modes[0]:+.4f}, {modes[1]:+.4f}; "
            f"coefficients changing sign in the last step: {flips}")

    # OLS / Ridge / Lasso comparison at this degree
    log(f"\nTest MSE at degree {DEGREE_GD} (seed {SEED}):")
    theta_ols = ols_fit(X, y)
    log(f"  OLS closed form: {mse(prob['y_test'], prob['scaler'].predict(prob['X_test'], theta_ols)):.4e}")
    for lam in LAMBDAS_LASSO:
        theta_r = ridge_fit(X, y, lam)
        log(f"  Ridge lambda = {lam:g}: {mse(prob['y_test'], prob['scaler'].predict(prob['X_test'], theta_r)):.4e}")
    for lam, label, value in rows:
        log(f"  Lasso lambda = {lam:g}, {label}: {value:.4e}")

    # Lasso path figure
    lambdas = np.logspace(-6, -0.5, 45)
    path = np.array([lasso_fit_sklearn(X, y, lam) for lam in lambdas])
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for j in range(p):
        colour = COLOURS[0] if (j + 1) % 2 == 0 else COLOURS[1]
        ax.semilogx(lambdas, path[:, j], "-", color=colour, lw=1.4,
                    label=("even power" if j == 1 else ("odd power" if j == 0 else None)))
    for lam in LAMBDAS_LASSO:
        theta_plain, theta_adam = gd_thetas[lam]
        first = lam == LAMBDAS_LASSO[0]
        ax.plot([lam] * p, theta_plain, "o", mfc="none", mec="#222222", ms=6,
                label=(r"plain GD, $0.99\,\gamma_{\max}$" if first else None))
        ax.plot([lam] * p, theta_adam, "x", color="#222222", ms=5, label=(r"Adam, $\gamma = 10^{-3}$" if first else None))
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\theta_j$ (standardised columns)")
    ax.legend(fontsize=8)
    save_figure(fig, "g_lasso_path")
    for lam in (1e-3, 1e-2, 1e-1):
        theta = lasso_fit_sklearn(X, y, lam)
        log(f"  path: lambda = {lam:g}: {int(np.sum(theta == 0))} of {p} coefficients exactly zero, "
            f"nonzero powers {[j + 1 for j in range(p) if theta[j] != 0]}")
    log(f"\nFigures saved in {FIGURE_DIR}/")
    log.save("part_g.txt")


# ---------------------------------------------------------------------------
# Part h) Stochastic gradient descent
# ---------------------------------------------------------------------------

def part_h() -> None:
    """
    Part h) SGD for OLS: dependence on batch size, epochs and schedule; the
    update rules with SGD compared with full-batch GD per pass through the data.

    LLM-assisted: Claude, see module docstring.
    """
    FIGURE_DIR.mkdir(exist_ok=True)
    log = Log()
    log("=" * 72)
    log("Part h) Stochastic gradient descent")
    log("=" * 72)
    prob = gd_problem()
    X, y = prob["X"], prob["y"]
    n, p = X.shape
    theta_hat = ols_fit(X, y)
    c_hat = float(cost(theta_hat, X, y))
    test_hat = mse(prob["y_test"], prob["scaler"].predict(prob["X_test"], theta_hat))
    excess = lambda t: float(cost(t, X, y)) - c_hat
    grad_batch = lambda Xb, yb, t: gradient_analytic(t, Xb, yb)
    n_epochs = 1000
    log(f"OLS, degree {DEGREE_GD}, {n} training points. Closed form: training MSE {c_hat:.4e}, test MSE {test_hat:.4e}")
    log(f"Largest per-sample curvature 2||x_i||^2 = {np.max(2 * np.sum(X**2, axis=1)):.1f} "
        f"(gamma < {2 / np.max(2 * np.sum(X**2, axis=1)):.4f} for stability with batch size 1)")

    log(f"\nPlain SGD, constant gamma = 0.05: excess training cost after 10 / 100 / 1000 epochs, test MSE after 1000")
    batch_runs = {}
    for M in [1, 5, 10, 20, n]:
        run = sgd(X, y, grad_batch, np.zeros(p), "plain", 0.05, M, n_epochs, SEED, record=excess)
        batch_runs[M] = run
        h = run["history"]
        test = mse(prob["y_test"], prob["scaler"].predict(prob["X_test"], run["theta"]))
        log(f"  M = {M:>3} ({run['updates'] // n_epochs:>3} updates/epoch): {h[9]:.2e} / {h[99]:.2e} / {h[-1]:.2e}, "
            f"test {test:.4e}")

    variants = {
        "plain, gamma = 0.05": ("plain", 0.05, None),
        "plain, schedule 50/(t+500)": ("plain", 0.0, (50.0, 500.0)),
        "momentum, gamma = 0.01": ("momentum", 0.01, None),
        "RMSProp, gamma = 1e-3": ("rmsprop", 1e-3, None),
        "Adam, gamma = 0.01": ("adam", 0.01, None),
    }
    M = 5
    log(f"\nBatch size M = {M}: excess training cost after 10 / 100 / 1000 epochs, test MSE after 1000")
    variant_runs = {}
    for label, (m, g, schedule) in variants.items():
        run = sgd(X, y, grad_batch, np.zeros(p), m, g, M, n_epochs, SEED, schedule=schedule, record=excess)
        variant_runs[label] = run
        h = run["history"]
        test = mse(prob["y_test"], prob["scaler"].predict(prob["X_test"], run["theta"]))
        log(f"  {label:<28}: {h[9]:.2e} / {h[99]:.2e} / {h[-1]:.2e}, test {test:.4e}")
    gamma_gd = 0.99 * 2 / np.linalg.eigvalsh(hessian(X)).max()
    full = {}
    for m, g in [("plain", gamma_gd), ("momentum", 0.1), ("adam", 0.05)]:
        run = gradient_descent(lambda t: gradient_analytic(t, X, y), np.zeros(p), m, g, n_epochs, record=excess)
        full[m] = (g, run["history"])
        log(f"  full-batch {m:<9} gamma = {g:.3g}: {run['history'][9]:.2e} / {run['history'][99]:.2e} / "
            f"{run['history'][-1]:.2e} after the same number of passes through the data")
    for target in (1e-3, 1e-4):
        log(f"\nPasses through the data (epochs) to reach excess cost < {target:g}:")
        for label, run in {**{f"SGD M={M_} plain 0.05": r for M_, r in batch_runs.items()},
                           **{f"SGD M=5 {k}": r for k, r in variant_runs.items()}}.items():
            hit = next((i + 1 for i, v in enumerate(run["history"]) if v < target), None)
            log(f"  {label:<40}: {hit if hit else '-'}")
        for m, (g, hist) in full.items():
            hit = next((i + 1 for i, v in enumerate(hist) if v < target), None)
            log(f"  {'full-batch ' + m:<40}: {hit if hit else '-'}")

    def rolling_median(values: list[float], window: int = 25) -> np.ndarray:
        values = np.asarray(values)
        return np.array([np.median(values[max(0, i - window + 1):i + 1]) for i in range(len(values))])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    epochs = np.arange(1, n_epochs + 1)
    ax1.set_ylim(1e-4, 1)
    for colour, (M_, run) in zip(COLOURS + ["#222222"], batch_runs.items()):
        ax1.loglog(epochs, rolling_median(run["history"]), color=colour, lw=1.6,
                   label=f"$M = {M_}$" + (" (full batch)" if M_ == n else ""))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(r"$C(\theta) - C(\hat\theta)$, training (running median, 25 epochs)")
    ax1.set_title(r"Plain SGD, $\gamma = 0.05$, batch size $M$", fontsize=10)
    ax1.legend(fontsize=8)
    shown_variants = {"plain, gamma = 0.05": COLOURS[0], "plain, schedule 50/(t+500)": COLOURS[2],
                      "Adam, gamma = 0.01": COLOURS[1]}
    for label, colour in shown_variants.items():
        nice = label.replace("gamma", "$\\gamma$")
        ax2.loglog(epochs, rolling_median(variant_runs[label]["history"]), color=colour, lw=1.6,
                   label=f"SGD $M=5$, {nice}")
    for m, colour in [("plain", "#222222"), ("adam", COLOURS[1])]:
        g, hist = full[m]
        ax2.loglog(epochs, hist, "--", color=colour, lw=1.4, label=f"full batch, {m}, $\\gamma = {g:.2g}$")
    ax2.set_ylim(1e-10, 1)
    ax2.set_xlabel("Epoch (passes through the training data)")
    ax2.set_title("SGD with $M = 5$ vs. full-batch GD", fontsize=10)
    ax2.legend(fontsize=7.5)
    save_figure(fig, "h_sgd")
    log(f"\nFigures saved in {FIGURE_DIR}/")
    log.save("part_h.txt")


# ---------------------------------------------------------------------------
# Part i) Final model selection with cross-validation
# ---------------------------------------------------------------------------

LAMBDAS_I = {"Ridge": np.logspace(-8, 0, 9), "Lasso": np.logspace(-6, -1, 6)}
LASSO_CONVERGED_MIN = 1e-3   # smallest Lasso lambda where scikit-learn's coordinate descent converges at high degree


def part_i() -> None:
    """
    Part i) OLS, Ridge and Lasso selected by 5-fold CV on the training part of
    each data set, refitted on the whole training part and tested once.
    Bootstrap bias-variance at a high degree for the three methods.

    LLM-assisted: Claude, see module docstring.
    """
    FIGURE_DIR.mkdir(exist_ok=True)
    log = Log()
    log("=" * 72)
    log("Part i) Final model selection with cross-validation")
    log("=" * 72)
    n, degrees, k = N_MAIN, DEGREES_C, 5
    log(f"n = {n}, sigma = {SIGMA_MAIN}, {len(SEEDS_RESAMPLING)} data sets. For each: 80/20 split; {k}-fold CV on the "
        f"80 training points over degree {degrees[0]}-{degrees[-1]} (and lambda); refit the selected model on all 80; "
        f"test once on the 20; MSE on new data from the known f.")
    log("Ridge lambdas: " + ", ".join(f"{l:.0e}" for l in LAMBDAS_I["Ridge"])
        + "; Lasso lambdas: " + ", ".join(f"{l:.0e}" for l in LAMBDAS_I["Lasso"]) + " (Lasso by scikit-learn)")

    methods = {"OLS": [0.0], "Ridge": list(LAMBDAS_I["Ridge"]), "Lasso": list(LAMBDAS_I["Lasso"])}
    restricted = "Lasso*"   # Lasso with lambda >= LASSO_CONVERGED_MIN, chosen from the same CV grid
    log(f"{restricted}: Lasso with lambda >= {LASSO_CONVERGED_MIN:g} only, chosen from the same CV grid")
    labels = list(methods) + [restricted]
    cv_curves = {m: np.empty((len(SEEDS_RESAMPLING), len(degrees))) for m in labels}
    chosen = {m: [] for m in labels}
    lasso_status = {"Lasso": [], restricted: []}
    x_grid = np.linspace(-1, 1, 2001)
    for r, seed in enumerate(SEEDS_RESAMPLING):
        x_train, x_test, y_train, y_test = split_data(n, SIGMA_MAIN, seed)
        for m, lams in methods.items():
            penalty = "lasso" if m == "Lasso" else "ridge"
            grid = np.empty((len(degrees), len(lams)))
            for i, degree in enumerate(degrees):
                for j, lam in enumerate(lams):
                    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
                    pipe = make_pipeline(FunctionTransformer(lambda x, d=degree: design_matrix(x.ravel(), d)),
                                         StandardScaler(), PolynomialRegression(lam, penalty))
                    grid[i, j] = -np.mean(cross_val_score(pipe, x_train.reshape(-1, 1), y_train, cv=kfold,
                                                          scoring="neg_mean_squared_error"))
            variants = [(m, grid, lams)]
            if m == "Lasso":
                keep = np.array(lams) >= LASSO_CONVERGED_MIN
                variants.append((restricted, grid[:, keep], list(np.array(lams)[keep])))
            for label, grid_v, lams_v in variants:
                cv_curves[label][r] = grid_v.min(axis=1)
                i, j = np.unravel_index(np.argmin(grid_v), grid_v.shape)
                degree, lam = degrees[i], lams_v[j]
                pipe = make_pipeline(FunctionTransformer(lambda x, d=degree: design_matrix(x.ravel(), d)),
                                     StandardScaler(), PolynomialRegression(lam, penalty))
                pipe.fit(x_train.reshape(-1, 1), y_train)
                test = mse(y_test, pipe.predict(x_test.reshape(-1, 1)))
                new = float(np.mean((pipe.predict(x_grid.reshape(-1, 1)) - runge(x_grid)) ** 2) + SIGMA_MAIN**2)
                nonzero = int(np.sum(pipe[-1].coef_ != 0))
                chosen[label].append((degree, lam, grid_v[i, j], test, new, nonzero))
                if penalty == "lasso":
                    X_s = pipe[:2].transform(x_train.reshape(-1, 1))
                    ok, gap = lasso_convergence(X_s, y_train - y_train.mean(), lam)
                    lasso_status[label].append((ok, gap, float(cost(pipe[-1].coef_, X_s, y_train - y_train.mean(),
                                                                    lam, "lasso"))))

    log(f"\nSelected models (median over data sets; [25th, 75th percentile])")
    log(f"{'method':>6} {'degree':>14} {'lambda':>9} {'CV MSE':>10} {'test MSE':>10} {'new data':>10} {'nonzero':>8}")
    for m, rows in chosen.items():
        arr = np.array(rows, dtype=float)
        q = lambda c: np.percentile(arr[:, c], [25, 50, 75])
        d = q(0)
        log(f"{m:>6} {d[1]:>5g} [{d[0]:g}, {d[2]:g}]  {np.median(arr[:, 1]):>9.1e} {np.median(arr[:, 2]):>10.4e} "
            f"{np.median(arr[:, 3]):>10.4e} {np.median(arr[:, 4]):>10.4e} {np.median(arr[:, 5]):>8g}")
    for m, rows in chosen.items():
        arr = np.array(rows, dtype=float)
        new_q = np.percentile(arr[:, 4], [25, 75])
        zeros = int(np.sum(arr[:, 5] < arr[:, 0]))
        log(f"  {m:>6}: MSE on new data, 25th-75th percentile over data sets {new_q[0]:.4e} - {new_q[1]:.4e}; "
            f"largest {arr[:, 4].max():.3e}; selected models with at least one zero coefficient: {zeros} of {len(arr)}")
    for m in labels:
        degs, lams = [row[0] for row in chosen[m]], [row[1] for row in chosen[m]]
        log(f"  {m:>6}: selected degrees {sorted(degs)}"
            + ("" if m == "OLS" else "; lambdas " + ", ".join(f"{l:.0e} x{lams.count(l)}" for l in sorted(set(lams)))))
    for label, rows in lasso_status.items():
        status = np.array(rows)
        log(f"  Selected {label} fits (scikit-learn, at most 100000 iterations): {int(np.sum(status[:, 0] == 0))} of "
            f"{len(status)} stopped at the iteration limit; duality gap (our cost units) median "
            f"{np.median(status[:, 1]):.1e}, max {np.max(status[:, 1]):.1e}, against a median cost of "
            f"{np.median(status[:, 2]):.1e}")
    wins = np.argmin(np.array([[row[4] for row in chosen[m]] for m in methods]), axis=0)
    log("  best on new data per data set (OLS, Ridge, Lasso): "
        + ", ".join(f"{m} {int(np.sum(wins == i))}" for i, m in enumerate(methods)))
    log("  Median CV MSE (best lambda per degree) against degree " + ", ".join(str(d) for d in degrees))
    for m in labels:
        med = np.median(cv_curves[m], axis=0)
        log(f"  {m:>6}: " + " ".join(f"{v:.4f}" for v in med))

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    names = {"OLS": "OLS", "Ridge": r"Ridge, best $\lambda$", "Lasso": r"Lasso, best $\lambda$",
             restricted: rf"Lasso, best $\lambda \geq 10^{{{int(np.log10(LASSO_CONVERGED_MIN))}}}$"}
    for colour, m in zip(COLOURS, labels):
        plot_median_band(ax, degrees, cv_curves[m], colour, names[m], band_alpha=0.12)
    ax.axhline(SIGMA_MAIN**2, color=GREY, ls=":", lw=1, label=r"$\sigma^2$")
    ax.set_yscale("log")
    ax.set_ylim(8e-3, 0.2)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(f"{k}-fold CV MSE")
    ax.legend(fontsize=8)
    save_figure(fig, "i_cv_methods")

    # Bootstrap bias-variance at a high degree for the three methods
    degree = 14
    lam_r = float(np.median([row[1] for row in chosen["Ridge"]]))
    lam_l = float(np.median([row[1] for row in chosen["Lasso"]]))
    lam_c = float(np.median([row[1] for row in chosen[restricted]]))
    fits = {"OLS": fit_ols,
            f"Ridge (lambda={lam_r:.1e})": lambda x_, y_, d: fit_ridge(x_, y_, d, lam_r),
            f"Lasso (lambda={lam_l:.1e})": lambda x_, y_, d: fit_lasso(x_, y_, d, lam_l),
            f"Lasso (lambda={lam_c:.1e})": lambda x_, y_, d: fit_lasso(x_, y_, d, lam_c)}
    log(f"\nBootstrap at degree {degree} ({N_BOOT} resamples; median over the {len(SEEDS_RESAMPLING)} data sets)")
    log(f"{'method':>24} {'error':>10} {'bias^2':>10} {'bias^2 (f)':>11} {'variance':>10}")
    for label, fit in fits.items():
        values = []
        for seed in SEEDS_RESAMPLING:
            x_train, x_test, y_train, y_test = split_data(n, SIGMA_MAIN, seed)
            out = bootstrap_bias_variance(x_train, y_train, x_test, y_test, degree, N_BOOT,
                                          np.random.default_rng(seed), fit=fit)
            values.append((out["error"], out["bias2"], np.mean((runge(x_test) - out["mean_prediction"]) ** 2),
                           out["variance"]))
        med = np.median(np.array(values), axis=0)
        log(f"{label:>24} {med[0]:>10.3e} {med[1]:>10.3e} {med[2]:>11.3e} {med[3]:>10.3e}")
    for lam in (lam_l, lam_c):
        gaps = []
        for seed in SEEDS_RESAMPLING:
            x_train, _, y_train, _ = split_data(n, SIGMA_MAIN, seed)
            X_train = design_matrix(x_train, degree)
            scaler = Scaler().fit(X_train, y_train)
            gaps.append(lasso_convergence(scaler.transform_X(X_train), scaler.transform_y(y_train), lam))
        log(f"  Lasso at degree {degree}, lambda = {lam:.1e}, on the full training sets: "
            f"{sum(not ok for ok, _ in gaps)} of {len(gaps)} stopped at the iteration limit; "
            f"median duality gap {np.median([g for _, g in gaps]):.1e} (our cost units)")
    log(f"\nFigures saved in {FIGURE_DIR}/")
    log.save("part_i.txt")


def fit_lasso(x_train: np.ndarray, y_train: np.ndarray, degree: int, lam: float) -> tuple[Scaler, np.ndarray]:
    """
    Scaled Lasso fit (scikit-learn coordinate descent), same preprocessing as fit_ols.

    LLM-assisted: Claude, see module docstring.
    """
    X_train = design_matrix(x_train, degree)
    scaler = Scaler().fit(X_train, y_train)
    return scaler, lasso_fit_sklearn(scaler.transform_X(X_train), scaler.transform_y(y_train), lam)


def lasso_convergence(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[bool, float]:
    """
    Refit scikit-learn's Lasso as in lasso_fit_sklearn and report whether it
    stopped before the iteration limit, and the duality gap converted to our
    cost (scikit-learn's objective is half of ours, so the gap is doubled).
    The gap bounds how far the cost can be above the true Lasso minimum.

    LLM-assisted: Claude, see module docstring.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Lasso
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = Lasso(alpha=lam / 2, fit_intercept=False, max_iter=100_000, tol=1e-8).fit(X, y)
    return bool(model.n_iter_ < model.max_iter), 2 * float(model.dual_gap_)


PARTS = {"a": part_a, "b": part_b, "c": part_c, "d": part_d, "e": part_e, "f": part_f, "g": part_g,
         "h": part_h, "i": part_i}


SLOW_PARTS = {"a": "about 3.5 min", "f": "about 10 min", "g": "about 5 min", "i": "about 9 min"}


def main() -> None:
    """
    Run the parts named on the command line, all parts with "all", or, without
    arguments, all parts except the slow ones in SLOW_PARTS.

    LLM-assisted: Claude, see module docstring.
    """
    args = [a.lower() for a in sys.argv[1:]]
    if args == ["all"]:
        selected = list(PARTS)
    elif args:
        unknown = [a for a in args if a not in PARTS]
        if unknown:
            sys.exit(f"Unknown part(s): {', '.join(unknown)}. Choose from {', '.join(PARTS)} or 'all'.")
        selected = args
    else:
        selected = [p for p in PARTS if p not in SLOW_PARTS]
        print("Running the fast parts " + ", ".join(selected) + ". Skipping the slow parts "
              + ", ".join(f"{p} ({t})" for p, t in SLOW_PARTS.items())
              + "; run them with, for example, 'uv run main.py a f', or everything with 'uv run main.py all'.")
    for part in selected:
        PARTS[part]()


if __name__ == "__main__":
    main()
