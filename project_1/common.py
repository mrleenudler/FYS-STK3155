"""
Shared functions for FYS-STK3155 Project 1: data generation, splitting,
design matrix, scaling, regression solvers (OLS, Ridge, and Lasso through
scikit-learn), error measures, the bootstrap, and a scikit-learn wrapper for
our own estimator.

All functions follow the course convention with 1/n on the data term of the
cost function, C(theta) = (1/n) ||y - X theta||^2.

LLM-assisted
------------
Tool: Claude (Opus 5.5, claude.ai, September 2026)
Role: Wrote the module (level 4) after the choices of data generation,
scaling, centering and intercept handling had been discussed and agreed on
with the author. Every function below carries a short LLM-assisted tag that
refers to this note.
Verification: tests.py (OLS against scikit-learn, exact recovery of a
noise-free polynomial, invariance of OLS predictions under scaling, recovery
of the intercept).
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2


def runge(x: np.ndarray) -> np.ndarray:
    """
    Runge's function f(x) = 1 / (1 + 25 x^2).

    LLM-assisted: Claude, see module docstring.
    """
    return 1.0 / (1.0 + 25.0 * x**2)


def make_data(n: int, noise_std: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw n points x ~ U[-1, 1] and return (x, y) with y = f(x) + eps,
    eps ~ N(0, noise_std^2).

    LLM-assisted: Claude, see module docstring.
    """
    x = rng.uniform(-1.0, 1.0, n)
    y = runge(x) + rng.normal(0.0, noise_std, n)
    return x, y


def split_data(n: int, noise_std: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one data set and split it once into training and test data.
    The seed fixes both the data (x values and noise) and the split, so a
    given seed always gives the same experiment.
    Returns x_train, x_test, y_train, y_test.

    LLM-assisted: Claude, see module docstring.
    """
    rng = np.random.default_rng(seed)
    x, y = make_data(n, noise_std, rng)
    return train_test_split(x, y, test_size=TEST_SIZE, random_state=seed)


def design_matrix(x: np.ndarray, degree: int, intercept: bool = False) -> np.ndarray:
    """
    Polynomial design matrix with columns x^1, ..., x^degree.
    With intercept=True a leading column of ones (x^0) is added.

    LLM-assisted: Claude, see module docstring.
    """
    first_power = 0 if intercept else 1
    return np.column_stack([x**k for k in range(first_power, degree + 1)])


class Scaler:
    """
    Standardises the feature columns and centres y, using statistics from
    the training data only. The same statistics are then applied to test data.

    LLM-assisted: Claude, see module docstring.
    """

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "Scaler":
        self.x_mean = X_train.mean(axis=0)
        self.x_std = X_train.std(axis=0)
        self.y_mean = y_train.mean()
        return self

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / self.x_std

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return y - self.y_mean

    def predict(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Prediction in original units: y_mean + X_scaled @ theta."""
        return self.y_mean + self.transform_X(X) @ theta

    def original_coefficients(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Convert coefficients fitted on standardised columns back to the
        original polynomial in x: returns (theta_0, theta_1..theta_d), with
        theta_j = theta_scaled_j / x_std_j and
        theta_0 = y_mean - sum_j x_mean_j * theta_j.
        """
        theta_orig = theta / self.x_std
        theta_0 = self.y_mean - self.x_mean @ theta_orig
        return float(theta_0), theta_orig


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    OLS parameters theta = pinv(X) y. The pseudoinverse of X is computed by
    SVD directly, which avoids squaring the condition number as forming
    X^T X would.

    LLM-assisted: Claude, see module docstring.
    """
    return np.linalg.pinv(X) @ y


def fit_ols(x_train: np.ndarray, y_train: np.ndarray, degree: int) -> tuple[Scaler, np.ndarray]:
    """
    Scaled OLS fit of a polynomial of the given degree: builds the design
    matrix without intercept, standardises it and centres y with training
    statistics, and solves for theta. Returns the fitted scaler and theta
    (coefficients on the standardised columns).

    LLM-assisted: Claude, see module docstring.
    """
    X_train = design_matrix(x_train, degree)
    scaler = Scaler().fit(X_train, y_train)
    theta = ols_fit(scaler.transform_X(X_train), scaler.transform_y(y_train))
    return scaler, theta


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Ridge parameters for the cost C = (1/n)||y - X theta||^2 + lam ||theta||^2,
    i.e. the solution of (X^T X + n lam I) theta = X^T y. Solved as a linear
    system, not by forming the inverse. X is assumed standardised and y
    centred, so no intercept is penalised.

    LLM-assisted: Claude, see module docstring.
    """
    n, p = X.shape
    return np.linalg.solve(X.T @ X + n * lam * np.eye(p), X.T @ y)


def ridge_fit_svd(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    The same Ridge solution written with the SVD X = U Sigma V^T:
    theta = sum_i sigma_i / (sigma_i^2 + n lam) (u_i^T y) v_i.
    With lam = 0 this is the OLS solution sum_i (u_i^T y) / sigma_i v_i.
    Used to check ridge_fit and to show the shrinkage per direction.

    LLM-assisted: Claude, see module docstring.
    """
    n = X.shape[0]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ (s / (s**2 + n * lam) * (U.T @ y))


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, degree: int, lam: float) -> tuple[Scaler, np.ndarray]:
    """
    Scaled Ridge fit of a polynomial of the given degree, with the same
    preprocessing as fit_ols. lam = 0 falls back to OLS.

    LLM-assisted: Claude, see module docstring.
    """
    if lam == 0:
        return fit_ols(x_train, y_train, degree)
    X_train = design_matrix(x_train, degree)
    scaler = Scaler().fit(X_train, y_train)
    theta = ridge_fit(scaler.transform_X(X_train), scaler.transform_y(y_train), lam)
    return scaler, theta


def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean squared error.

    LLM-assisted: Claude, see module docstring.
    """
    return float(np.mean((y - y_pred) ** 2))


def r2(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination R^2.

    LLM-assisted: Claude, see module docstring.
    """
    return float(1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))


def bootstrap_bias_variance(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                            y_test: np.ndarray, degree: int, n_boot: int,
                            rng: np.random.Generator, fit=None) -> dict[str, float]:
    """
    Bootstrap estimate of test error, squared bias and variance for scaled OLS
    (course notes, Sec. 2.10). The test set is fixed; the training set is
    resampled with replacement n_boot times, and the scaling is refitted on
    every bootstrap sample. With predictions p (n_test x n_boot):
        error    = mean over test points and resamples of (y_test - p)^2
        bias^2   = mean over test points of (y_test - mean_b p)^2
        variance = mean over test points of var_b(p)
    so that error = bias^2 + variance exactly. Since y_test contains noise,
    bias^2 here includes sigma^2. fit(x, y, degree) -> (scaler, theta)
    defaults to scaled OLS (fit_ols).

    LLM-assisted: Claude, see module docstring.
    """
    n = len(x_train)
    X_test = design_matrix(x_test, degree)
    predictions = np.empty((len(x_test), n_boot))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        scaler, theta = (fit or fit_ols)(x_train[idx], y_train[idx], degree)
        predictions[:, b] = scaler.predict(X_test, theta)
    y_col = y_test[:, None]
    mean_pred = np.mean(predictions, axis=1, keepdims=True)
    return {
        "error": float(np.mean((y_col - predictions) ** 2)),
        "bias2": float(np.mean((y_col - mean_pred) ** 2)),
        "variance": float(np.mean(np.var(predictions, axis=1))),
        "mean_prediction": mean_pred.ravel(),
    }


def lasso_fit_sklearn(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Lasso with scikit-learn's coordinate descent, translated to our cost
    (1/n)||y - X theta||^2 + lam ||theta||_1. scikit-learn minimises
    (1/(2n))||y - X w||^2 + alpha ||w||_1, so alpha = lam / 2.
    The ConvergenceWarning is silenced because the cross-validation grid in
    part i produces many; convergence of the selected models is checked
    separately with lasso_convergence() in main.py.

    LLM-assisted: Claude, see module docstring.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Lasso
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return Lasso(alpha=lam / 2, fit_intercept=False, max_iter=100_000, tol=1e-8).fit(X, y).coef_


class PolynomialRegression(BaseEstimator, RegressorMixin):
    """
    scikit-learn compatible wrapper around our own OLS/Ridge (and Lasso via
    scikit-learn), for use inside a
    Pipeline with cross-validation. Expects standardised features (the
    pipeline scales them); centres y with the mean of the training fold,
    fits theta with ols_fit (lam = 0) or ridge_fit, and adds the mean back
    when predicting. lam follows our 1/n convention.

    LLM-assisted: Claude, see module docstring.
    """

    def __init__(self, lam: float = 0.0, penalty: str = "ridge"):
        self.lam = lam
        self.penalty = penalty

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialRegression":
        self.y_mean_ = float(np.mean(y))
        y_c = y - self.y_mean_
        if self.lam == 0:
            self.coef_ = ols_fit(X, y_c)
        elif self.penalty == "lasso":
            self.coef_ = lasso_fit_sklearn(X, y_c, self.lam)
        else:
            self.coef_ = ridge_fit(X, y_c, self.lam)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.y_mean_
