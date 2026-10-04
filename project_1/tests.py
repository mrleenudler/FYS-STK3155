"""
Tests of the shared functions in common.py and the optimisers in optim.py.

Run with
    uv run tests.py
Each test prints PASS or raises an AssertionError with an explanation.

LLM-assisted
------------
Tool: Claude (Opus 5.5, claude.ai, September 2026)
Role: Wrote the file (level 4). The choice of what to test follows the
checks discussed with the author: agreement with scikit-learn, exact
recovery of a known polynomial, invariance of OLS predictions under
scaling, recovery of the intercept after centering, and for Ridge the
agreement with scikit-learn and with the SVD form of the solution. For
optim.py: analytic gradients against JAX, gradient descent against the
closed-form solutions, SGD with one full batch against plain gradient
descent, and Lasso by gradient descent against scikit-learn.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from common import (Scaler, design_matrix, fit_ols, fit_ridge, lasso_fit_sklearn, mse, ols_fit, r2,
                    ridge_fit, ridge_fit_svd, split_data)
from optim import cost, gradient_analytic, gradient_descent, gradient_jax, hessian, sgd

TOL = 1e-10


def test_ols_against_sklearn() -> None:
    """
    Own OLS on scaled data equals scikit-learn's LinearRegression without intercept.

    LLM-assisted: Claude, see module docstring.
    """
    x_train, _, y_train, _ = split_data(100, 0.1, seed=42)
    for degree in [1, 5, 10]:
        scaler, theta = fit_ols(x_train, y_train, degree)
        X_s = scaler.transform_X(design_matrix(x_train, degree))
        theta_skl = LinearRegression(fit_intercept=False).fit(X_s, scaler.transform_y(y_train)).coef_
        assert np.max(np.abs(theta - theta_skl)) < 1e-8, f"OLS differs from scikit-learn at degree {degree}"


def test_exact_polynomial() -> None:
    """
    Without noise, data from a known polynomial are reproduced exactly, and the
    coefficients converted back to the original polynomial are the true ones.

    LLM-assisted: Claude, see module docstring.
    """
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, 50)
    theta_true = np.array([0.5, -1.0, 2.0, 0.0, 3.0])  # theta_0 ... theta_4
    y = design_matrix(x, 4, intercept=True) @ theta_true
    scaler, theta = fit_ols(x, y, 4)
    theta_0, theta_orig = scaler.original_coefficients(theta)
    assert abs(theta_0 - theta_true[0]) < TOL, "Intercept not recovered"
    assert np.max(np.abs(theta_orig - theta_true[1:])) < TOL, "Coefficients not recovered"
    assert mse(y, scaler.predict(design_matrix(x, 4), theta)) < TOL, "Noise-free fit not exact"


def test_scaling_invariance_ols() -> None:
    """
    Scaled OLS (centred, no intercept column) and unscaled OLS (intercept
    column, no centering) give the same predictions.

    LLM-assisted: Claude, see module docstring.
    """
    x_train, x_test, y_train, _ = split_data(100, 0.1, seed=42)
    degree = 8
    scaler, theta = fit_ols(x_train, y_train, degree)
    pred_scaled = scaler.predict(design_matrix(x_test, degree), theta)
    theta_u = ols_fit(design_matrix(x_train, degree, intercept=True), y_train)
    pred_unscaled = design_matrix(x_test, degree, intercept=True) @ theta_u
    assert np.max(np.abs(pred_scaled - pred_unscaled)) < 1e-10, "Scaling changed the OLS predictions"


def test_scaler_uses_training_statistics() -> None:
    """
    Transformed training columns have mean 0 and standard deviation 1, and
    the test data are transformed with the training statistics (so they are
    in general not exactly standardised themselves).

    LLM-assisted: Claude, see module docstring.
    """
    x_train, x_test, y_train, _ = split_data(100, 0.1, seed=42)
    X_train = design_matrix(x_train, 5)
    scaler = Scaler().fit(X_train, y_train)
    X_s = scaler.transform_X(X_train)
    assert np.max(np.abs(X_s.mean(axis=0))) < TOL, "Training columns not centred"
    assert np.max(np.abs(X_s.std(axis=0) - 1)) < TOL, "Training columns not unit variance"
    assert abs(scaler.transform_y(y_train).mean()) < TOL, "Training targets not centred"
    X_test_s = scaler.transform_X(design_matrix(x_test, 5))
    assert np.max(np.abs(X_test_s.mean(axis=0))) > 1e-3, "Test data seem to be scaled with their own statistics"


def test_r2_limits() -> None:
    """
    R2 = 1 for a perfect prediction and R2 = 0 for predicting the mean.

    LLM-assisted: Claude, see module docstring.
    """
    y = np.array([1.0, 2.0, 4.0, 7.0])
    assert abs(r2(y, y) - 1.0) < TOL
    assert abs(r2(y, np.full_like(y, y.mean()))) < TOL


def test_reproducible_split() -> None:
    """
    The same seed gives identical data and split; a different seed does not.

    LLM-assisted: Claude, see module docstring.
    """
    first = split_data(100, 0.1, seed=42)
    second = split_data(100, 0.1, seed=42)
    other = split_data(100, 0.1, seed=43)
    assert all(np.array_equal(a, b) for a, b in zip(first, second)), "Same seed gave different data"
    assert not np.array_equal(first[0], other[0]), "Different seeds gave the same data"
    assert len(first[0]) == 80 and len(first[1]) == 20, "Split is not 80/20"


def test_ridge_against_sklearn() -> None:
    """
    Own Ridge equals scikit-learn's Ridge. With our cost (1/n on the data
    term) scikit-learn's alpha corresponds to n * lambda.

    LLM-assisted: Claude, see module docstring.
    """
    x_train, _, y_train, _ = split_data(100, 0.1, seed=42)
    n = len(x_train)
    for degree in [5, 15]:
        for lam in [1e-4, 1e-2, 1.0]:
            scaler, theta = fit_ridge(x_train, y_train, degree, lam)
            X_s = scaler.transform_X(design_matrix(x_train, degree))
            skl = Ridge(alpha=n * lam, fit_intercept=False).fit(X_s, scaler.transform_y(y_train))
            assert np.max(np.abs(theta - skl.coef_)) < 1e-8, f"Ridge differs from scikit-learn ({degree=}, {lam=})"


def test_ridge_svd_form() -> None:
    """
    The Ridge solution from the linear system equals the SVD form, and the
    SVD form with lambda = 0 equals OLS.

    LLM-assisted: Claude, see module docstring.
    """
    x_train, _, y_train, _ = split_data(100, 0.1, seed=42)
    degree = 10
    X_s = Scaler().fit(design_matrix(x_train, degree), y_train).transform_X(design_matrix(x_train, degree))
    y_c = y_train - y_train.mean()
    for lam in [1e-4, 1e-2, 1.0]:
        assert np.max(np.abs(ridge_fit(X_s, y_c, lam) - ridge_fit_svd(X_s, y_c, lam))) < 1e-8, \
            f"Ridge and its SVD form differ ({lam=})"
    assert np.max(np.abs(ridge_fit_svd(X_s, y_c, 0.0) - ols_fit(X_s, y_c))) < 1e-8, "SVD form with lambda = 0 is not OLS"


def test_ridge_limits() -> None:
    """
    Ridge approaches OLS as lambda -> 0 and shrinks the coefficients towards
    zero as lambda grows.

    LLM-assisted: Claude, see module docstring.
    """
    x_train, _, y_train, _ = split_data(100, 0.1, seed=42)
    degree = 5
    _, theta_ols = fit_ols(x_train, y_train, degree)
    _, theta_small = fit_ridge(x_train, y_train, degree, 1e-12)
    assert np.max(np.abs(theta_small - theta_ols)) < 1e-6, "Ridge with tiny lambda is not OLS"
    norms = [np.linalg.norm(fit_ridge(x_train, y_train, degree, lam)[1]) for lam in [1e-3, 1e-1, 10.0, 1e3]]
    assert all(a > b for a, b in zip(norms, norms[1:])), "Coefficient norm does not decrease with lambda"
    assert norms[-1] < 1e-2, "Coefficients do not go to zero for large lambda"


def scaled_problem(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardised design matrix and centred targets for the seed-42 training set.

    LLM-assisted: Claude, see module docstring.
    """
    x_train, _, y_train, _ = split_data(100, 0.1, seed=42)
    X = design_matrix(x_train, degree)
    scaler = Scaler().fit(X, y_train)
    return scaler.transform_X(X), scaler.transform_y(y_train)


def test_gradients_against_jax() -> None:
    """
    The analytic gradients of the OLS, Ridge and Lasso costs equal the JAX
    gradients (at a point with no zero coefficients, where |theta| is
    differentiable).

    LLM-assisted: Claude, see module docstring.
    """
    X, y = scaled_problem(6)
    theta = np.random.default_rng(3).normal(size=X.shape[1])
    for lam, penalty in [(0.0, "ridge"), (1e-2, "ridge"), (1e-2, "lasso")]:
        diff = np.max(np.abs(gradient_analytic(theta, X, y, lam, penalty) - gradient_jax(theta, X, y, lam, penalty)))
        assert diff < 1e-12, f"Analytic and JAX gradients differ ({lam=}, {penalty=})"


def test_gradient_descent_closed_form() -> None:
    """
    Plain GD, momentum and Adam converge to the closed-form OLS and Ridge
    solutions, and plain GD diverges just above gamma_max = 2 / mu_max (mu: Hessian eigenvalues).

    LLM-assisted: Claude, see module docstring.
    """
    X, y = scaled_problem(5)
    p = X.shape[1]
    for lam in [0.0, 1e-2]:
        theta_hat = ols_fit(X, y) if lam == 0 else ridge_fit(X, y, lam)
        gamma_max = 2 / np.linalg.eigvalsh(hessian(X, lam)).max()
        close = lambda t: np.linalg.norm(t - theta_hat) < 1e-8
        grad = lambda t: gradient_analytic(t, X, y, lam)
        for method, gamma in [("plain", 0.9 * gamma_max), ("momentum", 0.1), ("adam", 0.05)]:
            run = gradient_descent(grad, np.zeros(p), method, gamma, 200_000, converged=close)
            assert run["converged"], f"{method} did not reach the closed-form solution ({lam=})"
        run = gradient_descent(grad, np.zeros(p), "plain", 1.05 * gamma_max, 200_000)
        assert run["diverged"], f"Plain GD did not diverge above gamma_max ({lam=})"


def test_sgd_full_batch_is_gd() -> None:
    """
    SGD with one batch containing all rows takes the same steps as plain
    gradient descent.

    LLM-assisted: Claude, see module docstring.
    """
    X, y = scaled_problem(5)
    p = X.shape[1]
    grad_batch = lambda Xb, yb, t: gradient_analytic(t, Xb, yb)
    theta_sgd = sgd(X, y, grad_batch, np.zeros(p), "plain", 0.1, X.shape[0], 500, seed=1)["theta"]
    theta_gd = gradient_descent(lambda t: gradient_analytic(t, X, y), np.zeros(p), "plain", 0.1, 500)["theta"]
    assert np.max(np.abs(theta_sgd - theta_gd)) < 1e-12, "Full-batch SGD differs from plain GD"


def test_lasso_gd_against_sklearn() -> None:
    """
    Plain subgradient descent for Lasso reaches the cost and coefficients of
    scikit-learn's coordinate descent (alpha = lambda / 2) when no
    coefficient is zero at the optimum.

    LLM-assisted: Claude, see module docstring.
    """
    X, y = scaled_problem(5)
    lam = 1e-3
    theta_skl = lasso_fit_sklearn(X, y, lam)
    assert np.all(theta_skl != 0), "Test assumes no zero coefficients"
    gamma = 0.5 * 2 / np.linalg.eigvalsh(hessian(X)).max()
    theta = gradient_descent(lambda t: gradient_analytic(t, X, y, lam, "lasso"), np.zeros(X.shape[1]), "plain",
                             gamma, 50_000)["theta"]
    assert abs(float(cost(theta, X, y, lam, "lasso")) - float(cost(theta_skl, X, y, lam, "lasso"))) < 1e-10, \
        "Lasso cost from GD differs from scikit-learn"
    assert np.max(np.abs(theta - theta_skl)) < 1e-5, "Lasso coefficients from GD differ from scikit-learn"


if __name__ == "__main__":
    tests = [test_ols_against_sklearn, test_exact_polynomial, test_scaling_invariance_ols,
             test_scaler_uses_training_statistics, test_r2_limits, test_reproducible_split,
             test_ridge_against_sklearn, test_ridge_svd_form, test_ridge_limits,
             test_gradients_against_jax, test_gradient_descent_closed_form, test_sgd_full_batch_is_gd,
             test_lasso_gd_against_sklearn]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
