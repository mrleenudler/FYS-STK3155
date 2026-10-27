"""
Tests of the neural network in nn.py.

Run with
    python tests.py
Each test prints PASS or raises an AssertionError with an explanation.

LLM-assisted
------------
Tool: Claude (Opus 5.5, claude.ai, October 2026)
Role: Wrote the file (level 4). The two checks are the ones the project text
asks for (learning goal 10): the backpropagation gradient against automatic
differentiation, and a linear network against the closed-form OLS solution.
"""

import numpy as np

from common import design_matrix, ols_fit, split_data, Scaler
from nn import NeuralNetwork, one_hot, train


def test_gradient_against_jax() -> None:
    """
    Backpropagation agrees with jax.grad for every hidden activation, for MSE
    with linear output, cross entropy with softmax output and binary cross
    entropy with sigmoid output, with and without L1/L2 penalty.

    LLM-assisted: Claude, see module docstring.
    """
    rng = np.random.default_rng(3)
    X = rng.normal(size=(25, 4))
    y_reg = rng.normal(size=(25, 2))
    y_cls = one_hot(rng.integers(0, 3, 25), 3)
    y_bin = rng.integers(0, 2, (25, 1)).astype(float)
    for hidden in ["sigmoid", "relu", "leaky_relu"]:
        for cost, out, y in [("mse", "linear", y_reg), ("cross_entropy", "softmax", y_cls),
                             ("binary_cross_entropy", "sigmoid", y_bin)]:
            for penalty in [None, "l2", "l1"]:
                net = NeuralNetwork(4, [6, 5, y.shape[1]], [hidden, hidden, out], cost, seed=7)
                own = net.compute_gradient(X, y, lam=0.1, penalty=penalty)
                ref = net.jax_gradient(X, y, lam=0.1, penalty=penalty)
                for l, ((dW, db), (dW_j, db_j)) in enumerate(zip(own, ref)):
                    assert np.allclose(dW, dW_j, atol=1e-12) and np.allclose(db, db_j, atol=1e-12), \
                        f"Gradient differs from JAX: {hidden}, {cost}, penalty={penalty}, layer {l}"


def test_linear_network_is_ols() -> None:
    """
    A network with no hidden layer and a linear output is z = X W + b, which
    is the OLS model with an intercept. Trained on the standardised design
    matrix of Project 1 (degree 5, seed 42), plain gradient descent must reach
    the closed-form OLS parameters, and the bias must be the mean of y.

    LLM-assisted: Claude, see module docstring.
    """
    x_train, _, y_train, _ = split_data(100, 0.1, seed=42)
    X = design_matrix(x_train, 5)
    scaler = Scaler().fit(X, y_train)
    X_s = scaler.transform_X(X)
    theta_ols = ols_fit(X_s, scaler.transform_y(y_train))

    # Learning rate from the largest eigenvalue of the Hessian (2/n) A^T A of the
    # cost in (W, b), A = [X_s, 1]; plain GD converges for gamma < 2 / mu_max.
    A = np.column_stack([X_s, np.ones(len(X_s))])
    mu_max = np.linalg.eigvalsh(2.0 / len(A) * A.T @ A).max()

    net = NeuralNetwork(X_s.shape[1], [1], ["linear"], "mse", seed=0)
    train(net, X_s, y_train.reshape(-1, 1), method="plain", gamma=0.99 / mu_max, n_epochs=20000)
    W, b = net.layers[0]
    assert np.max(np.abs(W.ravel() - theta_ols)) < 1e-6, \
        f"Linear network does not reproduce OLS: max |W - theta| = {np.max(np.abs(W.ravel() - theta_ols)):.1e}"
    assert abs(b[0] - y_train.mean()) < 1e-6, "Bias is not the mean of y"


if __name__ == "__main__":
    for test in [test_gradient_against_jax, test_linear_network_is_ols]:
        test()
        print(f"PASS {test.__name__}")
