"""
Gradient-based optimisation for FYS-STK3155 Project 1: cost functions and
gradients for OLS, Ridge and Lasso (analytic and by automatic
differentiation with JAX), the update rules plain / momentum / AdaGrad /
RMSProp / Adam, full-batch gradient descent and stochastic gradient descent.

Conventions as in common.py: C(theta) = (1/n)||y - X theta||^2 + penalty,
with lam ||theta||^2 (Ridge) or lam ||theta||_1 (Lasso). X is standardised
and y centred, so there is no intercept.

The design follows the course notes (Secs. 4.3-4.12): the update rule, the
data loop and the gradient are three independent pieces.

LLM-assisted
------------
Tool: Claude (Opus 5.5, claude.ai, September 2026)
Role: Wrote the module (level 4) following the course's reference
implementation (optimiser_step, optimise, sgd in the week-38/39 notebooks)
and the conventions agreed with the authors.
Verification: tests.py (analytic gradients against JAX, gradient descent
against the closed-form OLS and Ridge solutions, Lasso against scikit-learn),
and the checks printed by part_e-part_h in main.py.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

METHODS = ["plain", "momentum", "adagrad", "rmsprop", "adam"]


# ---------------------------------------------------------------------------
# Cost functions and gradients
# ---------------------------------------------------------------------------

def cost(theta, X, y, lam: float = 0.0, penalty: str = "ridge"):
    """
    Cost (1/n)||y - X theta||^2 + lam ||theta||^2 (penalty='ridge') or
    + lam ||theta||_1 (penalty='lasso'). Written with jax.numpy so that JAX
    can differentiate it; works on NumPy arrays as well.

    LLM-assisted: Claude, see module docstring.
    """
    residual = y - X @ theta
    data_term = jnp.mean(residual**2)
    if penalty == "lasso":
        return data_term + lam * jnp.sum(jnp.abs(theta))
    return data_term + lam * jnp.sum(theta**2)


def gradient_analytic(theta: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float = 0.0,
                      penalty: str = "ridge") -> np.ndarray:
    """
    Analytic gradient: (2/n) X^T (X theta - y) + 2 lam theta (Ridge), or
    + lam sign(theta) (Lasso subgradient, sign(0) = 0).

    LLM-assisted: Claude, see module docstring.
    """
    n = X.shape[0]
    grad = (2.0 / n) * X.T @ (X @ theta - y)
    if penalty == "lasso":
        return grad + lam * np.sign(theta)
    return grad + 2.0 * lam * theta


_gradient_jax = jax.jit(jax.grad(cost, argnums=0), static_argnames=("penalty",))


def gradient_jax(theta: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float = 0.0,
                 penalty: str = "ridge") -> np.ndarray:
    """
    Gradient of cost() by automatic differentiation (jax.grad, reverse mode),
    returned as a NumPy array.

    LLM-assisted: Claude, see module docstring.
    """
    return np.asarray(_gradient_jax(jnp.asarray(theta), jnp.asarray(X), jnp.asarray(y), lam, penalty=penalty))


def hessian(X: np.ndarray, lam: float = 0.0) -> np.ndarray:
    """
    Hessian of the OLS/Ridge cost: (2/n) X^T X + 2 lam I (constant in theta).

    LLM-assisted: Claude, see module docstring.
    """
    n, p = X.shape
    return (2.0 / n) * X.T @ X + 2.0 * lam * np.eye(p)


# ---------------------------------------------------------------------------
# Update rules
# ---------------------------------------------------------------------------

def optimiser_step(method: str, theta: np.ndarray, g: np.ndarray, state: dict, t: int, gamma: float,
                   beta: float = 0.9, rho: float = 0.99, beta1: float = 0.9, beta2: float = 0.999,
                   eps: float = 1e-8) -> np.ndarray:
    """
    One update of theta with gradient g. state carries the memory of the
    method between calls (v for momentum, r for AdaGrad/RMSProp, m and r for
    Adam); t is the step number from 1 (used by Adam's bias correction).

    plain:    theta - gamma g
    momentum: v = beta v + gamma g;          theta - v
    adagrad:  r = r + g^2;                   theta - gamma g / (sqrt(r) + eps)
    rmsprop:  r = rho r + (1 - rho) g^2;     theta - gamma g / (sqrt(r) + eps)
    adam:     m = beta1 m + (1 - beta1) g;   r = beta2 r + (1 - beta2) g^2,
              bias-corrected m / (1 - beta1^t), r / (1 - beta2^t);
              theta - gamma m_hat / (sqrt(r_hat) + eps)

    LLM-assisted: Claude, see module docstring.
    """
    if method == "plain":
        return theta - gamma * g
    if method == "momentum":
        state["v"] = beta * state.get("v", np.zeros_like(theta)) + gamma * g
        return theta - state["v"]
    if method == "adagrad":
        state["r"] = state.get("r", np.zeros_like(theta)) + g**2
        return theta - gamma * g / (np.sqrt(state["r"]) + eps)
    if method == "rmsprop":
        state["r"] = rho * state.get("r", np.zeros_like(theta)) + (1 - rho) * g**2
        return theta - gamma * g / (np.sqrt(state["r"]) + eps)
    if method == "adam":
        state["m"] = beta1 * state.get("m", np.zeros_like(theta)) + (1 - beta1) * g
        state["r"] = beta2 * state.get("r", np.zeros_like(theta)) + (1 - beta2) * g**2
        m_hat = state["m"] / (1 - beta1**t)
        r_hat = state["r"] / (1 - beta2**t)
        return theta - gamma * m_hat / (np.sqrt(r_hat) + eps)
    raise ValueError(f"Unknown method {method}")


# ---------------------------------------------------------------------------
# Full-batch gradient descent and SGD
# ---------------------------------------------------------------------------

def gradient_descent(grad: Callable[[np.ndarray], np.ndarray], theta0: np.ndarray, method: str,
                     gamma: float, max_iter: int, converged: Callable[[np.ndarray], bool] | None = None,
                     record: Callable[[np.ndarray], float] | None = None, **kw) -> dict:
    """
    Full-batch gradient descent. Stops when converged(theta) is True (checked
    every iteration), when theta is no longer finite (divergence), or after
    max_iter iterations. record(theta), if given, is stored every iteration.
    Returns theta, the number of iterations, whether it converged/diverged,
    and the recorded history.

    LLM-assisted: Claude, see module docstring.
    """
    theta = theta0.copy()
    state: dict = {}
    history = []
    for t in range(1, max_iter + 1):
        theta = optimiser_step(method, theta, grad(theta), state, t, gamma, **kw)
        if record is not None:
            history.append(record(theta))
        if not np.all(np.isfinite(theta)) or np.max(np.abs(theta)) > 1e8:
            return {"theta": theta, "iterations": t, "converged": False, "diverged": True, "history": history}
        if converged is not None and converged(theta):
            return {"theta": theta, "iterations": t, "converged": True, "diverged": False, "history": history}
    return {"theta": theta, "iterations": max_iter, "converged": False, "diverged": False, "history": history}


def sgd(X: np.ndarray, y: np.ndarray, grad_batch: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        theta0: np.ndarray, method: str, gamma: float, batch_size: int, n_epochs: int, seed: int,
        schedule: tuple[float, float] | None = None,
        record: Callable[[np.ndarray], float] | None = None, **kw) -> dict:
    """
    Stochastic gradient descent with minibatches. Each epoch reshuffles the
    row indices once and walks through them in batches of batch_size rows
    (drawn without replacement). grad_batch(X_b, y_b, theta) is the gradient
    on the batch (mean over its rows). With schedule = (t0, t1) the learning
    rate at update t is t0 / (t + t1); otherwise gamma is constant.
    record(theta) is stored after every epoch.

    LLM-assisted: Claude, see module docstring.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    theta = theta0.copy()
    state: dict = {}
    history = []
    t = 0
    for _ in range(n_epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = order[start:start + batch_size]
            t += 1
            rate = schedule[0] / (t + schedule[1]) if schedule is not None else gamma
            theta = optimiser_step(method, theta, grad_batch(X[batch], y[batch], theta), state, t, rate, **kw)
        if not np.all(np.isfinite(theta)) or np.max(np.abs(theta)) > 1e8:
            return {"theta": theta, "updates": t, "diverged": True, "history": history}
        if record is not None:
            history.append(record(theta))
    return {"theta": theta, "updates": t, "diverged": False, "history": history}
