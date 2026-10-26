"""
Feed-forward neural network for FYS-STK3155 Project 2.

What is in this file
--------------------
1. Activation functions and their derivatives: sigmoid, ReLU, Leaky ReLU,
   linear and softmax.
2. Cost functions: the mean squared error (MSE), the binary cross entropy and
   the multiclass cross entropy.
3. The class NeuralNetwork: initialisation of the weights, the forward pass,
   the cost, the gradient by backpropagation (compute_gradient), and the same
   gradient from JAX (only used to test compute_gradient).
4. The function train, which trains a network with the update rules from
   Project 1 (optim.optimiser_step): plain gradient descent, RMSprop, Adam, ...

Conventions (lecture notes Sec. 8.5 and the exercises of weeks 41 and 42):
- the samples are the rows of the input, so X has shape (samples, features);
- the weight matrix W of a layer has shape (inputs, outputs);
- every layer computes z = a @ W + b and then a = f(z).

The cost is a mean over the samples, as in Project 1. The optional penalty is
lam * (sum of all squared weights) for L2, or lam * (sum of all |weights|)
for L1. The biases are not penalised, just as the intercept was not penalised
in Project 1.

LLM-assisted
------------
Tool: Claude (Opus 5.5, claude.ai, October 2026)
Role: Wrote the module (level 4) around andreand's own batched
backpropagation from the week 42 exercises (backprop in
NeuralNetwork.compute_gradient is his code, moved into the class with
the output-layer case for the cross entropies and the penalty term
added). Initialisation follows the class in Sec. 8.13 of the lecture notes.
Verification: tests.py (gradient against jax.grad for every activation and
all three costs, with and without L1/L2; a network without hidden layers and
with linear output trained by gradient descent reproduces the closed-form OLS
solution of Project 1).
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from optim import optimiser_step

# JAX uses 32-bit numbers unless we ask for 64-bit. We want 64-bit, so that the
# gradient test can compare with our NumPy gradient to many decimals.
jax.config.update("jax_enable_x64", True)

# A network is stored as a list with one (W, b) pair per layer
Layers = list[tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------
# 1. Activation functions and their derivatives (lecture notes Sec. 8.6)
# ---------------------------------------------------------------------------

LEAKY_SLOPE = 0.01   # the slope alpha of Leaky ReLU for z <= 0


def _xp(z):
    """
    Choose the library that matches the array z.

    The same activation functions are used in two places:
    - in the training, with NumPy arrays (fast);
    - in the test against jax.grad, where JAX sends its own kind of arrays
      through the functions, and only jax.numpy functions work on those.
    So we use NumPy for NumPy arrays and jax.numpy for everything else.
    Example: _xp(z).tanh(z) is np.tanh(z) or jnp.tanh(z).
    """
    if isinstance(z, np.ndarray):
        return np
    return jnp


def sigmoid(z):
    """
    1 / (1 + e^(-z)).

    Written as (1 + tanh(z/2)) / 2, which is exactly the same function, but
    does not overflow: for large negative z, e^(-z) is too large for a float,
    while tanh(z/2) just becomes -1.
    """
    return 0.5 * (1.0 + _xp(z).tanh(0.5 * z))


def sigmoid_der(z):
    """sigma'(z) = sigma(z) (1 - sigma(z))."""
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z):
    """max(0, z) for each element."""
    return _xp(z).where(z > 0, z, 0.0)


def relu_der(z):
    """1 where z > 0, otherwise 0 (we use 0 at z = 0, where ReLU has no derivative)."""
    return _xp(z).where(z > 0, 1.0, 0.0)


def leaky_relu(z):
    """z where z > 0, otherwise alpha * z."""
    return _xp(z).where(z > 0, z, LEAKY_SLOPE * z)


def leaky_relu_der(z):
    """1 where z > 0, otherwise alpha."""
    return _xp(z).where(z > 0, 1.0, LEAKY_SLOPE)


def linear(z):
    """The identity, f(z) = z (output layer for regression)."""
    return z


def linear_der(z):
    """f'(z) = 1 for each element."""
    return _xp(z).ones_like(z)


def softmax(z):
    """
    Softmax of each row of z (one row of scores per sample):
    a_k = e^(z_k) / sum_j e^(z_j).

    The largest z of each row is subtracted first. This does not change the
    result (the same factor appears above and below the fraction line), but
    all exponents become <= 0, so e^z cannot overflow.
    """
    xp = _xp(z)
    e_z = xp.exp(z - xp.max(z, axis=1, keepdims=True))
    return e_z / xp.sum(e_z, axis=1, keepdims=True)


# Look-up table: name -> (function, derivative). The softmax has no derivative
# here: it is only used in the output layer together with the cross entropy,
# where compute_gradient uses delta = (a - y) / N directly.
ACTIVATIONS: dict[str, tuple[Callable, Callable | None]] = {
    "sigmoid": (sigmoid, sigmoid_der),
    "relu": (relu, relu_der),
    "leaky_relu": (leaky_relu, leaky_relu_der),
    "linear": (linear, linear_der),
    "softmax": (softmax, None),
}


# ---------------------------------------------------------------------------
# 2. Cost functions (the data term only; the penalty is added in
#    NeuralNetwork.cost and NeuralNetwork.compute_gradient)
# ---------------------------------------------------------------------------

def mse(predict, target):
    """Mean squared error: the mean over all elements of (predict - target)^2."""
    return _xp(predict).mean((predict - target) ** 2)


def mse_der(predict, target):
    """dC/da for the MSE: 2 (a - y) / (number of elements)."""
    return 2.0 * (predict - target) / predict.size


def cross_entropy(predict, target):
    """
    Multiclass cross entropy, mean over the samples:
    C = -(1/N) sum_n sum_k y_nk ln(a_nk), with one-hot targets y.
    The tiny number 1e-15 avoids ln(0) if a probability is exactly 0.
    """
    xp = _xp(predict)
    return -xp.sum(target * xp.log(predict + 1e-15)) / predict.shape[0]


def binary_cross_entropy(predict, target):
    """
    Binary cross entropy, mean over the samples (one output column, y = 0 or 1):
    C = -(1/N) sum_n [y ln(a) + (1 - y) ln(1 - a)].
    The tiny number 1e-15 avoids ln(0).
    """
    xp = _xp(predict)
    return -xp.sum(target * xp.log(predict + 1e-15) + (1 - target) * xp.log(1 - predict + 1e-15)) / predict.shape[0]


# Look-up table: name -> (cost, derivative with respect to the output a).
# The cross entropies have no derivative here, since compute_gradient uses
# delta = (a - y) / N for them directly (Methods, "Matched pairs").
COSTS = {
    "mse": (mse, mse_der),
    "cross_entropy": (cross_entropy, None),
    "binary_cross_entropy": (binary_cross_entropy, None),
}


def accuracy(predict: np.ndarray, target: np.ndarray) -> float:
    """
    Share of the samples where the largest output is the correct class.
    target can be one-hot vectors (one row per sample) or class labels (0-9).
    """
    if target.ndim == 1:
        labels = target
    else:
        labels = np.argmax(target, axis=1)
    return float(np.mean(np.argmax(predict, axis=1) == labels))


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Class labels, e.g. [2, 0], as one-hot rows, e.g. [[0, 0, 1], [1, 0, 0]]."""
    out = np.zeros((len(labels), n_classes))
    out[np.arange(len(labels)), labels] = 1.0
    return out


# ---------------------------------------------------------------------------
# 3. The network
# ---------------------------------------------------------------------------

class NeuralNetwork:
    """
    Fully connected feed-forward network.

    n_inputs     : number of features (1 for the Runge data, 784 for MNIST)
    layer_sizes  : number of outputs of each layer; the last one is the output
                   layer. Example: [50, 1] is one hidden layer of 50 nodes and
                   one output node.
    activations  : one name per layer from ACTIVATIONS, e.g. ["sigmoid", "linear"]
    cost         : "mse", "cross_entropy" (needs a softmax output layer) or
                   "binary_cross_entropy" (needs one sigmoid output node)
    seed         : seed for the random initial weights
    bias_init    : the value all biases start at

    The weights are drawn from a normal distribution with mean 0 and variance
    2/n_in for ReLU and Leaky ReLU (He) and 1/n_in otherwise, where n_in is
    the number of inputs to the layer (lecture notes Sec. 8.10 and 8.13).

    LLM-assisted: Claude, see module docstring.
    """

    def __init__(self, n_inputs: int, layer_sizes: list[int], activations: list[str], cost: str = "mse",
                 seed: int = 0, bias_init: float = 0.01):
        # Stop early with a clear message if the combination does not make sense
        if len(layer_sizes) != len(activations):
            raise ValueError("One activation per layer is needed")
        if cost == "cross_entropy" and activations[-1] != "softmax":
            raise ValueError("Cross entropy is only implemented with a softmax output layer")
        if cost == "binary_cross_entropy" and (activations[-1] != "sigmoid" or layer_sizes[-1] != 1):
            raise ValueError("Binary cross entropy is only implemented with one sigmoid output node")
        if "softmax" in activations[:-1]:
            raise ValueError("Softmax is only allowed in the output layer")

        # Look up the functions once, so that the forward and backward passes
        # can use them directly
        self.activation_names = activations
        self.activation_funcs = [ACTIVATIONS[name][0] for name in activations]
        self.activation_ders = [ACTIVATIONS[name][1] for name in activations]
        self.cost_name = cost
        self.cost_func, self.cost_der = COSTS[cost]

        # Initial weights and biases, layer by layer
        rng = np.random.default_rng(seed)
        self.layers: Layers = []
        i_size = n_inputs                       # number of inputs to the current layer
        for size, activation in zip(layer_sizes, activations):
            if activation in ("relu", "leaky_relu"):
                std = np.sqrt(2.0 / i_size)     # He: variance 2 / n_in
            else:
                std = np.sqrt(1.0 / i_size)     # variance 1 / n_in
            W = rng.normal(0.0, std, (i_size, size))
            b = np.full(size, bias_init)
            self.layers.append((W, b))
            i_size = size                       # the outputs of this layer are the inputs to the next

    # -- forward pass -------------------------------------------------------

    def predict(self, inputs: np.ndarray, layers: Layers | None = None):
        """
        The forward pass: returns the output of the last layer.
        Normally the network's own weights are used. The test against JAX
        passes its own copy of the weights as layers, so that jax.grad can
        differentiate with respect to them.
        """
        if layers is None:
            layers = self.layers
        a = inputs
        for (W, b), f in zip(layers, self.activation_funcs):
            a = f(a @ W + b)
        return a

    def _feed_forward_saver(self, inputs: np.ndarray):
        """
        The forward pass that also stores what backpropagation needs: the input
        a to every layer and its z. Returns (layer_inputs, zs, output).
        """
        layer_inputs, zs = [], []
        a = inputs
        for (W, b), f in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = f(z)
            zs.append(z)
        return layer_inputs, zs, a

    # -- cost and gradient ---------------------------------------------------

    def cost(self, inputs: np.ndarray, targets: np.ndarray, lam: float = 0.0, penalty: str | None = None,
             layers: Layers | None = None):
        """
        The cost: data term plus lam * sum W^2 (penalty="l2") or lam * sum |W|
        (penalty="l1"), summed over the weights of all layers (not the biases).
        Only used by the JAX test; the training uses compute_gradient.
        """
        if layers is None:
            layers = self.layers
        c = self.cost_func(self.predict(inputs, layers), targets)
        if penalty == "l2":
            c = c + lam * sum(_xp(W).sum(W ** 2) for W, _ in layers)
        elif penalty == "l1":
            c = c + lam * sum(_xp(W).sum(_xp(W).abs(W)) for W, _ in layers)
        return c

    def compute_gradient(self, inputs: np.ndarray, targets: np.ndarray, lam: float = 0.0,
                         penalty: str | None = None) -> Layers:
        """
        Backpropagation for a whole batch (andreand's backprop_batch from week 42).
        Returns [(dC/dW, dC/db), ...], one pair per layer, in the same order as
        self.layers.

        Output layer: delta = dC/dz = f'(z) * dC/da in general. For softmax with
        cross entropy and for sigmoid with binary cross entropy the two factors
        cancel, and delta = (a - y) / N is used directly.
        Penalty: 2 lam W (L2) or lam sign(W) (L1) is added to dC/dW.
        """
        # Forward pass, keeping the input and z of every layer
        layer_inputs, zs, predict = self._feed_forward_saver(inputs)
        layer_grads: Layers = [() for _ in self.layers]

        # Go backwards from the output layer to the first layer
        for i in reversed(range(len(self.layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_ders[i]

            if i == len(self.layers) - 1:
                # Output layer
                if self.cost_name in ("cross_entropy", "binary_cross_entropy"):
                    # matched pairs: the derivatives of cost and activation cancel
                    dC_dz = (predict - targets) / predict.shape[0]
                else:
                    dC_dz = activation_der(z) * self.cost_der(predict, targets)
            else:
                # Hidden layer: pull the delta of the next layer back through its weights
                W_next, _ = self.layers[i + 1]
                dC_da = dC_dz @ W_next.T                       # (N, outputs of layer i)
                dC_dz = activation_der(z) * dC_da              # delta of layer i

            dC_dW = np.asarray(layer_input.T @ dC_dz)          # (inputs, outputs): sums over the samples
            dC_db = np.asarray(np.sum(dC_dz, axis=0))          # (outputs,): sum over the samples

            # The penalty only adds a term to the weight gradient
            W, _ = self.layers[i]
            if penalty == "l2":
                dC_dW = dC_dW + 2.0 * lam * W
            elif penalty == "l1":
                dC_dW = dC_dW + lam * np.sign(W)

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads

    def jax_gradient(self, inputs: np.ndarray, targets: np.ndarray, lam: float = 0.0,
                     penalty: str | None = None) -> Layers:
        """
        The same gradient by automatic differentiation of self.cost (only for
        testing compute_gradient). jax.grad differentiates the cost with respect
        to its argument, here the list of (W, b) pairs.
        """
        def cost_of_layers(layers):
            return self.cost(inputs, targets, lam, penalty, layers)

        grads = jax.grad(cost_of_layers)(self.layers)
        return [(np.asarray(dW), np.asarray(db)) for dW, db in grads]


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------

def train(network: NeuralNetwork, X: np.ndarray, y: np.ndarray, method: str, gamma: float,
          n_epochs: int, batch_size: int | None = None, seed: int = 0, lam: float = 0.0,
          penalty: str | None = None, record: Callable[[NeuralNetwork], float] | None = None,
          **kw) -> dict:
    """
    Train the network in place.

    method     : update rule from optim.py: "plain", "momentum", "adagrad",
                 "rmsprop" or "adam"
    gamma      : learning rate
    n_epochs   : number of passes through the training data
    batch_size : None (or at least the number of samples) is plain gradient
                 descent with all the data in every step; otherwise stochastic
                 gradient descent with minibatches, in a new random order every
                 epoch, as sgd() in Project 1
    seed       : seed for the order of the minibatches
    lam, penalty : L1/L2 penalty, see NeuralNetwork.cost
    record     : a function of the network; if given, its value is stored after
                 every epoch (for training curves)
    **kw       : other settings for optimiser_step, e.g. rho for RMSprop

    Every weight matrix and every bias vector has its own memory in the
    optimiser. Training stops early if a parameter becomes infinite or NaN
    (divergence). Returns {"diverged": ..., "updates": ..., "history": [...]}.

    LLM-assisted: Claude, see module docstring.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if batch_size is None or batch_size >= n:
        batch_size = n                  # plain gradient descent: one batch with all the data

    # The memory of the optimiser (running averages in RMSprop and Adam):
    # one dict for W and one for b in every layer
    states = [({}, {}) for _ in network.layers]
    history = []
    t = 0                               # number of updates so far (Adam needs it)

    for epoch in range(n_epochs):
        # SGD goes through the rows in a new random order every epoch; GD keeps the order
        if batch_size < n:
            order = rng.permutation(n)
        else:
            order = np.arange(n)

        for start in range(0, n, batch_size):
            batch = order[start:start + batch_size]
            t += 1
            grads = network.compute_gradient(X[batch], y[batch], lam, penalty)

            # One update of every weight matrix and bias vector
            for l in range(len(network.layers)):
                W, b = network.layers[l]
                dW, db = grads[l]
                state_W, state_b = states[l]
                W = optimiser_step(method, W, dW, state_W, t, gamma, **kw)
                b = optimiser_step(method, b, db, state_b, t, gamma, **kw)
                network.layers[l] = (W, b)

        # Stop if the training has diverged
        if not all(np.all(np.isfinite(W)) for W, _ in network.layers):
            return {"diverged": True, "updates": t, "history": history}
        if record is not None:
            history.append(record(network))

    return {"diverged": False, "updates": t, "history": history}
