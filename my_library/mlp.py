"""
Multi-Layer Perceptron — built from scratch using NumPy only.
Full forward pass + backpropagation through time.

Supports:
  - Arbitrary layer sizes
  - Activations: ReLU, Sigmoid, Tanh, Softmax
  - Optimizers: SGD, SGD+Momentum, Adam
  - L2 Regularization
  - Mini-batch training
  - Dropout

Author : Sajid Ansari
Project: Handwritten Digit Recognizer
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════
#  Activation Functions
# ══════════════════════════════════════════════════════════════════

class ReLU:
    def forward(self, z):
        self.mask = z > 0
        return z * self.mask

    def backward(self, dA):
        return dA * self.mask


class Sigmoid:
    def forward(self, z):
        self.out = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return self.out

    def backward(self, dA):
        return dA * self.out * (1 - self.out)


class Tanh:
    def forward(self, z):
        self.out = np.tanh(z)
        return self.out

    def backward(self, dA):
        return dA * (1 - self.out ** 2)


class Softmax:
    def forward(self, z):
        # Numerically stable softmax
        z_shift = z - np.max(z, axis=1, keepdims=True)
        exp_z   = np.exp(z_shift)
        self.out = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.out

    def backward(self, dA):
        # Combined with cross-entropy loss: gradient is (pred - y)
        # so this is handled in the loss backward directly
        return dA


ACTIVATIONS = {
    "relu":    ReLU,
    "sigmoid": Sigmoid,
    "tanh":    Tanh,
    "softmax": Softmax,
}


# ══════════════════════════════════════════════════════════════════
#  Loss Functions
# ══════════════════════════════════════════════════════════════════

def cross_entropy_loss(y_pred, y_true):
    """
    Categorical cross-entropy.
    y_pred: (batch, n_classes) — softmax probabilities
    y_true: (batch,) — integer class labels
    """
    n = y_pred.shape[0]
    eps = 1e-12
    correct = y_pred[np.arange(n), y_true.astype(int)]
    return -np.mean(np.log(correct + eps))


def cross_entropy_grad(y_pred, y_true):
    """Gradient of cross-entropy + softmax combined."""
    n = y_pred.shape[0]
    dZ = y_pred.copy()
    dZ[np.arange(n), y_true.astype(int)] -= 1
    return dZ / n


# ══════════════════════════════════════════════════════════════════
#  Dense Layer
# ══════════════════════════════════════════════════════════════════

class DenseLayer:
    def __init__(self, n_in, n_out, activation="relu", l2=0.0):
        # He initialization for ReLU, Xavier for others
        if activation == "relu":
            scale = np.sqrt(2.0 / n_in)
        else:
            scale = np.sqrt(1.0 / n_in)

        self.W  = np.random.randn(n_in, n_out) * scale
        self.b  = np.zeros((1, n_out))
        self.l2 = l2

        self.activation = ACTIVATIONS[activation]()

        # Adam moment estimates
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

        # Gradients (populated during backward)
        self.dW = None
        self.db = None

    def forward(self, A_prev, training=True):
        self.A_prev = A_prev
        self.Z      = A_prev @ self.W + self.b
        self.A      = self.activation.forward(self.Z)
        return self.A

    def backward(self, dA):
        dZ = self.activation.backward(dA)
        m  = self.A_prev.shape[0]

        self.dW = (self.A_prev.T @ dZ) / m + (self.l2 / m) * self.W
        self.db = np.mean(dZ, axis=0, keepdims=True)
        dA_prev = dZ @ self.W.T
        return dA_prev


# ══════════════════════════════════════════════════════════════════
#  Dropout Layer
# ══════════════════════════════════════════════════════════════════

class DropoutLayer:
    def __init__(self, keep_prob=0.8):
        self.keep_prob = keep_prob
        self.mask = None

    def forward(self, A, training=True):
        if training:
            self.mask = (np.random.rand(*A.shape) < self.keep_prob) / self.keep_prob
            return A * self.mask
        return A

    def backward(self, dA):
        return dA * self.mask


# ══════════════════════════════════════════════════════════════════
#  MLP Classifier
# ══════════════════════════════════════════════════════════════════

class MLPClassifier:
    """
    Multi-Layer Perceptron for classification.

    Parameters
    ----------
    layer_sizes  : list[int]   e.g. [784, 256, 128, 10]
    activations  : list[str]   e.g. ['relu', 'relu', 'softmax']
    optimizer    : str         'sgd' | 'momentum' | 'adam'
    learning_rate: float
    l2           : float       L2 regularization strength
    dropout_rate : float       dropout keep probability (1.0 = no dropout)
    epochs       : int
    batch_size   : int
    random_state : int
    """

    def __init__(
        self,
        layer_sizes,
        activations=None,
        optimizer="adam",
        learning_rate=0.001,
        l2=1e-4,
        dropout_rate=1.0,
        epochs=20,
        batch_size=64,
        random_state=42,
        verbose=True,
    ):
        self.layer_sizes   = layer_sizes
        self.optimizer     = optimizer
        self.lr            = learning_rate
        self.l2            = l2
        self.dropout_rate  = dropout_rate
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.random_state  = random_state
        self.verbose       = verbose

        # Default activations: ReLU for hidden, softmax for output
        if activations is None:
            activations = ["relu"] * (len(layer_sizes) - 2) + ["softmax"]
        self.activations = activations

        self.layers       = []
        self.train_losses = []
        self.val_losses   = []
        self.train_accs   = []
        self.val_accs     = []
        self._t           = 0   # Adam timestep

    def _build(self):
        np.random.seed(self.random_state)
        self.layers = []
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(
                DenseLayer(
                    self.layer_sizes[i],
                    self.layer_sizes[i + 1],
                    activation=self.activations[i],
                    l2=self.l2,
                )
            )
            # Add dropout after every hidden layer (not output)
            if i < len(self.layer_sizes) - 2 and self.dropout_rate < 1.0:
                self.layers.append(DropoutLayer(self.dropout_rate))

    def _forward(self, X, training=True):
        A = X
        for layer in self.layers:
            A = layer.forward(A, training=training)
        return A

    def _backward(self, y_pred, y_true):
        # Gradient from softmax + cross-entropy combined
        dA = cross_entropy_grad(y_pred, y_true)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def _update_sgd(self):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

    def _update_momentum(self, beta=0.9):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.mW = beta * layer.mW + (1 - beta) * layer.dW
                layer.mb = beta * layer.mb + (1 - beta) * layer.db
                layer.W -= self.lr * layer.mW
                layer.b -= self.lr * layer.mb

    def _update_adam(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                # Biased moment estimates
                layer.mW = beta1 * layer.mW + (1 - beta1) * layer.dW
                layer.vW = beta2 * layer.vW + (1 - beta2) * layer.dW ** 2
                layer.mb = beta1 * layer.mb + (1 - beta1) * layer.db
                layer.vb = beta2 * layer.vb + (1 - beta2) * layer.db ** 2
                # Bias correction
                mW_hat = layer.mW / (1 - beta1 ** self._t)
                vW_hat = layer.vW / (1 - beta2 ** self._t)
                mb_hat = layer.mb / (1 - beta1 ** self._t)
                vb_hat = layer.vb / (1 - beta2 ** self._t)
                # Update
                layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)
                layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def _update(self):
        if self.optimizer == "sgd":
            self._update_sgd()
        elif self.optimizer == "momentum":
            self._update_momentum()
        else:
            self._update_adam()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self._build()
        n = X_train.shape[0]

        for epoch in range(1, self.epochs + 1):
            # Shuffle
            idx = np.random.permutation(n)
            X_s, y_s = X_train[idx], y_train[idx]

            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n, self.batch_size):
                Xb = X_s[start:start + self.batch_size]
                yb = y_s[start:start + self.batch_size]

                y_pred = self._forward(Xb, training=True)
                loss   = cross_entropy_loss(y_pred, yb)
                epoch_loss += loss
                n_batches  += 1

                self._backward(y_pred, yb)
                self._update()

            # Epoch metrics
            train_pred  = self._forward(X_train, training=False)
            train_loss  = cross_entropy_loss(train_pred, y_train)
            train_acc   = np.mean(np.argmax(train_pred, axis=1) == y_train)

            self.train_losses.append(float(train_loss))
            self.train_accs.append(float(train_acc))

            val_info = ""
            if X_val is not None:
                val_pred = self._forward(X_val, training=False)
                val_loss = cross_entropy_loss(val_pred, y_val)
                val_acc  = np.mean(np.argmax(val_pred, axis=1) == y_val)
                self.val_losses.append(float(val_loss))
                self.val_accs.append(float(val_acc))
                val_info = f"  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"

            if self.verbose and (epoch % 2 == 0 or epoch == 1):
                print(f"Epoch {epoch:>3}/{self.epochs} "
                      f" loss: {train_loss:.4f}  acc: {train_acc:.4f}{val_info}")

        return self

    def predict_proba(self, X):
        return self._forward(X, training=False)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))
