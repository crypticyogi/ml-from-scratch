import numpy as np
import matplotlib.pyplot as plt


class StandardScaler:
    """Minimal standard scaler so the example is fully self-contained."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class LogisticRegressionGD:
    """
    Binary Logistic Regression trained with Batch Gradient Descent.

    Parameters
    ----------
    lr : float
        Learning rate.
    epochs : int
        Maximum passes over the training set.
    C : float
        Inverse of L2 regularization strength. Use ``np.inf`` to disable.
    tol : float
        Stop early if loss improvement is below this threshold.
    fit_intercept : bool
        Whether to learn a bias term.
    verbose : int
        If > 0, prints loss every ``verbose`` epochs.
    """

    def __init__(
        self,
        lr: float = 0.1,
        epochs: int = 2000,
        C: float = 1.0,
        tol: float = 1e-4,
        fit_intercept: bool = True,
        verbose: int = 0,
    ):
        self.lr = lr
        self.epochs = epochs
        self.C = C
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.verbose = verbose

        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid (no overflow/underflow)."""
        out = np.empty_like(z, dtype=np.float64)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[neg])
        out[neg] = exp_z / (1.0 + exp_z)
        return out

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise ValueError("Call fit before prediction.")
        return X @ self.w + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def _loss(self, z: np.ndarray, y: np.ndarray) -> float:
        """
        Stable binary cross-entropy without clipping probabilities:
        loss = mean( max(z,0) - y*z + log1p(exp(-|z|)) ) + L2_term
        """
        pointwise = np.maximum(z, 0.0) - y * z + np.log1p(np.exp(-np.abs(z)))
        reg = 0.0
        if not np.isinf(self.C) and self.C > 0 and self.w is not None:
            reg = (1.0 / (2.0 * self.C)) * np.sum(self.w ** 2)
        return float(np.mean(pointwise) + reg)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be 2-D.")
        if y.ndim != 1:
            raise ValueError("y must be 1-D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0
        self.loss_history.clear()

        reg_coef = 0.0 if np.isinf(self.C) or self.C <= 0 else 1.0 / self.C

        for epoch in range(self.epochs):
            # Forward pass
            z = self.decision_function(X)
            y_prob = self.sigmoid(z)

            # Gradients (BCE + L2)
            error = y_prob - y
            dw = (1.0 / n_samples) * (X.T @ error) + reg_coef * self.w
            db = (1.0 / n_samples) * np.sum(error) if self.fit_intercept else 0.0

            # Update
            self.w -= self.lr * dw
            if self.fit_intercept:
                self.b -= self.lr * db

            # Loss after update
            z = self.decision_function(X)
            loss = self._loss(z, y)
            self.loss_history.append(loss)

            if self.verbose and (epoch + 1) % self.verbose == 0:
                print(f"Epoch {epoch + 1:5d}/{self.epochs}  Loss: {loss:.6f}")

            # Early stopping
            if epoch > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                if self.verbose:
                    print(f"Converged at epoch {epoch + 1}")
                break

        return self


def make_toy_data(n: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0

    X0 = rng.normal(loc=(-2.0, -2.0), scale=1.0, size=(n0, 2))
    X1 = rng.normal(loc=(2.0, 2.0), scale=1.0, size=(n1, 2))

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    idx = rng.permutation(n)
    return X[idx], y[idx]


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Pure-NumPy classification metrics (no external dependencies)."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def plot_decision_boundary(model, X_train, y_train, X_test, y_test, scaler=None):
    X_all = np.vstack([X_train, X_test])
    x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
    y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    if scaler is not None:
        grid = scaler.transform(grid)

    probs = model.predict_proba(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(xx, yy, probs, levels=20, cmap="RdBu_r", alpha=0.7)
    plt.colorbar(contour, label="P(y=1)")

    plt.scatter(
        X_train[:, 0], X_train[:, 1], c=y_train, cmap="RdBu_r",
        edgecolors="k", s=50, marker="o", label="Train",
    )
    plt.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap="RdBu_r",
        edgecolors="k", s=90, marker="*", label="Test",
    )

    plt.title("Decision Surface")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()


def demo():
    # 1. Generate data
    X, y = make_toy_data(n=200, seed=42)

    # 2. Split FIRST (before any scaling or fitting)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    split = int(0.8 * len(y))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # 3. Scale using training statistics only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4. Train (higher lr is safe now because data is scaled)
    model = LogisticRegressionGD(lr=0.5, epochs=2000, C=1e4, tol=1e-6, verbose=200)
    model.fit(X_train_s, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test_s)
    m = _evaluate(y_test, y_pred)

    print("Test set metrics:")
    print(f"Accuracy : {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall   : {m['recall']:.4f}")
    print(f"F1       : {m['f1']:.4f}")
    print(f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")
    print(f"\nFinal loss: {model.loss_history[-1]:.6f}")
    print(f"Learned weights: {model.w.round(4)}, bias: {model.b:.4f}")

    # 6. Plot
    plot_decision_boundary(model, X_train, y_train, X_test, y_test, scaler=scaler)

    plt.figure(figsize=(8, 4))
    plt.plot(model.loss_history, lw=2)
    plt.title("Training Loss (Binary Cross-Entropy)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    demo()