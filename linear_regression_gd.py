import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionGD:
    """
    Linear Regression trained with Batch Gradient Descent.
    Supports single- and multi-feature inputs.
    Model: y_hat = X @ w + b
    """

    def __init__(
        self,
        lr: float = 0.01,
        epochs: int = 2000,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        alpha: float = 0.0,
        verbose: int = 0,
    ):
        self.lr = lr
        self.epochs = epochs
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.alpha = alpha          # L2 regularization strength
        self.verbose = verbose

        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.loss_history: list[float] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise ValueError("Call fit before prediction.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.w + self.b

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.loss_history.clear()

        for epoch in range(self.epochs):
            y_pred = self.predict(X)

            # Gradients of MSE
            error = y_pred - y
            dw = (2.0 / n_samples) * (X.T @ error)
            db = (2.0 / n_samples) * np.sum(error) if self.fit_intercept else 0.0

            # L2 regularization (don't regularize bias)
            if self.alpha > 0:
                dw += 2.0 * self.alpha * self.w

            # Parameter update
            self.w -= self.lr * dw
            if self.fit_intercept:
                self.b -= self.lr * db

            # Loss after update (includes L2 penalty for monitoring)
            loss = self.mse(y, self.predict(X))
            if self.alpha > 0:
                loss += self.alpha * np.sum(self.w ** 2)
            self.loss_history.append(loss)

            if self.verbose and (epoch + 1) % self.verbose == 0:
                print(f"Epoch {epoch + 1:5d}/{self.epochs}  Loss: {loss:.6f}")

            # Early stopping
            if epoch > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                if self.verbose:
                    print(f"Converged at epoch {epoch + 1}")
                break

        return self


def normal_equation(X: np.ndarray, y: np.ndarray):
    """Closed-form Ordinary Least Squares (for verification)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
    return theta[1:], float(theta[0])  # w, b


def demo():
    rng = np.random.default_rng(42)
    n = 100

    # True relationship: y = 3x + 2 + noise
    X = np.linspace(0, 10, n)
    noise = rng.normal(0, 2.0, size=n)
    y = 3.0 * X + 2.0 + noise

    # Train / test split
    idx = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit
    model = LinearRegressionGD(lr=0.01, epochs=2000, tol=1e-7, alpha=0.0, verbose=0)
    model.fit(X_train, y_train)

    # Benchmark against closed-form solution
    w_ne, b_ne = normal_equation(X_train, y_train)
    print(f"Normal Eq:  w={w_ne[0]:.4f}, b={b_ne:.4f}")
    print(f"GD:         w={model.w[0]:.4f}, b={model.b:.4f}")
    print(f"Final MSE (train): {model.loss_history[-1]:.4f}")

    # Generalization
    y_pred_test = model.predict(X_test)
    print(f"Test MSE:   {model.mse(y_test, y_pred_test):.4f}")

    # Plotting
    X_plot = np.linspace(X.min(), X.max(), 200)

    plt.figure(figsize=(12, 5))

    # Fit plot
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, c="steelblue", edgecolors="k", s=50, label="Train", zorder=3)
    plt.scatter(X_test, y_test, c="forestgreen", edgecolors="k", s=70, marker="s", label="Test", zorder=3)
    plt.plot(X_plot, model.predict(X_plot), "r-", lw=2, label="GD Fit", zorder=2)
    plt.plot(X_plot, 3.0 * X_plot + 2.0, "k--", lw=1.5, label="True Line", zorder=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.4)

    # Loss curve
    plt.subplot(1, 2, 2)
    plt.plot(model.loss_history, lw=2)
    plt.title("Training Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo()