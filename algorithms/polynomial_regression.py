import numpy as np
import matplotlib.pyplot as plt


class PolynomialFeatures:
    """
    Generate polynomial features from a single input feature.
    X_poly = [1, x, x^2, x^3, ..., x^degree]
    """

    def __init__(self, degree: int = 2):
        if degree < 0:
            raise ValueError("degree must be >= 0")
        self.degree = degree

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X).reshape(-1, 1)
        X_poly = np.ones((X.shape[0], self.degree + 1))
        for i in range(1, self.degree + 1):
            X_poly[:, i] = X[:, 0] ** i
        return X_poly


class LinearRegressionClosedForm:
    """
    Simple linear regression using the Normal Equation.
    theta = (X^T X)^{-1} X^T y
    """

    def __init__(self):
        self.theta = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        # Add bias column if not present
        if X.shape[1] == 1:
            X = np.c_[np.ones(X.shape[0]), X]
        self.theta = np.linalg.pinv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] == 1:
            X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.theta


def make_sine_data(n: int = 50, noise_std: float = 0.3, seed: int = 42):
    """Generate y = sin(x) + noise"""
    rng = np.random.default_rng(seed)
    X = np.linspace(0, 2 * np.pi, n)
    y = np.sin(X) + rng.normal(0, noise_std, size=n)
    return X, y


def plot_fit(X_train, y_train, X_test, y_test, X_plot, y_plot, degree, title_suffix=""):
    plt.figure(figsize=(12, 5))

    # Data + fit
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, c="steelblue", edgecolors="k", s=50, label="Train", zorder=3)
    plt.scatter(X_test, y_test, c="forestgreen", edgecolors="k", s=70, marker="s", label="Test", zorder=3)
    plt.plot(X_plot, y_plot, "r-", lw=2, label=f"Degree {degree} Fit", zorder=2)
    plt.plot(X_plot, np.sin(X_plot), "k--", lw=1.5, label="True sin(x)", zorder=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Polynomial Fit (degree={degree}) {title_suffix}")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.4)

    # Residuals
    plt.subplot(1, 2, 2)
    y_pred = np.interp(X_train, X_plot, y_plot)  # approximate prediction on train points
    residuals = y_train - y_pred
    plt.scatter(y_pred, residuals, c="purple", edgecolors="k", s=50)
    plt.axhline(0, color="red", ls="--", lw=1.5)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()


def demo():
    X, y = make_sine_data(n=30, noise_std=0.3, seed=42)

    # Train/test split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    split = int(0.7 * len(X))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Smooth curve for plotting
    X_plot = np.linspace(0, 2 * np.pi, 200)

    # Try different degrees
    for degree in [1, 3, 10]:
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.transform(X_train)
        X_plot_poly = poly.transform(X_plot)

        model = LinearRegressionClosedForm()
        model.fit(X_train_poly, y_train)

        y_plot = model.predict(X_plot_poly)
        y_test_pred = model.predict(poly.transform(X_test))
        train_mse = np.mean((model.predict(X_train_poly) - y_train) ** 2)
        test_mse = np.mean((y_test_pred - y_test) ** 2)

        print(f"\nDegree {degree}:")
        print(f"  Train MSE: {train_mse:.4f}")
        print(f"  Test MSE : {test_mse:.4f}")

        title_suffix = f"(Train MSE={train_mse:.3f}, Test MSE={test_mse:.3f})"
        plot_fit(X_train, y_train, X_test, y_test, X_plot, y_plot, degree, title_suffix)

    plt.show()


if __name__ == "__main__":
    demo()