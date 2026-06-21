import numpy as np
import matplotlib.pyplot as plt


class StandardScaler:
    """Z-score scaling: critical for polynomial features."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class PolynomialFeatures:
    """
    Generate polynomial features from a single input feature.
    Returns [1, x, x^2, ..., x^degree]
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


class PolynomialRegression:
    """
    Polynomial Regression using the Normal Equation.
    Includes feature scaling and optional Ridge (L2) regularization.
    """

    def __init__(self, degree: int = 2, alpha: float = 0.0):
        """
        Parameters
        ----------
        degree : int
            Polynomial degree.
        alpha : float
            Ridge regularization strength. alpha=0 is pure OLS.
        """
        self.degree = degree
        self.alpha = alpha
        self.poly = PolynomialFeatures(degree)
        self.scaler = StandardScaler()
        self.theta = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X).reshape(-1)
        y = np.asarray(y).reshape(-1)

        # Build polynomial features (includes bias column of ones)
        X_poly = self.poly.transform(X)

        # Scale ONLY the raw polynomial terms, NOT the bias column
        X_scaled = X_poly.copy()
        if self.degree > 0:
            X_scaled[:, 1:] = self.scaler.fit_transform(X_poly[:, 1:])

        # Ridge closed-form: (X^T X + alpha*I)^{-1} X^T y
        # Do NOT regularize the bias term (index 0)
        reg_matrix = np.eye(X_scaled.shape[1])
        reg_matrix[0, 0] = 0.0

        self.theta = np.linalg.pinv(X_scaled.T @ X_scaled + self.alpha * reg_matrix) @ X_scaled.T @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X).reshape(-1)
        X_poly = self.poly.transform(X)

        X_scaled = X_poly.copy()
        if self.degree > 0:
            X_scaled[:, 1:] = self.scaler.transform(X_poly[:, 1:])

        return X_scaled @ self.theta


def make_sine_data(n: int = 50, noise_std: float = 0.3, seed: int = 42):
    """Generate y = sin(x) + noise"""
    rng = np.random.default_rng(seed)
    X = np.linspace(0, 2 * np.pi, n)
    y = np.sin(X) + rng.normal(0, noise_std, size=n)
    return X, y


def plot_fit(X_train, y_train, X_test, y_test, X_plot, y_plot,
             y_train_pred, degree, train_mse, test_mse, theta):
    """Three-panel plot: fit, residuals, and coefficient magnitudes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Data + Fit
    ax = axes[0]
    ax.scatter(X_train, y_train, c="steelblue", edgecolors="k", s=50, label="Train", zorder=3)
    ax.scatter(X_test, y_test, c="forestgreen", edgecolors="k", s=70, marker="s", label="Test", zorder=3)
    ax.plot(X_plot, y_plot, "r-", lw=2, label=f"Degree {degree} Fit", zorder=2)
    ax.plot(X_plot, np.sin(X_plot), "k--", lw=1.5, label="True sin(x)", zorder=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Fit (degree={degree})\nTrain MSE={train_mse:.4f}, Test MSE={test_mse:.4f}")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)

    # 2. Residuals (FIXED: uses actual predictions, not interpolation)
    ax = axes[1]
    residuals = y_train - y_train_pred
    ax.scatter(y_train_pred, residuals, c="purple", edgecolors="k", s=50, alpha=0.7)
    ax.axhline(0, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Plot (Train)")
    ax.grid(True, ls="--", alpha=0.4)

    # 3. Coefficient magnitudes (shows overfitting via exploding weights)
    ax = axes[2]
    if theta is not None and len(theta) > 1:
        coeffs = np.abs(theta[1:])  # exclude bias term
        ax.bar(range(1, len(theta)), coeffs, color="coral", edgecolor="k")
        ax.set_xlabel("Power of x")
        ax.set_ylabel("|Coefficient|")
        ax.set_title("Coefficient Magnitudes (excl. bias)")
        ax.set_xticks(range(1, len(theta)))
    ax.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    return fig


def plot_complexity_curve(degrees, train_mses, test_mses):
    """Classic bias-variance visualization."""
    plt.figure(figsize=(8, 5))
    plt.plot(degrees, train_mses, "o-", color="steelblue", label="Train MSE", lw=2, markersize=8)
    plt.plot(degrees, test_mses, "s-", color="forestgreen", label="Test MSE", lw=2, markersize=8)
    plt.xlabel("Polynomial Degree")
    plt.ylabel("MSE")
    plt.title("Model Complexity: Bias-Variance Tradeoff")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.4)
    plt.yscale("log")
    plt.tight_layout()


def demo():
    X, y = make_sine_data(n=30, noise_std=0.3, seed=42)

    # Train/test split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    split = int(0.7 * len(X))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[train_idx]

    X_plot = np.linspace(0, 2 * np.pi, 200)

    degrees = [1, 3, 10]
    train_mses, test_mses = [], []

    for degree in degrees:
        model = PolynomialRegression(degree=degree, alpha=0.0)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_plot = model.predict(X_plot)

        train_mse = float(np.mean((y_train_pred - y_train) ** 2))
        test_mse = float(np.mean((y_test_pred - y_test) ** 2))
        train_mses.append(train_mse)
        test_mses.append(test_mse)

        print(f"\nDegree {degree}:")
        print(f"  Train MSE: {train_mse:.4f}")
        print(f"  Test MSE : {test_mse:.4f}")

        plot_fit(X_train, y_train, X_test, y_test, X_plot, y_plot,
                 y_train_pred, degree, train_mse, test_mse, model.theta)

    # Summary complexity curve
    plot_complexity_curve(degrees, train_mses, test_mses)

    plt.show()


if __name__ == "__main__":
    demo()