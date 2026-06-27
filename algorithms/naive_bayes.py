import numpy as np
import matplotlib.pyplot as plt


class GaussianNB:
    """
    Gaussian Naive Bayes classifier (from scratch).

    Assumes each feature is normally distributed within each class.
    """

    def __init__(self):
        self.classes_ = None
        self.priors_ = None      # P(y)
        self.means_ = None       # mean per class, per feature
        self.vars_ = None        # variance per class, per feature

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.means_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[idx] = X_c.mean(axis=0)
            self.vars_[idx] = X_c.var(axis=0)
            self.priors_[idx] = X_c.shape[0] / X.shape[0]

        return self

    @staticmethod
    def _gaussian_pdf(x, mean, var):
        """Probability density of Gaussian."""
        eps = 1e-9
        var = var + eps  # avoid division by zero
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
        exponent = np.exp(-((x - mean) ** 2) / (2.0 * var))
        return coeff * exponent

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # Posterior = prior * likelihood, done in log-space for stability
        posteriors = np.zeros((n_samples, n_classes))

        for idx, c in enumerate(self.classes_):
            prior = np.log(self.priors_[idx])
            likelihood = np.sum(
                np.log(self._gaussian_pdf(X, self.means_[idx], self.vars_[idx])),
                axis=1,
            )
            posteriors[:, idx] = prior + likelihood

        return self.classes_[np.argmax(posteriors, axis=1)]


def make_toy_data(n: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0

    # Two Gaussian blobs with slightly different spreads
    X0 = rng.normal(loc=(-2, -2), scale=1.0, size=(n0, 2))
    X1 = rng.normal(loc=(2, 2), scale=1.5, size=(n1, 2))

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    idx = rng.permutation(n)
    return X[idx], y[idx]


def train_test_split(X, y, test_size=0.25, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    test_n = int(n * test_size)
    return X[idx[test_n:]], X[idx[:test_n]], y[idx[test_n:]], y[idx[:test_n]]


def _evaluate(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    acc = (tp + tn) / len(y_true)
    return {"accuracy": acc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def plot_decision_boundary(model, X_train, y_train, X_test, y_test):
    X_all = np.vstack([X_train, X_test])
    pad = 1.0
    x_min, x_max = X_all[:, 0].min() - pad, X_all[:, 0].max() + pad
    y_min, y_max = X_all[:, 1].min() - pad, X_all[:, 1].max() + pad

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5],
                 colors=["#4c72b0", "#dd8452"], alpha=0.3)
    plt.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.5, linestyles="--")

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm",
                edgecolors="k", s=50, marker="o", label="Train")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm",
                edgecolors="k", s=90, marker="*", label="Test")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("Gaussian Naive Bayes Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()


def demo():
    X, y = make_toy_data(n=300, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, seed=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    m = _evaluate(y_test, y_pred)

    print("Gaussian Naive Bayes")
    print(f"Accuracy: {m['accuracy']:.4f}")
    print(f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")

    # Show learned parameters
    for idx, c in enumerate(model.classes_):
        print(f"\nClass {int(c)}:")
        print(f"  Prior     : {model.priors_[idx]:.4f}")
        print(f"  Means     : {model.means_[idx]}")
        print(f"  Variances : {model.vars_[idx]}")

    plot_decision_boundary(model, X_train, y_train, X_test, y_test)
    plt.show()


if __name__ == "__main__":
    demo()