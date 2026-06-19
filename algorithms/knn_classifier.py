import numpy as np
import matplotlib.pyplot as plt


class StandardScaler:
    """Minimal standard scaler — critical for distance-based methods."""

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


class KNNClassifier:
    """
    k-Nearest Neighbors classifier (vectorized brute-force).

    Parameters
    ----------
    k : int
        Number of neighbors.
    weights : {'uniform', 'distance'}
        'uniform' = majority vote.
        'distance' = weight neighbors by inverse distance.
    """

    def __init__(self, k: int = 5, weights: str = "uniform"):
        if k <= 0:
            raise ValueError("k must be >= 1")
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")
        self.k = k
        self.weights = weights
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y).reshape(-1)

        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if self.k > self.X_train.shape[0]:
            raise ValueError(f"k={self.k} > n_samples={self.X_train.shape[0]}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None:
            raise ValueError("Call fit before predict")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Vectorized Euclidean distances:
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2a·b
        test_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)          # (n_test, 1)
        train_sq = np.sum(self.X_train ** 2, axis=1).reshape(1, -1)  # (1, n_train)
        cross_term = X @ self.X_train.T                           # (n_test, n_train)
        dists = np.sqrt(np.maximum(test_sq + train_sq - 2 * cross_term, 0.0))

        # k nearest neighbors
        nn_idx = np.argsort(dists, axis=1)[:, : self.k]           # (n_test, k)
        nn_labels = self.y_train[nn_idx]                          # (n_test, k)
        nn_dists = np.take_along_axis(dists, nn_idx, axis=1)      # (n_test, k)

        # Guard against division by zero in distance weighting
        nn_dists = np.where(nn_dists == 0, 1e-10, nn_dists)

        preds = np.empty(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            labels = nn_labels[i]
            if self.weights == "uniform":
                values, counts = np.unique(labels, return_counts=True)
                # Tie-break: highest count wins; ties go to smaller label
                order = np.lexsort((values, -counts))
                preds[i] = values[order[0]]
            else:  # distance-weighted
                w = 1.0 / nn_dists[i]
                values = np.unique(labels)
                best_val, best_score = -1, -1.0
                for v in values:
                    score = np.sum(w[labels == v])
                    if score > best_score or (abs(score - best_score) < 1e-12 and v < best_val):
                        best_score = score
                        best_val = v
                preds[i] = best_val

        return preds


def make_toy_data(n: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0
    X0 = rng.normal(loc=(-2, -2), scale=1.2, size=(n0, 2))
    X1 = rng.normal(loc=(2, 2), scale=1.2, size=(n1, 2))
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


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def plot_decision_boundary(model, X_train, y_train, X_test, y_test, scaler=None):
    X_all = np.vstack([X_train, X_test])
    pad = 1.0
    x_min, x_max = X_all[:, 0].min() - pad, X_all[:, 0].max() + pad
    y_min, y_max = X_all[:, 1].min() - pad, X_all[:, 1].max() + pad

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                         np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if scaler is not None:
        grid = scaler.transform(grid)

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
    plt.title(f"KNN Decision Boundary (k={model.k}, weights='{model.weights}')")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()


def demo():
    X, y = make_toy_data(n=300, seed=7)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, seed=7)

    # Feature scaling is critical for KNN (distance-based)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = KNNClassifier(k=7, weights="uniform")
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    m = _evaluate(y_test, y_pred)

    print(f"KNN (k={model.k}, weights='{model.weights}')")
    print(f"Accuracy : {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall   : {m['recall']:.4f}")
    print(f"F1       : {m['f1']:.4f}")
    print(f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")

    # Plot in original feature space
    plot_decision_boundary(model, X_train, y_train, X_test, y_test, scaler=scaler)

    plt.show()


if __name__ == "__main__":
    demo()