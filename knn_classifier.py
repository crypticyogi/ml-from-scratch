import numpy as np
import matplotlib.pyplot as plt


class KNNClassifier:
    """
    k-Nearest Neighbors classifier (from scratch).

    - Stores training data (lazy learning)
    - Predicts by majority vote among k nearest points
    """

    def __init__(self, k: int = 5):
        if k <= 0:
            raise ValueError("k must be >= 1")
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y).reshape(-1)
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        preds = [self._predict_one(x) for x in X]
        return np.array(preds, dtype=int)

    def _predict_one(self, x: np.ndarray) -> int:
        # Euclidean distances to all training points
        dists = np.linalg.norm(self.X_train - x, axis=1)

        # indices of k nearest neighbors
        nn_idx = np.argsort(dists)[: self.k]
        nn_labels = self.y_train[nn_idx]

        # majority vote
        values, counts = np.unique(nn_labels, return_counts=True)
        return int(values[np.argmax(counts)])


def make_toy_data(n: int = 300, seed: int = 42):
    """
    Make a simple 2D dataset with 2 classes (two blobs).
    """
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
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def plot_points(X, y, title):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")


def demo():
    X, y = make_toy_data(n=300, seed=7)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, seed=7)

    model = KNNClassifier(k=7)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = float(np.mean(y_pred == y_test))
    print(f"KNN (k={model.k}) test accuracy: {acc:.4f}")

    plot_points(X_train, y_train, "Training data")
    plot_points(X_test, y_test, "Test data (true labels)")
    plot_points(X_test, y_pred, "Test data (predicted labels)")

    plt.show()


if __name__ == "__main__":
    demo()
