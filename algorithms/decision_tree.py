import numpy as np
import matplotlib.pyplot as plt


class Node:
    """One node in the decision tree."""

    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0.0
        self.left = None
        self.right = None


class DecisionTreeClassifier:
    """
    Basic Decision Tree Classifier (binary splits, Gini impurity).
    """

    def __init__(self, max_depth: int = 3, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None
        self.classes_ = None
        self.n_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        # Remap class labels to 0, 1, ... for internal use
        self.classes_, y_mapped = np.unique(y, return_inverse=True)
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(np.asarray(X), np.asarray(y_mapped))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.array([self._predict_one(x) for x in X])

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        """Gini impurity: 1 - sum(p_i^2). Lower is purer."""
        m = len(y)
        if m == 0:
            return 0.0
        return 1.0 - np.sum((np.bincount(y) / m) ** 2)

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """Find the feature and threshold that give the best Gini reduction."""
        m = len(y)
        if m <= 1:
            return None, None

        num_parent = np.bincount(y, minlength=len(self.classes_))
        best_gini = 1.0 - np.sum((num_parent / m) ** 2)
        best_idx, best_thr = None, None

        # Try every feature
        for idx in range(self.n_features_):
            # Sort by this feature
            order = np.argsort(X[:, idx])
            X_sorted, y_sorted = X[order, idx], y[order]

            num_left = np.zeros(len(self.classes_), dtype=int)
            num_right = num_parent.copy()

            # Try every possible split point between distinct values
            for i in range(1, m):
                c = y_sorted[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                # Skip if values are identical (no valid threshold between them)
                if X_sorted[i] == X_sorted[i - 1]:
                    continue

                gini_left = 1.0 - np.sum((num_left / i) ** 2)
                gini_right = 1.0 - np.sum((num_right / (m - i)) ** 2)
                gini = (i * gini_left + (m - i) * gini_right) / m

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (X_sorted[i] + X_sorted[i - 1]) / 2

        return best_idx, best_thr

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        """Recursively build the tree."""
        num_samples_per_class = [np.sum(y == i) for i in range(len(self.classes_))]
        predicted_class = int(np.argmax(num_samples_per_class))

        node = Node(
            gini=self._gini(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        # Split only if we haven't hit the stopping criteria
        if depth < self.max_depth and len(y) >= self.min_samples_split:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                left_mask = X[:, idx] < thr
                X_left, y_left = X[left_mask], y[left_mask]
                X_right, y_right = X[~left_mask], y[~left_mask]

                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)

        return node

    def _predict_one(self, x: np.ndarray):
        """Walk down the tree."""
        node = self.tree_
        while node.left:
            if x[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return self.classes_[node.predicted_class]


def make_toy_data(n: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0
    X0 = rng.normal(loc=(-2, -2), scale=1.0, size=(n0, 2))
    X1 = rng.normal(loc=(2, 2), scale=1.0, size=(n1, 2))
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

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                         np.linspace(y_min, y_max, 250))
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
    plt.title(f"Decision Tree Boundary (max_depth={model.max_depth})")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()


def demo():
    X, y = make_toy_data(n=200, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, seed=42)

    model = DecisionTreeClassifier(max_depth=3, min_samples_split=2)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    m = _evaluate(y_test, y_pred)

    print(f"Decision Tree (max_depth={model.max_depth})")
    print(f"Accuracy: {m['accuracy']:.4f}")
    print(f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")

    plot_decision_boundary(model, X_train, y_train, X_test, y_test)
    plt.show()


if __name__ == "__main__":
    demo()