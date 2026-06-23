import numpy as np
import matplotlib.pyplot as plt


class Node:
    """One node in the decision tree."""

    def __init__(self, impurity, num_samples, num_samples_per_class, predicted_class):
        self.impurity = impurity
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0.0
        self.left = None
        self.right = None
        self.info_gain = 0.0


class DecisionTreeClassifier:
    """
    Improved Decision Tree Classifier from scratch.

    Parameters
    ----------
    criterion : {'gini', 'entropy'}
        Function to measure split quality.
    max_depth : int
        Maximum depth of the tree.
    min_samples_split : int
        Minimum samples required to split an internal node.
    min_samples_leaf : int
        Minimum samples required to be at a leaf node.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ):
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None
        self.classes_ = None
        self.n_features_ = None
        self.feature_importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        self.classes_, y_mapped = np.unique(y, return_inverse=True)
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_, dtype=float)
        self.tree_ = self._grow_tree(np.asarray(X), np.asarray(y_mapped))

        total = np.sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ /= total
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.array([self._predict_one(x) for x in X])

    def _impurity(self, y: np.ndarray) -> float:
        """Gini or Entropy impurity."""
        m = len(y)
        if m == 0:
            return 0.0
        p = np.bincount(y, minlength=len(self.classes_)) / m
        if self.criterion == "gini":
            return 1.0 - np.sum(p ** 2)
        # entropy
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """Find the feature and threshold that give the best impurity reduction."""
        m = len(y)
        if m <= 1:
            return None, None, None

        num_parent = np.bincount(y, minlength=len(self.classes_))
        best_impurity = self._impurity(y)
        best_idx, best_thr = None, None

        for idx in range(self.n_features_):
            order = np.argsort(X[:, idx])
            X_sorted, y_sorted = X[order, idx], y[order]

            num_left = np.zeros(len(self.classes_), dtype=int)
            num_right = num_parent.copy()

            for i in range(1, m):
                c = y_sorted[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                if X_sorted[i] == X_sorted[i - 1]:
                    continue

                # Enforce min_samples_leaf
                if i < self.min_samples_leaf or (m - i) < self.min_samples_leaf:
                    continue

                impurity_left = self._impurity(y_sorted[:i])
                impurity_right = self._impurity(y_sorted[i:])
                impurity = (i * impurity_left + (m - i) * impurity_right) / m

                if impurity < best_impurity:
                    best_impurity = impurity
                    best_idx = idx
                    best_thr = (X_sorted[i] + X_sorted[i - 1]) / 2

        return best_idx, best_thr, best_impurity

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        """Recursively build the tree."""
        num_samples_per_class = [np.sum(y == i) for i in range(len(self.classes_))]
        predicted_class = int(np.argmax(num_samples_per_class))
        current_impurity = self._impurity(y)

        node = Node(
            impurity=current_impurity,
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if depth < self.max_depth and len(y) >= self.min_samples_split:
            result = self._best_split(X, y)
            if result[0] is not None:
                idx, thr, child_impurity = result
                left_mask = X[:, idx] < thr
                X_left, y_left = X[left_mask], y[left_mask]
                X_right, y_right = X[~left_mask], y[~left_mask]

                node.feature_index = idx
                node.threshold = thr
                node.info_gain = current_impurity - child_impurity
                self.feature_importances_[idx] += len(y) * node.info_gain

                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)

        return node

    def _predict_one(self, x: np.ndarray):
        """Walk from root to leaf."""
        node = self.tree_
        while node.left:
            if x[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return self.classes_[node.predicted_class]

    def print_tree(self, feature_names=None, class_names=None, node=None, indent=""):
        """Pretty print the tree structure."""
        if node is None:
            node = self.tree_
            if feature_names is None:
                feature_names = [f"x[{i}]" for i in range(self.n_features_)]
            if class_names is None:
                class_names = [str(c) for c in self.classes_]
            print(f"\n{self.criterion.upper()} Tree (max_depth={self.max_depth})")
            print("-" * 50)

        if node.left is None and node.right is None:
            cname = class_names[node.predicted_class]
            print(f"{indent}LEAF → predict={cname}, samples={node.num_samples}, "
                  f"counts={node.num_samples_per_class}")
        else:
            fname = feature_names[node.feature_index]
            print(f"{indent}IF {fname} < {node.threshold:.3f}:")
            print(f"{indent}  THEN (left):")
            self.print_tree(feature_names, class_names, node.left, indent + "    ")
            print(f"{indent}  ELSE (right):")
            self.print_tree(feature_names, class_names, node.right, indent + "    ")


def make_toy_data(n=200, seed=42):
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
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def plot_decision_boundary(model, X_train, y_train, X_test, y_test, title_suffix=""):
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
    plt.title(f"Decision Tree Boundary\n{title_suffix}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()


def demo():
    X, y = make_toy_data(n=200, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, seed=42)

    configs = [
        {"criterion": "gini", "max_depth": 2, "label": "Underfit (depth=2)"},
        {"criterion": "gini", "max_depth": 5, "label": "Good (depth=5)"},
        {"criterion": "gini", "max_depth": 15, "label": "Overfit (depth=15)"},
        {"criterion": "entropy", "max_depth": 5, "label": "Entropy (depth=5)"},
    ]

    for cfg in configs:
        model = DecisionTreeClassifier(
            criterion=cfg["criterion"],
            max_depth=cfg["max_depth"],
            min_samples_split=2,
            min_samples_leaf=1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        m = _evaluate(y_test, y_pred)

        print(f"\n=== {cfg['label']} ===")
        print(f"Accuracy : {m['accuracy']:.4f}")
        print(f"Precision: {m['precision']:.4f}")
        print(f"Recall   : {m['recall']:.4f}")
        print(f"F1       : {m['f1']:.4f}")
        print(f"Feature importances: {model.feature_importances_.round(4)}")

        if cfg["max_depth"] <= 5:
            model.print_tree()

        plot_decision_boundary(
            model, X_train, y_train, X_test, y_test,
            title_suffix=f"{cfg['label']}, acc={m['accuracy']:.3f}"
        )

    # Feature importance bar chart (from last fitted model)
    plt.figure(figsize=(6, 4))
    plt.bar(
        [f"x[{i}]" for i in range(model.n_features_)],
        model.feature_importances_,
        color="teal",
        edgecolor="k",
    )
    plt.title(f"Feature Importances ({model.criterion})")
    plt.ylabel("Importance")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    demo()