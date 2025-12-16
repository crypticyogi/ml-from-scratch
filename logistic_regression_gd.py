import numpy as np
import matplotlib.pyplot as plt


class LogisticRegressionGD:
    """
    Binary Logistic Regression trained with Batch Gradient Descent.
    Model: p(y=1|x) = sigmoid(w^T x + b)
    """

    def __init__(self, lr: float = 0.1, epochs: int = 2000):
        self.lr = lr
        self.epochs = epochs
        self.w = None  # shape: (n_features,)
        self.b = 0.0
        self.loss_history = []

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        # numerically stable sigmoid
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w + self.b
        return self.sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.epochs):
            y_prob = self.predict_proba(X)

            # gradients
            dw = (1.0 / n_samples) * (X.T @ (y_prob - y))
            db = (1.0 / n_samples) * np.sum(y_prob - y)

            # update
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # loss
            self.loss_history.append(self.binary_cross_entropy(y, y_prob))

        return self


def make_toy_data(n: int = 200, seed: int = 42):
    """
    Create a simple 2D linearly separable dataset.
    Two Gaussian blobs.
    """
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0

    X0 = rng.normal(loc=(-2, -2), scale=1.0, size=(n0, 2))
    X1 = rng.normal(loc=(2, 2), scale=1.0, size=(n1, 2))

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    # shuffle
    idx = rng.permutation(n)
    return X[idx], y[idx]


def plot_decision_boundary(model: LogisticRegressionGD, X: np.ndarray, y: np.ndarray):
    # Create a grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, probs, levels=20)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("Logistic Regression Decision Surface (probability of class 1)")
    plt.xlabel("x1")
    plt.ylabel("x2")


def demo():
    X, y = make_toy_data(n=200, seed=42)

    model = LogisticRegressionGD(lr=0.1, epochs=2000)
    model.fit(X, y)

    # Train/test split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    split = int(0.8 * len(y))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = LogisticRegressionGD(lr=0.1, epochs=2000)
    model.fit(X_train, y_train)

    from metrics import classification_report
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    print("Test set metrics:")
    print(f"Accuracy : {report['accuracy']:.4f}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall   : {report['recall']:.4f}")
    print(f"F1       : {report['f1']:.4f}")
    print(f"TP={report['tp']} TN={report['tn']} FP={report['fp']} FN={report['fn']}")

    print(f"Final loss: {model.loss_history[-1]:.6f}")
    print(f"Learned weights: {model.w}, bias: {model.b:.4f}")

    # Plot decision surface + data
    plot_decision_boundary(model, X, y)

    # Plot loss curve
    plt.figure()
    plt.plot(model.loss_history)
    plt.title("Training Loss (Binary Cross-Entropy)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.show()


if __name__ == "__main__":
    demo()
