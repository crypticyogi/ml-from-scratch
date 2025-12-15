import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionGD:
    """
    Linear Regression trained with Batch Gradient Descent.
    Model: y_hat = w * x + b
    """

    def __init__(self, lr: float = 0.01, epochs: int = 2000):
        self.lr = lr
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0
        self.loss_history = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.w * X + self.b

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        # Ensure 1D arrays
        X = X.reshape(-1)
        y = y.reshape(-1)

        n = X.shape[0]

        for _ in range(self.epochs):
            y_pred = self.predict(X)

            # Gradients for MSE loss
            dw = (-2.0 / n) * np.sum(X * (y - y_pred))
            db = (-2.0 / n) * np.sum(y - y_pred)

            # Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # Track loss
            self.loss_history.append(self.mse(y, y_pred))

        return self


def demo():
    # Create synthetic data: y = 3x + 2 + noise
    rng = np.random.default_rng(42)
    X = np.linspace(0, 10, 80)
    noise = rng.normal(0, 2.0, size=X.shape[0])
    y = 3 * X + 2 + noise

    model = LinearRegressionGD(lr=0.01, epochs=2000)
    model.fit(X, y)

    print(f"Learned parameters: w={model.w:.4f}, b={model.b:.4f}")
    print(f"Final MSE: {model.loss_history[-1]:.4f}")

    # Plot fit
    plt.figure()
    plt.scatter(X, y)
    plt.plot(X, model.predict(X))
    plt.title("Linear Regression (Gradient Descent)")

    # Plot loss curve
    plt.figure()
    plt.plot(model.loss_history)
    plt.title("Training Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")

    plt.show()


if __name__ == "__main__":
    demo()
