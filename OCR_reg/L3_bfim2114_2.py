import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.core.fromnumeric")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.core._methods")


class GradientDescentLinearRegressionBinaryClassifier:
    def __init__(self, learning_rate=0.00000001, n_iterations=10000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
        self.loss_history = []

    def _mse_loss(self, X, y, theta):
        return np.mean((X @ theta - y) ** 2)

    def fit(self, x, y):
        y = np.array(y).reshape(-1, 1)
        m = x.shape[0]
        x_b = np.c_[np.ones((m, 1)), x]
        self.theta = np.zeros((x_b.shape[1], 1))

        for _ in range(self.n_iterations):
            gradients = 2 * x_b.T.dot(x_b.dot(self.theta) - y)
            self.theta -= self.learning_rate * gradients
            current_loss = self._mse_loss(x_b, y, self.theta)
            self.loss_history.append(current_loss)

    def predict(self, x):
        x_b = np.c_[np.ones((x.shape[0], 1)), x]
        y_pred = x_b.dot(self.theta)
        return (y_pred > 0.5).astype(int)

    def get_loss_history(self):
        return self.loss_history
