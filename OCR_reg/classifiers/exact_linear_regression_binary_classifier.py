import numpy as np


class ExactLinearRegressionBinaryClassifier:
    def __init__(self, lambda_reg=0.01):
        self.theta_best = None
        self.lambda_reg = lambda_reg

    def fit(self, x, y):
        x_b = np.c_[np.ones((x.shape[0], 1)), x]
        reg_matrix = self.lambda_reg * np.eye(x_b.shape[1])
        self.theta_best = np.linalg.pinv(x_b.T.dot(x_b) + reg_matrix).dot(x_b.T).dot(y)

    def predict(self, x):
        x_b = np.c_[np.ones((x.shape[0], 1)), x]
        y_pred = x_b.dot(self.theta_best)
        return (y_pred > 0.5).astype(int)
