import numpy as np

class CrossEntropyLoss:
    def forward(self, Y_hat, Y):
        self.Y_hat = Y_hat
        self.Y = Y
        m = Y.shape[1]
        return -np.sum(Y * np.log(Y_hat + 1e-9)) / m

    def backward(self):
        return self.Y_hat - self.Y
