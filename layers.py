import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2 / input_dim)
        self.b = np.zeros((output_dim, 1))

    def forward(self, A):
        self.A_prev = A
        return self.W @ A + self.b

    def backward(self, dZ):
        m = self.A_prev.shape[1]
        self.dW = (1/m) * dZ @ self.A_prev.T
        self.db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        return self.W.T @ dZ
