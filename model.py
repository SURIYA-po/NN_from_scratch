class Model:
    def __init__(self, layers, loss, optimizer):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dLoss):
        for layer in reversed(self.layers):
            dLoss = layer.backward(dLoss)

    def train(self, X, Y, epochs=1000):
        for epoch in range(epochs):
            Y_hat = self.forward(X)
            loss = self.loss.forward(Y_hat, Y)
            dLoss = self.loss.backward()
            self.backward(dLoss)

            params, grads = {}, {}
            for i, layer in enumerate(self.layers):
                if hasattr(layer, "W"):
                    params[f"W{i}"] = layer.W
                    grads[f"W{i}"] = layer.dW

            self.optimizer.update(params, grads)

            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss {loss:.4f}")
