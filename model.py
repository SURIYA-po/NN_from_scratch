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

    def step(self):
        params = {}
        grads = {}

        for i, layer in enumerate(self.layers):
            if hasattr(layer, "W"):
                params[f"W{i}"] = layer.W
                grads[f"W{i}"] = layer.dW

                params[f"b{i}"] = layer.b
                grads[f"b{i}"] = layer.db

        self.optimizer.update(params, grads)
