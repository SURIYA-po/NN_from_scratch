import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


from layers import Dense
from activations import ReLU, Sigmoid
from losses import CrossEntropyLoss
from optimizers import Adam
from model import Model


# Generate non-linear dataset
X, Y = make_moons(n_samples=800, noise=0.25)
X = X.T
Y = Y.reshape(1, -1)


# Define model architecture
layers = [
Dense(2, 16),
ReLU(),
Dense(16, 8),
ReLU(),
Dense(8, 1),
Sigmoid()
]


model = Model(
layers=layers,
loss=CrossEntropyLoss(),
optimizer=Adam(lr=0.01)
)


# Training loop
losses = []
for epoch in range(2000):
	Y_hat = model.forward(X)
	loss = model.loss.forward(Y_hat, Y)
	dLoss = model.loss.backward()
	model.backward(dLoss)
	model.step()
	losses.append(loss)

	if epoch % 200 == 0:
		print(f"Epoch {epoch} | Loss: {loss:.4f}")


# Plot training loss
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()