import torch
import matplotlib.pyplot as plt


class RegressionModel:
    def __init__(self):
        # Initialize parameters (w and b) to 1
        self.w = torch.ones(1, requires_grad=True)
        self.b = torch.ones(1, requires_grad=True)

    def forward(self, x):
        # Compute the forward pass (wx + b)
        return self.w * x + self.b

    def update(self, learning_rate):
        # Update parameters using gradient descent
        with torch.no_grad():
            self.w -= learning_rate * self.w.grad
            self.b -= learning_rate * self.b.grad
        # Reset gradients after the update
        self.reset_grad()

    def reset_grad(self):
        # Reset the gradients to zero
        self.w.grad.zero_()
        self.b.grad.zero_()

    def criterion(self, y, yp):
        # Compute Mean Squared Error Loss
        return torch.mean((yp - y) ** 2)


# Training Data
x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0], dtype=torch.float32)
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0], dtype=torch.float32)

# Learning rate
learning_rate = torch.tensor(0.001)

# Create an instance of the RegressionModel
model = RegressionModel()

# List to track loss values
losses = []

# Training loop (100 epochs)
epochs = 100
for epoch in range(epochs):
    # Forward pass: Compute predicted y
    y_pred = model.forward(x)

    # Compute the loss
    loss = model.criterion(y, y_pred)

    # Backpropagation: Compute gradients
    loss.backward()

    # Update parameters (w and b)
    model.update(learning_rate)

    # Store the loss value for plotting
    losses.append(loss.item())

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')

# Plot the loss curve
plt.plot(range(epochs), losses, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Epoch vs Loss")
plt.legend()
plt.show()
