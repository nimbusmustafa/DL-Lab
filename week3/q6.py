import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Data: Subject X1, X2, Y
x_data = torch.tensor([
    [3.0, 8.0],
    [4.0, 5.0],
    [5.0, 7.0],
    [6.0, 3.0],
    [2.0, 1.0]
], dtype=torch.float32)

y_data = torch.tensor([-3.7, 3.5, 2.5, 11.5, 5.7], dtype=torch.float32).view(-1, 1)


# Define the Multiple Linear Regression Model using nn.Linear
class MultipleLinearRegressionModel(nn.Module):
    def __init__(self):
        super(MultipleLinearRegressionModel, self).__init__()
        # Input size = 2 (X1, X2), output size = 1 (Y)
        self.linear = nn.Linear(2, 1)  # Two input features (X1, X2)

    def forward(self, x):
        return self.linear(x)


# Initialize the model
model = MultipleLinearRegressionModel()

# Define the loss function (Mean Squared Error)
criterion = nn.MSELoss()

# Define the optimizer (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 200
losses = []

for epoch in range(epochs):
    # Forward pass: Compute predicted y
    y_pred = model(x_data)

    # Compute the loss
    loss = criterion(y_pred, y_data)

    # Backward pass and optimization
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update the parameters

    # Store the loss for plotting
    losses.append(loss.item())

    # Print loss every 20 epochs
    if epoch % 20 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

# Plot the loss vs epoch graph
plt.plot(range(epochs), losses, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Epoch vs Loss for Multiple Linear Regression")
plt.legend()
plt.show()

# Print the learned parameters (weights and bias)
w1 = model.linear.weight[0][0].item()
w2 = model.linear.weight[0][1].item()
b = model.linear.bias.item()

print(f"Learned parameters: w1 = {w1}, w2 = {w2}, b = {b}")

# Verifying the model's prediction for X1 = 3, X2 = 2
x_new = torch.tensor([[3.0, 2.0]], dtype=torch.float32)  # New data point for verification
y_new_pred = model(x_new)

print(f"Prediction for X1=3, X2=2: {y_new_pred.item()}")
