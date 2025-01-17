import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Data: X and y
x_data = torch.tensor([1, 5, 10, 10, 25, 50, 70, 75, 100], dtype=torch.float32).view(-1, 1)  # Reshape for 1 feature
y_data = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32).view(-1, 1)  # Labels


# Define the Logistic Regression Model using nn.Linear
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        # Input size = 1 (x), output size = 1 (y), use sigmoid activation
        self.linear = nn.Linear(1, 1)  # One input feature

    def forward(self, x):
        # Apply the sigmoid function to output probabilities
        return torch.sigmoid(self.linear(x))


# Initialize the model
model = LogisticRegressionModel()

# Define the loss function (Binary Cross-Entropy Loss)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for classification

# Define the optimizer (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 100
losses = []

for epoch in range(epochs):
    # Forward pass: Compute predicted y (probability)
    y_pred = model(x_data)

    # Compute the loss
    loss = criterion(y_pred, y_data)

    # Backward pass and optimization
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update the parameters

    # Store the loss for plotting
    losses.append(loss.item())

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

# Plot the loss vs epoch graph
plt.plot(range(epochs), losses, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Epoch vs Loss for Logistic Regression")
plt.legend()
plt.show()

# Print the learned parameters (weights and bias)
w = model.linear.weight.item()
b = model.linear.bias.item()
print(f"Learned parameters: w = {w}, b = {b}")

# Prediction for new values (e.g., X=60)
x_new = torch.tensor([[60.0]], dtype=torch.float32)
y_new_pred = model(x_new)

print(f"Prediction for X=60: {y_new_pred.item()}")
