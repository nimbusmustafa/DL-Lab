import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Data from the question
x = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
                  19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2], dtype=torch.float32).view(-1, 1)
y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
                  16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6], dtype=torch.float32).view(-1, 1)


# Define the Linear Regression model using nn.Linear
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input feature, one output

    def forward(self, x):
        return self.linear(x)


# Instantiate the model
model = LinearRegressionModel()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop
epochs = 100
losses = []

for epoch in range(epochs):
    # Forward pass
    y_pred = model(x)

    # Compute the loss
    loss = criterion(y_pred, y)

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
plt.title("Epoch vs Loss for Linear Regression")
plt.legend()
plt.show()

# Print the learned parameters (w and b)
print(f"Learned parameters: w = {model.linear.weight.item()}, b = {model.linear.bias.item()}")
