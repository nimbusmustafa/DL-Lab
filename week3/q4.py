import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# Define a custom Dataset class
class LinearRegressionDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


# Define the model class extending nn.Module
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        # Initialize parameters w and b as tensors, which will be learned
        self.w = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Linear regression formula
        return self.w * x + self.b


# Training data
x_data = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0], dtype=torch.float32)
y_data = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0], dtype=torch.float32)

# Create Dataset and DataLoader
dataset = LinearRegressionDataset(x_data, y_data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Initialize the model
model = RegressionModel()

# Loss function: Mean Squared Error (MSE)
criterion = nn.MSELoss()

# Optimizer: Stochastic Gradient Descent (SGD)
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop
epochs = 101
losses = []

for epoch in range(epochs):
    epoch_loss = 0.0
    for x_batch, y_batch in dataloader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: Compute predicted y
        y_pred = model(x_batch)

        # Compute the loss
        loss = criterion(y_pred, y_batch)

        # Backpropagation: Compute gradients
        loss.backward()

        # Update the parameters (w and b)
        optimizer.step()

        # Accumulate loss for the epoch
        epoch_loss += loss.item()

    # Average loss for the epoch
    epoch_loss /= len(dataloader)
    losses.append(epoch_loss)

    # Print the loss for every 10th epoch
    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_loss}')

# Plotting the loss curve
plt.plot(range(epochs), losses, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Epoch vs Loss")
plt.legend()
plt.show()

# Print the final learned parameters
print(f"Learned parameters: w = {model.w.item()}, b = {model.b.item()}")
