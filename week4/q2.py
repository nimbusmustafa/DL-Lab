import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Step 1: Define XOR truth table as tensors
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
Y = torch.tensor([0, 1, 1, 0], dtype=torch.float32).view(-1, 1)  # Reshaping Y to be a column vector


# Step 2: Define the XORModel class
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.hidden = nn.Linear(2, 2)  # 2 input neurons, 2 hidden neurons
        self.output = nn.Linear(2, 1)  # 2 hidden neurons, 1 output neuron

    def forward(self, x):
        x = torch.relu(self.hidden(x))  # Apply Sigmoid activation to hidden layer
        x = torch.relu(self.output(x))  # Apply Sigmoid activation to output layer
        return x


# Step 3: Define the Dataset class for XOR data
class MyDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# Step 4: Create DataLoader
full_dataset = MyDataset(X, Y)
batch_size = 1
train_data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

# Step 5: Find the available device (CPU or GPU) and load model to that device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XORModel().to(device)
print(model)

# Step 6: Define the loss function and optimizer
loss_fn = torch.nn.MSELoss()  # Mean Squared Error loss
optimizer = torch.optim.adam(model.parameters(), lr=0.03)  # Stochastic Gradient Descent optimizer

# Step 7: Training loop
epochs = 10000
loss_list = []  # List to store loss values for plotting

for epoch in range(epochs):
    total_loss = 0
    for inputs, targets in train_data_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # Move to device (GPU/CPU)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(inputs)

        # Compute the loss
        loss = loss_fn(predictions, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Store the average loss for this epoch
    loss_list.append(total_loss / len(train_data_loader))

    # Print the loss every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_data_loader):.4f}')

# Step 8: Plot the loss curve
plt.plot(loss_list)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

# Step 9: Model Inference (Test the model)
with torch.no_grad():
    model.eval()  # Set model to evaluation mode
    for inputs in X:
        inputs = inputs.to(device)  # Move input to device
        output = model(inputs)
        print(f'Input: {inputs.cpu().numpy()}, Predicted Output: {output.cpu().numpy()}')

# Step 10: Verify and print all learnable parameters
print("\nLearnable parameters of the model:")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data}")

# Number of learnable parameters
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\nTotal number of learnable parameters: {params}')
