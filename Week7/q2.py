import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

# Define the base directory where the dataset is stored
base_dir = 'cats_and_dogs_filtered/cats_and_dogs_filtered'  # Adjust based on your extracted folder structure

# Correct directory paths
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'validation')

# Print to verify correct paths
print("Train directory:", train_dir)
print("Validation directory:", valid_dir)

# Step 1: Set up data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])

# Step 2: Load the dataset
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

# Step 3: Define a simple neural network for cat-dog classification
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3 * 224 * 224, 1000)  # Assuming image size is 224x224 and 3 color channels
        self.fc2 = nn.Linear(1000, 2)  # Binary classification (cat vs dog)

    def forward(self, x):
        x = x.view(-1, 3 * 224 * 224)  # Flattening the input image
        x = torch.relu(self.fc1(x))  # ReLU activation
        x = self.fc2(x)  # Output layer
        return x

# Step 4: Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)

criterion = nn.CrossEntropyLoss()  # Cross entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # No weight_decay for L1 in optimizer

# Step 5: L1 Regularization using weight_decay in optimizer
def train_with_l1_weight_decay(model, train_loader, criterion, optimizer, num_epochs=5, lambda_l1=0.01):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Add L1 regularization term to the loss
            l1_loss = l1_regularization_loss(model, lambda_l1)
            total_loss = loss + l1_loss

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/5], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

# Step 6: L1 Regularization using a loop to calculate L1 norm of weights
def l1_regularization_loss(model, lambda_l1=0.01):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))  # L1 norm of each weight
    return lambda_l1 * l1_loss

def train_with_manual_l1_regularization(model, train_loader, criterion, optimizer, lambda_l1=0.01, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Add L1 regularization manually
            l1_loss = l1_regularization_loss(model, lambda_l1)
            total_loss = loss + l1_loss

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/5], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

# Step 7: Train the model with L1 regularization via optimizer weight_decay
print("Training with L1 regularization (optimizer weight_decay):")
train_with_l1_weight_decay(model, train_loader, criterion, optimizer)

# Step 8: Train the model with L1 regularization manually
print("\nTraining with L1 regularization (manual):")
train_with_manual_l1_regularization(model, train_loader, criterion, optimizer, lambda_l1=0.01)

# Step 9: Evaluate the model on the validation set
def evaluate_model(model, valid_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Validation Loss: {running_loss / len(valid_loader):.4f}, Accuracy: {accuracy:.2f}%")

# Evaluate the model on validation set
evaluate_model(model, valid_loader, criterion)

# Step 10: Save the trained model
