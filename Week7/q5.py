import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define base directory where dataset is stored
base_dir = 'cats_and_dogs_filtered/cats_and_dogs_filtered'  # Adjust based on your extracted folder structure

# Correct directory paths
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'validation')

# Verify paths
print("Train directory:", train_dir)
print("Validation directory:", valid_dir)

# Set up image transformations (without data augmentation)
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard for pretrained networks
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)

# DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Define the model architecture
class SimpleCNN(nn.Module):
    def __init__(self, use_dropout=False, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        self.use_dropout = use_dropout
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 18 * 18, 512)
        self.fc2 = nn.Linear(512, 2)  # 2 output classes (cat, dog)

        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)  # PyTorch built-in dropout
        else:
            self.dropout = nn.Identity()  # No dropout

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, kernel_size=2)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout (standard or custom)
        x = self.fc2(x)
        return x

# Training function with Early Stopping
def train_model_with_early_stopping(model, train_loader, valid_loader, num_epochs=5, lr=0.001, patience=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_validation_loss = float('inf')
    current_patience = 0

    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()  # Move to GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        valid_loss = 0.0
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.cuda(), labels.cuda()  # Move to GPU
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_valid += (predicted == labels).sum().item()
                total_valid += labels.size(0)

        valid_loss = valid_loss / len(valid_loader)
        valid_accuracy = 100 * correct_valid / total_valid
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.2f}%")

        # Early stopping logic
        if valid_loss < best_validation_loss:
            best_validation_loss = valid_loss
            current_patience = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Save best model
        else:
            current_patience += 1
            if current_patience > patience:
                print(f"Early stopping! No improvement for {patience} epochs.")
                break

    return train_losses, valid_losses, train_accuracies, valid_accuracies

# Training without Early Stopping (for comparison)
def train_model_without_early_stopping(model, train_loader, valid_loader, num_epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()  # Move to GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        valid_loss = 0.0
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.cuda(), labels.cuda()  # Move to GPU
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_valid += (predicted == labels).sum().item()
                total_valid += labels.size(0)

        valid_loss = valid_loss / len(valid_loader)
        valid_accuracy = 100 * correct_valid / total_valid
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.2f}%")

    return train_losses, valid_losses, train_accuracies, valid_accuracies

# Initialize models
model_with_early_stopping = SimpleCNN(use_dropout=True).cuda()  # Model with early stopping
model_without_early_stopping = SimpleCNN(use_dropout=True).cuda()  # Model without early stopping

# Train models
print("\nTraining Model With Early Stopping...")
train_losses_es, valid_losses_es, train_accuracies_es, valid_accuracies_es = train_model_with_early_stopping(
    model_with_early_stopping, train_loader, valid_loader, num_epochs=10, patience=5
)

print("\nTraining Model Without Early Stopping...")
train_losses_no_es, valid_losses_no_es, train_accuracies_no_es, valid_accuracies_no_es = train_model_without_early_stopping(
    model_without_early_stopping, train_loader, valid_loader, num_epochs=10
)

# Plot the results for comparison
epochs = range(1, 11)

plt.figure(figsize=(12, 6))

# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses_es, label='Train Loss (Early Stopping)')
plt.plot(epochs, valid_losses_es, label='Valid Loss (Early Stopping)')
plt.plot(epochs, train_losses_no_es, label='Train Loss (No Early Stopping)')
plt.plot(epochs, valid_losses_no_es, label='Valid Loss (No Early Stopping)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves Comparison')

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies_es, label='Train Accuracy (Early Stopping)')
plt.plot(epochs, valid_accuracies_es, label='Valid Accuracy (Early Stopping)')
plt.plot(epochs, train_accuracies_no_es, label='Train Accuracy (No Early Stopping)')
plt.plot(epochs, valid_accuracies_no_es, label='Valid Accuracy (No Early Stopping)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curves Comparison')

plt.tight_layout()
plt.show()
