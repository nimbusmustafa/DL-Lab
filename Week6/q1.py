import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


# Define a more complex CNN model with Batch Normalization and ReLU activation
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # Convolutional layer 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization after conv1
        # Convolutional layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization after conv2
        # Convolutional layer 3 (additional layer for more complexity)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Batch normalization after conv3
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)  # Dropout regularization to reduce overfitting


    def forward(self, x):
            # Conv1 -> BatchNorm -> ReLU -> MaxPool
            x = self.pool(torch.relu(self.bn1(self.conv1(x))))
            # Conv2 -> BatchNorm -> ReLU -> MaxPool
            x = self.pool(torch.relu(self.bn2(self.conv2(x))))
            # Conv3 -> BatchNorm -> ReLU -> MaxPool
            x = self.pool(torch.relu(self.bn3(self.conv3(x))))

            # Flatten the tensor to feed into fully connected layers
            x = x.view(-1, 128 * 3 * 3)  # This assumes the size is 128x7x7

            # Fully connected layer 1 -> ReLU activation
            x = torch.relu(self.fc1(x))

            # Apply dropout after the fully connected layer to prevent overfitting
            x = self.dropout(x)

            # Fully connected layer 2 (output layer)
            x = self.fc2(x)
            return x


# Set up data loaders for FashionMNIST with data augmentation
transform_train = transforms.Compose([
    transforms.RandomRotation(10),  # Random rotation between -10 and 10 degrees
    transforms.RandomHorizontalFlip(),  # Randomly flip the images
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Use FashionMNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, optimizer, and learning rate scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier().to(device)  # Move the model to the GPU or CPU

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce the learning rate by a factor of 0.5 every 5 epochs

# Training the model
num_epochs = 15  # Increased number of epochs for better training
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # Zero the gradients

        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    scheduler.step()  # Update the learning rate
    train_accuracy = 100 * correct / total
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

# Evaluate on the test set
model.eval()  # Set model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Save the trained model
torch.save(model, "./ModelFiles/model.pt")
print("Optimized model saved to ./ModelFiles/model.pt")
