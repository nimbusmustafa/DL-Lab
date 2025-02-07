import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Define the base directory where the dataset is stored
base_dir = 'cats_and_dogs_filtered/cats_and_dogs_filtered'  # Adjust based on your extracted folder structure

# Correct directory paths
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'validation')

# Print to verify correct paths
print("Train directory:", train_dir)
print("Validation directory:", valid_dir)

# Image Preprocessing and Augmentation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),  # Crop the image to 227x227
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Pre-trained AlexNet normalization
])

# Load the datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)

# Dataloaders for batching
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Load the pre-trained AlexNet model
model = models.alexnet(pretrained=True)

# Freeze all layers in AlexNet to use transfer learning (so only the classifier part will be trained)
for param in model.parameters():
    param.requires_grad = False

# Modify the classifier to suit the binary classification problem (cats vs. dogs)
model.classifier[6] = nn.Linear(in_features=4096, out_features=2)  # Change to 2 output classes

# Move the model to the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

# Train the model
epochs = 10  # Number of epochs
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Track running loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print statistics
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")

# Evaluate the model on the validation set
model.eval()  # Set model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # No need to track gradients during validation
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Get predicted class label
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate and print accuracy on the validation set
accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")
