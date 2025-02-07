import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the CNN model (same as in MNIST)
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
            x = x.view(-1, 128 * 3 * 3)  # This assumes the size is 128x3x3

            # Fully connected layer 1 -> ReLU activation
            x = torch.relu(self.fc1(x))

            # Apply dropout after the fully connected layer to prevent overfitting
            x = self.dropout(x)

            # Fully connected layer 2 (output layer)
            x = self.fc2(x)
            return x


# Step 1: Load FashionMNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

mnist_testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

# Step 2: Load the pre-trained model from disk
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the entire model (which includes both architecture and trained parameters)
model = torch.load("./ModelFiles/model.pt")
model.to(device)  # Move the model to the correct device (GPU/CPU)

# Step 3: Print model state_dict (inspect parameters)
print("Model's state_dict:")
for param_tensor in model.state_dict().keys():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
print()

# Step 4: Evaluate the model on the FashionMNIST test set
model.eval()  # Set model to evaluation mode

correct = 0
total = 0
for i, (inputs, labels) in enumerate(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Perform forward pass
    outputs = model(inputs)

    # Get predicted class label (highest value in the output layer)
    _, predicted = torch.max(outputs, 1)

    # Print the true and predicted labels
    print("True label:{}".format(labels))
    print('Predicted: {}'.format(predicted))

    # Calculate total number of labels
    total += labels.size(0)

    # Calculate total correct predictions
    correct += (predicted == labels).sum()

# Calculate and print accuracy
accuracy = 100.0 * correct / total
print("The overall accuracy is {:.2f}%".format(accuracy))
