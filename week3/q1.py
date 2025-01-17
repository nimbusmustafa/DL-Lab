import torch
import matplotlib.pyplot as plt

# Initialize data
x = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
                  19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2], dtype=torch.float32)
y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
                  16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6], dtype=torch.float32)

# Initialize parameters
w = torch.ones(1, requires_grad=True)
b = torch.ones(1, requires_grad=True)

# Learning rate
lr = 0.001
epochs = 10

# List to track loss values
losses = []

# Training loop
for epoch in range(epochs):
    # Prediction (y = wx + b)
    y_pred = w * x + b

    # Compute the loss (Mean Squared Error)
    loss = torch.mean((y_pred - y) ** 2)

    # Backpropagation
    loss.backward()

    # Update parameters (gradient descent)
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

        # Zero the gradients after the update
        w.grad.zero_()
        b.grad.zero_()

    # Store the loss
    losses.append(loss.item())

    # Print the loss for every 100th epoch
    if epoch % 2 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')

# Plotting the loss over epochs
print(losses)
plt.plot(range(epochs), losses, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.show()
