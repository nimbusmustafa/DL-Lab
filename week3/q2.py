import torch

# Initialize data
x = torch.tensor([2.0, 4.0], dtype=torch.float32)
y = torch.tensor([20.0, 40.0], dtype=torch.float32)

# Initialize parameters
w = torch.ones(1, requires_grad=True)
b = torch.ones(1, requires_grad=True)

# Learning rate and epochs
lr = 0.001
epochs = 2

# Training loop
for epoch in range(epochs):
    # Prediction
    y_pred = w * x + b

    # Compute the loss (Mean Squared Error)
    loss = torch.mean((y_pred - y) ** 2)

    # Backpropagation
    loss.backward()

    # Print gradients
    print(f"Epoch {epoch + 1}:")
    print(f"w.grad = {w.grad}")
    print(f"b.grad = {b.grad}")

    # Update parameters
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

        # Zero the gradients
        w.grad.zero_()
        b.grad.zero_()

    # Print updated parameters
    print(f"Updated w = {w.item()}, b = {b.item()}\n")
