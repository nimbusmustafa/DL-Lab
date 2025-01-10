import torch

x = torch.tensor(2.0, requires_grad=True)  # Example value for x

# Define the function y = 8x^4 + 3x^3 + 7x^2 + 6x + 3
y = 8*x**4 + 3*x**3 + 7*x**2 + 6*x + 3
y.backward()

print("PyTorch computed gradient:", x.grad)

# Analytical gradient: 32x^3 + 9x^2 + 14x + 6
analytical_grad = 32*x**3 + 9*x**2 + 14*x + 6

# Output the analytical gradient
print("Analytical gradient:", analytical_grad)
