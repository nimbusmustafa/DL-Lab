import torch

x = torch.tensor(1.0, requires_grad=True)  # Example value for x

f_x = torch.exp(-x**2 - 2*x - torch.sin(x))

f_x.backward()

print("PyTorch computed gradient:", x.grad)

# Analytical gradient: exp(-x^2 - 2x - sin(x)) * (-2x - 2 - cos(x))
analytical_grad = torch.exp(-x**2 - 2*x - torch.sin(x)) * (-2*x - 2 - torch.cos(x))

# Output the analytical gradient
print("Analytical gradient:", analytical_grad)
