import torch

a = torch.tensor(2.0, requires_grad=True)  # Example value for a
b = torch.tensor(3.0, requires_grad=False) # b does not require gradient
x = 2 * a + 3 * b
y = 5 * a**2 + 3 * b**3
z = 2 * x + 3 * y

z.backward()  # This computes all the gradients

# Output the gradient dz/da
print("Gradient dz/da:", a.grad)
