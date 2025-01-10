import torch

w = torch.tensor(1.0, requires_grad=True)  # Example value for w
x = torch.tensor(2.0,requires_grad=True)  # Example value for x
b = torch.tensor(1.0,requires_grad=True)  # Example value for b

def manual(x1):
    return max(x1,0)
# Forward pass
u = w * x
v = u + b
a = torch.relu(v)

# Compute the gradient da/dw
a.backward()

# Output the gradient da/dw
print("Gradient da/dw:", w.grad)
print("Gradient da/dw manually:", manual(x))

