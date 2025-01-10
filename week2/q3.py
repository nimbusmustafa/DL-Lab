import torch
import math
# Define the variables
w = torch.tensor(1.0, requires_grad=True)  # Example value for w
x = torch.tensor(2.0)  # Example value for x
b = torch.tensor(1.0)  # Example value for b
def manual(w,x,b):
    return x*(1/(1+math.exp(-w*x-b)))*(1-(1/(1+math.exp(-w*x-b))))
# Forward pass
u = w * x
v = u + b
a = torch.sigmoid(v)  # Sigmoid activation function

# Compute the gradient da/dw
a.backward()

# Output the gradient da/dw
print("Gradient da/dw:", w.grad)
print("Gradient da/dw manually:", manual(w,x,b))

