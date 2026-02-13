import torch

# input vector
x = torch.tensor([2.0,3.0])

# weights
W = torch.tensor([0.5,1.0])

# bias
b = torch.tensor(1.0)

# liner tranformation (vecetor)->(scalar)
z = torch.dot(W,x) + b

# activation
y = torch.relu(z)

# print
print("z = ",z)
print("y = ",y)
