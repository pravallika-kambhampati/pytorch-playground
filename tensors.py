"""
Tensors, similar to arrays and matrices. 
To encode the inputs and outputs of a model, as well as the model’s parameters.
Can run on GPUs or other hardware accelerators. 
Tensors and NumPy arrays can often share the same underlying memory, eliminating the need to copy data.
Tensors are also optimized for automatic differentiation.
"""
import torch
import numpy as np

# Initializing a Tensor 

# 1. Directly from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# 2. From a numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 3. From another tensor:
# The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# 4. With random or constant values:
# shape is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# Attributes of a Tensor
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


"""
Operations on Tensors
https://pytorch.org/docs/stable/torch.html
Each of these operations can be run on the GPU (at typically higher speeds than on a CPU).
By default, tensors are created on the CPU. We need to explicitly move tensors to the GPU using .to method (after checking for GPU availability). 
"""
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
else:
    print('nay')

print(tensor.device)

# Standard numpy-like indexing and slicing:

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# Joining tensors along a dimension
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Matrix multiplication btw two tensors
shape = (5,5,)
tensor = torch.ones(shape)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

print(y1)
print(y2)
print(y3)

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# Single-element tensors If you have a one-element tensor, 
# for example by aggregating all values of a tensor into one value, you can convert it to a Python numerical value using item():
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place operations Operations that store the result into the operand are called in-place. They are denoted by a _ suffix. 
# For example: x.copy_(y), x.t_(), will change x.
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# Bridge with NumPy
# Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# change in the tensor reflects in the NumPy array
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

# Changes in the NumPy array reflects in the tensor.
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")