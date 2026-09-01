# Install: pip install torch
import torch

tensor1 = torch.tensor([1.0, 2.0, 3.0])
tensor2 = torch.tensor([4.0, 5.0, 6.0])

print("Tensor 1:", tensor1)
print("Tensor 2:", tensor2)

print("\nAddition:", tensor1 + tensor2)
print("Subtraction:", tensor1 - tensor2)
print("Multiplication:", tensor1 * tensor2)
print("Division:", tensor1 / tensor2)
print("Dot Product:", torch.dot(tensor1, tensor2))
