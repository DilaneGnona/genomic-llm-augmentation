import torch
import torch.nn as nn
print(f"Torch version: {torch.__version__}")
model = nn.Linear(10, 1)
x = torch.randn(1, 10)
y = model(x)
print(f"Output shape: {y.shape}")
