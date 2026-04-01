import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import build_model

def test_transformer():
    input_dim = 102
    try:
        model = build_model("transformer", input_dim=input_dim)
        if model is None:
            print("Model is None")
            return
        print(f"Model built: {model}")
        x = torch.randn(32, input_dim)
        y = model(x)
        print(f"Output shape: {y.shape}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_transformer()
