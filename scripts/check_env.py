import sys
try:
    import torch
    print("torch OK")
except ImportError:
    print("torch MISSING")

try:
    import xgboost
    print("xgboost OK")
except ImportError:
    print("xgboost MISSING")

try:
    import pandas as pd
    print("pandas OK")
except ImportError:
    print("pandas MISSING")
