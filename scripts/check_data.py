import pandas as pd
from pathlib import Path

INPUT_DIR = Path('02_processed_data/pepper/')

print("Checking X.csv structure...")
X = pd.read_csv(INPUT_DIR / 'X.csv', nrows=5)
print(X)
print(f"X shape: {X.shape}")
print("Columns:", X.columns.tolist()[:5])  # Show first 5 columns

print("\nChecking y.csv structure...")
y = pd.read_csv(INPUT_DIR / 'y.csv', nrows=5)
print(y)
print(f"y shape: {y.shape}")
print("Columns:", y.columns.tolist())