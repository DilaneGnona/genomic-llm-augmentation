import pandas as pd
import numpy as np

# Paths to the data files
X_path = "02_processed_data/pepper/X.csv"
y_path = "02_processed_data/pepper/y.csv"
pca_path = "02_processed_data/pepper/pca_covariates.csv"

# Inspect X.csv
print("Inspecting X.csv:")
X_df = pd.read_csv(X_path)
print(f"Shape: {X_df.shape}")
print(f"First few columns: {list(X_df.columns[:5])}")
print(f"Data types: {X_df.dtypes[:5]}")
print(f"First few rows:\n{X_df.head(2)}")
print(f"Contains NaN: {X_df.isna().any().any()}")

# Inspect y.csv
print("\nInspecting y.csv:")
y_df = pd.read_csv(y_path)
print(f"Shape: {y_df.shape}")
print(f"Columns: {list(y_df.columns)}")
print(f"Data types: {y_df.dtypes}")
print(f"First few rows:\n{y_df.head(2)}")
print(f"Contains NaN: {y_df.isna().any().any()}")

# Inspect pca_covariates.csv
print("\nInspecting pca_covariates.csv:")
pca_df = pd.read_csv(pca_path)
print(f"Shape: {pca_df.shape}")
print(f"First few columns: {list(pca_df.columns[:5])}")
print(f"Data types: {pca_df.dtypes[:5]}")
print(f"First few rows:\n{pca_df.head(2)}")
print(f"Contains NaN: {pca_df.isna().any().any()}")

# Check for index column issues
print("\nChecking for index column issues:")
if 'Unnamed: 0' in X_df.columns:
    print("X.csv has an unnamed index column")
    print(f"Index column values: {X_df['Unnamed: 0'].head(5).tolist()}")

if 'Unnamed: 0' in y_df.columns:
    print("y.csv has an unnamed index column")
    print(f"Index column values: {y_df['Unnamed: 0'].head(5).tolist()}")

if 'Unnamed: 0' in pca_df.columns:
    print("pca_covariates.csv has an unnamed index column")
    print(f"Index column values: {pca_df['Unnamed: 0'].head(5).tolist()}")