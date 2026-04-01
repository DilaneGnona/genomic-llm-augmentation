import os
import pandas as pd
import numpy as np

# Directories
BASE_DIR = r"c:\Users\OMEN\Desktop\experiment_snp"
REAL_X_PATH = os.path.join(BASE_DIR, "02_processed_data", "pepper", "X.csv")
SYNTHETIC_SNPS_PATH = os.path.join(BASE_DIR, "04_augmentation", "pepper", "synthetic_snps.csv")
SYNTHETIC_Y_PATH = os.path.join(BASE_DIR, "04_augmentation", "pepper", "synthetic_y.csv")
REAL_Y_PATH = os.path.join(BASE_DIR, "02_processed_data", "pepper", "y.csv")

# Step 1: Load the first 100 columns from real X.csv
print("Loading reference SNP columns from real data...")
# Read only the header row to get column names
real_cols = pd.read_csv(REAL_X_PATH, nrows=0).columns.tolist()

# Take the first 100 SNP columns
reference_cols = real_cols[:100]
print(f"Selected {len(reference_cols)} reference SNP columns")
print(f"First few reference columns: {reference_cols[:5]}")

# Step 2: Load synthetic SNPs data
print("\nLoading synthetic SNPs data...")
synthetic_df = pd.read_csv(SYNTHETIC_SNPS_PATH)
print(f"Synthetic SNPs shape before alignment: {synthetic_df.shape}")
print(f"First few synthetic columns: {synthetic_df.columns.tolist()[:5]}")

# Step 3: Align columns
print("\nAligning synthetic SNP columns with reference...")

# Check if synthetic data has appropriate columns
if len(synthetic_df.columns) >= 100:
    # If synthetic data uses different column names, rename them to match reference
    if not all(col in synthetic_df.columns for col in reference_cols[:len(synthetic_df.columns)]):
        print("Synthetic data has different column names. Renaming columns to match reference...")
        # Create a mapping from old column names to new reference names
        rename_dict = {old_col: new_col for old_col, new_col in 
                      zip(synthetic_df.columns[:len(reference_cols)], reference_cols)}
        synthetic_df = synthetic_df.rename(columns=rename_dict)
    
    # Keep only the reference columns
    synthetic_df_aligned = synthetic_df[reference_cols].copy()
    print(f"Synthetic SNPs shape after alignment: {synthetic_df_aligned.shape}")
    
    # Save the aligned synthetic data
    synthetic_df_aligned.to_csv(SYNTHETIC_SNPS_PATH, index=False)
    print(f"Saved aligned synthetic SNPs to {SYNTHETIC_SNPS_PATH}")
else:
    print(f"Warning: Synthetic data has fewer columns ({len(synthetic_df.columns)}) than required (100)")
    # Still try to align with available columns
    available_reference_cols = reference_cols[:len(synthetic_df.columns)]
    rename_dict = {old_col: new_col for old_col, new_col in 
                  zip(synthetic_df.columns, available_reference_cols)}
    synthetic_df = synthetic_df.rename(columns=rename_dict)
    synthetic_df_aligned = synthetic_df[available_reference_cols].copy()
    synthetic_df_aligned.to_csv(SYNTHETIC_SNPS_PATH, index=False)
    print(f"Saved partially aligned synthetic SNPs to {SYNTHETIC_SNPS_PATH}")

# Step 4: Load real X subset for validation
print("\nLoading real X subset for validation...")
real_ref_X = pd.read_csv(REAL_X_PATH, usecols=reference_cols)
print(f"Real reference X shape: {real_ref_X.shape}")

# Step 5: Load real y and synthetic y for validation
print("\nLoading phenotype data...")
real_y = pd.read_csv(REAL_Y_PATH)
synthetic_y = pd.read_csv(SYNTHETIC_Y_PATH)
print(f"Real y shape: {real_y.shape}")
print(f"Synthetic y shape: {synthetic_y.shape}")

# Step 6: Validate synthetic SNP data
print("\nValidating synthetic SNP data...")

# Separate sample ID column from SNP columns
sample_id_col = synthetic_df_aligned.columns[0]  # First column is sample ID
snp_columns = synthetic_df_aligned.columns[1:]   # All other columns are SNPs

# Check data types for SNP columns only
snp_dtypes = synthetic_df_aligned[snp_columns].dtypes
print(f"Synthetic SNP dtypes: {snp_dtypes.unique()}")

# Check value domain (should be 0/1/2) for SNP columns only
print("Checking unique values in synthetic SNPs...")
# Sample a few SNP columns to check values
if len(snp_columns) > 0:
    sample_cols = snp_columns[:5]  # Check first 5 SNP columns
    for col in sample_cols:
        col_values = synthetic_df_aligned[col].unique()
        print(f"Column {col} unique values: {sorted([int(v) for v in col_values if pd.notna(v)])}")
    
    # Get all unique values across all SNP columns
    all_snp_values = synthetic_df_aligned[snp_columns].values.ravel()
    unique_snp_values = sorted(set([int(v) for v in all_snp_values if pd.notna(v)]))
    print(f"\nAll unique values across all SNP columns: {unique_snp_values}")
else:
    print("No SNP columns found to validate")
    unique_snp_values = []

# Check for NaN values
nan_count = synthetic_df_aligned[snp_columns].isna().sum().sum()
print(f"Number of NaN values in synthetic SNPs: {nan_count}")

# Step 7: Convert SNP columns to integer type if needed
if not all(snp_dtypes == 'int64'):
    print("\nConverting synthetic SNP columns to integer type...")
    # Keep sample ID column as is, convert only SNP columns
    synthetic_df_aligned[snp_columns] = synthetic_df_aligned[snp_columns].astype(int)
    synthetic_df_aligned.to_csv(SYNTHETIC_SNPS_PATH, index=False)
    print(f"Saved integer-typed synthetic SNPs to {SYNTHETIC_SNPS_PATH}")
    # Re-check unique values after conversion
    all_snp_values = synthetic_df_aligned[snp_columns].values.ravel()
    unique_snp_values = sorted(set([int(v) for v in all_snp_values if pd.notna(v)]))

# Generate final report
print("\n=====================================")
print("NORMALIZATION REPORT FOR PEPPER DATASET")
print("=====================================")
print(f"Real reference X shape: {real_ref_X.shape}")
print(f"Synthetic X shape: {synthetic_df_aligned.shape}")
print(f"Real y shape: {real_y.shape}")
print(f"Synthetic y shape: {synthetic_y.shape}")
print(f"\nColumn alignment: {'COMPLETED' if synthetic_df_aligned.columns.tolist() == reference_cols else 'PARTIAL'}")
print(f"Synthetic SNPs data type (excluding sample ID): {snp_dtypes.iloc[0]}")
print(f"Unique values in synthetic SNPs: {unique_snp_values}")
print(f"\nFinal file paths:")
print(f"- Synthetic SNPs: {SYNTHETIC_SNPS_PATH}")
print(f"- Synthetic phenotypes: {SYNTHETIC_Y_PATH}")
print(f"- Augmentation log: {os.path.join(BASE_DIR, '04_augmentation', 'pepper', 'augmentation_log.txt')}")
print("=====================================")