"""
Align X and y datasets by Sample_ID
"""
import pandas as pd
import os

BASE = r'c:\Users\OMEN\Desktop\experiment_snp'

# Read cleaned X
x_path = os.path.join(BASE, '02_processed_data', 'pepper', 'X_cleaned.csv')
print("Reading X_cleaned.csv...")
df_x = pd.read_csv(x_path)
print(f"X shape: {df_x.shape}")
print(f"X samples: {df_x['Sample_ID'].head(5).tolist()}")

# Read y
y_path = os.path.join(BASE, '02_processed_data', 'pepper', 'y.csv')
print("\nReading y.csv...")
df_y = pd.read_csv(y_path)
print(f"y shape: {df_y.shape}")
print(f"y samples: {df_y['Sample_ID'].head(5).tolist()}")

# Find common samples
x_samples = set(df_x['Sample_ID'])
y_samples = set(df_y['Sample_ID'])
common_samples = x_samples.intersection(y_samples)

print(f"\nX samples: {len(x_samples)}")
print(f"Y samples: {len(y_samples)}")
print(f"Common samples: {len(common_samples)}")

# Filter to common samples
print("\nAligning datasets...")
df_x_aligned = df_x[df_x['Sample_ID'].isin(common_samples)].sort_values('Sample_ID')
df_y_aligned = df_y[df_y['Sample_ID'].isin(common_samples)].sort_values('Sample_ID')

# Save aligned files
x_out = os.path.join(BASE, '02_processed_data', 'pepper', 'X_aligned.csv')
y_out = os.path.join(BASE, '02_processed_data', 'pepper', 'y_aligned.csv')

print(f"\nSaving X_aligned.csv ({len(df_x_aligned)} samples)...")
df_x_aligned.to_csv(x_out, index=False)

print(f"Saving y_aligned.csv ({len(df_y_aligned)} samples)...")
df_y_aligned.to_csv(y_out, index=False)

# Check Yield_BV
if 'Yield_BV' in df_y_aligned.columns:
    non_null = df_y_aligned['Yield_BV'].notna().sum()
    print(f"\nYield_BV non-null values: {non_null}/{len(df_y_aligned)}")

print("\n✓ Done! Files saved:")
print(f"  - {x_out}")
print(f"  - {y_out}")
