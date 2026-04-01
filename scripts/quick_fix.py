"""Quick fix for pepper dataset"""
import pandas as pd
import os

BASE = r'c:\Users\OMEN\Desktop\experiment_snp'
proc_dir = os.path.join(BASE, '02_processed_data', 'pepper')

# Read only first 20 rows to check structure
x_path = os.path.join(proc_dir, 'X.csv')
print('Reading first 20 rows of X.csv...')
df = pd.read_csv(x_path, nrows=20, low_memory=False)
print(f'Shape: {df.shape}')
print(f'Columns (first 5): {list(df.columns[:5])}')
print(f'Sample_ID values: {df["Sample_ID"].tolist()}')

# Keep only rows where Sample_ID starts with SAMEA
mask = df['Sample_ID'].astype(str).str.startswith('SAMEA', na=False)
df_clean = df[mask].copy()
print(f'\nAfter filtering: {df_clean.shape}')
print(f'Clean Sample_IDs: {df_clean["Sample_ID"].tolist()}')
