"""
Fix preprocessing issues for pepper and ipk_out_raw datasets
Removes metadata rows and aligns X and y data
"""
import os
import pandas as pd
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def fix_pepper_dataset():
    """Fix pepper dataset - remove metadata rows (POS, REF, ALT)"""
    print("="*70)
    print("FIXING PEPPER DATASET")
    print("="*70)
    
    proc_dir = os.path.join(BASE, '02_processed_data', 'pepper')
    
    # Read X.csv in chunks to handle large file
    x_path = os.path.join(proc_dir, 'X.csv')
    print(f"\nReading {x_path}...")
    
    # Read full file but filter out metadata rows
    print("This may take a few minutes (4.3 GB file)...")
    df = pd.read_csv(x_path, low_memory=False)
    print(f"Original shape: {df.shape}")
    
    # Remove metadata rows - keep only rows where Sample_ID starts with SAMEA
    mask = df['Sample_ID'].astype(str).str.startswith('SAMEA', na=False)
    df_clean = df[mask].copy()
    print(f"After removing metadata rows: {df_clean.shape}")
    
    # Convert feature columns to numeric
    feature_cols = [c for c in df_clean.columns if c != 'Sample_ID']
    print(f"Converting {len(feature_cols)} feature columns to numeric...")
    for col in feature_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Save cleaned X
    output_path = os.path.join(proc_dir, 'X_cleaned.csv')
    print(f"\nSaving cleaned X to: {output_path}")
    df_clean.to_csv(output_path, index=False)
    print("✓ Saved X_cleaned.csv")
    
    # Also save as parquet
    parquet_path = os.path.join(proc_dir, 'X_cleaned.parquet')
    df_clean.to_parquet(parquet_path, index=False)
    print("✓ Saved X_cleaned.parquet")
    
    # Read and align y.csv
    y_path = os.path.join(proc_dir, 'y.csv')
    y_df = pd.read_csv(y_path)
    print(f"\ny.csv shape: {y_df.shape}")
    
    # Find common samples
    x_samples = set(df_clean['Sample_ID'])
    y_samples = set(y_df['Sample_ID'])
    common_samples = x_samples.intersection(y_samples)
    
    print(f"\nX samples: {len(x_samples)}")
    print(f"Y samples: {len(y_samples)}")
    print(f"Common samples: {len(common_samples)}")
    
    # Filter to common samples
    df_aligned = df_clean[df_clean['Sample_ID'].isin(common_samples)].sort_values('Sample_ID')
    y_aligned = y_df[y_df['Sample_ID'].isin(common_samples)].sort_values('Sample_ID')
    
    # Save aligned files
    x_aligned_path = os.path.join(proc_dir, 'X_aligned.csv')
    y_aligned_path = os.path.join(proc_dir, 'y_aligned.csv')
    
    df_aligned.to_csv(x_aligned_path, index=False)
    y_aligned.to_csv(y_aligned_path, index=False)
    
    print(f"\n✓ Saved X_aligned.csv ({len(df_aligned)} samples)")
    print(f"✓ Saved y_aligned.csv ({len(y_aligned)} samples)")
    
    # Check Yield_BV
    if 'Yield_BV' in y_aligned.columns:
        non_null = y_aligned['Yield_BV'].notna().sum()
        print(f"\nYield_BV non-null values: {non_null}/{len(y_aligned)}")
    
    return df_aligned, y_aligned

def fix_ipk_dataset():
    """Fix ipk_out_raw dataset - just align X and y"""
    print("\n" + "="*70)
    print("FIXING IPK_OUT_RAW DATASET")
    print("="*70)
    
    proc_dir = os.path.join(BASE, '02_processed_data', 'ipk_out_raw')
    
    # Read X.csv
    x_path = os.path.join(proc_dir, 'X.csv')
    print(f"\nReading {x_path}...")
    df = pd.read_csv(x_path)
    print(f"X shape: {df.shape}")
    
    # Check SNP values
    feature_cols = [c for c in df.columns if c != 'Sample_ID']
    sample_vals = df[feature_cols[0]].unique()
    print(f"Unique values in first SNP: {sorted(sample_vals)[:10]}")
    
    # Read y.csv
    y_path = os.path.join(proc_dir, 'y.csv')
    y_df = pd.read_csv(y_path)
    print(f"y.csv shape: {y_df.shape}")
    print(f"y.csv columns: {list(y_df.columns)}")
    
    # Find common samples
    x_samples = set(df['Sample_ID'])
    y_samples = set(y_df['Sample_ID'])
    common_samples = x_samples.intersection(y_samples)
    
    print(f"\nX samples: {len(x_samples)}")
    print(f"Y samples: {len(y_samples)}")
    print(f"Common samples: {len(common_samples)}")
    
    # Filter to common samples
    df_aligned = df[df['Sample_ID'].isin(common_samples)].sort_values('Sample_ID')
    y_aligned = y_df[y_df['Sample_ID'].isin(common_samples)].sort_values('Sample_ID')
    
    # Save aligned files
    x_aligned_path = os.path.join(proc_dir, 'X_aligned.csv')
    y_aligned_path = os.path.join(proc_dir, 'y_aligned.csv')
    
    df_aligned.to_csv(x_aligned_path, index=False)
    y_aligned.to_csv(y_aligned_path, index=False)
    
    print(f"\n✓ Saved X_aligned.csv ({len(df_aligned)} samples)")
    print(f"✓ Saved y_aligned.csv ({len(y_aligned)} samples)")
    
    # Check phenotype
    if 'Yield_BV' in y_aligned.columns:
        non_null = y_aligned['Yield_BV'].notna().sum()
        print(f"Yield_BV non-null: {non_null}/{len(y_aligned)}")
    elif 'YR_LS' in y_aligned.columns:
        non_null = y_aligned['YR_LS'].notna().sum()
        print(f"YR_LS non-null: {non_null}/{len(y_aligned)}")
    
    return df_aligned, y_aligned

def main():
    print("="*70)
    print("FIXING PREPROCESSING - VCF TO CSV")
    print("="*70)
    print()
    
    # Fix pepper
    try:
        print("Starting pepper dataset fix...")
        x_pepper, y_pepper = fix_pepper_dataset()
        print("\n✓ Pepper dataset fixed!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Fix ipk
    try:
        print("\nStarting ipk_out_raw dataset fix...")
        x_ipk, y_ipk = fix_ipk_dataset()
        print("\n✓ IPK dataset fixed!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("FIX COMPLETE!")
    print("="*70)
    print("""
New files created:
- 02_processed_data/pepper/X_cleaned.csv
- 02_processed_data/pepper/X_cleaned.parquet
- 02_processed_data/pepper/X_aligned.csv
- 02_processed_data/pepper/y_aligned.csv
- 02_processed_data/ipk_out_raw/X_aligned.csv
- 02_processed_data/ipk_out_raw/y_aligned.csv

Use X_aligned.csv and y_aligned.csv for training.
""")

if __name__ == '__main__':
    main()
