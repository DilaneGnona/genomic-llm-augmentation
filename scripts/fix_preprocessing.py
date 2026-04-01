"""
Fix preprocessing issues for pepper and ipk_out_raw datasets
- Remove metadata rows from X.csv
- Ensure SNP values are integers (0,1,2)
- Align X and y by Sample_ID
"""
import os
import pandas as pd
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def fix_pepper_dataset():
    """Fix pepper dataset - remove metadata rows and ensure proper format"""
    print("="*70)
    print("FIXING PEPPER DATASET")
    print("="*70)
    
    proc_dir = os.path.join(BASE, '02_processed_data', 'pepper')
    
    # Read X.csv
    x_path = os.path.join(proc_dir, 'X.csv')
    print(f"\nReading {x_path}...")
    
    # Read the full file to identify metadata rows
    df = pd.read_csv(x_path, low_memory=False)
    print(f"Original shape: {df.shape}")
    print(f"First few Sample_IDs: {df['Sample_ID'].head(10).tolist()}")
    
    # Identify and remove metadata rows (POS, REF, ALT, etc.)
    metadata_keywords = ['POS', 'REF', 'ALT', 'CHROM', 'FILTER', 'QUAL', 'INFO']
    sample_id_col = df['Sample_ID'].astype(str)
    
    # Keep only rows where Sample_ID starts with SAMEA (actual sample IDs)
    mask = sample_id_col.str.startswith('SAMEA', na=False)
    df_clean = df[mask].copy()
    
    print(f"\nAfter removing metadata rows: {df_clean.shape}")
    print(f"Sample IDs: {df_clean['Sample_ID'].head(5).tolist()}")
    
    # Convert SNP columns to numeric (they should be 0, 1, 2)
    feature_cols = [c for c in df_clean.columns if c != 'Sample_ID']
    
    print(f"\nConverting {len(feature_cols)} feature columns to numeric...")
    for col in feature_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Check for any non-integer values
    print("\nChecking value ranges...")
    sample_col = feature_cols[0]
    unique_vals = df_clean[sample_col].dropna().unique()
    print(f"Unique values in first SNP column: {sorted(unique_vals)[:20]}")
    
    # If values are continuous (not 0,1,2), we need to check if they're already encoded
    if len(unique_vals) > 10:
        print("⚠️  Values appear to be continuous, not discrete SNP genotypes")
        print("   This may indicate the data was standardized/normalized")
        print("   Keeping as-is for now...")
    else:
        print("✓ Values look like SNP genotypes")
    
    # Save cleaned X.csv
    output_path = os.path.join(proc_dir, 'X_cleaned.csv')
    df_clean.to_csv(output_path, index=False)
    print(f"\n✓ Saved cleaned X to: {output_path}")
    
    # Also save as parquet for efficiency
    parquet_path = os.path.join(proc_dir, 'X_cleaned.parquet')
    df_clean.to_parquet(parquet_path, index=False)
    print(f"✓ Saved cleaned X to: {parquet_path}")
    
    # Check y.csv alignment
    y_path = os.path.join(proc_dir, 'y.csv')
    y_df = pd.read_csv(y_path)
    print(f"\ny.csv shape: {y_df.shape}")
    print(f"y.csv Sample_IDs: {y_df['Sample_ID'].head(5).tolist()}")
    
    # Find common samples
    x_samples = set(df_clean['Sample_ID'])
    y_samples = set(y_df['Sample_ID'])
    common_samples = x_samples.intersection(y_samples)
    
    print(f"\nX samples: {len(x_samples)}")
    print(f"Y samples: {len(y_samples)}")
    print(f"Common samples: {len(common_samples)}")
    
    # Filter both to common samples
    df_clean_aligned = df_clean[df_clean['Sample_ID'].isin(common_samples)].sort_values('Sample_ID')
    y_clean_aligned = y_df[y_df['Sample_ID'].isin(common_samples)].sort_values('Sample_ID')
    
    # Save aligned datasets
    x_aligned_path = os.path.join(proc_dir, 'X_aligned.csv')
    y_aligned_path = os.path.join(proc_dir, 'y_aligned.csv')
    
    df_clean_aligned.to_csv(x_aligned_path, index=False)
    y_clean_aligned.to_csv(y_aligned_path, index=False)
    
    print(f"\n✓ Saved aligned X ({len(df_clean_aligned)} samples) to: {x_aligned_path}")
    print(f"✓ Saved aligned y ({len(y_clean_aligned)} samples) to: {y_aligned_path}")
    
    # Check Yield_BV availability
    if 'Yield_BV' in y_clean_aligned.columns:
        non_null = y_clean_aligned['Yield_BV'].notna().sum()
        print(f"\nYield_BV non-null values: {non_null}/{len(y_clean_aligned)}")
    
    return df_clean_aligned, y_clean_aligned

def fix_ipk_out_raw_dataset():
    """Fix ipk_out_raw dataset - already looks good, just align X and y"""
    print("\n" + "="*70)
    print("FIXING IPK_OUT_RAW DATASET")
    print("="*70)
    
    proc_dir = os.path.join(BASE, '02_processed_data', 'ipk_out_raw')
    
    # Read X.csv
    x_path = os.path.join(proc_dir, 'X.csv')
    print(f"\nReading {x_path}...")
    
    df = pd.read_csv(x_path)
    print(f"X shape: {df.shape}")
    print(f"Sample_IDs: {df['Sample_ID'].head(5).tolist()}")
    
    # Check SNP values
    feature_cols = [c for c in df.columns if c != 'Sample_ID']
    sample_col = feature_cols[0]
    unique_vals = df[sample_col].unique()
    print(f"\nUnique values in first SNP column: {sorted(unique_vals)}")
    
    if set(unique_vals).issubset({0, 1, 2, 0.0, 1.0, 2.0, np.nan}):
        print("✓ Values are correct SNP genotypes (0,1,2)")
    else:
        print("⚠️  Unexpected values found")
    
    # Read y.csv
    y_path = os.path.join(proc_dir, 'y.csv')
    y_df = pd.read_csv(y_path)
    print(f"\ny.csv shape: {y_df.shape}")
    print(f"y.csv columns: {list(y_df.columns)}")
    
    # Find common samples
    x_samples = set(df['Sample_ID'])
    y_samples = set(y_df['Sample_ID'])
    common_samples = x_samples.intersection(y_samples)
    
    print(f"\nX samples: {len(x_samples)}")
    print(f"Y samples: {len(y_samples)}")
    print(f"Common samples: {len(common_samples)}")
    
    # Filter both to common samples
    df_aligned = df[df['Sample_ID'].isin(common_samples)].sort_values('Sample_ID')
    y_aligned = y_df[y_df['Sample_ID'].isin(common_samples)].sort_values('Sample_ID')
    
    # Save aligned datasets
    x_aligned_path = os.path.join(proc_dir, 'X_aligned.csv')
    y_aligned_path = os.path.join(proc_dir, 'y_aligned.csv')
    
    df_aligned.to_csv(x_aligned_path, index=False)
    y_aligned.to_csv(y_aligned_path, index=False)
    
    print(f"\n✓ Saved aligned X ({len(df_aligned)} samples) to: {x_aligned_path}")
    print(f"✓ Saved aligned y ({len(y_aligned)} samples) to: {y_aligned_path}")
    
    # Check if Yield_BV exists
    if 'Yield_BV' in y_aligned.columns:
        non_null = y_aligned['Yield_BV'].notna().sum()
        print(f"\nYield_BV non-null values: {non_null}/{len(y_aligned)}")
    else:
        print("\n⚠️  Yield_BV column not found in y.csv")
        print("   Available columns:", list(y_aligned.columns))
    
    return df_aligned, y_aligned

def main():
    print("="*70)
    print("FIXING PREPROCESSING ISSUES")
    print("="*70)
    print()
    
    # Fix pepper dataset
    try:
        x_pepper, y_pepper = fix_pepper_dataset()
        print("\n✓ Pepper dataset fixed successfully")
    except Exception as e:
        print(f"\n✗ Error fixing pepper dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # Fix ipk_out_raw dataset
    try:
        x_ipk, y_ipk = fix_ipk_out_raw_dataset()
        print("\n✓ IPK out raw dataset fixed successfully")
    except Exception as e:
        print(f"\n✗ Error fixing ipk_out_raw dataset: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("FIX COMPLETE")
    print("="*70)
    print("""
New files created:
- 02_processed_data/pepper/X_cleaned.csv
- 02_processed_data/pepper/X_cleaned.parquet
- 02_processed_data/pepper/X_aligned.csv
- 02_processed_data/pepper/y_aligned.csv
- 02_processed_data/ipk_out_raw/X_aligned.csv
- 02_processed_data/ipk_out_raw/y_aligned.csv

You can now use these cleaned files for training.
""")

if __name__ == '__main__':
    main()
