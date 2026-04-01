import pandas as pd
import numpy as np
import os
import sys

# Simplified config for prepare_contexts
class SimpleConfig:
    def get_target_column(self, dataset):
        targets = {"ipk_out_raw": "YR_LS", "pepper": "Yield_BV"}
        return targets.get(dataset)

cfg = SimpleConfig()
BASE_DIR = r"c:\Users\OMEN\Desktop\experiment_snp"

def prepare_dataset_contexts(dataset):
    print(f"\n--- Preparing Contexts for {dataset} ---")
    
    # Paths
    processed_dir = os.path.join(BASE_DIR, "02_processed_data", dataset)
    out_dir = os.path.join(BASE_DIR, "04_augmentation", dataset, "context_learning", "contexts")
    os.makedirs(out_dir, exist_ok=True)
    
    # Try different X file names
    x_candidates = ["X.csv", "X_cleaned.csv", "X_mini.csv", "X_aligned.csv"]
    X_df = None
    for x_name in x_candidates:
        x_path = os.path.join(processed_dir, x_name)
        if os.path.exists(x_path):
            try:
                print(f"Loading {x_name} (optimized)...")
                # For pepper, 30k SNPs * 7k samples is ~2GB in memory. 
                # Let's read only the first 1000 samples to avoid MemoryError in this environment
                X_df = pd.read_csv(x_path, header=0, low_memory=False, nrows=1000)
                break
            except Exception as e:
                print(f"  Failed to read {x_name}: {e}")
                continue
    
    if X_df is None:
        print(f"  [!] No X.csv found for {dataset}. Skipping.")
        return
    
    # Load Y
    y_path = os.path.join(processed_dir, "y.csv")
    if not os.path.exists(y_path):
        print(f"  [!] No y.csv found for {dataset}. Skipping.")
        return
    y_df = pd.read_csv(y_path)
    
    # Load PCA
    pca_path = os.path.join(processed_dir, "pca_covariates.csv")
    if not os.path.exists(pca_path):
        print(f"  [!] No pca_covariates.csv found for {dataset}. Skipping.")
        return
    pca_df = pd.read_csv(pca_path)

    # Clean Sample_ID
    X_df['Sample_ID'] = X_df['Sample_ID'].astype(str)
    y_df['Sample_ID'] = y_df['Sample_ID'].astype(str)
    pca_df['Sample_ID'] = pca_df['Sample_ID'].astype(str)
    
    # Check for metadata rows (POS, REF, ALT) in X
    metadata_rows = X_df[X_df['Sample_ID'].str.upper().isin(['POS', 'REF', 'ALT'])].index
    if len(metadata_rows) > 0:
        print(f"  Removing {len(metadata_rows)} metadata rows (POS/REF/ALT)")
        X_df = X_df.drop(metadata_rows).reset_index(drop=True)
    
    # Target column from config
    target_col = cfg.get_target_column(dataset)
    if target_col not in y_df.columns:
        # Fallback to any numeric column
        num_cols = y_df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if "ID" not in c.upper() and "SAMPLE" not in c.upper()]
        if not num_cols:
            print(f"  [!] No numeric target found in y.csv for {dataset}. Skipping.")
            return
        target_col = num_cols[0]
        print(f"  Using fallback target: {target_col}")

    # Merge
    df = pd.merge(X_df, y_df[['Sample_ID', target_col]], on='Sample_ID', how='inner')
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])
    
    print(f"  Shape after merge: {df.shape}")
    
    # Feature Selection (Top SNPs by variance)
    snp_cols = [c for c in X_df.columns if c != 'Sample_ID' and not c.startswith('Unnamed')]
    if len(snp_cols) > 120:
        print(f"  Selecting top 120 SNPs from {len(snp_cols)}...")
        variances = df[snp_cols].var()
        top_snps = variances.sort_values(ascending=False).head(120).index.tolist()
    else:
        top_snps = snp_cols
        
    final_cols = ['Sample_ID'] + top_snps + [target_col]
    df_final = df[final_cols].copy()
    
    # Generate Contexts
    print("  Generating context files (forcing int format)...")
    
    # Cast SNPs to int to avoid .0 in CSV
    df_final[top_snps] = df_final[top_snps].astype(int)
    
    # A. context_stats (Medians)
    median_val = df_final[target_col].median()
    df_final['dist_to_median'] = (df_final[target_col] - median_val).abs()
    stats_df = df_final.sort_values('dist_to_median').head(10).drop(columns=['dist_to_median'])
    stats_df.to_csv(os.path.join(out_dir, f'{dataset}_context_stats.csv'), index=False)
    
    # B. context_high_var (Top & Bottom)
    top_5 = df_final.sort_values(target_col, ascending=False).head(5)
    bot_5 = df_final.sort_values(target_col, ascending=True).head(5)
    high_var_df = pd.concat([top_5, bot_5])
    high_var_df.to_csv(os.path.join(out_dir, f'{dataset}_context_high_var.csv'), index=False)
    
    # C. context_short (Top 5)
    short_df = df_final.sort_values(target_col, ascending=False).head(5)
    short_df.to_csv(os.path.join(out_dir, f'{dataset}_context_short.csv'), index=False)
    
    # D. context_long (Mix 20)
    sorted_df = df_final.sort_values(target_col, ascending=False)
    n = len(sorted_df)
    if n >= 20:
        part_high = sorted_df.head(7)
        part_low = sorted_df.tail(7)
        med_idx = n // 2
        part_med = sorted_df.iloc[med_idx-3 : med_idx+3]
        long_df = pd.concat([part_high, part_med, part_low])
    else:
        long_df = sorted_df
    long_df.to_csv(os.path.join(out_dir, f'{dataset}_context_long.csv'), index=False)
    
    print(f"  Contexts generated for {dataset} in {out_dir}")

def main():
    datasets = ["ipk_out_raw", "pepper"]
    for ds in datasets:
        try:
            print(f"\n[Main] Starting {ds}...")
            prepare_dataset_contexts(ds)
            print(f"[Main] Finished {ds}.")
        except Exception as e:
            print(f"  [!] Error processing {ds}: {e}")

if __name__ == "__main__":
    main()