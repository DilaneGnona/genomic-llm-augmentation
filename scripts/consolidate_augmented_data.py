import os
import pandas as pd
import glob

def consolidate_ipk_data():
    print("--- CONSOLIDATING IPK AUGMENTED DATA ---")
    
    # 1. Load Real Data
    real_x = pd.read_csv("02_processed_data/ipk_out_raw/X.csv")
    real_y = pd.read_csv("02_processed_data/ipk_out_raw/y.csv")
    
    # Align real data
    real_df = real_x.merge(real_y, on="Sample_ID")
    target_col = "YR_LS"
    
    # 2. Find all synthetic files
    aug_base = "04_augmentation/ipk_out_raw/model_sources"
    synth_files = glob.glob(os.path.join(aug_base, "*", "context_*", "synth_*.csv"))
    
    dfs = [real_df]
    
    for f in synth_files:
        try:
            sdf = pd.read_csv(f)
            # Basic cleaning: ensure target col is numeric and SNPs are int
            sdf[target_col] = pd.to_numeric(sdf[target_col], errors='coerce')
            sdf = sdf.dropna(subset=[target_col])
            
            # Identify SNP columns (starting with SNP_)
            snp_cols = [c for c in sdf.columns if c.startswith("SNP_")]
            for c in snp_cols:
                sdf[c] = pd.to_numeric(sdf[c], errors='coerce').fillna(0).astype(int)
            
            model_name = f.split(os.sep)[-3]
            ctx_name = f.split(os.sep)[-2]
            sdf['Source'] = f"{model_name}_{ctx_name}"
            dfs.append(sdf)
            print(f"  Added {len(sdf)} samples from {model_name} ({ctx_name})")
        except Exception as e:
            print(f"  Error reading {f}: {e}")

    # 3. Final merge
    # We use the SNPs from the context file as our feature set
    context_file = "04_augmentation/ipk_out_raw/context_learning/contexts/ipk_out_raw_context_stats.csv"
    ctx_df = pd.read_csv(context_file)
    context_snps = [c for c in ctx_df.columns if c.startswith("SNP_")]
    print(f"  Target SNP count from context: {len(context_snps)}")
    
    keep_cols = ['Sample_ID', target_col] + context_snps
    
    final_dfs = []
    for d in dfs:
        # Fill missing SNPs with 0 (or median, but 0 is safe for SNPs)
        for c in context_snps:
            if c not in d.columns:
                d[c] = 0
        final_dfs.append(d[keep_cols])
    
    final_df = pd.concat(final_dfs, axis=0, ignore_index=True)
    
    # Save consolidated data
    out_dir = "02_processed_data/ipk_out_raw_augmented"
    os.makedirs(out_dir, exist_ok=True)
    final_df.to_csv(os.path.join(out_dir, "augmented_combined.csv"), index=False)
    
    print(f"\nSUCCESS: Consolidated {len(final_df)} samples (Real + Synth)")
    print(f"File saved to: {out_dir}/augmented_combined.csv")

if __name__ == "__main__":
    consolidate_ipk_data()
