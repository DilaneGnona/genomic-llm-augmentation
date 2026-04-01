import os
import pandas as pd
import glob
import sys

def consolidate_data(dataset, target_col):
    print(f"--- CONSOLIDATING {dataset.upper()} AUGMENTED DATA ---")
    
    # 1. Find all synthetic files FIRST to get the SNP list
    aug_base = f"04_augmentation/{dataset}/model_sources"
    search_pattern = os.path.join(aug_base, "*", "context_*", "synth_*.csv")
    synth_files = glob.glob(search_pattern)
    print(f"DEBUG: Found {len(synth_files)} synthetic files.")

    # 2. Get feature list from context
    context_dir = f"04_augmentation/{dataset}/context_learning/contexts"
    ctx_files = glob.glob(os.path.join(context_dir, "*.csv"))
    if not ctx_files:
        print("Error: No context files found.")
        return
    ctx_df = pd.read_csv(ctx_files[0])
    # Features are everything except Sample_ID and known target columns
    exclude_cols = ['Sample_ID', 'YR_LS', 'Yield_BV', 'Yield', 'YR_precision', 'WGS_BIOSAMPLE_ID']
    feature_cols = [c for c in ctx_df.columns if c not in exclude_cols]
    print(f"DEBUG: Target Feature count: {len(feature_cols)}")

    # 3. Load Real Data (ONLY columns that exist in X.csv)
    print(f"DEBUG: Loading real data for {dataset}...")
    # Peak first to see what columns exist in X.csv
    x_sample = pd.read_csv(f"02_processed_data/{dataset}/X.csv", nrows=1)
    available_cols = set(x_sample.columns)
    
    final_feature_cols = [c for c in feature_cols if c in available_cols]
    print(f"DEBUG: Features available in X.csv: {len(final_feature_cols)}")
    
    real_x = pd.read_csv(f"02_processed_data/{dataset}/X.csv", usecols=['Sample_ID'] + final_feature_cols)
    real_y = pd.read_csv(f"02_processed_data/{dataset}/y.csv")
    real_df = real_x.merge(real_y, on="Sample_ID")
    print(f"DEBUG: Real data loaded, shape {real_df.shape}")
    
    dfs = [real_df]
    
    for f in synth_files:
        try:
            sdf = pd.read_csv(f)
            # Find the actual target column in the synth file
            found_target = None
            for tc in [target_col, 'Yield_BV', 'YR_LS', 'Yield']:
                if tc in sdf.columns:
                    found_target = tc
                    break
            
            if not found_target:
                # Last resort: find any numeric column that isn't a feature
                num_cols = sdf.select_dtypes(include=['number']).columns
                potential_targets = [c for c in num_cols if c not in feature_cols and c != 'Sample_ID']
                if potential_targets:
                    found_target = potential_targets[0]
            
            if not found_target:
                print(f"  Warning: Target column not identified in {f}, skipping.")
                continue

            sdf = sdf.rename(columns={found_target: target_col})
            print(f"DEBUG: Found target {found_target} in {f}, renamed to {target_col}")
            sdf[target_col] = pd.to_numeric(sdf[target_col], errors='coerce')
            sdf = sdf.dropna(subset=[target_col])
            print(f"DEBUG: After cleaning target, {len(sdf)} rows remain in {f}")
            
            # Clean feature columns
            for c in final_feature_cols:
                if c in sdf.columns:
                    sdf[c] = pd.to_numeric(sdf[c], errors='coerce').fillna(0).astype(int)
            
            model_name = f.split(os.sep)[-3]
            ctx_name = f.split(os.sep)[-2]
            sdf['Source'] = f"{model_name}_{ctx_name}"
            dfs.append(sdf)
            print(f"  Added {len(sdf)} samples from {model_name} ({ctx_name})")
        except Exception as e:
            print(f"  Error reading {f}: {e}")

    keep_cols = ['Sample_ID', target_col] + final_feature_cols
    
    final_dfs = []
    for d in dfs:
        for c in final_feature_cols:
            if c not in d.columns:
                d[c] = 0
        final_dfs.append(d[keep_cols])
    
    final_df = pd.concat(final_dfs, axis=0, ignore_index=True)
    
    # Save consolidated data
    out_dir = f"02_processed_data/{dataset}_augmented"
    os.makedirs(out_dir, exist_ok=True)
    final_df.to_csv(os.path.join(out_dir, "augmented_combined.csv"), index=False)
    
    print(f"\nSUCCESS: Consolidated {len(final_df)} samples for {dataset}")
    print(f"File saved to: {out_dir}/augmented_combined.csv")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        target = sys.argv[2] if len(sys.argv) > 2 else "Yield"
        consolidate_data(dataset, target)
    else:
        # Default to pepper for this task
        consolidate_data("pepper", "Yield")
