import os
import pandas as pd
import numpy as np

# Paths
DATA_DIR = "02_processed_data/ipk_out_raw"
RAW_DATA_DIR = "01_raw_data/ipk_out_raw"

def fix_y_csv():
    # Load the current y.csv (which only has Sample_ID)
    current_y = pd.read_csv(os.path.join(DATA_DIR, "y.csv"), index_col=0)
    print(f"Current y.csv shape: {current_y.shape}")
    
    # Load the original phenotype data
    phenotypes = pd.read_csv(os.path.join(RAW_DATA_DIR, "Geno_IDs_and_Phenotypes.txt"), sep="\t")
    print(f"Original phenotype data shape: {phenotypes.shape}")
    print(f"Available columns: {list(phenotypes.columns)}")
    
    # Create a new y DataFrame with the sample IDs from current_y
    new_y = pd.DataFrame(index=current_y.index)
    
    # Add target columns from the phenotype file
    # Since we don't have a direct mapping between the ID formats,
    # we'll initialize with NaN values
    target_columns = ['YR_LS', 'YR_precision', 'Yield_BV']
    for col in target_columns:
        if col in phenotypes.columns:
            new_y[col] = np.nan
    
    # Try to find any possible matches based on ID patterns
    # This is a heuristic and might not find many matches
    print("\nAttempting to find ID matches...")
    matches_found = 0
    
    # For each sample in our genotype data
    for sample_id in new_y.index:
        # Simple heuristic: check if any part of the sample ID exists in the phenotype IDs
        sample_parts = sample_id.split('_')
        for part in sample_parts:
            if len(part) > 4:  # Only check parts that are long enough to be meaningful
                # Search in GBS_BIOSAMPLE_ID
                matching_rows = phenotypes[phenotypes['GBS_BIOSAMPLE_ID'].str.contains(part, na=False)]
                if len(matching_rows) == 1:
                    for col in target_columns:
                        if col in matching_rows.columns:
                            new_y.at[sample_id, col] = matching_rows.iloc[0][col]
                    matches_found += 1
                    break
    
    print(f"Found {matches_found} possible matches")
    
    # Save the new y.csv file
    new_y.to_csv(os.path.join(DATA_DIR, "y.csv"))
    print(f"\nNew y.csv saved with shape: {new_y.shape}")
    print(f"Columns added: {list(new_y.columns)}")
    
    # Check statistics for each numeric column
    for col in new_y.columns:
        if np.issubdtype(new_y[col].dtype, np.number):
            non_na_count = new_y[col].notna().sum()
            print(f"\n{col} statistics:")
            print(f"  Non-NA values: {non_na_count}")
            if non_na_count > 0:
                print(f"  Mean: {new_y[col].mean():.4f}")
                print(f"  Std: {new_y[col].std():.4f}")
                print(f"  Min: {new_y[col].min():.4f}")
                print(f"  Max: {new_y[col].max():.4f}")

if __name__ == "__main__":
    fix_y_csv()