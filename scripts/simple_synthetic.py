import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
DATASET = 'pepper'
INPUT_DIR = Path('02_processed_data/pepper/')
OUTPUT_DIR = Path('04_augmentation/pepper/qwen3coder')
REF_SAMPLES = 500
REF_FEATURES = 100
SYNTHETIC_SAMPLES = 1000
SEED = 42

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading reference data...")
    
    # Load reference data (only first REF_SAMPLES rows and REF_FEATURES columns to avoid large file issues)
    X_ref = pd.read_csv(INPUT_DIR / 'X.csv', index_col=0, nrows=REF_SAMPLES + 1)  # Read extra row to be safe
    X_ref = X_ref.iloc[:REF_SAMPLES, :REF_FEATURES]  # Get exactly REF_SAMPLES x REF_FEATURES
    y_ref = pd.read_csv(INPUT_DIR / 'y.csv', index_col=0)
    
    print(f"Reference X shape: {X_ref.shape}")
    print(f"Reference y shape: {y_ref.shape}")
    
    # Get common samples
    common_index = X_ref.index.intersection(y_ref.index)
    y_ref = y_ref.loc[common_index]
    
    print(f"Common samples: {len(common_index)}")
    
    # Generate synthetic SNPs with similar distribution
    print("Generating synthetic SNPs...")
    np.random.seed(SEED)
    
    synthetic_snps = []
    for i in range(SYNTHETIC_SAMPLES):
        sample = {'Sample_ID': f'SYNTH_{i+1}'}
        for feature in X_ref.columns:
            # Get the distribution of the current feature
            dist = X_ref[feature].value_counts(normalize=True)
            probs = [dist.get(0, 0), dist.get(1, 0), dist.get(2, 0)]
            
            # Normalize probabilities to sum to 1
            sum_probs = sum(probs)
            if sum_probs == 0:
                probs = [1/3, 1/3, 1/3]
            else:
                probs = [p/sum_probs for p in probs]
            
            # Generate SNP value
            sample[feature] = np.random.choice([0, 1, 2], p=probs)
        synthetic_snps.append(sample)
    
    # Convert to DataFrame and save
    df_snps = pd.DataFrame(synthetic_snps)
    df_snps.set_index('Sample_ID', inplace=True)
    df_snps.to_csv(OUTPUT_DIR / 'synthetic_snps.csv')
    
    print(f"Generated {len(df_snps)} synthetic SNP samples")
    print(f"Saved to {OUTPUT_DIR / 'synthetic_snps.csv'}")
    
    # Generate synthetic phenotypes
    print("Generating synthetic phenotypes...")
    target_col = 'Yield_BV'
    
    if len(y_ref) > 0:
        # Use the same distribution as reference
        y_mean = y_ref[target_col].mean()
        y_std = y_ref[target_col].std()
        y_min = y_ref[target_col].min()
        y_max = y_ref[target_col].max()
        
        synthetic_pheno = np.random.normal(y_mean, y_std, size=SYNTHETIC_SAMPLES)
        synthetic_pheno = np.clip(synthetic_pheno, y_min, y_max)
    else:
        # Fallback to default distribution
        synthetic_pheno = np.random.normal(5.0, 2.0, size=SYNTHETIC_SAMPLES)
    
    # Create DataFrame and save
    df_pheno = pd.DataFrame({
        'Sample_ID': [f'SYNTH_{i+1}' for i in range(SYNTHETIC_SAMPLES)],
        target_col: synthetic_pheno
    })
    df_pheno.set_index('Sample_ID', inplace=True)
    df_pheno.to_csv(OUTPUT_DIR / 'synthetic_y.csv')
    
    print(f"Generated {len(df_pheno)} synthetic phenotypes")
    print(f"Saved to {OUTPUT_DIR / 'synthetic_y.csv'}")
    
    print("Done!")

if __name__ == "__main__":
    main()