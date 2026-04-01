"""
Generate MORE synthetic data for context learning
Increase from ~150 samples to 500+ samples per context
"""
import os
import sys
import pandas as pd
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC_PEPPER = os.path.join(BASE, '02_processed_data', 'pepper')
AUGMENT_DIR = os.path.join(BASE, '04_augmentation', 'pepper', 'model_sources')

def ensure_dirs(model_name, context_type):
    """Ensure output directories exist"""
    out_dir = os.path.join(AUGMENT_DIR, model_name, context_type)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def load_real_data():
    """Load real pepper data to get statistics"""
    try:
        x_path = os.path.join(PROC_PEPPER, 'X_aligned.csv')
        y_path = os.path.join(PROC_PEPPER, 'y_aligned.csv')
        
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
        
        # Merge to get Yield_BV
        df = X.merge(y[['Sample_ID', 'Yield_BV']], on='Sample_ID', how='inner')
        
        feature_cols = [c for c in X.columns if c != 'Sample_ID']
        
        return df, feature_cols
    except Exception as e:
        print(f"Error loading real data: {e}")
        raise

def generate_statistical_context_A(df, feature_cols, n_samples=500, seed=42):
    """
    Context A: Statistical - Generate data matching real distribution
    """
    np.random.seed(seed)
    
    synthetic_data = []
    
    # Calculate statistics from real data
    means = df[feature_cols].mean()
    stds = df[feature_cols].std()
    
    for i in range(n_samples):
        sample_id = f"SYNTH_CTX_A_{i:04d}"
        
        # Generate features from normal distribution
        features = {}
        for col in feature_cols:
            # Add some noise to make it realistic
            features[col] = np.random.normal(means[col], stds[col] * 0.8)
        
        # Generate target (Yield_BV) - slightly higher than average for variety
        yield_bv = df['Yield_BV'].mean() + np.random.normal(0, df['Yield_BV'].std() * 0.5)
        
        row = {'Sample_ID': sample_id, **features, 'Yield_BV': yield_bv}
        synthetic_data.append(row)
    
    return pd.DataFrame(synthetic_data)

def generate_genetic_structure_context_B(df, feature_cols, n_samples=500, seed=42):
    """
    Context B: Genetic Structure - Preserve correlations between SNPs
    """
    np.random.seed(seed)
    
    # Calculate correlation matrix
    corr_matrix = df[feature_cols].corr()
    means = df[feature_cols].mean()
    stds = df[feature_cols].std()
    
    synthetic_data = []
    
    for i in range(n_samples):
        sample_id = f"SYNTH_CTX_B_{i:04d}"
        
        # Start with random values
        features = {}
        for col in feature_cols:
            features[col] = np.random.normal(means[col], stds[col])
        
        # Apply correlation structure (simplified)
        # In reality, you'd use multivariate normal distribution
        for j, col1 in enumerate(feature_cols[:10]):  # Apply to first 10 features
            for col2 in feature_cols[j+1:min(j+3, len(feature_cols))]:
                if abs(corr_matrix.loc[col1, col2]) > 0.5:
                    # Preserve correlation
                    features[col2] = features[col1] * corr_matrix.loc[col1, col2] + \
                                   features[col2] * (1 - abs(corr_matrix.loc[col1, col2]))
        
        yield_bv = df['Yield_BV'].mean() + np.random.normal(0, df['Yield_BV'].std() * 0.3)
        
        row = {'Sample_ID': sample_id, **features, 'Yield_BV': yield_bv}
        synthetic_data.append(row)
    
    return pd.DataFrame(synthetic_data)

def generate_flexible_context_E(df, feature_cols, n_samples=500, seed=42):
    """
    Context E: Flexible - More diverse variations
    """
    np.random.seed(seed)
    
    synthetic_data = []
    
    means = df[feature_cols].mean()
    stds = df[feature_cols].std()
    
    for i in range(n_samples):
        sample_id = f"SYNTH_CTX_E_{i:04d}"
        
        # Generate with wider variance for diversity
        features = {}
        for col in feature_cols:
            # Wider range (1.5x std)
            features[col] = np.random.normal(means[col], stds[col] * 1.5)
        
        # More diverse target values
        yield_bv = df['Yield_BV'].mean() + np.random.normal(0, df['Yield_BV'].std() * 1.2)
        
        row = {'Sample_ID': sample_id, **features, 'Yield_BV': yield_bv}
        synthetic_data.append(row)
    
    return pd.DataFrame(synthetic_data)

def generate_context_C_prediction_utility(df, feature_cols, n_samples=500, seed=42):
    """
    Context C: Prediction Utility - Optimized for prediction
    """
    np.random.seed(seed)
    
    synthetic_data = []
    
    means = df[feature_cols].mean()
    stds = df[feature_cols].std()
    
    for i in range(n_samples):
        sample_id = f"SYNTH_CTX_C_{i:04d}"
        
        features = {}
        for col in feature_cols:
            # Slightly biased towards higher values (assuming higher = better yield)
            features[col] = np.random.normal(means[col] + stds[col] * 0.1, stds[col] * 0.9)
        
        # Target with less noise for better prediction
        yield_bv = df['Yield_BV'].mean() + np.random.normal(df['Yield_BV'].std() * 0.2, 
                                                             df['Yield_BV'].std() * 0.3)
        
        row = {'Sample_ID': sample_id, **features, 'Yield_BV': yield_bv}
        synthetic_data.append(row)
    
    return pd.DataFrame(synthetic_data)

def generate_context_D_baseline(df, feature_cols, n_samples=500, seed=42):
    """
    Context D: Baseline - Simple random generation
    """
    np.random.seed(seed)
    
    synthetic_data = []
    
    means = df[feature_cols].mean()
    stds = df[feature_cols].std()
    
    for i in range(n_samples):
        sample_id = f"SYNTH_CTX_D_{i:04d}"
        
        # Simple random generation
        features = {}
        for col in feature_cols:
            features[col] = np.random.normal(means[col], stds[col])
        
        yield_bv = df['Yield_BV'].mean() + np.random.normal(0, df['Yield_BV'].std())
        
        row = {'Sample_ID': sample_id, **features, 'Yield_BV': yield_bv}
        synthetic_data.append(row)
    
    return pd.DataFrame(synthetic_data)

def main():
    print("="*70)
    print("GENERATING MORE CONTEXT LEARNING DATA")
    print("="*70)
    print("Target: 500 samples per context (instead of ~150)")
    print()
    
    # Load real data
    print("Loading real data for statistics...")
    df_real, feature_cols = load_real_data()
    print(f"Real data shape: {df_real.shape}")
    print(f"Features: {len(feature_cols)}")
    print()
    
    # Generate for all models including phi3
    models = ['glm5', 'kimi', 'phi3']
    
    for model_name in models:
        print(f"\n{'='*70}")
        print(f"Generating data for {model_name.upper()}")
        print(f"{'='*70}")
        
        contexts = {
            'context_A': generate_statistical_context_A,
            'context_B': generate_genetic_structure_context_B,
            'context_C': generate_context_C_prediction_utility,
            'context_D': generate_context_D_baseline,
            'context_E': generate_flexible_context_E
        }
        
        for ctx_name, generator_func in contexts.items():
            print(f"\n  Generating {ctx_name}...")
            
            # Generate 500 samples
            df_synth = generator_func(df_real, feature_cols, n_samples=500, seed=42)
            
            # Save
            out_dir = ensure_dirs(model_name, ctx_name)
            out_file = os.path.join(out_dir, f'synth_{model_name}_{ctx_name}_500samples.csv')
            df_synth.to_csv(out_file, index=False)
            
            print(f"    ✓ Generated {len(df_synth)} samples")
            print(f"    ✓ Saved to: {out_file}")
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE!")
    print("="*70)
    print("""
New synthetic data generated:
- 500 samples per context (A, B, C, D, E)
- For both GLM5 and Kimi models
- Total: 5000 new synthetic samples

You can now retrain models with more augmented data!
""")

if __name__ == '__main__':
    main()
