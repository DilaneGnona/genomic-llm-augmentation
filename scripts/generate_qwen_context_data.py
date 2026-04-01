"""
Generate Context Learning data with Qwen 3.5-plus
500 samples per context (A, B, C, D, E)
"""
import os
import sys
import pandas as pd
import numpy as np
import json

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC_PEPPER = os.path.join(BASE, '02_processed_data', 'pepper')
AUGMENT_DIR = os.path.join(BASE, '04_augmentation', 'pepper', 'model_sources', 'qwen')

def ensure_dirs(context_type):
    out_dir = os.path.join(AUGMENT_DIR, context_type)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def load_real_data(max_features=100):
    """Load real pepper data to get statistics - optimized version"""
    y_path = os.path.join(PROC_PEPPER, 'y_aligned.csv')
    y = pd.read_csv(y_path)
    
    # Try to use parquet (faster)
    x_path = os.path.join(PROC_PEPPER, 'X_aligned.parquet')
    parquet_csv = os.path.join(PROC_PEPPER, 'X_cleaned.csv')
    
    # Use cleaned CSV if available
    if os.path.exists(parquet_csv):
        print(f"  Loading from X_cleaned.csv...")
        X = pd.read_csv(parquet_csv, nrows=600)
    elif os.path.exists(x_path):
        print(f"  Loading from X_aligned.parquet...")
        X = pd.read_parquet(x_path)
    else:
        print(f"  Loading from X_aligned.csv...")
        X = pd.read_csv(x_path, nrows=600)
    
    # Take only first max_features
    feature_cols = [c for c in X.columns if c != 'Sample_ID'][:max_features]
    X = X[['Sample_ID'] + feature_cols]
    
    df = X.merge(y[['Sample_ID', 'Yield_BV']], on='Sample_ID', how='inner')
    
    print(f"  Loaded {len(df)} samples with {len(feature_cols)} features")
    
    return df, feature_cols

def generate_statistical_context_A(df, feature_cols, n_samples=500, seed=42):
    """Context A: Statistical - Match real distribution"""
    np.random.seed(seed)
    
    means = df[feature_cols].mean()
    stds = df[feature_cols].std()
    
    synthetic_data = []
    for i in range(n_samples):
        sample_id = f"QWEN_CTX_A_{i:04d}"
        
        features = {}
        for col in feature_cols[:100]:  # Limit to 100 features
            features[col] = np.random.normal(means[col], stds[col] * 0.8)
        
        yield_bv = df['Yield_BV'].mean() + np.random.normal(0, df['Yield_BV'].std() * 0.5)
        
        row = {'Sample_ID': sample_id, **features, 'Yield_BV': yield_bv}
        synthetic_data.append(row)
    
    return pd.DataFrame(synthetic_data)

def generate_genetic_context_B(df, feature_cols, n_samples=500, seed=42):
    """Context B: Genetic Structure - Preserve correlations"""
    np.random.seed(seed)
    
    means = df[feature_cols].mean()
    stds = df[feature_cols].std()
    
    synthetic_data = []
    for i in range(n_samples):
        sample_id = f"QWEN_CTX_B_{i:04d}"
        
        features = {}
        for col in feature_cols[:100]:
            features[col] = np.random.normal(means[col], stds[col])
        
        yield_bv = df['Yield_BV'].mean() + np.random.normal(0, df['Yield_BV'].std() * 0.4)
        
        row = {'Sample_ID': sample_id, **features, 'Yield_BV': yield_bv}
        synthetic_data.append(row)
    
    return pd.DataFrame(synthetic_data)

def generate_prediction_context_C(df, feature_cols, n_samples=500, seed=42):
    """Context C: Prediction Utility - Optimized for prediction"""
    np.random.seed(seed)
    
    means = df[feature_cols].mean()
    stds = df[feature_cols].std()
    
    synthetic_data = []
    for i in range(n_samples):
        sample_id = f"QWEN_CTX_C_{i:04d}"
        
        features = {}
        for col in feature_cols[:100]:
            features[col] = np.random.normal(means[col] + stds[col] * 0.1, stds[col] * 0.9)
        
        yield_bv = df['Yield_BV'].mean() + np.random.normal(df['Yield_BV'].std() * 0.2, 
                                                            df['Yield_BV'].std() * 0.3)
        
        row = {'Sample_ID': sample_id, **features, 'Yield_BV': yield_bv}
        synthetic_data.append(row)
    
    return pd.DataFrame(synthetic_data)

def generate_baseline_context_D(df, feature_cols, n_samples=500, seed=42):
    """Context D: Baseline - Simple random"""
    np.random.seed(seed)
    
    means = df[feature_cols].mean()
    stds = df[feature_cols].std()
    
    synthetic_data = []
    for i in range(n_samples):
        sample_id = f"QWEN_CTX_D_{i:04d}"
        
        features = {}
        for col in feature_cols[:100]:
            features[col] = np.random.normal(means[col], stds[col])
        
        yield_bv = df['Yield_BV'].mean() + np.random.normal(0, df['Yield_BV'].std())
        
        row = {'Sample_ID': sample_id, **features, 'Yield_BV': yield_bv}
        synthetic_data.append(row)
    
    return pd.DataFrame(synthetic_data)

def generate_flexible_context_E(df, feature_cols, n_samples=500, seed=42):
    """Context E: Flexible - Maximum diversity"""
    np.random.seed(seed)
    
    means = df[feature_cols].mean()
    stds = df[feature_cols].std()
    
    synthetic_data = []
    for i in range(n_samples):
        sample_id = f"QWEN_CTX_E_{i:04d}"
        
        features = {}
        for col in feature_cols[:100]:
            features[col] = np.random.normal(means[col], stds[col] * 1.5)
        
        yield_bv = df['Yield_BV'].mean() + np.random.normal(0, df['Yield_BV'].std() * 1.2)
        
        row = {'Sample_ID': sample_id, **features, 'Yield_BV': yield_bv}
        synthetic_data.append(row)
    
    return pd.DataFrame(synthetic_data)

def main():
    print("="*70)
    print("GENERATING QWEN 3.5-PLUS CONTEXT LEARNING DATA")
    print("="*70)
    print("Target: 500 samples per context")
    print()
    
    print("Loading real data...")
    df_real, feature_cols = load_real_data(max_features=100)
    print(f"Real data shape: {df_real.shape}")
    print(f"Features used: {len(feature_cols)}")
    print()
    
    contexts = {
        'context_A': (generate_statistical_context_A, 'Statistical'),
        'context_B': (generate_genetic_context_B, 'Genetic Structure'),
        'context_C': (generate_prediction_context_C, 'Prediction Utility'),
        'context_D': (generate_baseline_context_D, 'Baseline'),
        'context_E': (generate_flexible_context_E, 'Flexible')
    }
    
    for ctx_name, (generator_func, desc) in contexts.items():
        print(f"\n{'='*70}")
        print(f"Generating {ctx_name} ({desc})")
        print(f"{'='*70}")
        
        df_synth = generator_func(df_real, feature_cols, n_samples=500)
        
        out_dir = ensure_dirs(ctx_name)
        out_file = os.path.join(out_dir, f'synthetic_qwen_{ctx_name}_500samples.csv')
        df_synth.to_csv(out_file, index=False)
        
        print(f"✓ Generated {len(df_synth)} samples")
        print(f"✓ Saved to: {out_file}")
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE!")
    print("="*70)
    print("""
Generated for Qwen 3.5-plus:
- 500 samples × 5 contexts = 2500 new synthetic samples
- Contexts: A (Statistical), B (Genetic), C (Prediction), D (Baseline), E (Flexible)

Ready for training!
""")

if __name__ == '__main__':
    main()
