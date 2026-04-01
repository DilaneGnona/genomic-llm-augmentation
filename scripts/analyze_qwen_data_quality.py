"""
Analyze Qwen generated data quality
Compare with real data statistics
"""
import os
import pandas as pd
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC_PEPPER = os.path.join(BASE, '02_processed_data', 'pepper')
QWEN_DIR = os.path.join(BASE, '04_augmentation', 'pepper', 'model_sources', 'qwen')
OUTDIR = os.path.join(BASE, '03_modeling_results', 'pepper', 'qwen_data_analysis')

def ensure_dirs():
    os.makedirs(OUTDIR, exist_ok=True)

def load_real_data(max_features=100):
    """Load real pepper data"""
    x_csv = os.path.join(PROC_PEPPER, 'X_aligned.csv')
    y_path = os.path.join(PROC_PEPPER, 'y_aligned.csv')
    
    X = pd.read_csv(x_csv)
    y = pd.read_csv(y_path)
    
    feature_cols = [c for c in X.columns if c != 'Sample_ID'][:max_features]
    X = X[['Sample_ID'] + feature_cols]
    
    df = X.merge(y[['Sample_ID', 'Yield_BV']], on='Sample_ID', how='inner')
    return df, feature_cols

def load_qwen_context(context_type, real_cols):
    """Load Qwen data for specific context"""
    file_path = os.path.join(QWEN_DIR, context_type, f'synthetic_qwen_{context_type}_500samples.csv')
    
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    
    # Align columns
    for c in real_cols:
        if c not in df.columns:
            df[c] = 0
    
    keep_cols = ['Sample_ID'] + real_cols + ['Yield_BV']
    df = df[[c for c in keep_cols if c in df.columns]]
    
    return df

def analyze_data_quality():
    """Analyze and compare data quality"""
    print("="*70)
    print("QWEN DATA QUALITY ANALYSIS")
    print("="*70)
    print()
    
    ensure_dirs()
    
    # Load real data
    print("Loading real data...")
    df_real, feature_cols = load_real_data(max_features=100)
    print(f"Real data: {len(df_real)} samples, {len(feature_cols)} features")
    print()
    
    # Analyze each context
    contexts = ['context_A', 'context_B', 'context_C', 'context_D', 'context_E']
    
    print("="*70)
    print("STATISTICAL COMPARISON")
    print("="*70)
    print()
    
    # Real data stats
    print("REAL DATA:")
    print(f"  Yield_BV mean: {df_real['Yield_BV'].mean():.4f}")
    print(f"  Yield_BV std:  {df_real['Yield_BV'].std():.4f}")
    print(f"  Yield_BV min:  {df_real['Yield_BV'].min():.4f}")
    print(f"  Yield_BV max:  {df_real['Yield_BV'].max():.4f}")
    
    # Sample SNP stats
    sample_snp = feature_cols[0]
    print(f"\n  {sample_snp} mean: {df_real[sample_snp].mean():.4f}")
    print(f"  {sample_snp} std:  {df_real[sample_snp].std():.4f}")
    print()
    
    results_summary = []
    
    for ctx in contexts:
        print(f"\n{'='*70}")
        print(f"QWEN {ctx.upper()}")
        print(f"{'='*70}")
        
        df_qwen = load_qwen_context(ctx, feature_cols)
        
        if df_qwen is None:
            print(f"  Data not found!")
            continue
        
        print(f"  Samples: {len(df_qwen)}")
        
        # Yield stats
        print(f"\n  Yield_BV:")
        print(f"    mean: {df_qwen['Yield_BV'].mean():.4f} (real: {df_real['Yield_BV'].mean():.4f})")
        print(f"    std:  {df_qwen['Yield_BV'].std():.4f} (real: {df_real['Yield_BV'].std():.4f})")
        print(f"    min:  {df_qwen['Yield_BV'].min():.4f} (real: {df_real['Yield_BV'].min():.4f})")
        print(f"    max:  {df_qwen['Yield_BV'].max():.4f} (real: {df_real['Yield_BV'].max():.4f})")
        
        # SNP stats
        print(f"\n  {sample_snp}:")
        print(f"    mean: {df_qwen[sample_snp].mean():.4f} (real: {df_real[sample_snp].mean():.4f})")
        print(f"    std:  {df_qwen[sample_snp].std():.4f} (real: {df_real[sample_snp].std():.4f})")
        
        # Check for NaN
        nan_count = df_qwen.isna().sum().sum()
        print(f"\n  NaN values: {nan_count}")
        
        # Store for summary
        results_summary.append({
            'context': ctx,
            'n_samples': len(df_qwen),
            'yield_mean': df_qwen['Yield_BV'].mean(),
            'yield_std': df_qwen['Yield_BV'].std(),
            'snp_mean': df_qwen[sample_snp].mean(),
            'snp_std': df_qwen[sample_snp].std(),
            'nan_count': nan_count
        })
        
        # Print sample data
        print(f"\n  Sample data (first 3 rows):")
        print(df_qwen[['Sample_ID', sample_snp, 'Yield_BV']].head(3).to_string(index=False))
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON TABLE")
    print("="*70)
    
    summary_df = pd.DataFrame(results_summary)
    
    # Add real data row
    real_row = pd.DataFrame([{
        'context': 'REAL_DATA',
        'n_samples': len(df_real),
        'yield_mean': df_real['Yield_BV'].mean(),
        'yield_std': df_real['Yield_BV'].std(),
        'snp_mean': df_real[sample_snp].mean(),
        'snp_std': df_real[sample_snp].std(),
        'nan_count': df_real.isna().sum().sum()
    }])
    
    summary_df = pd.concat([real_row, summary_df], ignore_index=True)
    
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(OUTDIR, 'qwen_quality_summary.csv'), index=False)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {OUTDIR}")
    
    return summary_df

if __name__ == '__main__':
    analyze_data_quality()
