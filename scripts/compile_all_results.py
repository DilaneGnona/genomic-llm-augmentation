"""
Compile all training results from context learning models into comprehensive tables
"""
import os
import pandas as pd
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE, '03_modeling_results', 'pepper')
OUTDIR = os.path.join(BASE, '03_modeling_results', 'pepper', 'compiled_results')

def ensure_dirs():
    os.makedirs(OUTDIR, exist_ok=True)

def load_results():
    """Load all available results"""
    all_results = []
    
    # Kimi MLP results
    kimi_mlp_path = os.path.join(RESULTS_DIR, 'kimi_context', 'results.csv')
    if os.path.exists(kimi_mlp_path):
        df = pd.read_csv(kimi_mlp_path)
        df['llm_model'] = 'kimi'
        df['dl_model'] = 'mlp'
        all_results.append(df)
        print(f"Loaded Kimi MLP: {len(df)} rows")
    
    # GLM5 MLP results
    glm5_mlp_path = os.path.join(RESULTS_DIR, 'glm5_context', 'results.csv')
    if os.path.exists(glm5_mlp_path):
        df = pd.read_csv(glm5_mlp_path)
        df['llm_model'] = 'glm5'
        df['dl_model'] = 'mlp'
        all_results.append(df)
        print(f"Loaded GLM5 MLP: {len(df)} rows")
    
    # CNN/LSTM PyTorch results
    cnn_lstm_path = os.path.join(RESULTS_DIR, 'cnn_lstm_pytorch', 'cnn_lstm_pytorch_results.csv')
    if os.path.exists(cnn_lstm_path):
        df = pd.read_csv(cnn_lstm_path)
        # Extract llm_model from context column
        def extract_llm(context):
            if 'kimi' in context:
                return 'kimi'
            elif 'glm5' in context:
                return 'glm5'
            elif 'phi3' in context:
                return 'phi3'
            else:
                return 'baseline'
        df['llm_model'] = df['context'].apply(extract_llm)
        df['dl_model'] = df['model_type']
        all_results.append(df)
        print(f"Loaded CNN/LSTM PyTorch: {len(df)} rows")
    
    # Phi3 MLP results
    phi3_mlp_path = os.path.join(RESULTS_DIR, 'phi3_mlp', 'phi3_mlp_results.csv')
    if os.path.exists(phi3_mlp_path):
        df = pd.read_csv(phi3_mlp_path)
        df['llm_model'] = 'phi3'
        df['dl_model'] = 'mlp'
        all_results.append(df)
        print(f"Loaded Phi3 MLP: {len(df)} rows")
    
    # Phi3 DL results
    phi3_dl_path = os.path.join(RESULTS_DIR, 'phi3_dl_pytorch', 'phi3_dl_results.csv')
    if os.path.exists(phi3_dl_path):
        df = pd.read_csv(phi3_dl_path)
        df['llm_model'] = 'phi3'
        df['dl_model'] = df['model_type']
        all_results.append(df)
        print(f"Loaded Phi3 DL: {len(df)} rows")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return None

def standardize_columns(df):
    """Standardize column names across different result files"""
    column_mapping = {
        'r2': 'r2_holdout',
        'rmse': 'rmse_holdout',
        'cv_r2': 'cv_r2_mean',
        'cv_r2_std': 'cv_r2_std',
        'train_time': 'train_time_s',
        'context_type': 'context',
        'n_synthetic': 'n_samples'
    }
    
    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    return df

def create_summary_table(df):
    """Create a comprehensive summary table"""
    
    # Standardize context names
    def standardize_context(context):
        if pd.isna(context) or context == 'baseline':
            return 'baseline'
        if 'context_A' in context or 'prompt_A' in context:
            return 'context_A'
        elif 'context_B' in context or 'prompt_B' in context:
            return 'context_B'
        elif 'context_C' in context or 'prompt_C' in context:
            return 'context_C'
        elif 'context_D' in context or 'prompt_D' in context:
            return 'context_D'
        elif 'context_E' in context or 'prompt_E' in context:
            return 'context_E'
        return context
    
    df['context_std'] = df['context'].apply(standardize_context)
    
    # Create summary
    summary = df.groupby(['llm_model', 'context_std', 'dl_model']).agg({
        'r2_holdout': 'mean',
        'rmse_holdout': 'mean',
        'cv_r2_mean': 'mean',
        'n_samples': 'first'
    }).reset_index()
    
    return summary

def create_best_results_table(df):
    """Create table showing best results per LLM model"""
    
    # Filter valid results (not NaN)
    df_valid = df[df['r2_holdout'].notna()]
    
    best_results = []
    for llm in df_valid['llm_model'].unique():
        llm_data = df_valid[df_valid['llm_model'] == llm]
        
        # Find best by R2
        best_idx = llm_data['r2_holdout'].idxmax()
        best_row = llm_data.loc[best_idx]
        
        best_results.append({
            'LLM Model': llm.upper(),
            'Best DL Model': best_row['dl_model'].upper(),
            'Context': best_row['context'],
            'R²': f"{best_row['r2_holdout']:.4f}",
            'RMSE': f"{best_row['rmse_holdout']:.4f}",
            'CV R²': f"{best_row['cv_r2_mean']:.4f}" if pd.notna(best_row['cv_r2_mean']) else 'N/A',
            'N Samples': int(best_row['n_samples']) if pd.notna(best_row['n_samples']) else 'N/A'
        })
    
    return pd.DataFrame(best_results)

def create_comparison_matrix(df):
    """Create comparison matrix of all models"""
    
    df_valid = df[df['r2_holdout'].notna()]
    
    # Pivot table: LLM model vs DL model
    pivot = df_valid.pivot_table(
        values='r2_holdout',
        index='llm_model',
        columns='dl_model',
        aggfunc='max'
    )
    
    return pivot

def create_context_comparison(df):
    """Compare different contexts for each LLM model"""
    
    df_valid = df[df['r2_holdout'].notna()]
    
    contexts = ['baseline', 'context_A', 'context_B', 'context_C', 'context_D', 'context_E']
    llm_models = ['kimi', 'glm5', 'phi3']
    
    comparison_data = []
    for llm in llm_models:
        row = {'LLM Model': llm.upper()}
        llm_data = df_valid[df_valid['llm_model'] == llm]
        
        for ctx in contexts:
            ctx_data = llm_data[llm_data['context_std'] == ctx]
            if not ctx_data.empty:
                best_r2 = ctx_data['r2_holdout'].max()
                row[ctx] = f"{best_r2:.4f}"
            else:
                row[ctx] = '-'
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def main():
    print("="*80)
    print("COMPILING ALL CONTEXT LEARNING RESULTS")
    print("="*80)
    print()
    
    ensure_dirs()
    
    # Load all results
    df = load_results()
    if df is None:
        print("No results found!")
        return
    
    print(f"\nTotal records loaded: {len(df)}")
    print()
    
    # Standardize columns
    df = standardize_columns(df)
    
    # Save combined raw data
    combined_path = os.path.join(OUTDIR, 'all_results_combined.csv')
    df.to_csv(combined_path, index=False)
    print(f"Combined raw data saved to: {combined_path}")
    print()
    
    # Create summary table
    print("="*80)
    print("SUMMARY TABLE: All Results by LLM, Context, and DL Model")
    print("="*80)
    summary = create_summary_table(df)
    summary_path = os.path.join(OUTDIR, 'summary_by_model_context.csv')
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved to: {summary_path}")
    print()
    
    # Create best results table
    print("="*80)
    print("BEST RESULTS PER LLM MODEL")
    print("="*80)
    best_results = create_best_results_table(df)
    best_path = os.path.join(OUTDIR, 'best_results_per_llm.csv')
    best_results.to_csv(best_path, index=False)
    print(best_results.to_string(index=False))
    print(f"\nSaved to: {best_path}")
    print()
    
    # Create comparison matrix
    print("="*80)
    print("COMPARISON MATRIX: Max R² by LLM Model and DL Architecture")
    print("="*80)
    matrix = create_comparison_matrix(df)
    matrix_path = os.path.join(OUTDIR, 'comparison_matrix.csv')
    matrix.to_csv(matrix_path)
    print(matrix.to_string())
    print(f"\nSaved to: {matrix_path}")
    print()
    
    # Create context comparison
    print("="*80)
    print("CONTEXT COMPARISON: Best R² by Context Type")
    print("="*80)
    context_comp = create_context_comparison(df)
    context_path = os.path.join(OUTDIR, 'context_comparison.csv')
    context_comp.to_csv(context_path, index=False)
    print(context_comp.to_string(index=False))
    print(f"\nSaved to: {context_path}")
    print()
    
    # Overall statistics
    print("="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    df_valid = df[df['r2_holdout'].notna()]
    print(f"Total valid experiments: {len(df_valid)}")
    print(f"Best overall R²: {df_valid['r2_holdout'].max():.4f}")
    print(f"Worst overall R²: {df_valid['r2_holdout'].min():.4f}")
    print(f"Mean R²: {df_valid['r2_holdout'].mean():.4f}")
    print(f"Median R²: {df_valid['r2_holdout'].median():.4f}")
    print()
    
    # Best overall
    best_idx = df_valid['r2_holdout'].idxmax()
    best = df_valid.loc[best_idx]
    print("Best Overall Configuration:")
    print(f"  LLM Model: {best['llm_model'].upper()}")
    print(f"  DL Model: {best['dl_model'].upper()}")
    print(f"  Context: {best['context']}")
    print(f"  R²: {best['r2_holdout']:.4f}")
    print(f"  RMSE: {best['rmse_holdout']:.4f}")
    
    print()
    print("="*80)
    print(f"All results saved to: {OUTDIR}")
    print("="*80)

if __name__ == '__main__':
    main()
