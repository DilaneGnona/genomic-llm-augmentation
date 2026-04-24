import os
import sys
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.unified_genomic_pipeline import UnifiedGenomicPipeline

def run_few_shot_experiment(dataset, target, fractions=[0.1, 0.25, 0.5, 1.0]):
    logging.info(f"Starting Few-Shot Learning Experiment for {dataset}")
    
    pipeline = UnifiedGenomicPipeline(dataset, target, "few_shot_bench")
    
    # 1. Load data
    real_x_path = os.path.join(pipeline.base_dir, "02_processed_data", dataset, "X.csv")
    real_y_path = os.path.join(pipeline.base_dir, "02_processed_data", dataset, "y.csv")
    
    real_x = pd.read_csv(real_x_path)
    real_y = pd.read_csv(real_y_path)
    
    # Align
    common = list(set(real_x['Sample_ID']).intersection(set(real_y['Sample_ID'])))
    real_df = real_x[real_x['Sample_ID'].isin(common)].merge(real_y[['Sample_ID', target]], on='Sample_ID')
    
    # Get synthetic data
    # We need to use the consolidate_data logic but with controlled real data
    # To simplify, we'll first consolidate ALL data, then split real/synthetic
    all_data = pipeline.consolidate_data()
    
    # Identify which samples are real vs synthetic
    real_sample_ids = set(real_df['Sample_ID'])
    
    results = []
    
    for frac in fractions:
        logging.info(f"--- Evaluating fraction: {frac*100}% of real data ---")
        
        # Subsample real data
        if frac < 1.0:
            real_sub, _ = train_test_split(real_df, train_size=frac, random_state=42)
        else:
            real_sub = real_df
            
        real_sub_ids = set(real_sub['Sample_ID'])
        
        # Case A: Real Data Only
        df_only_real = all_data[all_data['Sample_ID'].isin(real_sub_ids)]
        logging.info(f"  Training 'Real Only' with {len(df_only_real)} samples...")
        try:
            summary_real = pipeline.train_ensemble(df_only_real, n_trials=5)
            r2_real = summary_real['metrics']['stacking_optimized']['r2']
        except:
            r2_real = -1
            
        # Case B: Real Data + Synthetic Data
        # Synthetic samples are those in all_data NOT in real_sample_ids
        df_aug = all_data[all_data['Sample_ID'].isin(real_sub_ids) | (~all_data['Sample_ID'].isin(real_sample_ids))]
        logging.info(f"  Training 'Augmented' with {len(df_aug)} samples ({len(df_aug)-len(df_only_real)} synthetic)...")
        try:
            summary_aug = pipeline.train_ensemble(df_aug, n_trials=5)
            r2_aug = summary_aug['metrics']['stacking_optimized']['r2']
        except:
            r2_aug = -1
            
        results.append({
            'fraction': frac,
            'n_real': len(df_only_real),
            'r2_real_only': r2_real,
            'r2_augmented': r2_aug
        })
        
    # Create report
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(pipeline.results_dir, "few_shot_results.csv"), index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['fraction']*100, res_df['r2_real_only'], marker='o', label='Real Data Only', color='red')
    plt.plot(res_df['fraction']*100, res_df['r2_augmented'], marker='s', label='Augmented (LLM)', color='green')
    plt.xlabel("Percentage of Real Training Data (%)")
    plt.ylabel("Prediction Accuracy (R2)")
    plt.title(f"Data Efficiency: Impact of LLM Augmentation - {dataset.upper()}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(pipeline.results_dir, "plots", "few_shot_curve.png"))
    
    logging.info(f"Few-shot experiment finished. Plot saved to {pipeline.results_dir}")
    return res_df

if __name__ == "__main__":
    print("Script started...")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
    # Use IPK for the demo as it had the biggest rescue effect
    try:
        run_few_shot_experiment("ipk_out_raw", "YR_LS")
        print("Script finished successfully.")
    except Exception as e:
        print(f"Script failed with error: {e}")
        import traceback
        traceback.print_exc()
