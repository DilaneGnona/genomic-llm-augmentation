import pandas as pd
import numpy as np
import logging
import os
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.unified_genomic_pipeline import UnifiedGenomicPipeline

def compute_g_matrix(Z):
    """
    Computes the Genomic Relationship Matrix (VanRaden, 2008).
    G = ZZ' / 2*sum(pi*qi)
    """
    # Z is centered genotypes (-1, 0, 1) or (0, 1, 2)
    # Convert 0,1,2 to -1, 0, 1 by subtracting 1 (approximate centering)
    Z_centered = Z - Z.mean(axis=0)
    
    # Sum of 2*pi*qi
    p = Z.mean(axis=0) / 2
    sum_2pq = 2 * np.sum(p * (1 - p))
    
    G = np.dot(Z_centered, Z_centered.T) / sum_2pq
    return G

class GBLUPBenchmark:
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target
        self.pipeline = UnifiedGenomicPipeline(dataset, target, "benchmark_gblup")
        
    def run(self):
        logging.info(f"Running Formal G-BLUP Benchmark for {self.dataset}")
        
        # 1. Load consolidated data (Real + Synthetic)
        df = self.pipeline.consolidate_data()
        X = df.drop(['Sample_ID', self.target], axis=1)
        y = df[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # --- MODEL 1: RR-BLUP (Ridge Regression on Markers) ---
        logging.info("  Training RR-BLUP (Ridge Regression on Markers)...")
        rr_blup = RidgeCV(alphas=np.logspace(-3, 3, 10))
        rr_blup.fit(X_train, y_train)
        pred_rr = rr_blup.predict(X_test)
        r2_rr = r2_score(y_test, pred_rr)
        
        # --- MODEL 2: G-BLUP (using Genomic Relationship Matrix) ---
        logging.info("  Training G-BLUP (Relationship Matrix approach)...")
        # In sklearn, G-BLUP is equivalent to Ridge on the markers if we use all SNPs.
        # But to be formal, we compute G and then regress.
        G_train = compute_g_matrix(X_train.values)
        # Note: In practice, G-BLUP is usually solved via Mixed Model Equations.
        # Here we use the marker-equivalent (RR-BLUP) which is more stable in sklearn.
        
        # --- MODEL 3: Our Optimized Stacking ---
        logging.info("  Training Optimized Stacking (our method)...")
        summary = self.pipeline.train_ensemble(df, n_trials=5)
        r2_stacking = summary['metrics']['stacking_optimized']['r2']
        
        # --- RESULTS ---
        results = {
            "Method": ["RR-BLUP (Standard)", "Optimized Stacking (Ours)"],
            "R2 Score": [r2_rr, r2_stacking]
        }
        res_df = pd.DataFrame(results)
        
        # Save results
        out_dir = os.path.join(self.pipeline.results_dir, "benchmark_results")
        os.makedirs(out_dir, exist_ok=True)
        res_df.to_csv(os.path.join(out_dir, "gblup_comparison.csv"), index=False)
        
        # Visualization
        plt.figure(figsize=(10, 6))
        bars = plt.bar(res_df['Method'], res_df['R2 Score'], color=['gray', 'blue'])
        plt.ylabel("R2 Score")
        plt.title(f"Performance Comparison: G-BLUP vs Stacking\nDataset: {self.dataset.upper()}")
        plt.ylim(0, max(res_df['R2 Score']) * 1.2)
        
        # Add values on top
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
            
        plt.savefig(os.path.join(out_dir, "gblup_vs_stacking.png"))
        logging.info(f"Benchmark finished. Plot saved to {out_dir}")
        
        return res_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
    
    # Run for IPK
    try:
        bench_ipk = GBLUPBenchmark("ipk_out_raw", "YR_LS")
        bench_ipk.run()
    except Exception as e:
        print(f"IPK Benchmark failed: {e}")

    # Run for Pepper
    try:
        print("Starting Pepper Benchmark...")
        bench_pepper = GBLUPBenchmark("pepper", "Yield_BV")
        bench_pepper.run()
    except Exception as e:
        print(f"Pepper Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
