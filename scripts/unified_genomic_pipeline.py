import os
import sys
import time
import json
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import ProjectConfig
from src.models import build_model
from scripts.generate_cl_automatic import CLGenerator

# Setup logging
def setup_logging(out_dir, dataset):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{dataset}_{run_id}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return run_id

class UnifiedGenomicPipeline:
    def __init__(self, dataset, target_col, run_id):
        self.dataset = dataset
        self.target_col = target_col
        self.run_id = run_id
        self.cfg = ProjectConfig("config.yaml")
        
        self.base_dir = r"c:\Users\OMEN\Desktop\experiment_snp"
        self.processed_dir = os.path.join(self.base_dir, "02_processed_data", f"{dataset}_augmented")
        self.results_dir = os.path.join(self.base_dir, "03_modeling_results", f"{dataset}_unified_{run_id}")
        
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "plots"), exist_ok=True)

    def run_augmentation(self, models=["gpt", "deepseek"], samples_per_context=200):
        logging.info(f"Starting Data Augmentation for {self.dataset}...")
        for model in models:
            logging.info(f"  >>> Generating with {model}...")
            gen = CLGenerator(self.dataset, model)
            contexts = [f"{self.dataset}_context_stats.csv", f"{self.dataset}_context_high_var.csv"]
            prompts = ["prompt_A_statistical.txt", "prompt_B_genetic_structure.txt"]
            
            for c, p in zip(contexts, prompts):
                try:
                    gen.generate(c, p, n_samples=samples_per_context)
                except Exception as e:
                    logging.error(f"Error during {model} generation for {c}: {e}")

    def consolidate_data(self):
        logging.info("Consolidating real and synthetic data...")
        # 1. Load Real Data
        real_x_path = os.path.join(self.base_dir, "02_processed_data", self.dataset, "X.csv")
        real_y_path = os.path.join(self.base_dir, "02_processed_data", self.dataset, "y.csv")
        
        if not os.path.exists(real_x_path) or not os.path.exists(real_y_path):
            logging.error("Real data files missing!")
            return None
            
        real_x = pd.read_csv(real_x_path)
        real_y = pd.read_csv(real_y_path)
        
        # Align real data
        common = list(set(real_x['Sample_ID']).intersection(set(real_y['Sample_ID'])))
        real_df = real_x[real_x['Sample_ID'].isin(common)].merge(real_y[['Sample_ID', self.target_col]], on='Sample_ID')
        
        real_features = [c for c in real_df.columns if c not in ['Sample_ID', self.target_col]]
        logging.info(f"Real data loaded: {len(real_df)} samples, {len(real_features)} features.")

        # 2. Load Synthetic Data
        synth_root = os.path.join(self.base_dir, "04_augmentation", self.dataset, "model_sources")
        synth_dfs = []
        synth_features = set()
        
        if os.path.exists(synth_root):
            import glob
            search_pattern = os.path.join(synth_root, "*", "*", "*.csv")
            synth_files = glob.glob(search_pattern)
            logging.info(f"Found {len(synth_files)} potential synthetic files.")
            
            for f in synth_files:
                try:
                    # Use index_col=False to handle trailing commas better
                    df = pd.read_csv(f, index_col=False)
                    
                    # If reading failed or produced 1 column, try with different separator
                    if df.shape[1] <= 1:
                        df = pd.read_csv(f, sep=None, engine='python')
                    
                    # Detect target column
                    target_var = None
                    for tc in [self.target_col, 'Yield_BV', 'YR_LS', 'Yield', 'Yield_bv']:
                        if tc in df.columns:
                            target_var = tc
                            break
                    
                    if not target_var:
                        # Fallback: check if target is the last column
                        last_col = df.columns[-1]
                        if "Sample" not in last_col and df[last_col].dtype in [np.float64, np.int64]:
                            target_var = last_col
                    
                    if not target_var: continue
                    df = df.rename(columns={target_var: self.target_col})
                    df[self.target_col] = pd.to_numeric(df[self.target_col], errors='coerce')
                    df = df.dropna(subset=[self.target_col])
                    if len(df) == 0: continue
                    
                    # Track which SNPs are actually in the synthetic data
                    for c in df.columns:
                        if c in real_features:
                            synth_features.add(c)
                    
                    # Ensure Sample_ID uniqueness by prefixing
                    model_name = f.split(os.sep)[-3]
                    ctx_name = f.split(os.sep)[-2]
                    
                    # Handle Sample_ID column missing or named differently
                    sid_col = None
                    for sc in ['Sample_ID', 'ID', 'sample_id', 'Sample']:
                        if sc in df.columns:
                            sid_col = sc
                            break
                    
                    if sid_col:
                        df['Sample_ID'] = f"{model_name}_{ctx_name}_" + df[sid_col].astype(str)
                    else:
                        # Generate IDs if missing
                        df['Sample_ID'] = [f"{model_name}_{ctx_name}_{i}" for i in range(len(df))]
                    
                    synth_dfs.append(df)
                    logging.info(f"  Loaded {len(df)} samples from {os.path.basename(f)}")
                except Exception as e:
                    logging.warning(f"Could not load {f}: {e}")
        
        # OPTIMIZATION: Only use features that appear in at least one synthetic file
        active_features = sorted(list(synth_features))
        logging.info(f"Using {len(active_features)} active features (present in synthetic data).")
        
        if not active_features:
            logging.error("No common features found between real and synthetic data!")
            return None
            
        # Align all dataframes to active_features
        aligned_dfs = []
        
        # Real data aligned
        real_aligned = real_df[['Sample_ID', self.target_col] + active_features]
        aligned_dfs.append(real_aligned)
        
        # Synthetic data aligned
        for sdf in synth_dfs:
            # Keep only available active features, fill others with 0
            cols_to_keep = ['Sample_ID', self.target_col] + [c for c in active_features if c in sdf.columns]
            sdf_aligned = sdf[cols_to_keep].copy()
            for mf in [c for c in active_features if c not in sdf.columns]:
                sdf_aligned[mf] = 0
            
            # Ensure numeric
            for c in active_features:
                sdf_aligned[c] = pd.to_numeric(sdf_aligned[c], errors='coerce').fillna(0).astype(int)
            
            # Reorder
            sdf_aligned = sdf_aligned[['Sample_ID', self.target_col] + active_features]
            aligned_dfs.append(sdf_aligned)
                                
        if len(aligned_dfs) > 1:
            combined_df = pd.concat(aligned_dfs, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['Sample_ID'])
            logging.info(f"Final dataset size: {len(combined_df)} samples, {combined_df.shape[1]} features.")
            
            out_path = os.path.join(self.processed_dir, "unified_dataset.csv")
            combined_df.to_csv(out_path, index=False)
            return combined_df
        return real_df

    def train_ensemble(self, df, k_features=1000):
        logging.info(f"Starting Ensemble Training (LightGBM + MLP) with k={k_features} features...")
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
        from sklearn.feature_selection import SelectKBest, f_regression
        import torch
        import joblib
        
        X = df.drop(['Sample_ID', self.target_col], axis=1)
        y = df[self.target_col]
        
        # Handle NaNs and Infs
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        y = y.fillna(y.mean())
        
        # Feature Selection
        if X.shape[1] > k_features:
            logging.info(f"  Performing Feature Selection (SelectKBest f_regression, k={k_features})...")
            selector = SelectKBest(f_regression, k=k_features)
            X_selected = selector.fit_transform(X, y)
            feature_names = X.columns[selector.get_support()].tolist()
            X = pd.DataFrame(X_selected, columns=feature_names)
            
            # Save feature names
            with open(os.path.join(self.results_dir, "models", "selected_features.json"), "w") as f:
                json.dump(feature_names, f)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = {}
        
        try:
            # 1. LightGBM
            logging.info("  Training LightGBM...")
            lgbm = build_model("lightgbm")
            if lgbm:
                lgbm.fit(X_train, y_train)
                y_pred_lgbm = lgbm.predict(X_test)
                results['lightgbm'] = {'r2': r2_score(y_test, y_pred_lgbm), 'pred': y_pred_lgbm}
                logging.info(f"    LightGBM R2: {results['lightgbm']['r2']:.4f}")
            else:
                logging.warning("    LightGBM not available, skipping.")
                results['lightgbm'] = {'r2': 0, 'pred': np.zeros_like(y_test)}
            
            # 2. MLP
            logging.info("  Training MLP (50 epochs)...")
            input_dim = X_train.shape[1]
            model_dl = build_model("mlp", input_dim=input_dim)
            
            if model_dl:
                # Simple PyTorch training loop
                X_t = torch.tensor(X_train.values, dtype=torch.float32)
                y_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
                optimizer = torch.optim.Adam(model_dl.parameters(), lr=0.001)
                criterion = torch.nn.MSELoss()
                
                model_dl.train()
                for epoch in range(50):
                    optimizer.zero_grad()
                    outputs = model_dl(X_t)
                    loss = criterion(outputs, y_t)
                    loss.backward()
                    optimizer.step()
                    if (epoch+1) % 10 == 0:
                        logging.info(f"    Epoch {epoch+1}/50, Loss: {loss.item():.4f}")
                    
                model_dl.eval()
                with torch.no_grad():
                    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
                    y_pred_dl = model_dl(X_test_t).numpy().flatten()
                    results['mlp'] = {'r2': r2_score(y_test, y_pred_dl), 'pred': y_pred_dl}
                    logging.info(f"    MLP R2: {results['mlp']['r2']:.4f}")
            else:
                logging.warning("    MLP not available, skipping.")
                results['mlp'] = {'r2': 0, 'pred': np.zeros_like(y_test)}
                
            # 3. Ensemble (Weighted Average)
            logging.info("  Creating Ensemble (Dynamic Weighting based on R2)...")
            # Heuristic: Use weights proportional to R2 if positive, else 0
            w_lgbm = max(0.001, results['lightgbm']['r2']) if results['lightgbm']['r2'] > 0 else 0.001
            w_dl = max(0.001, results.get('mlp', {'r2': 0})['r2']) if results.get('mlp', {'r2': 0})['r2'] > 0 else 0.001
            
            # If both are very low, default to 0.7/0.3 but warn
            if w_lgbm == 0.001 and w_dl == 0.001:
                logging.warning("    Both models have poor R2. Using default 0.7/0.3 weights.")
                w_lgbm, w_dl = 0.7, 0.3
            
            total_w = w_lgbm + w_dl
            y_pred_dl = results.get('mlp', {'pred': np.zeros_like(y_test)})['pred']
            y_pred_ensemble = (w_lgbm * results['lightgbm']['pred'] + w_dl * y_pred_dl) / total_w
            results['ensemble'] = {'r2': r2_score(y_test, y_pred_ensemble), 'pred': y_pred_ensemble}
            
            logging.info(f"    Weights: LightGBM={w_lgbm/total_w:.2f}, MLP={w_dl/total_w:.2f}")
            logging.info(f"Ensemble R2: {results['ensemble']['r2']:.4f}")
        except Exception as e:
            logging.error(f"Error during training: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None
        
        # Save results
        summary = {
            'dataset': self.dataset,
            'run_id': self.run_id,
            'samples': len(df),
            'features': X.shape[1],
            'metrics': {m: {'r2': results[m]['r2']} for m in results if 'r2' in results[m]}
        }
        
        with open(os.path.join(self.results_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
            
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_ensemble, alpha=0.5, color='teal')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("Actual Yield")
        plt.ylabel("Predicted Yield")
        plt.title(f"Genomic Ensemble Prediction - {self.dataset.upper()}\n(R2={results['ensemble']['r2']:.4f}, n={len(df)})")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.results_dir, "plots", "ensemble_scatter.png"))
        
        # Save model
        import joblib
        joblib.dump(lgbm, os.path.join(self.results_dir, "models", "lgbm_final.joblib"))
        if model_dl:
            torch.save(model_dl.state_dict(), os.path.join(self.results_dir, "models", "dl_final.pth"))
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="Unified Genomic Modeling Pipeline")
    parser.add_argument("--dataset", required=True, help="e.g. pepper, ipk_out_raw")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--skip_aug", action="store_true", help="Skip LLM augmentation")
    parser.add_argument("--samples", type=int, default=200, help="Samples per LLM context")
    
    args = parser.parse_args()
    
    run_id = setup_logging("03_modeling_results", args.dataset)
    logging.info(f"Starting Unified Pipeline - Run ID: {run_id}")
    
    pipeline = UnifiedGenomicPipeline(args.dataset, args.target, run_id)
    
    if not args.skip_aug:
        pipeline.run_augmentation(samples_per_context=args.samples)
    
    df = pipeline.consolidate_data()
    if df is not None:
        summary = pipeline.train_ensemble(df)
        if summary:
            logging.info("Pipeline Execution Finished Successfully!")
        else:
            logging.error("Pipeline failed during training.")
            sys.exit(1)
    else:
        logging.error("Pipeline failed at data consolidation step.")
        sys.exit(1)

if __name__ == "__main__":
    main()
