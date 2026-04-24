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
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
import torch
import joblib
import shap

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

    def optimize_hyperparameters(self, X, y, n_trials=20):
        logging.info(f"Starting Bayesian Optimization (Optuna) for LightGBM - {n_trials} trials...")
        
        def objective(trial):
            param = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42
            }
            
            from lightgbm import LGBMRegressor
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
                y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
                
                model = LGBMRegressor(**param)
                model.fit(X_t, y_t)
                preds = model.predict(X_v)
                scores.append(r2_score(y_v, preds))
                
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        logging.info(f"  Best Trial R2: {study.best_value:.4f}")
        logging.info(f"  Best Params: {study.best_params}")
        return study.best_params

    def train_ensemble(self, df, k_features=1000, n_folds=5, n_trials=20):
        logging.info(f"Starting ADVANCED Ensemble Training (Stacking + K-Fold) with k={k_features} features...")
        
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
        
        # Hold-out set for final verification
        X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(X, y, test_size=0.15, random_state=42)
        
        # 1. Hyperparameter Optimization for LightGBM
        best_lgbm_params = self.optimize_hyperparameters(X_train_full, y_train_full, n_trials=n_trials)
        
        # 2. K-Fold Stacking
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        meta_features = np.zeros((len(X_train_full), 2)) # Columns: [LGBM_pred, MLP_pred]
        
        logging.info(f"  Performing {n_folds}-Fold Stacking...")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
            X_tr, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
            y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
            
            # Base Model 1: LightGBM
            from lightgbm import LGBMRegressor
            lgbm = LGBMRegressor(**best_lgbm_params)
            lgbm.fit(X_tr, y_tr)
            meta_features[val_idx, 0] = lgbm.predict(X_val)
            
            # Base Model 2: MLP (PyTorch)
            input_dim = X_tr.shape[1]
            mlp = build_model("mlp", input_dim=input_dim)
            if mlp:
                X_tr_t = torch.tensor(X_tr.values, dtype=torch.float32)
                y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32).view(-1, 1)
                optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
                criterion = torch.nn.MSELoss()
                
                mlp.train()
                for _ in range(30): # Fewer epochs for fold training
                    optimizer.zero_grad()
                    loss = criterion(mlp(X_tr_t), y_tr_t)
                    loss.backward()
                    optimizer.step()
                
                mlp.eval()
                with torch.no_grad():
                    X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
                    meta_features[val_idx, 1] = mlp(X_val_t).numpy().flatten()
            
            logging.info(f"    Fold {fold+1} complete.")
            
        # 3. Train Meta-Learner (Ridge)
        logging.info("  Training Meta-Learner (Ridge)...")
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(meta_features, y_train_full)
        
        # 4. RR-BLUP Baseline (Master Thesis Standard)
        logging.info("  Training RR-BLUP Baseline (G-BLUP equivalent)...")
        from sklearn.linear_model import RidgeCV
        rr_blup = RidgeCV(alphas=np.logspace(-3, 3, 10))
        rr_blup.fit(X_train_full, y_train_full)
        pred_rrblup = rr_blup.predict(X_holdout)
        r2_rrblup = r2_score(y_holdout, pred_rrblup)
        logging.info(f"    RR-BLUP (Standard) Hold-out R2: {r2_rrblup:.4f}")

        # 5. Final Models (Train on full X_train_full)
        logging.info("  Training final base models on full training set...")
        final_lgbm = LGBMRegressor(**best_lgbm_params)
        final_lgbm.fit(X_train_full, y_train_full)
        
        final_mlp = build_model("mlp", input_dim=X.shape[1])
        if final_mlp:
            X_full_t = torch.tensor(X_train_full.values, dtype=torch.float32)
            y_full_t = torch.tensor(y_train_full.values, dtype=torch.float32).view(-1, 1)
            optimizer = torch.optim.Adam(final_mlp.parameters(), lr=0.001)
            criterion = torch.nn.MSELoss()
            final_mlp.train()
            for _ in range(50):
                optimizer.zero_grad()
                loss = criterion(final_mlp(X_full_t), y_full_t)
                loss.backward()
                optimizer.step()
        
        # 5. Final Prediction on Hold-out Set
        logging.info("  Evaluating on Hold-out set...")
        pred_lgbm = final_lgbm.predict(X_holdout)
        
        final_mlp.eval()
        with torch.no_grad():
            X_holdout_t = torch.tensor(X_holdout.values, dtype=torch.float32)
            pred_mlp = final_mlp(X_holdout_t).numpy().flatten()
            
        holdout_meta = np.column_stack([pred_lgbm, pred_mlp])
        pred_stacking = meta_model.predict(holdout_meta)
        
        r2_stack = r2_score(y_holdout, pred_stacking)
        r2_lgbm = r2_score(y_holdout, pred_lgbm)
        r2_mlp = r2_score(y_holdout, pred_mlp)
        
        logging.info(f"    LGBM Hold-out R2: {r2_lgbm:.4f}")
        logging.info(f"    MLP Hold-out R2: {r2_mlp:.4f}")
        logging.info(f"    STACKING Hold-out R2: {r2_stack:.4f}")
        
        # 6. SHAP Interpretability
        logging.info("  Performing SHAP Interpretability analysis...")
        try:
            explainer = shap.TreeExplainer(final_lgbm)
            shap_values = explainer.shap_values(X_holdout)
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_holdout, show=False)
            plt.title(f"SHAP SNP Importance - {self.dataset.upper()}")
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "plots", "shap_importance.png"))
            plt.close()
            
            # Save top 20 SNPs
            if isinstance(shap_values, list): # For multiclass/some versions
                vals = np.abs(shap_values[0]).mean(0)
            else:
                vals = np.abs(shap_values).mean(0)
            
            feature_importance = pd.DataFrame(list(zip(X_holdout.columns, vals)), columns=['SNP', 'feature_importance_vals'])
            feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
            feature_importance.head(20).to_csv(os.path.join(self.results_dir, "models", "top_20_snps.csv"), index=False)
            
        except Exception as e:
            logging.warning(f"  SHAP analysis failed: {e}")
        
        # Save results
        results = {
            'lightgbm': {'r2': r2_lgbm},
            'mlp': {'r2': r2_mlp},
            'rr_blup_standard': {'r2': r2_rrblup},
            'stacking_optimized': {'r2': r2_stack}
        }
        
        summary = {
            'dataset': self.dataset,
            'run_id': self.run_id,
            'samples': len(df),
            'features': X.shape[1],
            'metrics': results,
            'best_params_lgbm': best_lgbm_params,
            'meta_weights': meta_model.coef_.tolist(),
            'meta_intercept': meta_model.intercept_
        }
        
        with open(os.path.join(self.results_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
            
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(y_holdout, pred_stacking, alpha=0.5, color='darkblue', label='Stacking Predictions')
        plt.plot([y_holdout.min(), y_holdout.max()], [y_holdout.min(), y_holdout.max()], 'r--', lw=2)
        plt.xlabel("Actual Yield")
        plt.ylabel("Predicted Yield")
        plt.title(f"ADVANCED Stacking Prediction - {self.dataset.upper()}\n(Hold-out R2={r2_stack:.4f})")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.results_dir, "plots", "ensemble_scatter.png"))
        
        # Save models
        joblib.dump(final_lgbm, os.path.join(self.results_dir, "models", "lgbm_final.joblib"))
        joblib.dump(meta_model, os.path.join(self.results_dir, "models", "meta_learner.joblib"))
        if final_mlp:
            torch.save(final_mlp.state_dict(), os.path.join(self.results_dir, "models", "dl_final.pth"))
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="Unified Genomic Modeling Pipeline")
    parser.add_argument("--dataset", required=True, help="e.g. pepper, ipk_out_raw")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--skip_aug", action="store_true", help="Skip LLM augmentation")
    parser.add_argument("--samples", type=int, default=200, help="Samples per LLM context")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    
    args = parser.parse_args()
    
    run_id = setup_logging("03_modeling_results", args.dataset)
    logging.info(f"Starting Unified Pipeline - Run ID: {run_id}")
    
    pipeline = UnifiedGenomicPipeline(args.dataset, args.target, run_id)
    
    if not args.skip_aug:
        pipeline.run_augmentation(samples_per_context=args.samples)
    
    df = pipeline.consolidate_data()
    if df is not None:
        summary = pipeline.train_ensemble(df, n_trials=args.trials)
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
