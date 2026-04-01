import os
import json
import logging
import time
import numpy as np
import pandas as pd
import joblib
import shutil
from datetime import datetime
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Configuration for Qwen3Coder random forest model
CONFIG = {
    "DATASET": "pepper",
    "OVERWRITE_PREVIOUS": True,
    "RANDOM_SEED": 42,
    "OUTDIR": "03_modeling_results/pepper_augmented/qwen3coder",
    "PROCESSED": "02_processed_data/pepper",
    "AUGMENTED": "04_augmentation/pepper/qwen3coder",
    "SYNTHETIC_Y": "filtered_synthetic_y.csv",
    "OUTER_CV_FOLDS": 2,
    "INNER_CV_FOLDS": 2,
    "MAX_FEATURES": 2000,
    "TARGET_COLUMN": "Yield_BV",
    "MODEL_NAME": "random_forest_qwen3coder"
}

# Set random seeds for reproducibility
np.random.seed(CONFIG["RANDOM_SEED"])

# Create output directories if they don't exist
if CONFIG["OVERWRITE_PREVIOUS"] and os.path.exists(CONFIG["OUTDIR"]):
    print(f"Overwriting previous results in {CONFIG['OUTDIR']}...")
    shutil.rmtree(CONFIG["OUTDIR"])

for subdir in ["logs", "models", "metrics"]:
    os.makedirs(os.path.join(CONFIG["OUTDIR"], subdir), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(CONFIG["OUTDIR"], "logs", "pipeline.log")),
        logging.StreamHandler()
    ]
)

logging.info("Starting Qwen3Coder Random Forest Modeling Pipeline")
logging.info(f"Configuration: {json.dumps(CONFIG, indent=2)}")

def load_data():
    """Load and prepare data for modeling"""
    logging.info("Loading data...")
    
    # Load real data
    real_X_path = os.path.join(CONFIG["PROCESSED"], "X.csv")
    real_y_path = os.path.join(CONFIG["PROCESSED"], "y.csv")
    
    # Load Qwen3Coder synthetic y data
    synthetic_y_path = os.path.join(CONFIG["AUGMENTED"], CONFIG["SYNTHETIC_Y"])
    
    # Read synthetic SNPs to get the list of SNP columns we need
    synthetic_snps_path = os.path.join(CONFIG["AUGMENTED"], "synthetic_snps.csv")
    synthetic_snps = pd.read_csv(synthetic_snps_path)
    
    # Get all SNP columns (exclude Sample_ID)
    snp_columns = [col for col in synthetic_snps.columns if col != "Sample_ID"]
    logging.info(f"Found {len(snp_columns)} SNP columns in synthetic SNPs")
    
    # Get column indices for faster loading
    logging.info(f"Getting column indices for SNP columns...")
    
    # Read just the header of real_X to get column names and their indices
    with open(real_X_path, 'r') as f:
        header_line = f.readline().strip()
        header_columns = header_line.split(',')
    
    # Create a mapping from column name to index
    column_index_map = {column: idx for idx, column in enumerate(header_columns)}
    
    # Get the indices of the columns we need: Sample_ID + snp_columns
    columns_to_load = ["Sample_ID"] + snp_columns
    column_indices = [column_index_map[col] for col in columns_to_load if col in column_index_map]
    
    logging.info(f"Loaded {len(column_indices)} out of {len(columns_to_load)} requested columns")
    
    # Read CSV files using column indices for faster loading
    real_X = pd.read_csv(real_X_path, skiprows=[1, 2, 3], usecols=column_indices)
    real_y = pd.read_csv(real_y_path)
    synthetic_y = pd.read_csv(synthetic_y_path)
    
    logging.info(f"Real X shape: {real_X.shape}")
    logging.info(f"Real y shape: {real_y.shape}")
    logging.info(f"Qwen3Coder synthetic y shape: {synthetic_y.shape}")
    
    # Filter numeric columns (exclude Sample_ID)
    numeric_cols = [col for col in real_X.columns if col != 'Sample_ID']
    real_X_numeric = real_X[numeric_cols]
    logging.info(f"Real X after filtering numeric columns: {real_X_numeric.shape}")
    
    # Align real X and y to have the same number of samples
    min_samples = min(len(real_X_numeric), len(real_y))
    real_X_numeric = real_X_numeric.head(min_samples)
    real_y_filtered = real_y[CONFIG["TARGET_COLUMN"]].head(min_samples)
    logging.info(f"Real X and y after alignment: {len(real_X_numeric)} samples")
    
    # Generate synthetic X by sampling from real X (same number as synthetic y)
    synthetic_X = real_X_numeric.sample(n=len(synthetic_y), replace=True, random_state=CONFIG["RANDOM_SEED"])
    logging.info(f"Generated synthetic X shape: {synthetic_X.shape}")
    
    # Merge real and synthetic data
    X_combined = pd.concat([real_X_numeric, synthetic_X], axis=0, ignore_index=True)
    y_combined = pd.concat([real_y_filtered, synthetic_y[CONFIG["TARGET_COLUMN"]]], axis=0, ignore_index=True)
    
    logging.info(f"Combined X shape: {X_combined.shape}")
    logging.info(f"Combined y shape: {y_combined.shape}")
    
    # Convert to numpy arrays
    X_array = X_combined.values.astype(float)
    y_array = y_combined.values.astype(float)
    
    # Remove NaN values from y
    valid_indices = ~np.isnan(y_array)
    X_clean = X_array[valid_indices].reshape(-1, X_array.shape[1])
    y_clean = y_array[valid_indices].reshape(-1, 1)
    logging.info(f"Data after removing NaNs: {X_clean.shape}, {y_clean.shape}")
    
    return X_clean, y_clean

def train_random_forest(X, y):
    """Train random forest model with hyperparameter tuning"""
    logging.info("Training random forest model...")
    
    # Define hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "max_features": [CONFIG["MAX_FEATURES"]],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "bootstrap": [True, False]
    }
    
    # Create outer CV folds
    outer_cv = KFold(n_splits=CONFIG["OUTER_CV_FOLDS"], shuffle=True, random_state=CONFIG["RANDOM_SEED"])
    
    results = {
        "cv_scores": [],
        "best_models": [],
        "best_params": [],
        "feature_importances": []
    }
    
    fold = 1
    for train_idx, test_idx in outer_cv.split(X):
        logging.info(f"Processing outer fold {fold}/{CONFIG['OUTER_CV_FOLDS']}")
        
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        
        logging.info(f"Fold {fold} - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Inner CV for hyperparameter tuning
        rf = RandomForestRegressor(random_state=CONFIG["RANDOM_SEED"])
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=CONFIG["INNER_CV_FOLDS"], scoring='r2', n_jobs=-1)
        
        grid_search.fit(X_train, y_train.ravel())
        best_rf = grid_search.best_estimator_
        
        # Evaluate on outer fold test set
        y_pred = best_rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        logging.info(f"Fold {fold} - R2: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
        logging.info(f"Fold {fold} - Best params: {grid_search.best_params_}")
        
        # Store results
        results["cv_scores"].append({
            "fold": fold,
            "r2": r2,
            "mse": mse,
            "mae": mae
        })
        
        results["best_models"].append(best_rf)
        results["best_params"].append(grid_search.best_params_)
        results["feature_importances"].append(best_rf.feature_importances_)
        
        fold += 1
    
    return results

def save_results(results):
    """Save modeling results to files"""
    logging.info("Saving results...")
    
    # Calculate mean and standard deviation of scores
    mean_r2 = np.mean([res['r2'] for res in results['cv_scores']])
    std_r2 = np.std([res['r2'] for res in results['cv_scores']])
    mean_mse = np.mean([res['mse'] for res in results['cv_scores']])
    std_mse = np.std([res['mse'] for res in results['cv_scores']])
    mean_mae = np.mean([res['mae'] for res in results['cv_scores']])
    std_mae = np.std([res['mae'] for res in results['cv_scores']])
    
    logging.info(f"Mean R2: {mean_r2:.4f} ± {std_r2:.4f}")
    logging.info(f"Mean MSE: {mean_mse:.4f} ± {std_mse:.4f}")
    logging.info(f"Mean MAE: {mean_mae:.4f} ± {std_mae:.4f}")
    
    # Save summary statistics
    summary = {
        "mean_r2": mean_r2,
        "std_r2": std_r2,
        "mean_mse": mean_mse,
        "std_mse": std_mse,
        "mean_mae": mean_mae,
        "std_mae": std_mae,
        "cv_scores": results["cv_scores"],
        "best_params": results["best_params"],
        "config": CONFIG
    }
    
    summary_path = os.path.join(CONFIG["OUTDIR"], "metrics", "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save best models
    for idx, model in enumerate(results["best_models"]):
        model_path = os.path.join(CONFIG["OUTDIR"], "models", f"rf_model_fold_{idx+1}.pkl")
        joblib.dump(model, model_path)
    
    logging.info(f"Saved summary to {summary_path}")
    logging.info(f"Saved {len(results['best_models'])} models")

def main():
    """Main pipeline execution"""
    try:
        X, y = load_data()
        results = train_random_forest(X, y)
        save_results(results)
        logging.info("Random Forest modeling pipeline completed successfully!")
    except Exception as e:
        logging.error(f"Error in pipeline: {e}")
        logging.exception(e)

if __name__ == "__main__":
    main()