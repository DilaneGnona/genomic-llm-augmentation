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

# Configuration for GLM46 random forest model
CONFIG = {
    "DATASET": "pepper",
    "OVERWRITE_PREVIOUS": True,
    "RANDOM_SEED": 42,
    "OUTDIR": "03_modeling_results/pepper_augmented/glm46",
    "PROCESSED": "02_processed_data/pepper",
    "AUGMENTED": "04_augmentation/pepper",
    "GLM46_Y": "model_sources/glm46/synthetic_y_glm46_filtered.csv",
    "OUTER_CV_FOLDS": 2,  # 2x2 CV outer folds
    "INNER_CV_FOLDS": 2,  # 2x2 CV inner folds for hyperparameter tuning
    "MAX_FEATURES": 2000,  # k=2000 as requested
    "TARGET_COLUMN": "YR_LS",  # Target variable column name
    "MODEL_NAME": "random_forest_glm46"
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

logging.info("Starting GLM46 Random Forest Modeling Pipeline")
logging.info(f"Configuration: {json.dumps(CONFIG, indent=2)}")

def load_data():
    """Load and prepare data for modeling"""
    logging.info("Loading data...")
    
    # Load real data
    real_X_path = os.path.join(CONFIG["PROCESSED"], "X.csv")
    real_y_path = os.path.join(CONFIG["PROCESSED"], "y.csv")
    
    # Load GLM46 synthetic y data
    synthetic_y_path = os.path.join(CONFIG["AUGMENTED"], CONFIG["GLM46_Y"])
    
    # Read CSV files
    real_X = pd.read_csv(real_X_path)
    real_y = pd.read_csv(real_y_path)
    synthetic_y = pd.read_csv(synthetic_y_path)
    
    logging.info(f"Real X shape: {real_X.shape}")
    logging.info(f"Real y shape: {real_y.shape}")
    logging.info(f"GLM46 synthetic y shape: {synthetic_y.shape}")
    
    # Filter numeric columns (exclude Sample_ID and non-numeric columns like POS, CHR)
    numeric_cols = [col for col in real_X.columns if col != 'Sample_ID' and real_X[col].dtype in ['int64', 'float64']]
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
    nan_mask = ~np.isnan(y_array)
    X_array = X_array[nan_mask]
    y_array = y_array[nan_mask]
    
    logging.info(f"After removing NaNs: X shape = {X_array.shape}, y shape = {y_array.shape}")
    
    return X_array, y_array

def train_random_forest(X, y):
    """Train random forest with 2x2 CV and k=2000"""
    logging.info("Training Random Forest model...")
    
    # Define outer and inner CV strategies
    outer_cv = KFold(n_splits=CONFIG["OUTER_CV_FOLDS"], shuffle=True, random_state=CONFIG["RANDOM_SEED"])
    inner_cv = KFold(n_splits=CONFIG["INNER_CV_FOLDS"], shuffle=True, random_state=CONFIG["RANDOM_SEED"])
    
    # Define random forest parameters with k=2000 max features
    rf_params = {
        'n_estimators': [100],  # Number of trees
        'max_depth': [10, 20],  # Max tree depth
        'min_samples_split': [2],  # Min samples to split a node
        'min_samples_leaf': [1],  # Min samples in a leaf node
        'max_features': [CONFIG["MAX_FEATURES"]]  # k=2000 as requested
    }
    
    # Create random forest estimator
    rf = RandomForestRegressor(random_state=CONFIG["RANDOM_SEED"])
    
    # Perform nested cross-validation with grid search
    logging.info(f"Performing nested CV with outer folds: {CONFIG['OUTER_CV_FOLDS']}, inner folds: {CONFIG['INNER_CV_FOLDS']}")
    logging.info(f"Random Forest parameters: {json.dumps(rf_params, indent=2)}")
    
    best_models = []
    outer_scores = []
    
    fold_idx = 0
    for train_idx, test_idx in outer_cv.split(X):
        fold_idx += 1
        logging.info(f"Processing outer fold {fold_idx}/{CONFIG['OUTER_CV_FOLDS']}")
        
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        
        # Grid search in inner CV
        grid_search = GridSearchCV(estimator=rf, param_grid=rf_params, cv=inner_cv, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train_outer, y_train_outer)
        
        best_rf = grid_search.best_estimator_
        best_models.append(best_rf)
        
        # Evaluate on outer test set
        y_pred_outer = best_rf.predict(X_test_outer)
        r2 = r2_score(y_test_outer, y_pred_outer)
        mse = mean_squared_error(y_test_outer, y_pred_outer)
        mae = mean_absolute_error(y_test_outer, y_pred_outer)
        rmse = np.sqrt(mse)
        
        outer_scores.append({
            'fold': fold_idx,
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'best_params': grid_search.best_params_
        })
        
        logging.info(f"Outer fold {fold_idx} results: R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
    
    # Calculate average scores
    avg_r2 = np.mean([s['r2'] for s in outer_scores])
    avg_mse = np.mean([s['mse'] for s in outer_scores])
    avg_mae = np.mean([s['mae'] for s in outer_scores])
    avg_rmse = np.sqrt(avg_mse)
    
    logging.info(f"Average results: R² = {avg_r2:.4f}, RMSE = {avg_rmse:.4f}, MAE = {avg_mae:.4f}")
    
    return {
        'models': best_models,
        'outer_scores': outer_scores,
        'average_scores': {
            'r2': avg_r2,
            'mse': avg_mse,
            'mae': avg_mae,
            'rmse': avg_rmse
        }
    }

def save_results(results):
    """Save results to files"""
    logging.info("Saving results...")
    
    # Save models separately (joblib for non-serializable objects)
    models = results.pop('models', [])
    for i, model in enumerate(models):
        model_path = os.path.join(CONFIG["OUTDIR"], "models", f"rf_model_fold_{i+1}.joblib")
        joblib.dump(model, model_path)
    
    # Save metrics (JSON for serializable data)
    metrics_path = os.path.join(CONFIG["OUTDIR"], "metrics", "rf_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Saved metrics to: {metrics_path}")
    logging.info(f"Saved {len(models)} models")

def main():
    """Main pipeline"""
    try:
        # Load data
        X, y = load_data()
        
        # Train random forest
        results = train_random_forest(X, y)
        
        # Save results
        save_results(results)
        
        logging.info("GLM46 Random Forest Modeling Pipeline completed successfully!")
        
    except Exception as e:
        logging.exception(f"Error during pipeline execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()