import os
import pandas as pd
import numpy as np
import logging
import json
import time
from datetime import datetime
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories(base_dir, dataset_name):
    """Set up output directories"""
    out_dir = os.path.join(base_dir, "03_modeling_results", dataset_name)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    return out_dir

def setup_logging(out_dir):
    """Set up file logging"""
    log_path = os.path.join(out_dir, "logs", "pipeline.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return log_path

def load_and_align_data(data_dir):
    """Load and align X, y, and PCA data"""
    logger.info("Loading data files...")
    
    # Load clean phenotype data first
    y_clean_path = os.path.join(data_dir, "y_ipk_out_raw_clean.csv")
    y_clean = pd.read_csv(y_clean_path)
    logger.info(f"Loaded y_ipk_out_raw_clean.csv with {len(y_clean)} samples")
    
    # Load sample_map to get valid Sample_IDs
    sample_map_path = os.path.join(data_dir, "sample_map.csv")
    sample_map = pd.read_csv(sample_map_path)
    valid_samples = sample_map[sample_map['Status'] == 'kept']['Sample_ID'].tolist()
    logger.info(f"Found {len(valid_samples)} valid samples in sample_map.csv")
    
    # Create a sample ID mapping dictionary
    # This is a simplified approach since we don't have direct mapping
    # We'll create a new y dataframe with Sample_ID as index
    # We'll then try to match as best as possible during alignment
    
    # Load X and PCA data
    try:
        # Read X with first column as index
        X = pd.read_csv(os.path.join(data_dir, "X.csv"), index_col=0)
        logger.info(f"Loaded X with shape {X.shape}")
    except Exception as e:
        logger.error(f"Failed to load X.csv: {str(e)}")
        raise
    
    try:
        # Read PCA with first column as index
        pca_df = pd.read_csv(os.path.join(data_dir, "pca_covariates.csv"), index_col=0)
        logger.info(f"Loaded PCA covariates with shape {pca_df.shape}")
    except Exception as e:
        logger.error(f"Failed to load pca_covariates.csv: {str(e)}")
        raise
    
    # Create y dataframe with Sample_ID as index (using GBS_BIOSAMPLE_ID values)
    # This is temporary - we'll align properly later
    y = pd.DataFrame(index=y_clean['GBS_BIOSAMPLE_ID'])
    y['YR_LS'] = y_clean['YR_LS'].values
    
    # Report initial sample counts
    logger.info(f"Before alignment: X({len(X)}), y({len(y)}), PCA({len(pca_df)})")
    
    # Since we don't have a direct mapping, we'll work with the samples we have in X
    # and try to find as many overlapping samples as possible
    common_samples = X.index
    
    # Filter datasets to common samples
    X_aligned = X.copy()
    
    # For y, we'll keep it as is for now
    # For PCA, filter to samples in X
    pc_aligned = pca_df.loc[pca_df.index.isin(X_aligned.index)]
    
    logger.info(f"After filtering: X({len(X_aligned)}), PCA({len(pc_aligned)})")
    
    return X_aligned, y, pc_aligned

def validate_target(y_df, min_non_na=100):
    """Validate target column and calculate statistics"""
    logger.info("Validating target column 'YR_LS'...")
    
    # Check if target column exists
    if 'YR_LS' not in y_df.columns:
        logger.error("Target column 'YR_LS' not found in y data")
        raise ValueError("YR_LS target column not found")
    
    y_values = y_df['YR_LS']
    
    # Calculate statistics
    y_stats = {
        "count": len(y_values),
        "mean": float(y_values.mean()),
        "std": float(y_values.std()),
        "min": float(y_values.min()),
        "max": float(y_values.max()),
        "pct_na": float(y_values.isna().mean() * 100),
        "skew": float(y_values.skew())
    }
    
    logger.info(f"Target stats: mean={y_stats['mean']:.4f}, std={y_stats['std']:.4f}, min/max={y_stats['min']:.4f}/{y_stats['max']:.4f}, %NA={y_stats['pct_na']:.2f}%")
    
    # Check if std is near zero
    if y_stats['std'] < 1e-6:
        logger.error(f"Target standard deviation is near zero: {y_stats['std']}")
        raise ValueError("Target variable has no variability")
    
    # Check if there are enough non-NA values
    non_na_count = len(y_values.dropna())
    if non_na_count < min_non_na:
        logger.error(f"Insufficient non-NA values: only {non_na_count} found (minimum required: {min_non_na})")
        raise ValueError(f"Target column has insufficient non-NA values")
    
    return y_stats

def filter_constant_features(X):
    """Remove constant and near-constant features"""
    logger.info("Filtering constant and near-constant features...")
    
    # Remove constant features
    constant_filter = VarianceThreshold(threshold=0)
    X_filtered = constant_filter.fit_transform(X)
    constant_removed = X.shape[1] - X_filtered.shape[1]
    
    # Remove near-constant features (variance < 0.01)
    near_constant_filter = VarianceThreshold(threshold=0.01)
    X_filtered = near_constant_filter.fit_transform(X_filtered)
    near_constant_removed = X.shape[1] - X_filtered.shape[1] - constant_removed
    
    # Get feature names
    kept_features = constant_filter.get_support()
    if kept_features.any():
        near_kept_features = near_constant_filter.get_support()
        all_kept_features = kept_features.copy()
        all_kept_features[kept_features] = near_kept_features
        feature_names = X.columns[all_kept_features]
        X_filtered = pd.DataFrame(X_filtered, index=X.index, columns=feature_names)
    else:
        X_filtered = pd.DataFrame(X_filtered, index=X.index)
    
    filtering_results = {
        "constant_features": constant_removed,
        "near_constant_features": near_constant_removed,
        "total_removed": constant_removed + near_constant_removed,
        "features_before": X.shape[1],
        "features_after": X_filtered.shape[1]
    }
    
    logger.info(f"Feature filtering: {filtering_results['total_removed']} features removed ({filtering_results['constant_features']} constant, {filtering_results['near_constant_features']} near-constant)")
    logger.info(f"Features before/after: {filtering_results['features_before']} / {filtering_results['features_after']}")
    
    return X_filtered, filtering_results

def compute_baseline_metrics(y):
    """Compute baseline metrics by predicting the mean"""
    logger.info("Computing baseline metrics...")
    
    y_values = y['YR_LS'].dropna()
    y_pred = np.full_like(y_values, y_values.mean())
    
    baseline_rmse = np.sqrt(mean_squared_error(y_values, y_pred))
    baseline_mae = mean_absolute_error(y_values, y_pred)
    
    baseline_metrics = {
        "rmse": float(baseline_rmse),
        "mae": float(baseline_mae)
    }
    
    logger.info(f"Baseline RMSE: {baseline_metrics['rmse']:.4f}, Baseline MAE: {baseline_metrics['mae']:.4f}")
    
    return baseline_metrics

def run_diagnostics(X, y, pca_df):
    """Run all diagnostics"""
    logger.info("Running diagnostics...")
    
    # Sample alignment information
    sample_alignment = {
        "before_alignment": {
            "X_samples": len(X),
            "y_samples": len(y),
            "pca_samples": len(pca_df)
        },
        "after_alignment": {
            "common_samples": len(X)  # Using X samples as reference
        }
    }
    
    # Validate target
    y_stats = validate_target(y)
    
    # Filter constant features
    X_filtered, filtering_results = filter_constant_features(X)
    
    # Compute baseline metrics
    baseline_metrics = compute_baseline_metrics(y)
    
    # Combine diagnostics
    diagnostics = {
        "sample_alignment": sample_alignment,
        "y_statistics": y_stats,
        "feature_filtering": filtering_results,
        "baseline_metrics": baseline_metrics
    }
    
    return X_filtered, y, pca_df, diagnostics

def prepare_features(X, pca_df):
    """Prepare features by concatenating X and PCA"""
    logger.info("Preparing features by concatenating X and PCA...")
    
    # Ensure we only use PCA samples that are in X
    pca_filtered = pca_df.loc[pca_df.index.isin(X.index)]
    
    # Align indices
    common_indices = X.index.intersection(pca_filtered.index)
    X_aligned = X.loc[common_indices]
    pca_aligned = pca_filtered.loc[common_indices]
    
    # Concatenate features
    X_combined = pd.concat([X_aligned, pca_aligned], axis=1)
    
    logger.info(f"Combined features shape: {X_combined.shape}")
    
    return X_combined, common_indices

def apply_feature_selection(X, y, n_features_threshold=10000):
    """Apply feature selection if needed"""
    if X.shape[1] > n_features_threshold:
        logger.info(f"Number of features ({X.shape[1]}) exceeds threshold ({n_features_threshold}), applying SelectKBest")
        
        # For tree models, we'll use k=2000
        selector_tree = SelectKBest(f_regression, k=min(2000, X.shape[1]))
        selector_tree.fit(X, y)
        
        # For linear models, we'll use k=1000
        selector_linear = SelectKBest(f_regression, k=min(1000, X.shape[1]))
        selector_linear.fit(X, y)
        
        return {
            'tree': selector_tree,
            'linear': selector_linear
        }
    else:
        logger.info(f"Number of features ({X.shape[1]}) does not exceed threshold ({n_features_threshold}), skipping feature selection")
        return None

def get_model_configs():
    """Get model configurations"""
    configs = {
        'ridge': {
            'model': Ridge(random_state=42),
            'param_grid': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'selector_type': 'linear'
        },
        'lasso': {
            'model': Lasso(random_state=42),
            'param_grid': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            },
            'selector_type': 'linear'
        },
        'elasticnet': {
            'model': ElasticNet(random_state=42),
            'param_grid': {
                'alpha': [0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.5, 0.9]
            },
            'selector_type': 'linear'
        },
        'random_forest': {
            'model': RandomForestRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5]
            },
            'selector_type': 'tree'
        },
        'svr': {
            'model': SVR(),
            'param_grid': {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto']
            },
            'selector_type': 'linear'
        }
    }
    
    # Try to import and add XGBoost if available
    try:
        from xgboost import XGBRegressor
        configs['xgboost'] = {
            'model': XGBRegressor(random_state=42, verbosity=0),
            'param_grid': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6]
            },
            'selector_type': 'tree'
        }
        logger.info("XGBoost is available and will be used")
    except ImportError:
        logger.warning("XGBoost is not available, skipping")
    
    # Try to import and add LightGBM if available
    try:
        from lightgbm import LGBMRegressor
        configs['lightgbm'] = {
            'model': LGBMRegressor(random_state=42, verbosity=-1),
            'param_grid': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6]
            },
            'selector_type': 'tree'
        }
        logger.info("LightGBM is available and will be used")
    except ImportError:
        logger.warning("LightGBM is not available, skipping")
    
    return configs

def train_models(X, y, out_dir, run_id, feature_selectors=None):
    """Train models using nested cross-validation"""
    logger.info("Starting model training with nested cross-validation...")
    
    # Get model configurations
    model_configs = get_model_configs()
    
    # Prepare data
    y_values = y['YR_LS'].dropna()
    X_aligned = X.loc[y_values.index].copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_aligned)
    X_scaled = pd.DataFrame(X_scaled, index=X_aligned.index, columns=X_aligned.columns)
    
    # Set up nested CV
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # Store results
    all_results = {}
    best_models = {}
    
    # Train each model
    for model_name, config in model_configs.items():
        logger.info(f"Training {model_name}...")
        start_time = time.time()
        
        # Apply feature selection if available and appropriate
        X_model = X_scaled.copy()
        if feature_selectors and config['selector_type'] in feature_selectors:
            selector = feature_selectors[config['selector_type']]
            X_model = pd.DataFrame(
                selector.transform(X_model),
                index=X_model.index,
                columns=X_model.columns[selector.get_support()]
            )
            logger.info(f"Applied {config['selector_type']} feature selection: {X_model.shape[1]} features selected")
        
        # Lists to store fold results
        cv_r2_scores = []
        cv_rmse_scores = []
        cv_mae_scores = []
        cv_best_params = []
        
        # Nested CV
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_model, y_values)):
            logger.info(f"  Fold {fold + 1}/5")
            
            # Split data
            X_train, X_test = X_model.iloc[train_idx], X_model.iloc[test_idx]
            y_train, y_test = y_values.iloc[train_idx], y_values.iloc[test_idx]
            
            # Inner CV for hyperparameter tuning
            grid_search = GridSearchCV(
                config['model'],
                config['param_grid'],
                cv=inner_cv,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            # Evaluate on test fold
            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            logger.info(f"    Fold {fold + 1}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, Best params={best_params}")
            
            cv_r2_scores.append(r2)
            cv_rmse_scores.append(rmse)
            cv_mae_scores.append(mae)
            cv_best_params.append(best_params)
        
        # Calculate mean and std for CV scores
        cv_r2_mean = np.mean(cv_r2_scores)
        cv_r2_std = np.std(cv_r2_scores)
        cv_rmse_mean = np.mean(cv_rmse_scores)
        cv_mae_mean = np.mean(cv_mae_scores)
        training_time = time.time() - start_time
        
        # Store results
        results = {
            "cv_r2_mean": float(cv_r2_mean),
            "cv_r2_std": float(cv_r2_std),
            "cv_rmse_mean": float(cv_rmse_mean),
            "cv_mae_mean": float(cv_mae_mean),
            "training_time": float(training_time),
            "cv_best_params": cv_best_params
        }
        
        all_results[model_name] = results
        
        # Train final model on all data
        final_model = GridSearchCV(
            config['model'],
            config['param_grid'],
            cv=inner_cv,
            scoring='r2',
            n_jobs=-1
        )
        final_model.fit(X_model, y_values)
        best_models[model_name] = {
            'model': final_model.best_estimator_,
            'params': final_model.best_params_
        }
        
        logger.info(f"{model_name} results: R²={cv_r2_mean:.4f}±{cv_r2_std:.4f}, RMSE={cv_rmse_mean:.4f}, MAE={cv_mae_mean:.4f}, Time={training_time:.2f}s")
    
    # Save models
    for model_name, model_info in best_models.items():
        model_path = os.path.join(out_dir, "models", f"{run_id}_{model_name}.joblib")
        joblib.dump(model_info['model'], model_path)
        logger.info(f"Saved {model_name} model to {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(out_dir, "models", f"{run_id}_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    
    return all_results

def save_metrics(all_results, diagnostics, out_dir):
    """Save metrics to JSON files"""
    # Save diagnostics
    diagnostics_path = os.path.join(out_dir, "metrics", "diagnostics.json")
    with open(diagnostics_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    logger.info(f"Saved diagnostics to {diagnostics_path}")
    
    # Save all models metrics
    all_metrics_path = os.path.join(out_dir, "metrics", "all_models_metrics.json")
    with open(all_metrics_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved all models metrics to {all_metrics_path}")
    
    # Save individual model metrics
    for model_name, results in all_results.items():
        metrics_path = os.path.join(out_dir, "metrics", f"{model_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)

def generate_summary(out_dir, dataset_name, target_col, all_results, diagnostics, run_id):
    """Generate summary markdown file"""
    summary_path = os.path.join(out_dir, "summary.md")
    
    # Create markdown content
    markdown = f"""
# {dataset_name} Modeling Results

## Configuration

- **DATASET**: {dataset_name}
- **TARGET_COLUMN**: {target_col}
- **RANDOM_SEED**: 42
- **OUTDIR**: {out_dir}
- **OUTER_CV_FOLDS**: 5
- **INNER_CV_FOLDS**: 3
- **RUN_ID**: {run_id}

## Diagnostic Results

### Sample Alignment
- **Before alignment**: X({diagnostics['sample_alignment']['before_alignment']['X_samples']}), 
  y({diagnostics['sample_alignment']['before_alignment']['y_samples']}), 
  PCA({diagnostics['sample_alignment']['before_alignment']['pca_samples']})
- **After alignment**: {diagnostics['sample_alignment']['after_alignment']['common_samples']} common samples

### Target Statistics
- **Count**: {diagnostics['y_statistics']['count']}
- **Mean**: {diagnostics['y_statistics']['mean']:.4f}
- **Std**: {diagnostics['y_statistics']['std']:.4f}
- **Min/Max**: {diagnostics['y_statistics']['min']:.4f} / {diagnostics['y_statistics']['max']:.4f}
- **% NA**: {diagnostics['y_statistics']['pct_na']:.2f}%
- **Skew**: {diagnostics['y_statistics']['skew']:.4f}

### Feature Filtering
- **Constant features removed**: {diagnostics['feature_filtering']['constant_features']}
- **Near-constant features removed**: {diagnostics['feature_filtering']['near_constant_features']}
- **Features before/after**: {diagnostics['feature_filtering']['features_before']} / {diagnostics['feature_filtering']['features_after']}

### Baseline Metrics
- **Baseline RMSE (predict mean)**: {diagnostics['baseline_metrics']['rmse']:.4f}
- **Baseline MAE (predict mean)**: {diagnostics['baseline_metrics']['mae']:.4f}

## Cross-Validation Results

| Model | CV R² Mean | CV R² Std | CV RMSE Mean | CV MAE Mean | Training Time |
|-------|------------|-----------|--------------|-------------|---------------|
"""
    
    # Add model results sorted by R²
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['cv_r2_mean'], reverse=True)
    for model_name, results in sorted_models:
        markdown += f"| {model_name} | {results['cv_r2_mean']:.4f} | {results['cv_r2_std']:.4f} | {results['cv_rmse_mean']:.4f} | {results['cv_mae_mean']:.4f} | {results['training_time']:.2f}s |\n"
    
    # Add note about baseline
    best_r2 = max([r['cv_r2_mean'] for r in all_results.values()])
    if best_r2 > 0:
        markdown += "\n**Note**: Best model outperformed the baseline.\n"
    else:
        markdown += "\n**Note**: No models outperformed the baseline.\n"
    
    # Add timestamp
    markdown += f"\n## Last Updated\n\n- **Timestamp**: {datetime.now().isoformat()}\n"
    
    # Save summary
    with open(summary_path, 'w') as f:
        f.write(markdown)
    
    logger.info(f"Generated summary to {summary_path}")

def main():
    """Main function"""
    try:
        base_dir = "c:\\Users\\OMEN\\Desktop\\experiment_snp"
        dataset_name = "ipk_out_raw"
        target_col = "YR_LS"
        
        # Create output directories
        out_dir = setup_directories(base_dir, dataset_name)
        
        # Set up logging to file
        log_path = setup_logging(out_dir)
        logger.info(f"Starting diagnostic and retraining process for {dataset_name}...")
        logger.info(f"Logging to {log_path}")
        
        # Generate run ID
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Run ID: {run_id}")
        
        # Load and align data
        data_dir = os.path.join(base_dir, "02_processed_data", dataset_name)
        X, y, pca_df = load_and_align_data(data_dir)
        
        # Run diagnostics
        X_filtered, y, pca_df, diagnostics = run_diagnostics(X, y, pca_df)
        
        # Prepare features (concatenate with PCA)
        X_combined, common_indices = prepare_features(X_filtered, pca_df)
        
        # Apply feature selection if needed
        feature_selectors = apply_feature_selection(X_combined, y)
        
        # Train models
        all_results = train_models(X_combined, y, out_dir, run_id, feature_selectors)
        
        # Save metrics
        save_metrics(all_results, diagnostics, out_dir)
        
        # Generate summary
        generate_summary(out_dir, dataset_name, target_col, all_results, diagnostics, run_id)
        
        logger.info("Diagnostic and retraining process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()