import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime

# Setup logging
import logging
log_dir = os.path.join('03_modeling_results', 'pepper_7268809', 'logs')
os.makedirs(log_dir, exist_ok=True)

log_filename = os.path.join(log_dir, f'diagnostics_pepper_7268809_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def verify_sample_alignment():
    """Verify sample alignment across data files"""
    logger.info("Verifying sample alignment across data files...")
    
    # Load data files
    X = pd.read_csv(os.path.join('02_processed_data', 'pepper_7268809', 'X.csv'), index_col=0)
    y = pd.read_csv(os.path.join('02_processed_data', 'pepper_7268809', 'y.csv'), index_col=0)
    pca_covariates = pd.read_csv(os.path.join('02_processed_data', 'pepper_7268809', 'pca_covariates.csv'), index_col=0)
    
    # Check for index names
    has_index = not (X.index.name is None or X.index.name == '')
    
    if has_index:
        logger.info(f"Sample IDs found in index with name: {X.index.name}")
        # Verify indices match
        if X.index.equals(y.index) and X.index.equals(pca_covariates.index):
            logger.info("✅ Sample IDs match across all data files")
            return True, X, y, pca_covariates
        else:
            # Find mismatches
            x_indices = set(X.index)
            y_indices = set(y.index)
            pca_indices = set(pca_covariates.index)
            
            x_y_diff = x_indices.symmetric_difference(y_indices)
            x_pca_diff = x_indices.symmetric_difference(pca_indices)
            
            logger.error(f"❌ Sample ID mismatch found!")
            logger.error(f"X vs y mismatches: {len(x_y_diff)} samples")
            if x_y_diff:
                logger.error(f"First 10 mismatched samples: {list(x_y_diff)[:10]}")
            logger.error(f"X vs PCA mismatches: {len(x_pca_diff)} samples")
            if x_pca_diff:
                logger.error(f"First 10 mismatched samples: {list(x_pca_diff)[:10]}")
            return False, X, y, pca_covariates
    else:
        logger.info("No sample IDs found in index, checking row order...")
        # Check row count
        if len(X) == len(y) == len(pca_covariates):
            logger.info(f"✅ Row counts match: {len(X)} samples in each file")
            return True, X, y, pca_covariates
        else:
            logger.error(f"❌ Row count mismatch!")
            logger.error(f"X rows: {len(X)}, y rows: {len(y)}, PCA rows: {len(pca_covariates)}")
            return False, X, y, pca_covariates

def compute_target_statistics(y):
    """Compute statistics for the target variable"""
    logger.info("Computing target variable statistics...")
    
    # Convert to numpy array if needed
    if isinstance(y, pd.DataFrame):
        y_values = y.iloc[:, 0].values
        target_name = y.columns[0]
    else:
        y_values = y
        target_name = "target"
    
    # Basic statistics
    stats = {
        "target_name": target_name,
        "count": int(np.count_nonzero(~np.isnan(y_values))),
        "count_nan": int(np.count_nonzero(np.isnan(y_values))),
        "mean": float(np.nanmean(y_values)),
        "std": float(np.nanstd(y_values)),
        "min": float(np.nanmin(y_values)),
        "max": float(np.nanmax(y_values)),
        "skew": float(pd.Series(y_values).skew()),
        "percentiles": {
            "25%": float(np.nanpercentile(y_values, 25)),
            "50%": float(np.nanpercentile(y_values, 50)),
            "75%": float(np.nanpercentile(y_values, 75))
        }
    }
    
    logger.info(f"Target stats: count={stats['count']}, mean={stats['mean']:.4f}, "
                f"std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")
    
    # Save to file
    metrics_dir = os.path.join('03_modeling_results', 'pepper_7268809', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    stats_file = os.path.join(metrics_dir, 'target_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Target statistics saved to {stats_file}")
    return stats

def compute_baseline_performance(X, y):
    """Compute baseline model performance (predict mean)"""
    logger.info("Computing baseline model performance (predict mean)...")
    
    # Handle NaN values in targets first
    if isinstance(y, pd.DataFrame):
        # Create a copy to avoid modifying original data
        y_clean = y.copy()
        # Remove rows with NaN in target
        valid_indices = y_clean.dropna().index
        X_clean = X.loc[valid_indices]
        y_clean = y_clean.loc[valid_indices]
        logger.info(f"Filtered out {len(y) - len(y_clean)} rows with NaN target values")
    else:
        # For numpy arrays
        valid_mask = ~np.isnan(y)
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        logger.info(f"Filtered out {len(y) - len(y_clean)} rows with NaN target values")
    
    # Create train/test split (70/30) with fixed seed
    np.random.seed(42)
    indices = np.arange(len(X_clean))
    np.random.shuffle(indices)
    train_size = int(0.7 * len(X_clean))
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Compute mean on train only
    if isinstance(y_clean, pd.DataFrame):
        y_train = y_clean.iloc[train_indices, 0].values
        y_test = y_clean.iloc[test_indices, 0].values
    else:
        y_train = y_clean[train_indices]
        y_test = y_clean[test_indices]
    
    # Filter NaN values from train (additional check)
    y_train_clean = y_train[~np.isnan(y_train)]
    if len(y_train_clean) < len(y_train):
        logger.warning(f"Found {len(y_train) - len(y_train_clean)} NaN values in train targets, filtering them")
    
    # Compute mean on clean train data
    mean_prediction = np.mean(y_train_clean) if len(y_train_clean) > 0 else 0
    
    # Filter NaN values from test (additional check)
    valid_test_mask = ~np.isnan(y_test)
    y_test_clean = y_test[valid_test_mask]
    
    if len(y_test_clean) < len(y_test):
        logger.warning(f"Found {len(y_test) - len(y_test_clean)} NaN values in test targets, filtering them")
    
    baseline_predictions = np.full_like(y_test_clean, mean_prediction)
    
    # Compute metrics only if we have valid test samples
    if len(y_test_clean) > 0:
        baseline_metrics = {
            "baseline_type": "predict_mean",
            "original_sample_count": len(X),
            "clean_sample_count": len(X_clean),
            "train_size": len(train_indices),
            "train_size_clean": len(y_train_clean),
            "test_size": len(test_indices),
            "test_size_clean": len(y_test_clean),
            "mean_prediction": float(mean_prediction),
            "test_r2": float(r2_score(y_test_clean, baseline_predictions)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test_clean, baseline_predictions))),
            "test_mae": float(mean_absolute_error(y_test_clean, baseline_predictions))
        }
        
        logger.info(f"Baseline performance (predict mean):")
        logger.info(f"  Test R²: {baseline_metrics['test_r2']:.4f}")
        logger.info(f"  Test RMSE: {baseline_metrics['test_rmse']:.4f}")
        logger.info(f"  Test MAE: {baseline_metrics['test_mae']:.4f}")
    else:
        logger.error("No valid test samples after filtering NaN values")
        baseline_metrics = {
            "baseline_type": "predict_mean",
            "original_sample_count": len(X),
            "clean_sample_count": len(X_clean),
            "train_size": len(train_indices),
            "train_size_clean": len(y_train_clean),
            "test_size": len(test_indices),
            "test_size_clean": 0,
            "mean_prediction": float(mean_prediction),
            "test_r2": None,
            "test_rmse": None,
            "test_mae": None
        }
    
    return baseline_metrics

def main():
    """Main diagnostics function"""
    logger.info("="*50)
    logger.info("STARTING DATASET DIAGNOSTICS FOR PEPPER_7268809")
    logger.info("="*50)
    
    # Verify sample alignment
    is_aligned, X, y, pca_covariates = verify_sample_alignment()
    
    if not is_aligned:
        logger.error("❌ Diagnostics failed due to sample alignment issues")
        logger.error("Please fix the data files before proceeding with modeling.")
        return False
    
    # Compute target statistics
    target_stats = compute_target_statistics(y)
    
    # Compute baseline performance
    baseline_metrics = compute_baseline_performance(X, y)
    
    # Save comprehensive diagnostics report
    diagnostics_report = {
        "timestamp": datetime.now().isoformat(),
        "sample_alignment": {
            "status": "aligned" if is_aligned else "misaligned",
            "sample_count": len(X)
        },
        "target_statistics": target_stats,
        "baseline_performance": baseline_metrics
    }
    
    report_file = os.path.join('03_modeling_results', 'pepper_7268809', 'metrics', 'diagnostics_report.json')
    with open(report_file, 'w') as f:
        json.dump(diagnostics_report, f, indent=2)
    
    logger.info(f"Comprehensive diagnostics report saved to {report_file}")
    logger.info("="*50)
    logger.info("DIAGNOSTICS COMPLETED SUCCESSFULLY")
    logger.info("="*50)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)