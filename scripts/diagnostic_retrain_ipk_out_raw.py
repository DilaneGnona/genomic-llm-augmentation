import os
import json
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "03_modeling_results/ipk_out_raw/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/pipeline_{run_id}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Check for XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost is not available, will skip XGBoost model")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM is not available, will skip LightGBM model")
    LIGHTGBM_AVAILABLE = False

# Constants
DATA_DIR = "02_processed_data/ipk_out_raw"
MODEL_DIR = "03_modeling_results/ipk_out_raw/models"
METRICS_DIR = "03_modeling_results/ipk_out_raw/metrics"
TARGET_COLUMN = "YR_LS"  # Target column as specified in requirements
RANDOM_SEED = 42
OUTER_CV_FOLDS = 5
INNER_CV_FOLDS = 3
OVERWRITE_PREVIOUS = False

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

def load_and_validate_data():
    """Load data files and perform initial validation"""
    logger.info("Loading and validating data files...")
    
    # Load data files
    X_path = os.path.join(DATA_DIR, "X.csv")
    y_path = os.path.join(DATA_DIR, "y_ipk_out_raw_clean.csv")
    pca_path = os.path.join(DATA_DIR, "pca_covariates.csv")
    
    for file_path in [X_path, y_path, pca_path]:
        if not os.path.exists(file_path):
            logger.error(f"Required file not found: {file_path}")
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Load X and PCA with index_col=0 (Sample_ID)
    X = pd.read_csv(X_path, index_col=0)
    pca = pd.read_csv(pca_path, index_col=0)
    
    # Load y with Sample_ID as index
    y = pd.read_csv(y_path, index_col="Sample_ID")
    
    # Note: We've already preprocessed y.csv to have Sample_ID as index
    
    logger.info(f"Loaded X with shape: {X.shape}")
    logger.info(f"Loaded y with shape: {y.shape}")
    logger.info(f"Loaded PCA with shape: {pca.shape}")
    
    # Check if target column exists
    if TARGET_COLUMN not in y.columns:
        logger.error(f"Target column '{TARGET_COLUMN}' not found in y.csv")
        logger.info(f"Available columns in y.csv: {list(y.columns)}")
        # Try to find a suitable numeric target column if YR_LS not found
        numeric_cols = y.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            logger.warning(f"YR_LS not found. Available numeric columns: {numeric_cols}")
            return X, y, pca, None  # Return None for target series to indicate issue
        else:
            raise ValueError(f"No numeric target columns found in y.csv")
    
    return X, y, pca

def run_diagnostics(X, y, pca):
    """Run all diagnostics and save results"""
    logger.info("Running diagnostics...")
    diagnostics = {}
    
    # 1. Sample alignment check
    logger.info("Checking sample alignment...")
    X_samples = set(X.index)
    y_samples = set(y.index)
    pca_samples = set(pca.index)
    
    diagnostics['sample_alignment'] = {
        'before_alignment': {
            'X_samples': len(X_samples),
            'y_samples': len(y_samples),
            'pca_samples': len(pca_samples)
        }
    }
    
    # Find common samples
    common_samples = X_samples.intersection(y_samples).intersection(pca_samples)
    diagnostics['sample_alignment']['after_alignment'] = {
        'common_samples': len(common_samples)
    }
    
    # Filter data to common samples
    X_aligned = X.loc[list(common_samples)].copy()
    y_aligned = y.loc[list(common_samples)].copy()
    pca_aligned = pca.loc[list(common_samples)].copy()
    
    # 2. Y statistics - validate YR_LS column
    logger.info(f"Validating target column: {TARGET_COLUMN}")
    
    # Check if target column exists
    if TARGET_COLUMN not in y_aligned.columns:
        logger.error(f"Target column '{TARGET_COLUMN}' not found in y data")
        raise ValueError(f"{TARGET_COLUMN} target column not found")
    
    y_series = y_aligned[TARGET_COLUMN]
    y_stats = {
        'count': len(y_series),
        'mean': float(y_series.mean()),
        'std': float(y_series.std()),
        'min': float(y_series.min()),
        'max': float(y_series.max()),
        'pct_na': float(y_series.isna().mean() * 100),
        'skew': float(y_series.skew())
    }
    diagnostics['y_statistics'] = y_stats
    
    # Check for issues with y
    non_na_count = len(y_series.dropna())
    if y_series.std() < 1e-6:
        logger.error(f"Target variable has very low variance: std = {y_series.std()}")
        raise ValueError(f"Target variable has very low variance")
    
    # Require at least 100 non-NA values
    if non_na_count < 100:
        logger.error(f"Insufficient non-NA values: only {non_na_count} found (minimum required: 100)")
        raise ValueError(f"Target column has insufficient non-NA values")
    
    # 3. Check for constant/near-constant features
    logger.info("Checking for constant/near-constant features...")
    constant_cols = []
    near_constant_cols = []
    
    for col in X_aligned.columns:
        std = X_aligned[col].std()
        if std == 0:
            constant_cols.append(col)
        elif std < 1e-6:
            near_constant_cols.append(col)
    
    cols_to_drop = constant_cols + near_constant_cols
    X_filtered = X_aligned.drop(columns=cols_to_drop)
    
    diagnostics['feature_filtering'] = {
        'constant_features': len(constant_cols),
        'near_constant_features': len(near_constant_cols),
        'total_removed': len(cols_to_drop),
        'features_before': X_aligned.shape[1],
        'features_after': X_filtered.shape[1]
    }
    
    # 4. Baseline metrics (predict mean)
    logger.info("Calculating baseline metrics...")
    y_clean = y_series.dropna()
    mean_prediction = y_clean.mean() * np.ones_like(y_clean)
    
    baseline_rmse = np.sqrt(mean_squared_error(y_clean, mean_prediction))
    baseline_mae = mean_absolute_error(y_clean, mean_prediction)
    
    diagnostics['baseline_metrics'] = {
        'rmse': float(baseline_rmse),
        'mae': float(baseline_mae)
    }
    
    # Save diagnostics
    with open(os.path.join(METRICS_DIR, "diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)
    
    logger.info(f"Diagnostics saved to {METRICS_DIR}/diagnostics.json")
    
    # Remove NA values from y and corresponding X and PCA
    valid_indices = y_series.dropna().index
    X_final = X_filtered.loc[valid_indices].copy()
    y_final = y_series.loc[valid_indices].copy()
    pca_final = pca_aligned.loc[valid_indices].copy()
    
    return X_final, y_final, pca_final, diagnostics

def prepare_features(X, pca):
    """Prepare features by combining X and PCA, applying feature selection if needed"""
    # Concatenate X and PCA
    X_combined = pd.concat([X, pca], axis=1)
    logger.info(f"Combined feature shape: {X_combined.shape}")
    
    # Check if feature selection is needed
    feature_selection = None
    if X_combined.shape[1] > 10000:
        logger.info("Number of features > 10000, will use SelectKBest for tree/boosting models")
        feature_selection = 'selectkbest'
    
    return X_combined, feature_selection

def train_models(X, y, diagnostics):
    """Train all models with nested CV"""
    logger.info("Starting model training...")
    
    # Model configurations - simplified for small dataset
    models_config = {
        'ridge': {
            'model': Ridge(),
            'params': {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            },
            'needs_selection': False
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [0.001, 0.01, 0.1, 1.0]
            },
            'needs_selection': False
        },
        'elasticnet': {
            'model': ElasticNet(),
            'params': {
                'alpha': [0.001, 0.01, 0.1],
                'l1_ratio': [0.5, 0.9]
            },
            'needs_selection': False
        },
        'random_forest': {
            'model': RandomForestRegressor(random_state=RANDOM_SEED),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10],
                'max_features': ['sqrt']
            },
            'needs_selection': X.shape[1] > 10000
        },
        'svr': {
            'model': SVR(),
            'params': {
                'C': [0.1, 1.0],
                'gamma': ['scale']
            },
            'needs_selection': False
        },
        'xgboost': {
            'model': xgb.XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1) if XGBOOST_AVAILABLE else None,
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.3]
            },
            'needs_selection': X.shape[1] > 10000
        } if XGBOOST_AVAILABLE else {},
        'lightgbm': {
            'model': lgb.LGBMRegressor(random_state=RANDOM_SEED, n_jobs=-1) if LIGHTGBM_AVAILABLE else None,
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.3]
            },
            'needs_selection': X.shape[1] > 10000
        } if LIGHTGBM_AVAILABLE else {}
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models_config['xgboost'] = {
            'model': xgb.XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.3]
            },
            'needs_selection': X.shape[1] > 10000
        }
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models_config['lightgbm'] = {
            'model': lgb.LGBMRegressor(random_state=RANDOM_SEED, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.3]
            },
            'needs_selection': X.shape[1] > 10000
        }
    
    # Initialize results storage
    all_results = {}
    all_models_metrics = {}
    
    # Adjust CV strategy based on sample size
    n_samples = len(y)
    if n_samples < 20:
        # For very small datasets, use fewer folds
        outer_folds = max(2, min(OUTER_CV_FOLDS, n_samples // 3))
        inner_folds = max(2, min(INNER_CV_FOLDS, n_samples // 5))
        logger.warning(f"Small dataset detected ({n_samples} samples). Using {outer_folds} outer folds and {inner_folds} inner folds")
    else:
        outer_folds = OUTER_CV_FOLDS
        inner_folds = INNER_CV_FOLDS
    
    # Outer CV
    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=RANDOM_SEED)
    
    for model_name, config in models_config.items():
        logger.info(f"Training {model_name}...")
        start_time = time.time()
        
        # Lists to store fold results
        fold_r2_scores = []
        fold_rmse_scores = []
        fold_mae_scores = []
        fold_predictions = []
        fold_true_values = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            logger.info(f"  Fold {fold_idx+1}/{OUTER_CV_FOLDS}")
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Feature selection for tree/boosting models if needed
            feature_selector = None
            selected_indices = None
            
            if config['needs_selection']:
                logger.info(f"    Applying SelectKBest for feature selection...")
                # Try different k values
                k_values = [1000, 2000]
                best_k = k_values[0]
                best_score = -np.inf
                
                for k in k_values:
                    selector = SelectKBest(f_regression, k=min(k, X_train.shape[1]))
                    X_train_selected = selector.fit_transform(X_train, y_train)
                    
                    # Quick validation with default model
                    temp_model = config['model'].__class__()
                    temp_model.fit(X_train_selected, y_train)
                    score = temp_model.score(X_train_selected, y_train)
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                
                logger.info(f"    Selected k={best_k} features")
                feature_selector = SelectKBest(f_regression, k=min(best_k, X_train.shape[1]))
                X_train = feature_selector.fit_transform(X_train, y_train)
                X_test = feature_selector.transform(X_test)
                selected_indices = np.where(feature_selector.get_support())[0]
            
            # Scale features for linear models
            scaler = None
            if model_name in ['ridge', 'lasso', 'elasticnet', 'svr']:
                logger.info(f"    Scaling features...")
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            # Inner CV for hyperparameter tuning
            inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_SEED)
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=inner_cv,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            
            logger.info(f"    Tuning hyperparameters...")
            grid_search.fit(X_train, y_train)
            
            # Best model
            best_model = grid_search.best_estimator_
            
            # Predict on test fold
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            fold_r2_scores.append(r2)
            fold_rmse_scores.append(rmse)
            fold_mae_scores.append(mae)
            fold_predictions.extend(y_pred)
            fold_true_values.extend(y_test)
            
            logger.info(f"    Fold results: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        
        # Calculate overall metrics
        cv_r2_mean = np.mean(fold_r2_scores)
        cv_r2_std = np.std(fold_r2_scores)
        cv_rmse_mean = np.mean(fold_rmse_scores)
        cv_rmse_std = np.std(fold_rmse_scores)
        cv_mae_mean = np.mean(fold_mae_scores)
        cv_mae_std = np.std(fold_mae_scores)
        
        total_time = time.time() - start_time
        
        # Store model metrics
        model_metrics = {
            'cv_r2_mean': float(cv_r2_mean),
            'cv_r2_std': float(cv_r2_std),
            'cv_rmse_mean': float(cv_rmse_mean),
            'cv_rmse_std': float(cv_rmse_std),
            'cv_mae_mean': float(cv_mae_mean),
            'cv_mae_std': float(cv_mae_std),
            'fold_r2_scores': [float(r2) for r2 in fold_r2_scores],
            'fold_rmse_scores': [float(rmse) for rmse in fold_rmse_scores],
            'fold_mae_scores': [float(mae) for mae in fold_mae_scores],
            'training_time': float(total_time),
            'n_features': X.shape[1]
        }
        
        # Save model metrics
        metrics_file = os.path.join(METRICS_DIR, f"{model_name}_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(model_metrics, f, indent=2)
        
        all_models_metrics[model_name] = {
            'cv_r2_mean': float(cv_r2_mean),
            'cv_r2_std': float(cv_r2_std),
            'cv_rmse_mean': float(cv_rmse_mean),
            'cv_mae_mean': float(cv_mae_mean),
            'training_time': float(total_time)
        }
        
        logger.info(f"{model_name} results: R²={cv_r2_mean:.4f}±{cv_r2_std:.4f}, "
                   f"RMSE={cv_rmse_mean:.4f}, MAE={cv_mae_mean:.4f}, "
                   f"Time={total_time:.2f}s")
    
    # Save all models metrics
    all_metrics_file = os.path.join(METRICS_DIR, "all_models_metrics.json")
    with open(all_metrics_file, "w") as f:
        json.dump(all_models_metrics, f, indent=2)
    
    return all_models_metrics

def update_summary(models_metrics, diagnostics):
    """Update the summary.md file with results"""
    logger.info("Updating summary.md...")
    
    # Prepare model results table sorted by R²
    sorted_models = sorted(models_metrics.items(), 
                          key=lambda x: x[1]['cv_r2_mean'], 
                          reverse=True)
    
    # Create summary content
    summary_content = f"""
# ipk_out_raw Modeling Results

## Configuration

- **DATASET**: ipk_out_raw
- **TARGET_COLUMN**: {TARGET_COLUMN}
- **OVERWRITE_PREVIOUS**: {OVERWRITE_PREVIOUS}
- **RANDOM_SEED**: {RANDOM_SEED}
- **OUTDIR**: 03_modeling_results/ipk_out_raw
- **PROCESSED**: 02_processed_data/ipk_out_raw
- **OUTER_CV_FOLDS**: {OUTER_CV_FOLDS}
- **INNER_CV_FOLDS**: {INNER_CV_FOLDS}
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
    
    # Add model rows
    for model_name, metrics in sorted_models:
        summary_content += f"| {model_name} | {metrics['cv_r2_mean']:.4f} | {metrics['cv_r2_std']:.4f} | "
        summary_content += f"{metrics['cv_rmse_mean']:.4f} | {metrics['cv_mae_mean']:.4f} | "
        summary_content += f"{metrics['training_time']:.2f}s |\n"
    
    # Add note about baseline comparison
    best_model_r2 = max(m['cv_r2_mean'] for m in models_metrics.values())
    if best_model_r2 > 0:
        baseline_note = "**Note**: Some models outperformed the baseline."
    else:
        baseline_note = "**Note**: No models outperformed the baseline."
    
    summary_content += f"\n{baseline_note}\n\n"
    
    # Add timestamp
    summary_content += f"## Last Updated\n\n- **Timestamp**: {datetime.now().isoformat()}\n"
    
    # Save summary
    summary_file = "03_modeling_results/ipk_out_raw/summary.md"
    with open(summary_file, "w") as f:
        f.write(summary_content)
    
    logger.info(f"Summary saved to {summary_file}")

def main():
    try:
        logger.info("Starting diagnostic and retraining process for ipk_out_raw...")
        
        # Load data
        try:
            X, y, pca = load_and_validate_data()
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            # Try to find an alternative target column
            X = pd.read_csv(os.path.join(DATA_DIR, "X.csv"), index_col=0)
            y = pd.read_csv(os.path.join(DATA_DIR, "y.csv"), index_col=0)
            pca = pd.read_csv(os.path.join(DATA_DIR, "pca_covariates.csv"), index_col=0)
            
            # Find numeric columns in y
            numeric_cols = y.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                logger.error("No numeric columns found in y.csv")
                raise ValueError("No numeric target columns available")
            
            # Use the numeric column with most non-NA values
            global TARGET_COLUMN
            max_non_na_col = None
            max_non_na_count = -1
            
            for col in numeric_cols:
                non_na_count = y[col].notna().sum()
                if non_na_count > max_non_na_count:
                    max_non_na_count = non_na_count
                    max_non_na_col = col
            
            TARGET_COLUMN = max_non_na_col
            logger.warning(f"Using {TARGET_COLUMN} as target column (has {max_non_na_count} non-NA values)"),
        
        # Run diagnostics
        X_final, y_final, pca_final, diagnostics = run_diagnostics(X, y, pca)
        
        # Prepare features
        X_combined, _ = prepare_features(X_final, pca_final)
        
        # Train models
        models_metrics = train_models(X_combined, y_final, diagnostics)
        
        # Update summary
        update_summary(models_metrics, diagnostics)
        
        logger.info("Diagnostic and retraining process completed successfully!")
        
        # Create concise results table as requested
        print("\n=== Concise Results Table ===")
        print("| Model | CV R² mean±std |")
        print("|-------|----------------|")
        for model_name, metrics in sorted(models_metrics.items(), 
                                         key=lambda x: x[1]['cv_r2_mean'], 
                                         reverse=True):
            print(f"| {model_name} | {metrics['cv_r2_mean']:.4f}±{metrics['cv_r2_std']:.4f} |")
        
        print(f"\nBaseline RMSE: {diagnostics['baseline_metrics']['rmse']:.4f}")
        print(f"Baseline MAE: {diagnostics['baseline_metrics']['mae']:.4f}")
        
        best_model_r2 = max(m['cv_r2_mean'] for m in models_metrics.values())
        if best_model_r2 > 0:
            print("\nNote: Some models outperformed the baseline.")
        else:
            print("\nNote: No models outperformed the baseline.")
            
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()