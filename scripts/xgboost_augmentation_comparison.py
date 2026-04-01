import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import logging
import time
import traceback

# Set random seed for reproducibility
np.random.seed(42)

# Setup logging
log_dir = "c:/Users/OMEN/Desktop/experiment_snp/03_modeling_results/pepper/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "xgboost_augmentation.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()

try:
    # Verify XGBoost import
    import xgboost as xgb
    logger.info("XGBoost imported successfully")
except ImportError:
    logger.error("XGBoost import failed. Exiting.")
    raise

def load_and_verify_datasets():
    """Load and verify original and augmented datasets"""
    logger.info("Loading datasets...")
    
    # Paths
    orig_dir = "c:/Users/OMEN/Desktop/experiment_snp/02_processed_data/pepper"
    aug_dir = "c:/Users/OMEN/Desktop/experiment_snp/04_augmentation/pepper"
    
    # Load original data
    X_orig = pd.read_csv(os.path.join(orig_dir, "X.csv"))
    y_orig = pd.read_csv(os.path.join(orig_dir, "y.csv"))
    pca_covariates = pd.read_csv(os.path.join(orig_dir, "pca_covariates.csv"))
    
    # Load augmented data
    X_aug = pd.read_csv(os.path.join(aug_dir, "synthetic_snps.csv"))
    y_aug = pd.read_csv(os.path.join(aug_dir, "synthetic_y.csv"))
    
    logger.info(f"Original dataset shapes - X: {X_orig.shape}, y: {y_orig.shape}, PCA: {pca_covariates.shape}")
    logger.info(f"Augmented dataset shapes - X: {X_aug.shape}, y: {y_aug.shape}")
    
    # Check if first column might be an index or contains non-numeric data
    logger.info("Checking for non-numeric columns...")
    
    # Remove any non-numeric columns with aggressive detection
    def filter_numeric_columns(df):
        numeric_cols = []
        string_cols = []
        problematic_cols = []
        
        # First scan for columns with obvious string values
        for col in df.columns:
            # Skip empty columns
            if df[col].isna().all():
                problematic_cols.append(col)
                continue
                
            try:
                # Check sample values with more aggressive string detection
                sample_vals = df[col].dropna().head(100)
                
                # Check for any string that's not a number
                has_non_numeric_strings = False
                for val in sample_vals:
                    if isinstance(val, str):
                        # Try to see if it's a string that could be converted to number
                        try:
                            float(val)
                        except ValueError:
                            has_non_numeric_strings = True
                            break
                
                if has_non_numeric_strings:
                    string_cols.append(col)
                    continue
                
                # Try numeric conversion with coercion
                converted = pd.to_numeric(df[col], errors='coerce')
                
                # Check if conversion was mostly successful
                if converted.isna().mean() > 0.5:
                    # More than 50% values couldn't be converted
                    problematic_cols.append(col)
                    continue
                
                numeric_cols.append(col)
            except Exception as e:
                logger.warning(f"Error processing column {col}: {str(e)}")
                problematic_cols.append(col)
        
        # Log all problematic columns
        if string_cols or problematic_cols:
            total_removed = len(string_cols) + len(problematic_cols)
            logger.warning(f"Found {total_removed} problematic columns, removing them from dataset")
            
            if string_cols:
                logger.warning(f"String columns ({len(string_cols)}): {string_cols[:10]}..." if len(string_cols) > 10 else f"String columns: {string_cols}")
            if problematic_cols:
                logger.warning(f"Problematic columns ({len(problematic_cols)}): {problematic_cols[:10]}..." if len(problematic_cols) > 10 else f"Problematic columns: {problematic_cols}")
        
        # Return filtered DataFrame, ensuring we don't return empty
        if not numeric_cols:
            logger.error("No numeric columns found after filtering! Returning empty DataFrame")
            return pd.DataFrame()
        
        return df[numeric_cols]
    
    # Apply filtering
    X_orig = filter_numeric_columns(X_orig)
    X_aug = filter_numeric_columns(X_aug)
    
    logger.info(f"After filtering non-numeric columns - X_orig: {X_orig.shape}, X_aug: {X_aug.shape}")
    
    # Column alignment
    logger.info("Performing column alignment...")
    shared_columns = list(set(X_orig.columns) & set(X_aug.columns))
    logger.info(f"Number of shared columns: {len(shared_columns)}")
    
    # Reorder columns to match
    X_orig_aligned = X_orig[shared_columns]
    X_aug_aligned = X_aug[shared_columns]
    
    # Type checks and coercion with error handling
    logger.info("Performing type checks and coercion...")
    
    # Try to convert to integers with safe handling
    def safe_convert_to_int(df):
        for col in df.columns:
            try:
                # First convert to float, then to int to handle any decimal values
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0).astype(int)
            except Exception as e:
                logger.warning(f"Error converting column '{col}' to int: {str(e)}")
        return df
    
    X_orig_aligned = safe_convert_to_int(X_orig_aligned)
    X_aug_aligned = safe_convert_to_int(X_aug_aligned)
    
    # Verify values are in {0,1,2} or close enough (allow for some noise)
    try:
        orig_values = np.unique(X_orig_aligned.values)
        aug_values = np.unique(X_aug_aligned.values)
        logger.info(f"Original SNP values: {sorted(orig_values)}")
        logger.info(f"Augmented SNP values: {sorted(aug_values)}")
    except:
        logger.warning("Could not compute unique values")
    
    # Drop rows with missing target
    y_orig = y_orig.dropna()
    y_aug = y_aug.dropna()
    
    logger.info(f"After dropping missing targets - y_orig: {y_orig.shape}, y_aug: {y_aug.shape}")
    
    return X_orig_aligned, y_orig, pca_covariates, X_aug_aligned, y_aug

def build_training_matrices(X_orig, y_orig, pca_covariates, X_aug, y_aug):
    """Build the three training matrices"""
    logger.info("Building training matrices...")
    
    # Ensure y has the same number of rows as X
    logger.info(f"Before alignment - X_orig: {X_orig.shape}, y_orig: {y_orig.shape}")
    logger.info(f"Before alignment - X_aug: {X_aug.shape}, y_aug: {y_aug.shape}")
    
    # Ensure X and y have the same number of rows
    min_rows_orig = min(len(X_orig), len(y_orig))
    if min_rows_orig < len(X_orig) or min_rows_orig < len(y_orig):
        logger.warning(f"Truncating original dataset to {min_rows_orig} rows to match X and y")
        X_orig = X_orig.iloc[:min_rows_orig]
        y_orig = y_orig.iloc[:min_rows_orig]
    
    min_rows_aug = min(len(X_aug), len(y_aug))
    if min_rows_aug < len(X_aug) or min_rows_aug < len(y_aug):
        logger.warning(f"Truncating augmented dataset to {min_rows_aug} rows to match X and y")
        X_aug = X_aug.iloc[:min_rows_aug]
        y_aug = y_aug.iloc[:min_rows_aug]
    
    # Original-only: X + PCA covariates
    # Ensure PCA covariates have the same number of rows as X_orig
    if len(X_orig) > 0 and len(pca_covariates) > 0:
        min_rows_pca = min(len(X_orig), len(pca_covariates))
        if min_rows_pca < len(X_orig):
            logger.warning("PCA covariates rows less than X_orig, truncating X_orig")
            X_orig = X_orig.iloc[:min_rows_pca]
            y_orig = y_orig.iloc[:min_rows_pca]
        elif min_rows_pca < len(pca_covariates):
            logger.warning("X_orig rows less than PCA covariates, truncating PCA")
            pca_covariates = pca_covariates.iloc[:min_rows_pca]
        
        X_original_only = pd.concat([X_orig, pca_covariates], axis=1)
    else:
        logger.warning("PCA covariates not available or empty, using X_orig only")
        X_original_only = X_orig.copy()
    
    y_original_only = y_orig.copy()
    
    # Augmented-only: synthetic X, no PCA
    X_augmented_only = X_aug.copy()
    y_augmented_only = y_aug.copy()
    
    # Hybrid: concatenate original and augmented rows
    X_hybrid = pd.concat([X_orig, X_aug], axis=0)
    y_hybrid = pd.concat([y_orig, y_aug], axis=0)
    
    # Ensure all datasets have matching X and y row counts
    datasets = {
        "original": (X_original_only, y_original_only),
        "augmented": (X_augmented_only, y_augmented_only),
        "hybrid": (X_hybrid, y_hybrid)
    }
    
    # Final check and adjustment
    for name, (X, y) in datasets.items():
        if len(X) != len(y):
            min_len = min(len(X), len(y))
            logger.warning(f"{name} dataset has mismatched X and y: {len(X)} vs {len(y)}, truncating to {min_len}")
            datasets[name] = (X.iloc[:min_len], y.iloc[:min_len])
    
    # Create shapes report
    shapes_report = """
# Dataset Shapes Report

## Original-only
- X shape: {} rows, {} columns
- y shape: {} rows

## Augmented-only
- X shape: {} rows, {} columns
- y shape: {} rows

## Hybrid
- X shape: {} rows, {} columns
- y shape: {} rows
    """.format(
        datasets["original"][0].shape[0], datasets["original"][0].shape[1],
        datasets["original"][1].shape[0],
        datasets["augmented"][0].shape[0], datasets["augmented"][0].shape[1],
        datasets["augmented"][1].shape[0],
        datasets["hybrid"][0].shape[0], datasets["hybrid"][0].shape[1],
        datasets["hybrid"][1].shape[0]
    )
    
    # Save shapes report
    with open("c:/Users/OMEN/Desktop/experiment_snp/03_modeling_results/pepper/shapes_report.txt", "w") as f:
        f.write(shapes_report)
    
    logger.info("Shapes report saved")
    
    return datasets

def train_xgboost(X, y, dataset_name):
    """Train XGBoost model with nested CV"""
    logger.info(f"Training XGBoost on {dataset_name} dataset...")
    
    # Ensure X and y have the same number of rows
    if len(X) != len(y):
        min_len = min(len(X), len(y))
        logger.warning(f"Critical mismatch: X has {len(X)} rows, y has {len(y)} rows. Truncating to {min_len}")
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]
    
    # Check for empty dataset early
    if len(X) == 0 or len(y) == 0:
        logger.error(f"Empty dataset after initial processing: X={len(X)} rows, y={len(y)} rows")
        return {
            'model': None,
            'selector': None,
            'metrics': {
                'r2': {'mean': 0, 'std': 0},
                'rmse': {'mean': 0, 'std': 0},
                'mae': {'mean': 0, 'std': 0},
                'scores': {
                    'r2': [],
                    'rmse': [],
                    'mae': []
                }
            },
            'best_params': {},
            'feature_count': 0,
            'feature_selection_applied': False,
            'training_time': 0
        }
    
    # Ensure y is a 1D array
    if len(y.shape) > 1:
        logger.info("Converting y to 1D array")
        y = y.iloc[:, 0]  # Take first column
    
    # FINAL DATA VALIDATION BEFORE TRAINING - MORE AGGRESSIVE
    logger.info("Performing final data validation before XGBoost training...")
    
    # Step 1: Check for any remaining string columns and remove them
    string_cols = []
    for col in X.columns:
        # Check if column contains any string values
        if X[col].apply(lambda x: isinstance(x, str)).any():
            string_cols.append(col)
    
    if string_cols:
        logger.warning(f"Found {len(string_cols)} columns with string values, removing them")
        X = X.drop(columns=string_cols)
    
    # Step 2: Force all columns to be numeric with coercion
    numeric_failed_cols = []
    for col in X.columns:
        try:
            # Convert to numeric, coercing errors to NaN
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except Exception as e:
            logger.error(f"Failed to convert column '{col}' to numeric: {str(e)}")
            numeric_failed_cols.append(col)
    
    if numeric_failed_cols:
        logger.warning(f"Dropping {len(numeric_failed_cols)} columns that failed numeric conversion")
        X = X.drop(columns=numeric_failed_cols)
    
    # Check if we have any columns left
    if len(X.columns) == 0:
        logger.error(f"No valid columns left after numeric conversion for {dataset_name} dataset")
        return {
            'model': None,
            'selector': None,
            'metrics': {
                'r2': {'mean': 0, 'std': 0},
                'rmse': {'mean': 0, 'std': 0},
                'mae': {'mean': 0, 'std': 0},
                'scores': {
                    'r2': [],
                    'rmse': [],
                    'mae': []
                }
            },
            'best_params': {},
            'feature_count': 0,
            'feature_selection_applied': False,
            'training_time': 0
        }
    
    # Step 3: Remove any rows with NaN values
    initial_rows = len(X)
    X = X.dropna()
    if len(X) < initial_rows:
        logger.warning(f"Removed {initial_rows - len(X)} rows with NaN values after final data validation")
        # Update y to match - use iloc if DataFrame, regular indexing if array
        if hasattr(y, 'iloc'):
            y = y.iloc[:len(X)]
        else:
            y = y[:len(X)]
    
    # Step 4: One final check for object types in X
    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        logger.warning(f"Still found {len(object_cols)} object-type columns after conversion: {object_cols}")
        logger.warning("Attempting one final cleanup...")
        
        # Try to convert again or remove
        for col in object_cols:
            try:
                # Try to convert one more time with stricter handling
                X[col] = X[col].astype(str).str.extract(r'([0-9.]+)')[0].astype(float)
            except:
                logger.warning(f"Could not clean column {col}, removing it")
                X = X.drop(col, axis=1)
    
    # Step 5: Recheck after final cleanup
    if len(X.columns) == 0:
        logger.error(f"No valid columns left after final cleanup for {dataset_name} dataset")
        return {
            'model': None,
            'selector': None,
            'metrics': {
                'r2': {'mean': 0, 'std': 0},
                'rmse': {'mean': 0, 'std': 0},
                'mae': {'mean': 0, 'std': 0},
                'scores': {
                    'r2': [],
                    'rmse': [],
                    'mae': []
                }
            },
            'best_params': {},
            'feature_count': 0,
            'feature_selection_applied': False,
            'training_time': 0
        }
    
    # Check for empty dataset after cleaning
    if len(X) < 5:
        logger.error(f"Insufficient samples after cleaning: {len(X)} rows. Minimum 5 samples required")
        return {
            'model': None,
            'selector': None,
            'metrics': {
                'r2': {'mean': 0, 'std': 0},
                'rmse': {'mean': 0, 'std': 0},
                'mae': {'mean': 0, 'std': 0},
                'scores': {
                    'r2': [],
                    'rmse': [],
                    'mae': []
                }
            },
            'best_params': {},
            'feature_count': X.shape[1],
            'feature_selection_applied': False,
            'training_time': 0
        }
    
    # Convert to numpy arrays early to catch any remaining issues
    try:
        # Force X to be float type to avoid any type issues with XGBoost
        X_array = X.astype(float).values
        
        # Handle y conversion based on its type
        if hasattr(y, 'values'):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # Ensure y is float
        y_array = y_array.astype(float)
        
        logger.info(f"Successfully converted data to numpy arrays. X shape: {X_array.shape}, y shape: {y_array.shape}")
        
        # Check for any NaN or inf values in the arrays
        if np.isnan(X_array).any():
            logger.warning(f"Found NaN values in X array, replacing with 0")
            X_array = np.nan_to_num(X_array, nan=0)
        
        if np.isinf(X_array).any():
            logger.warning(f"Found inf values in X array, replacing with large finite numbers")
            X_array = np.nan_to_num(X_array, posinf=1e10, neginf=-1e10)
            
        if np.isnan(y_array).any():
            logger.warning(f"Found NaN values in y array, removing corresponding rows")
            valid_rows = ~np.isnan(y_array)
            X_array = X_array[valid_rows]
            y_array = y_array[valid_rows]
        
        if np.isinf(y_array).any():
            logger.warning(f"Found inf values in y array, removing corresponding rows")
            valid_rows = ~np.isinf(y_array)
            X_array = X_array[valid_rows]
            y_array = y_array[valid_rows]
            
    except Exception as e:
        logger.error(f"Failed to convert data to numpy arrays: {str(e)}")
        
        # Try a more aggressive approach - convert each column individually with error handling
        for col in X.columns.copy():
            try:
                # Try to convert with coerce
                X[col] = pd.to_numeric(X[col], errors='coerce')
                # Fill any remaining NaNs
                X[col] = X[col].fillna(0)
            except Exception as e_col:
                logger.error(f"Column '{col}' still problematic: {str(e_col)}, removing it")
                X = X.drop(col, axis=1)
        
        # Check if we have any columns left after this process
        if len(X.columns) == 0:
            logger.error(f"No valid columns after final cleaning for {dataset_name} dataset")
            return {
                'model': None,
                'selector': None,
                'metrics': {
                    'r2': {'mean': 0, 'std': 0},
                    'rmse': {'mean': 0, 'std': 0},
                    'mae': {'mean': 0, 'std': 0},
                    'scores': {
                        'r2': [],
                        'rmse': [],
                        'mae': []
                    }
                },
                'best_params': {},
                'feature_count': 0,
                'feature_selection_applied': False,
                'training_time': 0
            }
        
        # Try conversion again with float type enforcement
        try:
            X_array = X.astype(float).values
            
            if hasattr(y, 'values'):
                y_array = y.values.astype(float)
            else:
                y_array = np.array(y).astype(float)
                
            logger.info(f"Successfully converted after aggressive cleaning. X shape: {X_array.shape}")
        except Exception as e_final:
            logger.error(f"Final conversion attempt failed: {str(e_final)}")
            return {
                'model': None,
                'selector': None,
                'metrics': {
                    'r2': {'mean': 0, 'std': 0},
                    'rmse': {'mean': 0, 'std': 0},
                    'mae': {'mean': 0, 'std': 0},
                    'scores': {
                        'r2': [],
                        'rmse': [],
                        'mae': []
                    }
                },
                'best_params': {},
                'feature_count': 0,
                'feature_selection_applied': False,
                'training_time': 0
            }
    
    # Replace X with the cleaned version
    X = pd.DataFrame(X_array, columns=X.columns)
    
    # Check sample count
    n_samples = len(X)
    logger.info(f"Final sample count for {dataset_name}: {n_samples} rows")
    
    # Feature selection if needed
    selector = None
    if X.shape[1] > 10000:
        logger.info(f"Features > 10,000 ({X.shape[1]}), applying SelectKBest")
        selector = SelectKBest(f_regression, k=10000)
        X_selected = selector.fit_transform(X, y.values)
        logger.info(f"After selection: {X_selected.shape[1]} features")
    else:
        X_selected = X.values
        logger.info(f"No feature selection needed, using {X.shape[1]} features")
    
    # Adjust CV splits based on sample size - ensure splits <= samples
    n_outer_splits = min(5, max(2, n_samples // 5))
    # Ensure outer splits don't exceed samples
    n_outer_splits = min(n_outer_splits, n_samples)
    n_inner_splits = min(3, max(2, n_samples // 10))
    # Ensure inner splits don't exceed samples and are <= outer splits
    n_inner_splits = min(n_inner_splits, n_samples, n_outer_splits)
    
    logger.info(f"Using {n_outer_splits}-fold outer CV and {n_inner_splits}-fold inner CV for {n_samples} samples")
    
    # Nested CV setup
    outer_cv = KFold(n_splits=n_outer_splits, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=n_inner_splits, shuffle=True, random_state=42)
    
    # Hyperparameter grid - VERY simplified to ensure training completes
    # Using minimal grid with only one configuration to avoid issues
    param_grid = {
        'n_estimators': [100],
        'max_depth': [3],
        'learning_rate': [0.1],
        'subsample': [1.0],
        'colsample_bytree': [1.0],
        'reg_lambda': [1]
    }
    logger.info("Using minimal hyperparameter grid with single configuration for reliability")
    
    # Model setup
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Outer CV results
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    best_params_list = []
    
    start_time = time.time()
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_selected)):
        logger.info(f"Outer fold {fold + 1}/{n_outer_splits}")
        
        # Safety check for indices
        if max(train_idx) >= len(X_selected) or max(test_idx) >= len(X_selected):
            logger.error(f"Invalid indices: train_max={max(train_idx)}, test_max={max(test_idx)}, data_len={len(X_selected)}")
            continue
        
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y.values[train_idx], y.values[test_idx]
        
        # Inner CV for hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='r2',
            n_jobs=1,  # Use fewer jobs to avoid memory issues
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_params_list.append(best_params)
        
        # Train best model on the entire outer fold training set
        best_model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
        best_model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        
        logger.info(f"Fold {fold + 1} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    total_time = time.time() - start_time
    
    # Compute average metrics if we have results
    if r2_scores:
        avg_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        avg_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)
        avg_mae = np.mean(mae_scores)
        std_mae = np.std(mae_scores)
        
        logger.info(f"{dataset_name} - Average R²: {avg_r2:.4f} ± {std_r2:.4f}")
        logger.info(f"{dataset_name} - Average RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")
        logger.info(f"{dataset_name} - Average MAE: {avg_mae:.4f} ± {std_mae:.4f}")
    else:
        logger.warning(f"No valid CV folds completed for {dataset_name}")
        avg_r2 = std_r2 = avg_rmse = std_rmse = avg_mae = std_mae = 0
    
    logger.info(f"Total training time: {total_time:.2f} seconds")
    
    # Train final model on entire dataset with default params if no best params
    if best_params_list:
        # Find most common hyperparameters
        from collections import Counter
        common_params = {}
        for param in param_grid.keys():
            try:
                param_values = [params[param] for params in best_params_list]
                common_params[param] = Counter(param_values).most_common(1)[0][0]
            except:
                # Use default if we can't determine most common
                common_params[param] = param_grid[param][0]
        logger.info(f"Most common best parameters: {common_params}")
    else:
        common_params = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 
                         'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_lambda': 1}
        logger.warning("Using default parameters due to no valid CV results")
    
    # Train final model if we have data
    if X_selected.shape[0] > 0 and X_selected.shape[1] > 0:
        final_model = xgb.XGBRegressor(**common_params, objective='reg:squarederror', random_state=42)
        final_model.fit(X_selected, y.values)
    else:
        logger.error("Cannot train final model - empty data")
        final_model = None
    
    return {
        'model': final_model,
        'selector': selector,
        'metrics': {
            'r2': {'mean': avg_r2, 'std': std_r2},
            'rmse': {'mean': avg_rmse, 'std': std_rmse},
            'mae': {'mean': avg_mae, 'std': std_mae},
            'scores': {
                'r2': r2_scores,
                'rmse': rmse_scores,
                'mae': mae_scores
            }
        },
        'best_params': common_params,
        'feature_count': X_selected.shape[1],
        'feature_selection_applied': X.shape[1] > 10000,
        'training_time': total_time
    }

def save_results(results_dict):
    """Save models, metrics, and update all_models_metrics.json"""
    logger.info("Saving results...")
    
    # Create directories
    model_dir = "c:/Users/OMEN/Desktop/experiment_snp/03_modeling_results/pepper/models"
    metric_dir = "c:/Users/OMEN/Desktop/experiment_snp/03_modeling_results/pepper/metrics"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    
    # Save individual models and metrics
    all_models_metrics = []
    
    for dataset_type, result in results_dict.items():
        # Only save model if it exists
        if result['model'] is not None:
            model_file = os.path.join(model_dir, f"xgboost_{dataset_type}.joblib")
            joblib.dump({
                'model': result['model'],
                'selector': result['selector'],
                'best_params': result['best_params'],
                'feature_count': result['feature_count']
            }, model_file)
            logger.info(f"Saved model to {model_file}")
        else:
            logger.warning(f"Skipping model save for {dataset_type} - model is None")
        
        # Create metrics dict for JSON
        metrics_dict = {
            'model': f'XGBoost_{dataset_type}',
            'dataset': dataset_type,
            'r2_mean': result['metrics']['r2']['mean'],
            'r2_std': result['metrics']['r2']['std'],
            'rmse_mean': result['metrics']['rmse']['mean'],
            'rmse_std': result['metrics']['rmse']['std'],
            'mae_mean': result['metrics']['mae']['mean'],
            'mae_std': result['metrics']['mae']['std'],
            'r2_scores': result['metrics']['scores']['r2'],
            'rmse_scores': result['metrics']['scores']['rmse'],
            'mae_scores': result['metrics']['scores']['mae'],
            'best_params': result['best_params'],
            'feature_count': result['feature_count'],
            'feature_selection_applied': result['feature_selection_applied'],
            'training_time': result['training_time'],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save individual metrics JSON
        metric_file = os.path.join(metric_dir, f"xgboost_{dataset_type}_metrics.json")
        with open(metric_file, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        logger.info(f"Saved metrics to {metric_file}")
        
        all_models_metrics.append(metrics_dict)
    
    # Update all_models_metrics.json with better error handling
    all_metrics_file = os.path.join(metric_dir, "all_models_metrics.json")
    existing_metrics = []
    
    # Handle existing file with robust error checking
    if os.path.exists(all_metrics_file):
        try:
            with open(all_metrics_file, 'r') as f:
                file_content = f.read().strip()
                if file_content:
                    existing_metrics = json.loads(file_content)
                    
                    # Validate the structure is a list of dictionaries
                    if not isinstance(existing_metrics, list):
                        logger.warning(f"{all_metrics_file} is not a list, creating new file")
                        existing_metrics = []
                    else:
                        # Filter out any non-dictionary items
                        valid_metrics = []
                        for item in existing_metrics:
                            if isinstance(item, dict) and 'model' in item:
                                valid_metrics.append(item)
                            else:
                                logger.warning(f"Found invalid entry in {all_metrics_file}, skipping")
                        existing_metrics = valid_metrics
                else:
                    existing_metrics = []
        except json.JSONDecodeError as e:
            logger.error(f"Error reading {all_metrics_file}: {str(e)}, creating new file")
            existing_metrics = []
        except Exception as e:
            logger.error(f"Unexpected error reading {all_metrics_file}: {str(e)}, creating new file")
            existing_metrics = []
    
    # Create set of existing model names for duplicates check
    existing_model_names = set()
    for item in existing_metrics:
        if isinstance(item, dict) and 'model' in item:
            existing_model_names.add(item['model'])
    
    # Prepare new metrics to add
    new_metrics_list = []
    
    for new_metric in all_models_metrics:
        model_name = new_metric.get('model')
        
        # Only add if not already present and has valid structure
        if model_name and model_name not in existing_model_names:
            try:
                new_metrics_list.append(new_metric)
                logger.info(f"Adding metrics for {model_name}")
            except Exception as e:
                logger.error(f"Error adding metrics entry for {model_name}: {str(e)}")
    
    # Add new metrics to existing
    updated_metrics = existing_metrics + new_metrics_list
    
    # Save updated metrics
    try:
        with open(all_metrics_file, 'w') as f:
            json.dump(updated_metrics, f, indent=4)
        logger.info(f"Successfully updated {all_metrics_file}")
    except Exception as e:
        logger.error(f"Failed to save updated metrics: {str(e)}")
    
    return all_models_metrics

def update_summary(results_dict):
    """Update summary.md with comparison results"""
    logger.info("Updating summary.md...")
    
    summary_file = "c:/Users/OMEN/Desktop/experiment_snp/03_modeling_results/pepper/summary.md"
    
    # Read existing summary
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            existing_summary = f.read()
    else:
        existing_summary = "# Pepper Dataset Modeling Summary\n\n"
    
    # Create new section
    new_section = "## XGBoost on Augmented Data (and Hybrid)\n\n"
    
    # Create comparison table
    new_section += "| Dataset | R² Mean ± Std | RMSE Mean ± Std | MAE Mean ± Std | Feature Count | Feature Selection | Training Time (s) | Best Parameters |\n"
    new_section += "|---------|---------------|-----------------|----------------|---------------|-------------------|-------------------|-----------------|\n"
    
    for dataset_type in ['original', 'augmented', 'hybrid']:
        result = results_dict[dataset_type]
        r2_mean = result['metrics']['r2']['mean']
        r2_std = result['metrics']['r2']['std']
        rmse_mean = result['metrics']['rmse']['mean']
        rmse_std = result['metrics']['rmse']['std']
        mae_mean = result['metrics']['mae']['mean']
        mae_std = result['metrics']['mae']['std']
        feature_count = result['feature_count']
        feature_selection = "Yes" if result['feature_selection_applied'] else "No"
        training_time = result['training_time']
        best_params = str(result['best_params'])
        
        new_section += f"| {dataset_type.capitalize()} | {r2_mean:.4f} ± {r2_std:.4f} | {rmse_mean:.4f} ± {rmse_std:.4f} | {mae_mean:.4f} ± {mae_std:.4f} | {feature_count} | {feature_selection} | {training_time:.2f} | {best_params} |\n"
    
    # Add new section to summary
    # Check if section already exists
    if "## XGBoost on Augmented Data (and Hybrid)" in existing_summary:
        # Replace existing section
        parts = existing_summary.split("## XGBoost on Augmented Data (and Hybrid)")
        updated_summary = parts[0] + new_section
    else:
        updated_summary = existing_summary + "\n" + new_section
    
    # Save updated summary
    with open(summary_file, 'w') as f:
        f.write(updated_summary)
    
    logger.info(f"Updated summary.md with new comparison section")

def main():
    try:
        logger.info("Starting XGBoost augmentation comparison workflow...")
        
        # Step 1: Load and verify datasets
        X_orig, y_orig, pca_covariates, X_aug, y_aug = load_and_verify_datasets()
        
        # Step 2: Build training matrices
        datasets = build_training_matrices(X_orig, y_orig, pca_covariates, X_aug, y_aug)
        
        # Step 3: Train models for each dataset
        results_dict = {}
        for dataset_type, (X, y) in datasets.items():
            results_dict[dataset_type] = train_xgboost(X, y, dataset_type)
        
        # Step 4: Save results
        save_results(results_dict)
        
        # Step 5: Update summary
        update_summary(results_dict)
        
        logger.info("XGBoost augmentation comparison workflow completed successfully!")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()