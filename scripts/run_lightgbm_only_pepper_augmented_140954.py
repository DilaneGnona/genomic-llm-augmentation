import os
import sys
import json
import time
import logging
import numpy as np
import joblib
import importlib.util
from datetime import datetime

# Fixed run ID to match the interrupted run
RUN_ID = "20251026_140954"
OUTDIR = "03_modeling_results/pepper_augmented"
METRICS_DIR = os.path.join(OUTDIR, "metrics")
MODELS_DIR = os.path.join(OUTDIR, "models")
LOGS_DIR = os.path.join(OUTDIR, "logs")
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

log_file = os.path.join(LOGS_DIR, f"lightgbm_only_{RUN_ID}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("Starting LightGBM-only training for pepper_augmented with fixed RUN_ID")

# Dynamically import the augmented pipeline to reuse data loading and config
spec = importlib.util.spec_from_file_location(
    "aug", os.path.join("scripts", "unified_modeling_pipeline_augmented.py")
)
aug = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aug)

# Ensure LightGBM is available
try:
    from lightgbm import LGBMRegressor
except Exception as e:
    logging.error(f"LightGBM import failed: {e}")
    sys.exit(1)

# Force desired config values
aug.CONFIG["RUN_ID"] = RUN_ID
aug.CONFIG["OVERWRITE_PREVIOUS"] = True  # ensure we re-train even if previous aggregated results exist
logging.info(f"Using RUN_ID={aug.CONFIG['RUN_ID']}")

# Load and align data exactly as the augmented pipeline does
X, y, sample_ids = aug.load_and_align_data()
logging.info(f"Loaded aligned data: X shape={X.shape}, y length={len(y)}")

# Mirror LightGBM configuration from augmented pipeline
model_name = "lightgbm"
model_dir = os.path.join(OUTDIR, "models", f"{model_name}_{RUN_ID}")
os.makedirs(model_dir, exist_ok=True)
metrics_path = os.path.join(METRICS_DIR, f"{model_name}_metrics_{RUN_ID}.json")

needs_feature_selection = X.shape[1] > aug.CONFIG.get("MAX_FEATURES", 10000)
model_info = {
    'estimator': LGBMRegressor(random_state=aug.CONFIG["RANDOM_SEED"]),
    'params': {
        'num_leaves': [31, 63],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'feature_fraction': [0.8, 1.0],
        'bagging_fraction': [0.8, 1.0]
    },
    'needs_scaling': False,
    'needs_feature_selection': needs_feature_selection,
    'early_stopping': True
}

# Feature selection (SelectKBest with f_regression) if required
X_selected = X
selector = None
selected_k = None
if model_info['needs_feature_selection']:
    from sklearn.feature_selection import SelectKBest, f_regression
    k = min(aug.CONFIG["MAX_FEATURES"], X.shape[1])
    selected_k = k
    logging.info(f"Performing feature selection: selecting k={k} of {X.shape[1]} features")
    selector = SelectKBest(f_regression, k=k)
    X_selected = selector.fit_transform(X, y)
    joblib.dump(selector, os.path.join(model_dir, "feature_selector.joblib"))

# Nested CV setup identical to the augmented pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
outer_cv = KFold(n_splits=aug.CONFIG["OUTER_CV_FOLDS"], shuffle=True, random_state=aug.CONFIG["RANDOM_SEED"])
inner_cv = KFold(n_splits=aug.CONFIG["INNER_CV_FOLDS"], shuffle=True, random_state=aug.CONFIG["RANDOM_SEED"])

all_r2_scores = []
all_rmse_scores = []
all_mae_scores = []
all_best_params = []

start_time = time.time()

for i, (train_idx, test_idx) in enumerate(outer_cv.split(X_selected)):
    logging.info(f"Outer fold {i+1}/{aug.CONFIG['OUTER_CV_FOLDS']}")
    X_train, X_test = X_selected[train_idx], X_selected[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Grid search without explicit fit params (matching version-compatible approach)
    grid = GridSearchCV(
        model_info['estimator'],
        model_info['params'],
        cv=inner_cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    # Evaluate on the test fold
    y_pred = grid.best_estimator_.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    all_r2_scores.append(r2)
    all_rmse_scores.append(rmse)
    all_mae_scores.append(mae)
    all_best_params.append(grid.best_params_)

    logging.info(f"  Fold {i+1}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

# Train final model on all selected features using best params from highest R2 fold
best_idx = int(np.argmax(all_r2_scores))
final_model = model_info['estimator']
final_model.set_params(**all_best_params[best_idx])
final_model.fit(X_selected, y)
joblib.dump(final_model, os.path.join(model_dir, f"{model_name}_final_model.joblib"))

training_time = time.time() - start_time

# Assemble metrics payload consistent with augmented pipeline
model_metrics = {
    "model_name": model_name,
    "cv_r2_mean": float(np.mean(all_r2_scores)),
    "cv_r2_std": float(np.std(all_r2_scores)),
    "cv_rmse_mean": float(np.mean(all_rmse_scores)),
    "cv_rmse_std": float(np.std(all_rmse_scores)),
    "cv_mae_mean": float(np.mean(all_mae_scores)),
    "cv_mae_std": float(np.std(all_mae_scores)),
    "training_time_seconds": float(training_time),
    "fold_metrics": {
        "r2_scores": all_r2_scores,
        "rmse_scores": all_rmse_scores,
        "mae_scores": all_mae_scores,
        "best_params_per_fold": all_best_params
    },
    "final_best_params": all_best_params[best_idx],
    "features_count": int(X.shape[1]),
    "selected_k": int(selected_k) if selected_k is not None else None,
    "needs_feature_selection": bool(model_info['needs_feature_selection']),
    "run_id": RUN_ID,
    "timestamp": datetime.now().isoformat()
}

with open(metrics_path, 'w') as f:
    json.dump(model_metrics, f, indent=2)

logging.info(f"Saved LightGBM metrics to {metrics_path}")
print(f"Saved metrics: {metrics_path}")