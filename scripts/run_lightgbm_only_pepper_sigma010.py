import os
import sys
import json
import time
import logging
import numpy as np
import joblib
import importlib.util
import argparse
from datetime import datetime

# Dynamic RUN_ID for this execution
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = "03_modeling_results/pepper_augmented"
METRICS_DIR = os.path.join(OUTDIR, "metrics")
MODELS_DIR = os.path.join(OUTDIR, "models")
LOGS_DIR = os.path.join(OUTDIR, "logs")
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

log_file = os.path.join(LOGS_DIR, f"lightgbm_sigma010_{RUN_ID}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("Starting LightGBM-only training for pepper_augmented with sigma=0.10")

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

# -----------------------------
# CLI argument parsing
# -----------------------------
def str_to_bool(s: str) -> bool:
    return str(s).lower() in {"1", "true", "t", "yes", "y"}

def parse_args():
    p = argparse.ArgumentParser(description="Run LightGBM-only augmented pipeline for pepper with sigma=0.10 and custom grid")
    # High-level config
    p.add_argument("--dataset", choices=["pepper", "pepper_10611831", "ipk_out_raw"], default="pepper")
    p.add_argument("--use_synthetic", type=str, default="true")
    p.add_argument("--synthetic_only", type=str, default="false")
    p.add_argument("--selected_k", type=int, default=5000)
    p.add_argument("--sigma_resid_factor", type=float, default=0.10)
    p.add_argument("--models", type=str, default="lightgbm")
    p.add_argument("--cross_validation_outer", type=int, default=3)
    p.add_argument("--cross_validation_inner", type=int, default=2)
    p.add_argument("--holdout_size", type=float, default=0.2)
    p.add_argument("--augment_file", type=str, default=os.path.join("04_augmentation", "pepper", "model_sources", "llama3", "synthetic_y_filtered_f5_s42_k5000.csv"))
    p.add_argument("--overwrite_previous", action="store_true")

    # LightGBM grid params (all optional; defaults applied if omitted)
    p.add_argument("--learning_rate", nargs="*", type=float)
    p.add_argument("--num_leaves", nargs="*", type=int)
    p.add_argument("--n_estimators", nargs="*", type=int)
    p.add_argument("--feature_fraction", nargs="*", type=float)
    p.add_argument("--bagging_fraction", nargs="*", type=float)
    p.add_argument("--bagging_freq", nargs="*", type=int)
    p.add_argument("--min_child_samples", nargs="*", type=int)
    p.add_argument("--lambda_l2", nargs="*", type=float)
    p.add_argument("--min_gain_to_split", nargs="*", type=float)
    return p.parse_args()

args = parse_args()

# Force desired config values for this run
aug.CONFIG["RUN_ID"] = RUN_ID
aug.CONFIG["DATASET"] = args.dataset
aug.CONFIG["AUGMENTED_DATASET"] = "pepper_augmented"
aug.CONFIG["AUGMENTED"] = os.path.join("04_augmentation", "pepper") if args.dataset == "pepper" else os.path.join("04_augmentation", args.dataset)
aug.CONFIG["PROCESSED"] = os.path.join("02_processed_data", args.dataset)
aug.CONFIG["USE_SYNTHETIC"] = str_to_bool(args.use_synthetic)
aug.CONFIG["TARGET_COLUMN"] = "Yield_BV"
aug.CONFIG["AUGMENT_MODE"] = "llama3"
aug.CONFIG["AUGMENT_SEED"] = 42
aug.CONFIG["AUGMENT_FILE"] = args.augment_file
aug.CONFIG["SELECTED_K"] = args.selected_k
aug.CONFIG["SYNTHETIC_ONLY"] = str_to_bool(args.synthetic_only)
aug.CONFIG["SIGMA_RESID_FACTOR"] = float(args.sigma_resid_factor)
aug.CONFIG["OVERWRITE_PREVIOUS"] = bool(args.overwrite_previous)
# Ensure OUTDIR points to pepper_augmented to avoid confusion from module import defaults
aug.CONFIG["OUTDIR"] = OUTDIR

# Point to the k=5000 filtered synthetic target file specifically
if args.augment_file:
    aug.CONFIG["AUGMENT_FILE"] = args.augment_file
else:
    # default to k=5000 filtered synthetic target (organized under llama3)
    aug.CONFIG["AUGMENT_FILE"] = os.path.join(aug.CONFIG["AUGMENTED"], "model_sources", "llama3", "synthetic_y_filtered_f5_s42_k5000.csv")

logging.info(f"Using RUN_ID={aug.CONFIG['RUN_ID']}")

# Wrap end-to-end flow in a guard to ensure we always emit a metrics JSON
status = "failed"
reason = None
X = None
y = None
sample_ids = None
X_train = None
y_train = None
X_holdout = None
y_holdout = None
X_holdout_selected = None
selected_k = None
selector = None
model_name = "lightgbm"
try:
    # Load and align data exactly as the augmented pipeline does
    X, y, sample_ids = aug.load_and_align_data()
    logging.info(f"Loaded aligned data: X shape={getattr(X, 'shape', None)}, y length={len(y)}")

    # Build masks for synthetic and real samples
    syn_mask = [str(sid).startswith("SYNTHETIC_") for sid in sample_ids]
    real_mask = [not str(sid).startswith("SYNTHETIC_") for sid in sample_ids]

    import numpy as np
    syn_mask_np = np.array(syn_mask)
    real_mask_np = np.array(real_mask)

    # Split and prepare datasets depending on synthetic-only flag
    if aug.CONFIG.get("SYNTHETIC_ONLY", False):
        try:
            # Handle numpy arrays or pandas DataFrames
            if hasattr(X, 'iloc'):
                X = X.iloc[syn_mask_np]
            else:
                X = X[syn_mask_np]
            y = y[syn_mask_np]
            sample_ids = [sid for sid, keep in zip(sample_ids, syn_mask) if keep]
            logging.info(f"Filtered to synthetic-only: X shape={getattr(X, 'shape', None)}, y length={len(y)}")
            X_train = X
            y_train = y
            X_holdout = None
            y_holdout = None
        except Exception as e:
            logging.warning(f"Synthetic-only filtering failed; proceeding without filter. Error: {e}")
            X_train = X
            y_train = y
            X_holdout = None
            y_holdout = None
    else:
        # Real-sample validated training: use all synthetic + a train split of real; hold out a test split of real
        from sklearn.model_selection import train_test_split
        real_indices = np.where(real_mask_np)[0]
        if real_indices.size < 5:
            logging.warning("Too few real samples for a robust holdout; proceeding without holdout split.")
            train_real_idx = real_indices
            test_real_idx = np.array([], dtype=int)
        else:
            train_real_idx, test_real_idx = train_test_split(real_indices, test_size=float(args.holdout_size), random_state=aug.CONFIG.get("RANDOM_SEED", 42), shuffle=True)

        syn_indices = np.where(syn_mask_np)[0]
        train_indices = np.sort(np.concatenate([syn_indices, train_real_idx]))

        # Build training set
        if hasattr(X, 'iloc'):
            X_train = X.iloc[train_indices]
            y_train = y[train_indices]
        else:
            X_train = X[train_indices]
            y_train = y[train_indices]

        # Build holdout real-only set
        if test_real_idx.size > 0:
            if hasattr(X, 'iloc'):
                X_holdout = X.iloc[test_real_idx]
                y_holdout = y[test_real_idx]
            else:
                X_holdout = X[test_real_idx]
                y_holdout = y[test_real_idx]
            logging.info(f"Constructed training set with {X_train.shape[0]} samples; holdout real-only set with {X_holdout.shape[0]} samples")
        else:
            X_holdout = None
            y_holdout = None
            logging.info(f"Constructed training set with {X_train.shape[0]} samples; no holdout set created")

# Mirror LightGBM configuration from augmented pipeline
    model_dir = os.path.join(OUTDIR, "models", f"{model_name}_{RUN_ID}")
    os.makedirs(model_dir, exist_ok=True)
    metrics_path = os.path.join(METRICS_DIR, f"{model_name}_metrics_{RUN_ID}.json")

# Helper to persist a metrics JSON safely (used for both success and failure)
    def _write_metrics_payload(payload: dict, path: str = metrics_path):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
            logging.info(f"Saved LightGBM metrics to {path}")
            print(f"Saved metrics: {path}")
        except Exception as write_err:
            logging.error(f"Failed to write metrics JSON: {write_err}")

    needs_feature_selection = True if aug.CONFIG.get("SELECTED_K") else (X.shape[1] > aug.CONFIG.get("MAX_FEATURES", 10000))
    model_info = {
        'estimator': LGBMRegressor(random_state=aug.CONFIG["RANDOM_SEED"]),
        'params': {
            'num_leaves': args.num_leaves if args.num_leaves else [31, 63, 127, 255],
            'learning_rate': args.learning_rate if args.learning_rate else [0.05, 0.075, 0.1],
            'n_estimators': args.n_estimators if args.n_estimators else [100, 200, 400],
            'feature_fraction': args.feature_fraction if args.feature_fraction else [0.6, 0.8, 1.0],
            'bagging_fraction': args.bagging_fraction if args.bagging_fraction else [0.8, 1.0]
        },
        'needs_scaling': False,
        'needs_feature_selection': needs_feature_selection,
        'early_stopping': True
    }

# Optional extra regularization parameters
    if args.bagging_freq:
        model_info['params']['bagging_freq'] = args.bagging_freq
    if args.min_child_samples:
        model_info['params']['min_child_samples'] = args.min_child_samples
    if args.lambda_l2:
        # Map lambda_l2 to LightGBM's reg_lambda alias
        model_info['params']['reg_lambda'] = args.lambda_l2
    if args.min_gain_to_split:
        # LightGBM uses 'min_split_gain' as the parameter name
        model_info['params']['min_split_gain'] = args.min_gain_to_split

# Wrap training in a guard to ensure we always emit a metrics JSON
    # Feature selection (SelectKBest with f_regression) if required
    X_selected = X_train
    if model_info['needs_feature_selection']:
        from sklearn.feature_selection import SelectKBest, f_regression
        # Prefer configured SELECTED_K; otherwise cap to MAX_FEATURES and available post-filtrage features
        requested_k = aug.CONFIG.get("SELECTED_K")
        nb_features_postfiltrage = X_train.shape[1]
        default_cap = aug.CONFIG.get("MAX_FEATURES", nb_features_postfiltrage)
        k_effective = int(min(requested_k if requested_k else default_cap, nb_features_postfiltrage))
        selected_k = k_effective
        logging.info(
            f"Selected_k clamp: requested_k={requested_k}, nb_features_postfiltrage={nb_features_postfiltrage}, k_effective={k_effective}"
        )
        selector = SelectKBest(f_regression, k=k_effective)
        X_selected = selector.fit_transform(X_train, y_train)
        # Transform holdout if present
        if X_holdout is not None:
            try:
                X_holdout_selected = selector.transform(X_holdout)
            except Exception:
                X_holdout_selected = None
        joblib.dump(selector, os.path.join(model_dir, "feature_selector.joblib"))
    else:
        X_holdout_selected = X_holdout if X_holdout is not None else None

# Nested CV setup identical to the augmented pipeline
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    aug.CONFIG["OUTER_CV_FOLDS"] = int(args.cross_validation_outer)
    aug.CONFIG["INNER_CV_FOLDS"] = int(args.cross_validation_inner)
    aug.CONFIG["RANDOM_SEED"] = 42
    outer_cv = KFold(n_splits=aug.CONFIG["OUTER_CV_FOLDS"], shuffle=True, random_state=aug.CONFIG["RANDOM_SEED"])
    inner_cv = KFold(n_splits=aug.CONFIG["INNER_CV_FOLDS"], shuffle=True, random_state=aug.CONFIG["RANDOM_SEED"])

    all_r2_scores = []
    all_rmse_scores = []
    all_mae_scores = []
    all_best_params = []

    start_time = time.time()

    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X_selected)):
        logging.info(f"Outer fold {i+1}/{aug.CONFIG['OUTER_CV_FOLDS']}")
        X_train_cv, X_test_cv = X_selected[train_idx], X_selected[test_idx]
        y_train_cv, y_test_cv = y_train[train_idx], y_train[test_idx]

        grid = GridSearchCV(
            model_info['estimator'],
            model_info['params'],
            cv=inner_cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        # Avoid passing early-stopping or callback args to ensure compatibility across LightGBM versions
        # Some environments reject both 'early_stopping_rounds' and 'callbacks'.
        # Rely solely on inner CV for hyperparameter selection.
        grid.fit(X_train_cv, y_train_cv)

        y_pred = grid.best_estimator_.predict(X_test_cv)
        r2 = float(r2_score(y_test_cv, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test_cv, y_pred)))
        mae = float(mean_absolute_error(y_test_cv, y_pred))

        all_r2_scores.append(r2)
        all_rmse_scores.append(rmse)
        all_mae_scores.append(mae)
        all_best_params.append(grid.best_params_)

        logging.info(f"  Fold {i+1}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    # Train final model on all selected features using best params from highest R2 fold
    best_idx = int(np.argmax(all_r2_scores))
    final_model = model_info['estimator']
    final_model.set_params(**all_best_params[best_idx])
    # Train final model without early stopping to avoid API incompatibilities
    final_model.fit(X_selected, y_train)
    joblib.dump(final_model, os.path.join(model_dir, f"{model_name}_final_model.joblib"))

    training_time = time.time() - start_time

    # Assemble metrics payload consistent with augmented pipeline, including σ metadata
    model_metrics = {
        "model_name": model_name,
        "status": "success",
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
        "features_count": int(X_train.shape[1]),
        "selected_k": int(selected_k) if selected_k is not None else None,
        "needs_feature_selection": bool(model_info['needs_feature_selection']),
        "run_id": RUN_ID,
        "augment_mode": aug.CONFIG.get("AUGMENT_MODE"),
        "augment_seed": aug.CONFIG.get("AUGMENT_SEED"),
        "augment_file": aug.CONFIG.get("AUGMENT_FILE"),
        "use_synthetic": aug.CONFIG.get("USE_SYNTHETIC"),
        "synthetic_only": aug.CONFIG.get("SYNTHETIC_ONLY", False),
        "sigma_resid_factor": aug.CONFIG.get("SIGMA_RESID_FACTOR"),
        "fallback_percent": aug.CONFIG.get("FALLBACK_PERCENT"),
        "timestamp": datetime.now().isoformat()
    }

    # Evaluate and attach holdout metrics if available (real-only validation)
    if X_holdout_selected is not None and y_holdout is not None:
        try:
            y_holdout_pred = final_model.predict(X_holdout_selected)
            holdout_r2 = float(r2_score(y_holdout, y_holdout_pred))
            holdout_rmse = float(np.sqrt(mean_squared_error(y_holdout, y_holdout_pred)))
            holdout_mae = float(mean_absolute_error(y_holdout, y_holdout_pred))
            model_metrics.update({
                "holdout_r2": holdout_r2,
                "holdout_rmse": holdout_rmse,
                "holdout_mae": holdout_mae,
                "holdout_size": int(len(y_holdout))
            })
            logging.info(f"Holdout metrics (real-only): R2={holdout_r2:.4f}, RMSE={holdout_rmse:.4f}, MAE={holdout_mae:.4f}, N={len(y_holdout)}")
        except Exception as e:
            logging.warning(f"Failed to compute holdout metrics: {e}")

    _write_metrics_payload(model_metrics, metrics_path)
    status = "success"
except BaseException as e:
    reason = str(e)
    logging.error(f"LightGBM runner failed prior to metrics write: {reason}")
    # Log full traceback for debugging
    try:
        import traceback
        logging.error(traceback.format_exc())
    except Exception:
        pass
    # Emit a failure metrics JSON to satisfy audit/aggregation pipelines
    metrics_path = os.path.join(METRICS_DIR, f"{model_name}_metrics_{RUN_ID}.json")
    fail_payload = {
        "model_name": model_name,
        "status": "failed",
        "reason": reason,
        "run_id": RUN_ID,
        "augment_mode": aug.CONFIG.get("AUGMENT_MODE"),
        "augment_seed": aug.CONFIG.get("AUGMENT_SEED"),
        "augment_file": aug.CONFIG.get("AUGMENT_FILE"),
        "use_synthetic": aug.CONFIG.get("USE_SYNTHETIC"),
        "synthetic_only": aug.CONFIG.get("SYNTHETIC_ONLY", False),
        "sigma_resid_factor": aug.CONFIG.get("SIGMA_RESID_FACTOR"),
        "selected_k_requested": int(args.selected_k),
        "selected_k": int(selected_k) if selected_k is not None else None,
        "timestamp": datetime.now().isoformat()
    }
    try:
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(fail_payload, f, indent=2)
            f.flush()
        logging.info(f"Saved failure metrics to {metrics_path}")
        print(f"Saved metrics: {metrics_path}")
    except Exception as write_err:
        logging.error(f"Failed to write failure metrics JSON: {write_err}")