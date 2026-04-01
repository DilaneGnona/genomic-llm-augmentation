import os
import json
import logging
import time
import numpy as np
import pandas as pd
import joblib
import sys
import inspect
from datetime import datetime
import argparse
import tempfile
import subprocess

# 设置命令行参数解析器
parser = argparse.ArgumentParser(description='Augmented modeling pipeline for SNP datasets')
parser.add_argument('--dataset', choices=['pepper', 'pepper_10611831', 'ipk_out_raw'], default='pepper_10611831',
                    help='Dataset to process')
# FIX: robust boolean parsing from string values
parser.add_argument('--use_synthetic', type=str, default='True',
                    help='Whether to use synthetic data for augmentation (True/False)')
parser.add_argument('--target_column', type=str, default=None,
                    help='Target column name (overrides dataset default)')
parser.add_argument('--overwrite_previous', action='store_true',
                    help='Overwrite previous model metrics and artifacts')
parser.add_argument('--augment_mode', choices=['llama3', 'pca', 'none', 'deepseek', 'glm46'], default='llama3',
                    help='Augmentation mode')
parser.add_argument('--augment_size', type=str, default=None,
                    help='Requested synthetic sample count (number or "auto")')
parser.add_argument('--augment_file', type=str, default=None,
                    help='Path to filtered synthetic target file (e.g., synthetic_y_filtered.csv)')
parser.add_argument('--augment_seed', type=int, default=42,
                    help='Seed for augmentation generation')
parser.add_argument('--selected_k', type=int, default=None,
                    help='Feature selection cap for models')
parser.add_argument('--synthetic_only', action='store_true',
                    help='Train on synthetic-only; exclude synthetic from validation/test')
# NEW: model filter and sigma residual factor metadata
parser.add_argument('--models', type=str, default=None,
                    help='Comma-separated model list to train (e.g., randomforest,svr,lightgbm,xgboost)')
parser.add_argument('--sigma_resid_factor', type=float, default=None,
                    help='Scale for residual noise in synthetic target generation; recorded as metadata.')
# NEW: CV folds and split sizing overrides
parser.add_argument('--cross_validation_outer', type=int, default=None,
                    help='Override outer CV folds (default 5)')
parser.add_argument('--cross_validation_inner', type=int, default=None,
                    help='Override inner CV folds (default 3)')
parser.add_argument('--training_size', type=str, default=None,
                    help='Training fraction/percent (e.g., 80% or 0.8). Used for logging.')
parser.add_argument('--holdout_size', type=str, default=None,
                    help='Real-only holdout fraction/percent (e.g., 20% or 0.2), defaults to 0.2')
# NEW: RandomForest overrides
parser.add_argument('--rf_n_estimators', type=int, default=None,
                    help='Override RandomForest n_estimators (single value)')
parser.add_argument('--rf_max_depth', type=str, default=None,
                    help='Override RandomForest max_depth (integer or "none")')
parser.add_argument('--rf_max_features', type=str, default=None,
                    help='Override RandomForest max_features (e.g., "sqrt")')
parser.add_argument('--rf_max_samples', type=float, default=None,
                    help='Override RandomForest max_samples (e.g., 0.8)')

# Only parse CLI args when running as a script, not when imported
if __name__ == "__main__":
    args = parser.parse_args()
    # normalize use_synthetic to boolean
    if isinstance(args.use_synthetic, str):
        args.use_synthetic = args.use_synthetic.strip().lower() in ('true', '1', 'yes', 'y', 't')
else:
    # Safe defaults when imported by other scripts; callers can override CONFIG after import
    class _Args:
        dataset = 'pepper_10611831'
        use_synthetic = 'True'
        target_column = None
        overwrite_previous = False
        augment_mode = 'llama3'
        augment_size = None
        augment_file = None
        augment_seed = 42
        selected_k = None
        synthetic_only = False
        models = None
        sigma_resid_factor = None
        cross_validation_outer = None
        cross_validation_inner = None
        training_size = None
        holdout_size = None
    args = _Args()
    # normalize string default for use_synthetic
    if isinstance(args.use_synthetic, str):
        args.use_synthetic = args.use_synthetic.strip().lower() in ('true', '1', 'yes', 'y', 't')

def setup_config():
    """设置配置信息"""
    # 创建唯一的run_id
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 基础配置
    base_config = {
        "DATASET": args.dataset,
        "AUGMENTED_DATASET": f"{args.dataset}_augmented",
        "USE_SYNTHETIC": args.use_synthetic,
        "OVERWRITE_PREVIOUS": False,
        "REQUIRE_XGBOOST": False,
        "REQUIRE_LIGHTGBM": False,
        "RANDOM_SEED": 42,
        "OUTDIR": f"03_modeling_results/{args.dataset}_augmented",
        "PROCESSED": f"02_processed_data/{args.dataset}",
        "AUGMENTED": f"04_augmentation/{args.dataset}",
        "ORIGINAL_RESULTS": f"03_modeling_results/{args.dataset}",
        "OUTER_CV_FOLDS": 5,
        "INNER_CV_FOLDS": 3,
        "MAX_FEATURES": 10000,  # 将在代码中根据需要调整
        "TARGET_COLUMN": None,  # 将在运行时确定
        "RUN_ID": run_id,
        "RUN_TYPE": "augmented",
        "AUGMENT_MODE": args.augment_mode,
        "AUGMENT_SIZE": args.augment_size,
        "AUGMENT_SEED": args.augment_seed,
        "AUGMENT_FILE": args.augment_file,
        "SELECTED_K": args.selected_k,
        "SYNTHETIC_ONLY": args.synthetic_only,
        # NEW keys for model selection and sigma metadata
        "MODELS": [m.strip().lower() for m in args.models.split(',')] if args.models else None,
        "SIGMA_RESID_FACTOR": args.sigma_resid_factor
    }

    # Apply CV overrides if provided
    if isinstance(args.cross_validation_outer, int) and args.cross_validation_outer > 0:
        base_config["OUTER_CV_FOLDS"] = args.cross_validation_outer
    if isinstance(args.cross_validation_inner, int) and args.cross_validation_inner > 0:
        base_config["INNER_CV_FOLDS"] = args.cross_validation_inner

    # Parse sizing for holdout/training
    def _parse_fraction(val, default=None):
        if val is None:
            return default
        try:
            s = str(val).strip()
            if s.endswith('%'):
                num = float(s[:-1])
                return max(0.0, min(1.0, num / 100.0))
            else:
                num = float(s)
                return max(0.0, min(1.0, num))
        except Exception:
            return default

    hold_frac = _parse_fraction(args.holdout_size, default=0.2)
    train_frac = _parse_fraction(args.training_size, default=None)
    base_config["HOLDOUT_FRACTION"] = hold_frac
    base_config["TRAINING_FRACTION_REQUESTED"] = train_frac
    
    # 根据数据集设置特定的目标列或命令行覆盖
    if args.target_column:
        base_config["TARGET_COLUMN"] = args.target_column
    elif args.dataset == 'pepper':
        base_config["TARGET_COLUMN"] = "Yield_BV"
    elif args.dataset == 'pepper_10611831':
        base_config["TARGET_COLUMN"] = None
    elif args.dataset == 'ipk_out_raw':
        base_config["TARGET_COLUMN"] = "YR_LS"
    
    # Apply CLI overwrite flag
    base_config["OVERWRITE_PREVIOUS"] = args.overwrite_previous

    # RandomForest overrides
    rf_override = {}
    if args.rf_n_estimators is not None:
        rf_override['n_estimators'] = [int(args.rf_n_estimators)]
    if args.rf_max_depth is not None:
        rf_override['max_depth'] = [args.rf_max_depth]
    if args.rf_max_features is not None:
        rf_override['max_features'] = [args.rf_max_features]
    if args.rf_max_samples is not None:
        rf_override['max_samples'] = float(args.rf_max_samples)
    if rf_override:
        base_config['RF_OVERRIDE'] = rf_override
    
    return base_config

# 设置配置
CONFIG = setup_config()

# 设置随机种子
np.random.seed(CONFIG["RANDOM_SEED"])

# Derive fallback_percent from augment_file naming convention if available (e.g., synthetic_y_filtered_f20_s42.csv)
try:
    if CONFIG.get("AUGMENT_FILE"):
        import re
        base_name = os.path.basename(CONFIG["AUGMENT_FILE"]) if isinstance(CONFIG["AUGMENT_FILE"], str) else ""
        m = re.search(r"_f(\d+)_s(\d+)", base_name)
        if m:
            CONFIG["FALLBACK_PERCENT"] = int(m.group(1))
        else:
            CONFIG["FALLBACK_PERCENT"] = None
    else:
        CONFIG["FALLBACK_PERCENT"] = None
except Exception:
    CONFIG["FALLBACK_PERCENT"] = None

# 创建目录（如果不存在）
for subdir in ["logs", "models", "metrics", "plots", "plots_and_tables"]:
    os.makedirs(os.path.join(CONFIG["OUTDIR"], subdir), exist_ok=True)

# --- Atomic file write helper for per-model metrics ---
def save_model_metrics_atomic(metrics_dir, model_name, run_id, payload):
    """Write per-model metrics atomically to avoid partial files.

    Creates a temporary file `<model>_metrics_<RUN_ID>.json.tmp`, fsyncs, closes,
    then replaces the final `<model>_metrics_<RUN_ID>.json`.
    """
    try:
        final_path = os.path.join(metrics_dir, f"{model_name}_metrics_{run_id}.json")
        tmp_path = final_path + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
        logging.info(f"Atomically wrote metrics: {final_path}")
        return final_path
    except Exception as e:
        logging.error(f"Atomic write failed for {model_name} ({run_id}): {e}")
        try:
            # Best-effort fallback direct write (non-atomic)
            final_path = os.path.join(metrics_dir, f"{model_name}_metrics_{run_id}.json")
            with open(final_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
            logging.warning(f"Fallback non-atomic write succeeded: {final_path}")
            return final_path
        except Exception as e2:
            logging.error(f"Fallback write failed for {model_name} ({run_id}): {e2}")
            return None

# 设置日志（robust: always attach file handler even if logging is pre-configured）
log_file = os.path.join(CONFIG["OUTDIR"], "logs", f"pipeline_{CONFIG['RUN_ID']}.log")
try:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Attach a FileHandler for this run_id if not already present
    need_file_handler = True
    for h in logger.handlers:
        try:
            if isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == os.path.abspath(log_file):
                need_file_handler = False
                break
        except Exception:
            pass
    if need_file_handler:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
    # Always ensure a StreamHandler exists for console output
    has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers)
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(sh)
    logging.info(f"Pipeline log initialized: {log_file}")
except Exception:
    # Fallback to basicConfig if anything goes wrong
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Pipeline log initialized (basicConfig): {log_file}")

# 导入sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR

# 导入模型注册中心
from model_registry import list_registered_models, get_model_config

# 尝试导入XGBoost
XGBOOST_AVAILABLE = False
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
    logging.info("XGBoost is available")
except ImportError:
    XGBOOST_AVAILABLE = False
    if CONFIG["REQUIRE_XGBOOST"]:
        logging.error("XGBoost required but not available")
        sys.exit(1)
    else:
        logging.warning("XGBoost not available, will skip")

# 尝试导入LightGBM
LIGHTGBM_AVAILABLE = False
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
    logging.info("LightGBM is available")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    if CONFIG["REQUIRE_LIGHTGBM"]:
        logging.error("LightGBM required but not available")
        sys.exit(1)
    else:
        logging.warning("LightGBM not available, will skip")

def log_environment():
    """记录环境信息"""
    logging.info(f"Python version: {sys.version}")
    logging.info(f"NumPy version: {np.__version__}")
    logging.info(f"Pandas version: {pd.__version__}")
    logging.info(f"Joblib version: {joblib.__version__}")
    logging.info(f"XGBoost available: {XGBOOST_AVAILABLE}")
    logging.info(f"LightGBM available: {LIGHTGBM_AVAILABLE}")
    logging.info(f"Dataset: {CONFIG['DATASET']}")
    logging.info(f"Using synthetic data: {CONFIG['USE_SYNTHETIC']}")
    logging.info(f"Augment mode: {CONFIG.get('AUGMENT_MODE')}")
    logging.info(f"Augment size: {CONFIG.get('AUGMENT_SIZE')}")
    logging.info(f"Augment seed: {CONFIG.get('AUGMENT_SEED')}")
    if CONFIG.get('AUGMENT_FILE'):
        logging.info(f"Augment file: {CONFIG.get('AUGMENT_FILE')}")
    logging.info(f"Selected_k: {CONFIG.get('SELECTED_K')}")
    # NEW environment logs
    logging.info(f"Selected models (requested): {CONFIG.get('MODELS')}")
    logging.info(f"Sigma residual factor (metadata): {CONFIG.get('SIGMA_RESID_FACTOR')}")
    if CONFIG.get('FALLBACK_PERCENT') is not None:
        logging.info(f"Fallback percent (parsed): {CONFIG.get('FALLBACK_PERCENT')}")
    logging.info(f"Outer CV folds: {CONFIG.get('OUTER_CV_FOLDS')}")
    logging.info(f"Inner CV folds: {CONFIG.get('INNER_CV_FOLDS')}")
    logging.info(f"Holdout fraction (real-only): {CONFIG.get('HOLDOUT_FRACTION')}")
    if CONFIG.get('TRAINING_FRACTION_REQUESTED') is not None:
        logging.info(f"Training fraction (requested, for logging): {CONFIG.get('TRAINING_FRACTION_REQUESTED')}")
    
    # 保存配置
    with open(os.path.join(CONFIG["OUTDIR"], "logs", f"config_{CONFIG['RUN_ID']}.json"), 'w') as f:
        json.dump(CONFIG, f, indent=2)

def _get_parquet_engine():
    try:
        import pyarrow  # noqa: F401
        return 'pyarrow'
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return 'fastparquet'
        except Exception:
            return None

def _read_parquet_columns(parquet_path, columns=None):
    engine = _get_parquet_engine()
    if engine is None:
        return None
    import pandas as pd
    try:
        df = pd.read_parquet(parquet_path, columns=columns, engine=engine)
        return df
    except Exception:
        try:
            df = pd.read_parquet(parquet_path, columns=columns)
            return df
        except Exception:
            return None

def _list_parquet_columns(parquet_path):
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(parquet_path)
        return [n for n in pf.schema_arrow.names]
    except Exception:
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path, nrows=0)
            return df.columns.tolist()
        except Exception:
            return None

def _detect_id_column(df):
    """Detect ID column and normalize to 'Sample_ID'"""
    if 'Sample_ID' in df.columns:
        return 'Sample_ID'
    if 'IID' in df.columns:
        return 'IID'
    # fallback: assume first column is ID
    return df.columns[0]

def preflight_checks():
    """执行飞行前检查"""
    logging.info("Performing pre-flight checks")
    
    # 检查原始数据集目录
    if not os.path.exists(CONFIG["PROCESSED"]):
        logging.error(f"Processed data directory not found: {CONFIG['PROCESSED']}")
        return False
    
    # 检查必要的原始数据文件
    required_files = ["X.csv", "y.csv", "pca_covariates.csv"]
    for file in required_files:
        if not os.path.exists(os.path.join(CONFIG["PROCESSED"], file)):
            logging.error(f"Required file not found: {os.path.join(CONFIG['PROCESSED'], file)}")
            return False
    
    # 如果使用合成数据，检查合成数据目录和文件
    if CONFIG["USE_SYNTHETIC"]:
        if not os.path.exists(CONFIG["AUGMENTED"]):
            logging.error(f"Augmented data directory not found: {CONFIG['AUGMENTED']}")
            return False
        
        synth_snps = os.path.join(CONFIG["AUGMENTED"], "synthetic_snps.csv")
        synth_y_default = os.path.join(CONFIG["AUGMENTED"], "synthetic_y.csv")
        synth_y = CONFIG.get("AUGMENT_FILE") or synth_y_default
        if not os.path.exists(synth_snps):
            logging.error(f"Synthetic file not found: {synth_snps}")
            return False
        
        needs_target = CONFIG["TARGET_COLUMN"] is not None
        if needs_target:
            if not os.path.exists(synth_y):
                logging.error(f"Synthetic target file not found: {synth_y}")
                return False
            # Confirm Sample_ID alignment and numeric target
            snps_df = pd.read_csv(synth_snps)
            y_df = pd.read_csv(synth_y)
            if 'Sample_ID' not in y_df.columns:
                logging.error("Synthetic target file must include 'Sample_ID' column")
                return False
            if 'Sample_ID' in snps_df.columns:
                snp_ids = set(snps_df['Sample_ID'].tolist())
                y_ids = set(y_df['Sample_ID'].tolist())
                if snp_ids != y_ids:
                    # Allow filtered synthetic_y to be a strict subset of synthetic_snps
                    if y_ids.issubset(snp_ids):
                        logging.info("Synthetic_y Sample_IDs are a subset of synthetic_snps; proceeding with alignment.")
                    else:
                        logging.error("Sample_ID mismatch between synthetic_snps and synthetic_y (or augment_file); not a subset.")
                        return False
            # Check target numeric
            tgt = CONFIG['TARGET_COLUMN']
            if tgt and tgt in y_df.columns:
                try:
                    pd.to_numeric(y_df[tgt], errors='raise')
                except Exception:
                    logging.error("Synthetic target column must be numeric")
                    return False
        else:
            logging.info("Target column is None; unsupervised augmentation allowed. synthetic_y file not required.")
    
    logging.info("Pre-flight checks passed")
    return True

def load_and_align_data():
    """加载并对齐数据，包括真实数据和合成数据"""
    logging.info("Loading and aligning data")
    
    # Synthetic-only fast path: avoid reading massive real X.csv
    if CONFIG.get("SYNTHETIC_ONLY", False):
        logging.info("Synthetic-only mode enabled; skipping real X.csv read and using manifest for column schema.")
        # Load variant manifest to define SNP columns
        manifest_path = os.path.join(CONFIG["PROCESSED"], "variant_manifest.csv")
        if not os.path.exists(manifest_path):
            logging.error("variant_manifest.csv not found; cannot construct SNP schema without real X.csv")
            sys.exit(1)
        manifest_df = pd.read_csv(manifest_path)
        if "VAR_ID" not in manifest_df.columns:
            logging.error("variant_manifest.csv missing VAR_ID column")
            sys.exit(1)
        real_snp_columns = manifest_df["VAR_ID"].astype(str).tolist()

        # Determine PCA covariate columns from header (fill zeros for synthetic)
        pca_path = os.path.join(CONFIG["PROCESSED"], "pca_covariates.csv")
        if not os.path.exists(pca_path):
            logging.error("pca_covariates.csv not found; required to construct full feature matrix")
            sys.exit(1)
        pca_header = pd.read_csv(pca_path, nrows=0)
        pca_cols = [c for c in pca_header.columns if c != "Sample_ID"]

        # Load synthetic SNPs and align to manifest-defined columns
        X_synthetic = pd.read_csv(os.path.join(CONFIG["AUGMENTED"], "synthetic_snps.csv"))
        if 'Sample_ID' not in X_synthetic.columns:
            synthetic_sample_ids = [f"SYNTHETIC_{i}" for i in range(len(X_synthetic))]
            X_synthetic.insert(0, 'Sample_ID', synthetic_sample_ids)
        else:
            synthetic_sample_ids = X_synthetic['Sample_ID'].tolist()
        synthetic_snp_columns = X_synthetic.drop('Sample_ID', axis=1).columns.tolist()

        # Load synthetic y (filtered file may be provided) and align rows to y order
        y_synthetic_values = None
        synth_y_default = os.path.join(CONFIG["AUGMENTED"], "synthetic_y.csv")
        synth_y_path = CONFIG.get("AUGMENT_FILE") or synth_y_default
        if CONFIG["TARGET_COLUMN"] is not None and os.path.exists(synth_y_path):
            try:
                y_synthetic = pd.read_csv(synth_y_path, engine='python')
            except Exception:
                y_synthetic = pd.read_csv(synth_y_path)
            if 'Sample_ID' in y_synthetic.columns:
                y_ids = y_synthetic['Sample_ID'].tolist()
                X_synthetic_indexed = X_synthetic.set_index('Sample_ID')
                common_ids = [sid for sid in y_ids if sid in X_synthetic_indexed.index]
                X_synthetic = X_synthetic_indexed.loc[common_ids].reset_index()
                synthetic_sample_ids = common_ids
                normalized_ids = [sid if str(sid).startswith('SYNTHETIC_') else f"SYNTHETIC_{sid}" for sid in synthetic_sample_ids]
                X_synthetic['Sample_ID'] = normalized_ids
                synthetic_sample_ids = normalized_ids
                y_syn_indexed = y_synthetic.set_index('Sample_ID')
                y_synthetic = y_syn_indexed.loc[common_ids].reset_index(drop=True)
                y_synthetic['Sample_ID'] = normalized_ids
                synthetic_snp_columns = X_synthetic.drop('Sample_ID', axis=1).columns.tolist()
            # Extract numeric target
            if CONFIG["TARGET_COLUMN"] in y_synthetic.columns:
                y_synthetic_values = pd.to_numeric(y_synthetic[CONFIG["TARGET_COLUMN"]], errors='coerce')
            else:
                fallback_series = y_synthetic.iloc[:, 1] if 'Sample_ID' in y_synthetic.columns and y_synthetic.shape[1] > 1 else y_synthetic.iloc[:, 0]
                y_synthetic_values = pd.to_numeric(fallback_series, errors='coerce')
                logging.warning(f"Using fallback column of synthetic y as target instead of {CONFIG['TARGET_COLUMN']}")
            CONFIG["AUGMENT_SIZE_EFFECTIVE"] = int(len(y_synthetic))
            if CONFIG["AUGMENT_SIZE_EFFECTIVE"] == 0:
                logging.error("Filtered synthetic targets are empty; cannot proceed in synthetic-only mode.")
                sys.exit(1)
        else:
            logging.error("Synthetic target required but not found or invalid for synthetic-only run.")
            sys.exit(1)

        # Build aligned synthetic feature matrix using manifest columns and zero PCA covariates
        X_synthetic_aligned = pd.DataFrame(0, index=range(len(X_synthetic)), columns=real_snp_columns)
        for col in synthetic_snp_columns:
            if col in real_snp_columns:
                X_synthetic_aligned[col] = X_synthetic[col]
        synthetic_pca = pd.DataFrame(0, index=range(len(X_synthetic_aligned)), columns=pca_cols)
        X_synthetic_combined = pd.concat([X_synthetic_aligned, synthetic_pca], axis=1)

        # Final target cleanup and NaN handling
        y_values = pd.to_numeric(y_synthetic_values, errors='coerce')
        nan_count = y_values.isna().sum()
        if nan_count > 0:
            logging.warning(f"Found {nan_count} NaN values in synthetic target; removing these samples.")
            valid_indices = ~y_values.isna()
            X_synthetic_combined = X_synthetic_combined[valid_indices].reset_index(drop=True)
            y_values = y_values[valid_indices].reset_index(drop=True)
            synthetic_sample_ids = [synthetic_sample_ids[i] for i, valid in enumerate(valid_indices) if valid]

        logging.info(f"Combined dataset shape (synthetic-only): {X_synthetic_combined.shape} samples with {X_synthetic_combined.shape[1]} features")
        logging.info(f"Effective augment size: {CONFIG.get('AUGMENT_SIZE_EFFECTIVE')}")

        # Convert to numpy arrays
        X_array = X_synthetic_combined.replace([np.inf, -np.inf], np.nan).fillna(X_synthetic_combined.median()).values.astype(float)
        y_array = y_values.values.astype(float)
        return X_array, y_array, synthetic_sample_ids

    # 加载原始数据（非 synthetic-only 路径）
    logging.info(f"Loading real data from {CONFIG['PROCESSED']}")
    # Prefer Parquet if available
    x_parquet_path = os.path.join(CONFIG["PROCESSED"], "X.parquet")
    x_path = os.path.join(CONFIG["PROCESSED"], "X.csv")
    start_time = time.time()
    X_real = None
    if os.path.exists(x_parquet_path) and _get_parquet_engine() is not None:
        logging.info("Using optimized Parquet loader for X features.")
        # Optional columnar path: use existing selector if present
        selector_path = None
        try:
            models_dir = os.path.join(CONFIG["OUTDIR"], 'models')
            if os.path.isdir(models_dir):
                # Find any feature_selector.joblib from latest run
                candidates = []
                for name in os.listdir(models_dir):
                    p = os.path.join(models_dir, name, 'feature_selector.joblib')
                    if os.path.exists(p):
                        candidates.append(p)
                if candidates:
                    # Choose the most recently modified
                    selector_path = max(candidates, key=lambda p: os.path.getmtime(p))
        except Exception:
            selector_path = None

        parquet_cols = _list_parquet_columns(x_parquet_path) or []
        # Read PCA header to potentially build combined order
        pca_path = os.path.join(CONFIG["PROCESSED"], "pca_covariates.csv")
        pca_header = pd.read_csv(pca_path, nrows=0)
        pca_cols = [c for c in pca_header.columns if c != 'Sample_ID']

        selected_parquet_cols = None
        selected_pca_cols = None
        if selector_path:
            try:
                import joblib as _jb
                selector = _jb.load(selector_path)
                indices = selector.get_support(indices=True)
                # Build combined order: SNP first, then PCA
                snp_cols = [c for c in parquet_cols if c != 'Sample_ID']
                combined = snp_cols + pca_cols
                chosen = [combined[i] for i in indices if i < len(combined)]
                selected_parquet_cols = [c for c in chosen if c in snp_cols]
                selected_pca_cols = [c for c in chosen if c in pca_cols]
            except Exception:
                selected_parquet_cols = None
                selected_pca_cols = None

        # Columnar read if we have selected cols; else full
        if selected_parquet_cols:
            X_real_snp = _read_parquet_columns(x_parquet_path, columns=selected_parquet_cols)
            X_real_snp = X_real_snp.astype(np.float32)
            # Read PCA selected
            pca_real_part = pd.read_csv(pca_path, usecols=['Sample_ID'] + selected_pca_cols)
            # Merge on Sample_ID
            X_real = pd.concat([X_real_snp, pca_real_part.drop('Sample_ID', axis=1)], axis=1)
        else:
            X_real_df = _read_parquet_columns(x_parquet_path)
            if X_real_df is None:
                logging.warning("Parquet read failed; falling back to CSV loader")
            else:
                # Ensure Sample_ID exists; if missing, infer no ID and keep columns as-is
                X_real = X_real_df
                # Force float32 for features (keep Sample_ID type)
                if 'Sample_ID' in X_real.columns:
                    feat_cols = [c for c in X_real.columns if c != 'Sample_ID']
                    X_real[feat_cols] = X_real[feat_cols].astype(np.float32)
                else:
                    X_real = X_real.astype(np.float32)
                logging.info(f"Loaded X.parquet in {time.time() - start_time:.2f} seconds")

    if X_real is None:
        try:
            # Try pyarrow engine first for faster loading if available
            if CONFIG["DATASET"] == 'pepper':
                try:
                    X_real = pd.read_csv(x_path, dtype={'ID_12': 'object'}, on_bad_lines='skip', engine='pyarrow')
                    logging.info(f"Loaded X.csv using pyarrow engine in {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    logging.warning(f"Pyarrow engine failed: {e}, falling back to python engine")
                    X_real = pd.read_csv(x_path, dtype={'ID_12': 'object'}, on_bad_lines='skip', engine='python')
            else:
                try:
                    X_real = pd.read_csv(x_path, on_bad_lines='skip', engine='pyarrow')
                    logging.info(f"Loaded X.csv using pyarrow engine in {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    logging.warning(f"Pyarrow engine failed: {e}, falling back to python engine")
                    X_real = pd.read_csv(x_path, on_bad_lines='skip', engine='python')
        except Exception as e:
            logging.error(f"Failed to read X.csv: {e}")
            sys.exit(1)
    # Read y.csv robustly
    y_path = os.path.join(CONFIG["PROCESSED"], "y.csv")
    try:
        y_real = pd.read_csv(y_path)
    except BaseException as e:
        logging.error(f"Failed to read y.csv: {e}")
        sys.exit(1)
    # Read pca_covariates.csv robustly
    pca_path = os.path.join(CONFIG["PROCESSED"], "pca_covariates.csv")
    try:
        pca_real = pd.read_csv(pca_path)
    except BaseException as e:
        logging.error(f"Failed to read pca_covariates.csv: {e}")
        sys.exit(1)
    
    # 统一ID列名
    xid = _detect_id_column(X_real)
    yid = _detect_id_column(y_real)
    pcid = _detect_id_column(pca_real)
    if xid != 'Sample_ID':
        X_real = X_real.rename(columns={xid: 'Sample_ID'})
    if yid != 'Sample_ID':
        y_real = y_real.rename(columns={yid: 'Sample_ID'})
    if pcid != 'Sample_ID':
        pca_real = pca_real.rename(columns={pcid: 'Sample_ID'})

    # If Sample_ID missing or incorrect in X_real after rename attempt, attach from sample_map
    try:
        sm_path = os.path.join(CONFIG['PROCESSED'], 'sample_map.csv')
        sm_df = pd.read_csv(sm_path)
        sid_series = sm_df['Sample_ID']
        # Force Sample_ID from sample_map for robust alignment
        if 'Sample_ID' in X_real.columns:
            X_real['Sample_ID'] = sid_series.iloc[:len(X_real)].values
        else:
            X_real.insert(0, 'Sample_ID', sid_series.iloc[:len(X_real)].values)
    except Exception:
        pass

    # Remove meta rows like POS/REF/ALT if present
    meta_ids = {'POS','REF','ALT'}
    if 'Sample_ID' in X_real.columns:
        X_real = X_real[~X_real['Sample_ID'].isin(meta_ids)].reset_index(drop=True)
    if 'Sample_ID' in pca_real.columns:
        pca_real = pca_real[~pca_real['Sample_ID'].isin(meta_ids)].reset_index(drop=True)
    if 'Sample_ID' in y_real.columns:
        y_real = y_real[~y_real['Sample_ID'].isin(meta_ids)].reset_index(drop=True)
    
    logging.info(f"Real data shapes - X: {X_real.shape}, y: {y_real.shape}, PCA: {pca_real.shape}")
    
    # 验证样本ID对齐
    X_sample_ids = set(X_real['Sample_ID'])
    y_sample_ids = set(y_real['Sample_ID'])
    pca_sample_ids = set(pca_real['Sample_ID'])
    
    common_ids = X_sample_ids.intersection(y_sample_ids).intersection(pca_sample_ids)
    logging.info(f"Found {len(common_ids)} common samples across all real datasets")
    
    # 过滤数据，只保留共同的样本ID
    X_filtered = X_real[X_real['Sample_ID'].isin(common_ids)].sort_values('Sample_ID').reset_index(drop=True)
    y_filtered = y_real[y_real['Sample_ID'].isin(common_ids)].sort_values('Sample_ID').reset_index(drop=True)
    pca_filtered = pca_real[pca_real['Sample_ID'].isin(common_ids)].sort_values('Sample_ID').reset_index(drop=True)
    
    # 确保样本ID完全对齐
    if not (X_filtered['Sample_ID'].equals(y_filtered['Sample_ID']) and 
            X_filtered['Sample_ID'].equals(pca_filtered['Sample_ID'])):
        logging.error("Sample IDs not properly aligned after filtering")
        sys.exit(1)
    
    # 验证目标列存在（允许合成-only目标或无监督）
    has_real_target = CONFIG["TARGET_COLUMN"] is not None and CONFIG["TARGET_COLUMN"] in y_filtered.columns
    if not has_real_target:
        if CONFIG["USE_SYNTHETIC"] and CONFIG["TARGET_COLUMN"] is not None:
            logging.warning(f"Target column {CONFIG['TARGET_COLUMN']} not found in real y.csv; will use synthetic-only target.")
        elif CONFIG["TARGET_COLUMN"] is None:
            logging.info("No target column specified; pipeline will skip supervised training.")
        else:
            logging.error(f"Target column {CONFIG['TARGET_COLUMN']} not found in y.csv")
            sys.exit(1)

    # 提取特征（排除Sample_ID）
    X_real_features = X_filtered.drop('Sample_ID', axis=1)
    pca_real_features = pca_filtered.drop('Sample_ID', axis=1)

    # 合并X和PCA协变量
    X_real_combined = pd.concat([X_real_features, pca_real_features], axis=1)

    # 提取目标变量或准备合成-only/无监督
    if has_real_target:
        y_real_values = y_filtered[CONFIG["TARGET_COLUMN"]]
        all_sample_ids = X_filtered['Sample_ID'].tolist()
        X_combined = X_real_combined.copy()
        y_values = y_real_values.copy()
    else:
        y_real_values = pd.Series([], dtype=float)
        all_sample_ids = []
        X_combined = pd.DataFrame(columns=X_real_combined.columns)
        y_values = pd.Series([], dtype=float)
    
    if CONFIG["USE_SYNTHETIC"]:
        logging.info(f"Loading synthetic data from {CONFIG['AUGMENTED']}")
        
        # 加载合成数据：préférence à augment_file si celui-ci contient les colonnes SNPs
        X_synth_path_default = os.path.join(CONFIG["AUGMENTED"], "synthetic_snps.csv")
        X_synthetic = None
        aug_file = CONFIG.get("AUGMENT_FILE")
        if aug_file and os.path.exists(aug_file):
            try:
                tmp_df = pd.read_csv(aug_file, nrows=1)
                # Heuristic: if more than 10 columns and includes non-['Sample_ID', target], treat as feature source
                if tmp_df.shape[1] > 10:
                    X_synthetic = pd.read_csv(aug_file)
                else:
                    X_synthetic = pd.read_csv(X_synth_path_default)
            except Exception:
                X_synthetic = pd.read_csv(X_synth_path_default)
        else:
            X_synthetic = pd.read_csv(X_synth_path_default)
        
        # 检查合成数据是否有Sample_ID列
        if 'Sample_ID' not in X_synthetic.columns:
            synthetic_sample_ids = [f"SYNTHETIC_{i}" for i in range(len(X_synthetic))]
            X_synthetic.insert(0, 'Sample_ID', synthetic_sample_ids)
        else:
            synthetic_sample_ids = X_synthetic['Sample_ID'].tolist()
        
        # 验证合成数据列是否与真实数据匹配（只考虑SNP列）
        real_snp_columns = X_real_features.columns.tolist()
        synthetic_snp_columns = X_synthetic.drop('Sample_ID', axis=1).columns.tolist()
        
        # 记录列数量差异
        logging.info(f"Real data has {len(real_snp_columns)} SNP columns, synthetic data has {len(synthetic_snp_columns)} columns")
        logging.info("Using columnar load for 3000 selected SNPs")
        
        # 检查列匹配情况 - 不再强制要求完全匹配，而是创建缺失的列
        missing_cols = set(real_snp_columns) - set(synthetic_snp_columns)
        extra_cols = set(synthetic_snp_columns) - set(real_snp_columns)
        
        if missing_cols:
            logging.warning(f"Synthetic data missing {len(missing_cols)} SNP columns, will create missing columns with default values")
        if extra_cols:
            logging.warning(f"Synthetic data has {len(extra_cols)} extra columns that won't be used")
        
        # 为合成数据创建一个与真实数据匹配的特征矩阵
        X_synthetic_aligned = pd.DataFrame(0, index=range(len(X_synthetic)), columns=real_snp_columns)
        for col in synthetic_snp_columns:
            if col in real_snp_columns:
                X_synthetic_aligned[col] = X_synthetic[col]
        
        # 为合成数据创建PCA协变量（使用0值填充，因为合成数据没有真正的PCA协变量）
        synthetic_pca = pd.DataFrame(0, index=range(len(X_synthetic_aligned)), columns=pca_real_features.columns)
        
        # 合并合成数据的SNP和PCA协变量
        X_synthetic_combined = pd.concat([X_synthetic_aligned, synthetic_pca], axis=1)
        
        # 目标变量：如果需要且存在，则加载；否则在无监督模式下跳过
        y_synthetic_values = None
        synth_y_default = os.path.join(CONFIG["AUGMENTED"], "synthetic_y.csv")
        synth_y_path = CONFIG.get("AUGMENT_FILE") or synth_y_default
        if CONFIG["TARGET_COLUMN"] is not None and os.path.exists(synth_y_path):
            # Robust CSV parsing for synthetic_y with potential irregularities
            try:
                y_synthetic = pd.read_csv(synth_y_path, engine='python')
            except Exception:
                y_synthetic = pd.read_csv(synth_y_path)
            # Align X_synthetic rows to y_synthetic Sample_IDs and filter to intersection
            if 'Sample_ID' in y_synthetic.columns:
                y_ids = y_synthetic['Sample_ID'].tolist()
                # Reindex X_synthetic_aligned to y_ids order
                X_synthetic_indexed = X_synthetic.set_index('Sample_ID')
                common_ids = [sid for sid in y_ids if sid in X_synthetic_indexed.index]
                X_synthetic = X_synthetic_indexed.loc[common_ids].reset_index()
                synthetic_sample_ids = common_ids
                # Normalize synthetic IDs to have SYNTHETIC_ prefix for leak guard
                normalized_ids = [sid if str(sid).startswith('SYNTHETIC_') else f"SYNTHETIC_{sid}" for sid in synthetic_sample_ids]
                # Apply normalized IDs to X_synthetic
                X_synthetic['Sample_ID'] = normalized_ids
                synthetic_sample_ids = normalized_ids
                # Reorder and apply normalized IDs to y_synthetic to match X order
                y_syn_indexed = y_synthetic.set_index('Sample_ID')
                y_synthetic = y_syn_indexed.loc[common_ids].reset_index(drop=True)
                y_synthetic['Sample_ID'] = normalized_ids
                # Recompute aligned and combined to reflect filtered X
                synthetic_snp_columns = X_synthetic.drop('Sample_ID', axis=1).columns.tolist()
                X_synthetic_aligned = pd.DataFrame(0, index=range(len(X_synthetic)), columns=real_snp_columns)
                for col in synthetic_snp_columns:
                    if col in real_snp_columns:
                        X_synthetic_aligned[col] = X_synthetic[col]
                synthetic_pca = pd.DataFrame(0, index=range(len(X_synthetic_aligned)), columns=pca_real_features.columns)
                X_synthetic_combined = pd.concat([X_synthetic_aligned, synthetic_pca], axis=1)
            # Extract target values
            if CONFIG["TARGET_COLUMN"] in y_synthetic.columns:
                y_synthetic_values = pd.to_numeric(y_synthetic[CONFIG["TARGET_COLUMN"]], errors='coerce')
            else:
                # if filtered file has two columns (Sample_ID + target), prefer second
                fallback_series = y_synthetic.iloc[:, 1] if 'Sample_ID' in y_synthetic.columns and y_synthetic.shape[1] > 1 else y_synthetic.iloc[:, 0]
                y_synthetic_values = pd.to_numeric(fallback_series, errors='coerce')
                logging.warning(f"Using fallback column of synthetic y as target instead of {CONFIG['TARGET_COLUMN']}")
            # Record effective augment size
            CONFIG["AUGMENT_SIZE_EFFECTIVE"] = int(len(y_synthetic))
            # If no synthetic targets after filtering, skip augmentation gracefully
            if CONFIG["AUGMENT_SIZE_EFFECTIVE"] == 0:
                logging.warning("Filtered synthetic targets are empty; skipping synthetic augmentation for this run.")
                X_synthetic_combined = pd.DataFrame(columns=X_real_combined.columns)
                synthetic_sample_ids = []
                y_synthetic_values = pd.Series([], dtype=float)
        elif CONFIG["TARGET_COLUMN"] is None:
            logging.info("Unsupervised mode: synthetic_y not used.")
        else:
            logging.error("Synthetic target required but not found or invalid.")
            sys.exit(1)
            
        # 合并真实数据和合成数据或使用合成-only
        if has_real_target:
            # Combine real and (possibly empty) synthetic blocks
            if X_synthetic_combined is not None and len(X_synthetic_combined) > 0:
                X_combined = pd.concat([X_real_combined, X_synthetic_combined], ignore_index=True)
                if y_synthetic_values is not None and len(y_synthetic_values) > 0:
                    y_values = pd.concat([y_real_values, y_synthetic_values], ignore_index=True)
                else:
                    logging.error("Synthetic target missing in supervised mode after combination")
                    sys.exit(1)
                all_sample_ids.extend(synthetic_sample_ids)
            else:
                logging.info("No synthetic samples included; proceeding with real-only supervised training.")
                X_combined = X_real_combined.copy()
                y_values = y_real_values.copy()
                # synthetic_sample_ids is empty; nothing to extend
        else:
            X_combined = X_synthetic_combined
            if y_synthetic_values is not None:
                y_values = y_synthetic_values.reset_index(drop=True)
            else:
                logging.error("Synthetic-only run requires synthetic target.")
                sys.exit(1)
            all_sample_ids = synthetic_sample_ids
    
        logging.info(f"Combined dataset shape: {X_combined.shape} samples with {X_combined.shape[1]} features")
        logging.info(f"Real samples: {len(X_real_combined)}, Synthetic samples: {len(X_synthetic_combined)}")
        if CONFIG.get("AUGMENT_SIZE_EFFECTIVE") is not None:
            logging.info(f"Effective augment size: {CONFIG.get('AUGMENT_SIZE_EFFECTIVE')}")
    
    # 检查目标变量中的NaN值和非数值型值（监督模式）
    if len(y_values) > 0 and not pd.api.types.is_numeric_dtype(y_values):
        logging.warning("Target variable contains non-numeric values. Converting to numeric.")
        y_values = pd.to_numeric(y_values, errors='coerce')
    
    if len(y_values) > 0:
        nan_count = y_values.isna().sum()
        if nan_count > 0:
            logging.warning(f"Found {nan_count} NaN values in target variable. Removing these samples.")
            valid_indices = ~y_values.isna()
            X_combined = X_combined[valid_indices].reset_index(drop=True)
            y_values = y_values[valid_indices].reset_index(drop=True)
            all_sample_ids = [all_sample_ids[i] for i, valid in enumerate(valid_indices) if valid]
            logging.info(f"After removing NaN targets: {X_combined.shape[0]} samples remaining")
    
    # 基础数据信息
    logging.info(f"Final data: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
    logging.info(f"Using target column: {CONFIG['TARGET_COLUMN']}")
    
    # 处理特征中的NaN/inf值
    X_combined = X_combined.replace([np.inf, -np.inf], np.nan)
    X_combined = X_combined.fillna(X_combined.median())
    
    # 转换为numpy数组
    # Use float32 to reduce memory footprint
    X_array = X_combined.values.astype(np.float32)
    y_array = y_values.values.astype(np.float32)
    
    return X_array, y_array, all_sample_ids

def build_leakage_guard_splits(sample_ids):
    """Build outer CV splits with leak guard and a real-only holdout.
    - Outer test folds contain only real samples.
    - Training folds contain real-train (+ synthetic unless synthetic_only).
    - Holdout is a separate real-only split excluded from CV.
    Returns: (outer_splits, holdout_indices, info)
    """
    logging.info("Building leakage-guarded splits with real-only holdout")

    # Identify real vs synthetic
    real_idx = [i for i, sid in enumerate(sample_ids) if not str(sid).startswith("SYNTHETIC_")]
    syn_idx = [i for i, sid in enumerate(sample_ids) if str(sid).startswith("SYNTHETIC_")]
    logging.info(f"Real indices: {len(real_idx)}, Synthetic indices: {len(syn_idx)}")

    # Create real-only holdout
    real_arr = np.array(real_idx)
    real_train_all, real_holdout = train_test_split(
        real_arr,
        test_size=CONFIG.get("HOLDOUT_FRACTION", 0.2),
        random_state=CONFIG["RANDOM_SEED"]
    )
    holdout_indices = sorted(real_holdout.tolist())
    real_for_cv = sorted(real_train_all.tolist())

    # Outer CV on real_for_cv only
    outer_kf = KFold(n_splits=CONFIG["OUTER_CV_FOLDS"], shuffle=True, random_state=CONFIG["RANDOM_SEED"])
    outer_splits = []
    for tr_real, te_real in outer_kf.split(np.arange(len(real_for_cv))):
        tr_real_idx = np.array(real_for_cv)[tr_real].tolist()
        te_real_idx = np.array(real_for_cv)[te_real].tolist()
        if CONFIG.get("SYNTHETIC_ONLY", False):
            train_idx = syn_idx.copy()
        else:
            train_idx = sorted(list(set(tr_real_idx) | set(syn_idx)))
        outer_splits.append({
            "train": train_idx,
            "test": te_real_idx,
            "train_real": tr_real_idx
        })

    # Save split info
    split_info = {
        "real_indices": real_idx,
        "synthetic_indices": syn_idx,
        "holdout_indices": holdout_indices,
        "outer_splits": outer_splits,
        "random_seed": CONFIG["RANDOM_SEED"]
    }
    with open(os.path.join(CONFIG["OUTDIR"], "logs", f"splits_{CONFIG['RUN_ID']}.json"), 'w') as f:
        json.dump(split_info, f, indent=2)

    return outer_splits, holdout_indices, split_info

def train_models(X, y, sample_ids, outer_splits, holdout_indices):
    """训练模型，使用泄漏保护CV和真实-only holdout评估"""
    # 在无监督模式下跳过训练
    if y is None or len(y) == 0:
        logging.info("No target available; skipping supervised training. Writing empty metrics.")
        metrics_path = os.path.join(CONFIG["OUTDIR"], "metrics", "all_models_metrics.json")
        run_metrics_path = os.path.join(CONFIG["OUTDIR"], "metrics", f"all_models_metrics_{CONFIG['RUN_ID']}.json")
        with open(metrics_path, 'w') as f:
            json.dump({}, f)
        with open(run_metrics_path, 'w') as f:
            json.dump({}, f)
        return {}

    logging.info("Starting model training")
    
    # 根据selected_k控制特征选择
    needs_feature_selection = X.shape[1] > (CONFIG["SELECTED_K"] or CONFIG["MAX_FEATURES"]) if CONFIG["SELECTED_K"] else X.shape[1] > CONFIG["MAX_FEATURES"]
    selected_k = CONFIG["SELECTED_K"] if CONFIG["SELECTED_K"] else (5000 if X.shape[1] > 6000 else None)
    logging.info(f"Feature count: {X.shape[1]}, Needs feature selection: {needs_feature_selection}, selected_k: {selected_k}")
    
    # 获取所有已注册模型
    registered_models = list_registered_models()
    models = {}
    
    # 随机森林参数标准化函数
    def _normalize_depth_list(values):
        if values is None:
            return None
        out = []
        for v in values:
            if v is None:
                out.append(None)
                continue
            vs = str(v).strip().lower()
            if vs in {"none", "null"}:
                out.append(None)
            else:
                out.append(int(v))
        return out

    def _normalize_max_features_list(values):
        if values is None:
            return None
        out = []
        for v in values:
            if v is None:
                out.append(None)
                continue
            vs = str(v).strip().lower()
            if vs in {"none", "null"}:
                out.append(None)
            else:
                out.append(vs)
        return out

    # 获取所有已注册模型
    registered_models = list_registered_models()
    models = {}
    
    for model_name, model_config in registered_models.items():
        # 初始化模型 - 仅对接受random_state参数的模型传递该参数
        estimator_class = model_config['estimator_class']
        sig = inspect.signature(estimator_class.__init__)
        if 'random_state' in sig.parameters:
            estimator = estimator_class(random_state=CONFIG["RANDOM_SEED"])
        else:
            estimator = estimator_class()
        
        # 获取超参数网格
        params = model_config['param_grid'].copy()
        
        # 为随机森林应用特殊处理
        if model_name == 'randomforest':
            # 保持随机森林的特殊处理逻辑
            rf_params = params.copy()
            rf_override = CONFIG.get('RF_OVERRIDE')
            if rf_override:
                if rf_override.get('max_depth'):
                    rf_params['max_depth'] = _normalize_depth_list(rf_override['max_depth'])
                if rf_override.get('max_features'):
                    rf_params['max_features'] = _normalize_max_features_list(rf_override['max_features'])
                if rf_override.get('n_estimators'):
                    rf_params['n_estimators'] = rf_override['n_estimators']
                # estimator-level override for max_samples (not part of grid)
                if rf_override.get('max_samples') is not None:
                    try:
                        estimator.set_params(max_samples=float(rf_override['max_samples']))
                    except Exception:
                        pass
            params = rf_params
        
        # 创建模型配置项
        models[model_name] = {
            'estimator': estimator,
            'params': params,
            'needs_scaling': model_config['needs_scaling'],
            'needs_feature_selection': needs_feature_selection,
            'early_stopping': model_config['early_stopping']
        }




    
    # NEW: filter models by --models
    if CONFIG.get('MODELS'):
        requested = CONFIG['MODELS']
        valid = set(models.keys())
        missing = [m for m in requested if m not in valid]
        if missing:
            logging.warning(f"Requested models not available or misspelled: {missing}")
        models = {k: v for k, v in models.items() if k in requested}
        if not models:
            logging.error("No valid models left to train after filtering. Exiting.")
            return {}
    
    logging.info(f"Will train the following models: {list(models.keys())}")
    
    # 加载现有结果
    all_metrics_path = os.path.join(CONFIG["OUTDIR"], "metrics", "all_models_metrics.json")
    results = {}
    if os.path.exists(all_metrics_path) and not CONFIG["OVERWRITE_PREVIOUS"]:
        try:
            with open(all_metrics_path, 'r') as f:
                results = json.load(f)
            logging.info("Loaded existing results")
        except Exception:
            logging.warning("Failed to load existing results, starting fresh")
            results = {}

    # 训练每个模型
    for model_name, model_info in models.items():
        model_dir = os.path.join(CONFIG["OUTDIR"], "models", f"{model_name}_{CONFIG['RUN_ID']}")
        os.makedirs(model_dir, exist_ok=True)
        metrics_path = os.path.join(CONFIG["OUTDIR"], "metrics", f"{model_name}_metrics_{CONFIG['RUN_ID']}.json")

        if model_name in results and not CONFIG["OVERWRITE_PREVIOUS"]:
            logging.info(f"Results for {model_name} already exist, skipping")
            continue

        logging.info(f"Training {model_name}...")
        start_time = time.time()

        # 特征选择
        X_selected = X
        selector = None
        selected_k_local = None
        if model_info['needs_feature_selection']:
            logging.info(f"Performing feature selection for {model_name}")
            k = CONFIG["SELECTED_K"] if CONFIG.get("SELECTED_K") else min(CONFIG["MAX_FEATURES"], X.shape[1])
            selected_k_local = k
            selector = SelectKBest(f_regression, k=k)
            X_selected = selector.fit_transform(X, y)
            joblib.dump(selector, os.path.join(model_dir, "feature_selector.joblib"))

        # 嵌套交叉验证（外层测试仅真实，训练包含真实+合成或仅合成）
        all_r2_scores, all_rmse_scores, all_mae_scores, all_best_params = [], [], [], []
        for i, split in enumerate(outer_splits):
            logging.info(f"  Fold {i+1}/{CONFIG['OUTER_CV_FOLDS']}")
            train_idx = np.array(split['train'])
            test_idx = np.array(split['test'])
            train_real_idx = np.array(split['train_real'])

            # 训练数据：真实训练折 + 合成（或仅合成）
            if len(train_real_idx) > 0 and not CONFIG.get("SYNTHETIC_ONLY", False):
                synthetic_train_idx = np.setdiff1d(train_idx, train_real_idx)
                X_train = np.vstack([X_selected[train_real_idx], X_selected[synthetic_train_idx]])
                y_train = np.concatenate([y[train_real_idx], y[synthetic_train_idx]])
                n_real_train = len(train_real_idx)
                n_total_train = len(train_idx)
                synthetic_pos = np.arange(n_real_train, n_total_train)
                # 构建内层CV，验证仅真实
                inner_kf = KFold(n_splits=CONFIG["INNER_CV_FOLDS"], shuffle=True, random_state=CONFIG["RANDOM_SEED"])
                inner_splits = []
                for tr_r, val_r in inner_kf.split(np.arange(n_real_train)):
                    tr_indices = np.concatenate([tr_r, synthetic_pos])
                    inner_splits.append((tr_indices, val_r))
            else:
                # 合成-only：训练和内层CV基于合成；外层测试仍为真实
                synthetic_train_idx = train_idx
                X_train = X_selected[synthetic_train_idx]
                y_train = y[synthetic_train_idx]
                n_syn_train = len(synthetic_train_idx)
                inner_kf = KFold(n_splits=max(2, CONFIG["INNER_CV_FOLDS"]), shuffle=True, random_state=CONFIG["RANDOM_SEED"])
                inner_splits = list(inner_kf.split(np.arange(n_syn_train)))

            # 测试数据：真实测试折
            X_test = X_selected[test_idx]
            y_test = y[test_idx]

            # 缩放（如需）
            scaler = None
            if model_info['needs_scaling']:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                joblib.dump(scaler, os.path.join(model_dir, f"scaler_fold_{i+1}.joblib"))

            # 网格搜索
            grid = GridSearchCV(
                model_info['estimator'],
                model_info['params'],
                cv=inner_splits,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid.fit(X_train, y_train)

            # 外层测试评估
            y_pred = grid.best_estimator_.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            all_r2_scores.append(r2)
            all_rmse_scores.append(rmse)
            all_mae_scores.append(mae)
            all_best_params.append(grid.best_params_)
            logging.info(f"  Fold {i+1}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

        # 最终模型训练：真实非holdout + 合成（或仅合成）
        real_indices_all = [i for i, sid in enumerate(sample_ids or []) if not str(sid).startswith("SYNTHETIC_")]
        synthetic_all = [i for i, sid in enumerate(sample_ids or []) if str(sid).startswith("SYNTHETIC_")]
        real_train_final = np.array(sorted(list(set(real_indices_all) - set(holdout_indices or []))))
        if CONFIG.get("SYNTHETIC_ONLY", False):
            final_train_indices = np.array(synthetic_all)
        else:
            final_train_indices = np.concatenate([real_train_final, np.array(synthetic_all)]) if len(synthetic_all) > 0 else real_train_final

        if model_info['needs_scaling']:
            final_scaler = StandardScaler()
            X_final = final_scaler.fit_transform(X_selected[final_train_indices])
            joblib.dump(final_scaler, os.path.join(model_dir, "final_scaler.joblib"))
        else:
            X_final = X_selected[final_train_indices]

        final_model = model_info['estimator']
        final_model.set_params(**all_best_params[np.argmax(all_r2_scores)])
        # Apply LightGBM early stopping callbacks for final training when enabled
        if model_info.get('early_stopping', False) and model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
            val_n = max(1, int(0.1 * len(X_final)))
            X_val, y_val = X_final[:val_n], y[final_train_indices][:val_n]
            fit_params = {
                'eval_set': [(X_val, y_val)],
                'eval_metric': 'rmse',
                'callbacks': [lgb.early_stopping(50, verbose=False)]
            }
            final_model.fit(X_final, y[final_train_indices], **fit_params)
        else:
            final_model.fit(X_final, y[final_train_indices])
        joblib.dump(final_model, os.path.join(model_dir, f"{model_name}_final_model.joblib"))

        # 真实-only holdout评估
        holdout_idx = np.array(holdout_indices or [])
        X_holdout = X_selected[holdout_idx]
        y_holdout = y[holdout_idx]
        if model_info['needs_scaling']:
            try:
                X_holdout = final_scaler.transform(X_holdout)
            except Exception:
                pass
        y_hold_pred = final_model.predict(X_holdout)
        holdout_r2 = r2_score(y_holdout, y_hold_pred)
        holdout_rmse = np.sqrt(mean_squared_error(y_holdout, y_hold_pred))
        holdout_mae = mean_absolute_error(y_holdout, y_hold_pred)
        logging.info(f"Holdout: R2={holdout_r2:.4f}, RMSE={holdout_rmse:.4f}, MAE={holdout_mae:.4f}")

        # 汇总折内指标
        cv_r2_mean = float(np.mean(all_r2_scores))
        cv_r2_std = float(np.std(all_r2_scores))
        cv_rmse_mean = float(np.mean(all_rmse_scores))
        cv_rmse_std = float(np.std(all_rmse_scores))
        cv_mae_mean = float(np.mean(all_mae_scores))
        cv_mae_std = float(np.std(all_mae_scores))

        training_time = time.time() - start_time
        model_metrics = {
            "model_name": model_name,
            "cv_r2_mean": cv_r2_mean,
            "cv_r2_std": cv_r2_std,
            "cv_rmse_mean": cv_rmse_mean,
            "cv_rmse_std": cv_rmse_std,
            "cv_mae_mean": cv_mae_mean,
            "cv_mae_std": cv_mae_std,
            "holdout_r2": float(holdout_r2),
            "holdout_rmse": float(holdout_rmse),
            "holdout_mae": float(holdout_mae),
            "training_time_seconds": training_time,
            "fold_metrics": {
                "r2_scores": all_r2_scores,
                "rmse_scores": all_rmse_scores,
                "mae_scores": all_mae_scores,
                "best_params_per_fold": all_best_params
            },
            "final_best_params": all_best_params[np.argmax(all_r2_scores)],
            "features_count": X.shape[1],
            "feature_count": X.shape[1],
            "selected_k": selected_k,
            "needs_feature_selection": model_info['needs_feature_selection'],
            "run_id": CONFIG["RUN_ID"],
            "augment_mode": CONFIG.get("AUGMENT_MODE"),
            "augment_size": CONFIG.get("AUGMENT_SIZE"),
            "augment_size_effective": CONFIG.get("AUGMENT_SIZE_EFFECTIVE"),
            "augment_seed": CONFIG.get("AUGMENT_SEED"),
            "augment_file": CONFIG.get("AUGMENT_FILE"),
            "use_synthetic": CONFIG.get("USE_SYNTHETIC"),
            "synthetic_only": CONFIG.get("SYNTHETIC_ONLY", False),
            # NEW metadata
            "sigma_resid_factor": CONFIG.get("SIGMA_RESID_FACTOR"),
            "fallback_percent": CONFIG.get("FALLBACK_PERCENT"),
            "selected_models": CONFIG.get("MODELS"),
            "samples_count": int(len(y)),
            "timestamp": datetime.now().isoformat()
        }
        # Ensure per-model metrics JSON is always written atomically.
        metrics_payload = model_metrics
        try:
            # Update in-memory aggregate for later write
            results[model_name] = model_metrics
            logging.info(f"Completed training {model_name} in {training_time:.2f} seconds")
            logging.info(f"  CV R2: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
            logging.info(f"  CV RMSE: {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}")
            logging.info(f"  CV MAE: {cv_mae_mean:.4f} ± {cv_mae_std:.4f}")
        finally:
            # Attempt atomic write regardless of subsequent failures
            save_model_metrics_atomic(os.path.join(CONFIG["OUTDIR"], "metrics"), model_name, CONFIG["RUN_ID"], metrics_payload)

    # 保存所有模型的指标（aggregate written last; per-model writes not blocked）
    try:
        with open(all_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logging.info("Model training completed")
    except Exception as e:
        logging.error(f"Failed to write aggregated metrics: {e}")
        logging.info("Continuing; per-model files exist.")
    return results

def update_summary(results):
    """更新摘要报告"""
    logging.info("Updating summary report")
    summary_path = os.path.join(CONFIG["OUTDIR"], "summary.md")
    
    # 排序模型
    sorted_models = sorted([(k, v) for k, v in results.items() if isinstance(v, dict) and 'cv_r2_mean' in v], key=lambda x: x[1]['cv_r2_mean'] if isinstance(x[1]['cv_r2_mean'], float) else -1, reverse=True)
    
    summary_content = f"# Augmented Modeling Summary for {CONFIG['DATASET']}\n\n"
    summary_content += f"Run ID: `{CONFIG['RUN_ID']}`\n\n"
    aug_file_note = f", file=`{CONFIG['AUGMENT_FILE']}`" if CONFIG.get('AUGMENT_FILE') else ""
    eff_size_note = f" (effective={CONFIG['AUGMENT_SIZE_EFFECTIVE']})" if CONFIG.get('AUGMENT_SIZE_EFFECTIVE') is not None else ""
    summary_content += f"Augmentation: mode=`{CONFIG['AUGMENT_MODE']}`, size=`{CONFIG['AUGMENT_SIZE']}{eff_size_note}`, seed=`{CONFIG['AUGMENT_SEED']}`{aug_file_note}\n\n"
    
    if CONFIG.get('SIGMA_RESID_FACTOR') is not None:
        summary_content += f"Sigma residual factor (metadata): `{CONFIG['SIGMA_RESID_FACTOR']}`\n\n"
    
    if len(results) == 0:
        summary_content += "Supervised training skipped (no target).\n\n"
    else:
        summary_content += "## Model Metrics\n\n"
        summary_content += "| Model | CV R2 Mean | CV R2 Std | CV RMSE Mean | CV RMSE Std | CV MAE Mean | CV MAE Std | Holdout R2 | Holdout RMSE | Holdout MAE | Features | selected_k |\n"
        summary_content += "|-------|------------|-----------|--------------|-------------|-------------|------------|------------|--------------|-------------|----------|------------|\n"
        for model_name, metrics in sorted_models:
            summary_content += (
                f"| {model_name} | {metrics['cv_r2_mean']:.4f} | {metrics['cv_r2_std']:.4f} | {metrics['cv_rmse_mean']:.4f} | {metrics['cv_rmse_std']:.4f} | "
                f"{metrics['cv_mae_mean']:.4f} | {metrics['cv_mae_std']:.4f} | {metrics.get('holdout_r2', float('nan')):.4f} | "
                f"{metrics.get('holdout_rmse', float('nan')):.4f} | {metrics.get('holdout_mae', float('nan')):.4f} | {metrics['features_count']} | {metrics['selected_k']} |\n"
            )

        summary_content += "\n## Hyperparameter Grids\n\n"
        for model_name, metrics in sorted_models:
            summary_content += f"### {model_name}\n\n"
            if 'final_best_params' in metrics:
                summary_content += "Best parameters:\n\n"
                for param, value in metrics['final_best_params'].items():
                    summary_content += f"- **{param}**: {value}\n"
                summary_content += "\n"
    
    # 生成的文件信息
    summary_content += "\n## Generated Artifacts\n\n"
    summary_content += f"- **Models**: `{os.path.join(CONFIG['OUTDIR'], 'models')}` directory\n"
    summary_content += f"- **Metrics**: `{os.path.join(CONFIG['OUTDIR'], 'metrics')}` directory\n"
    summary_content += f"- **Logs**: `{os.path.join(CONFIG['OUTDIR'], 'logs')}` directory with environment info and splits\n"
    config_file = os.path.join(CONFIG['OUTDIR'], 'logs', f'config_{CONFIG["RUN_ID"]}.json')
    summary_content += f"- **Configuration**: `{config_file}`\n"
    splits_file = os.path.join(CONFIG['OUTDIR'], 'logs', f'splits_{CONFIG["RUN_ID"]}.json')
    summary_content += f"- **Splits**: `{splits_file}`\n"

    norm_report = os.path.join(CONFIG["AUGMENTED"], "normalization_report.txt")
    summary_content += "\n## Augmentation Diagnostics\n\n"
    if os.path.exists(norm_report):
        try:
            with open(norm_report, 'r', encoding='utf-8') as f:
                summary_content += f.read()
        except Exception:
            summary_content += "Normalization report could not be read.\n"
    else:
        summary_content += "Normalization report not found.\n"

    summary_content += "\n## Notes\n\n"
    summary_content += "- Leakage guard: synthetic excluded from validation/test; real-only holdout evaluated.\n"
    if CONFIG.get('SYNTHETIC_ONLY'):
        summary_content += "- Synthetic-only diagnostic run (training/inner-CV on synthetic).\n"

    with open(summary_path, "w", encoding='utf-8') as f:
        f.write(summary_content)

    logging.info(f"Summary report updated: {summary_path}")
    return summary_path

def _load_json_safe(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def generate_comparison_table(results):
    """生成与原始模型的比较表格，稳健处理不同schema并排除非模型键"""
    logging.info("Generating comparison table")
    comparison_content = "# Comparison: Original vs Augmented\n\n"
    comparison_content += f"Dataset: `{CONFIG['DATASET']}` | Augmentation: `{CONFIG['AUGMENT_MODE']}` size=`{CONFIG['AUGMENT_SIZE']}`\n\n"

    # 加载原始结果（可能存在不同的schema）
    original_metrics_path = os.path.join(CONFIG["ORIGINAL_RESULTS"], "metrics", "all_models_metrics.json")
    original_results_raw = _load_json_safe(original_metrics_path)

    # 归一化原始结果为 {model_name: {cv_r2_mean, cv_rmse_mean, cv_mae_mean}}
    original_models_map = {}
    if isinstance(original_results_raw, dict):
        # 从 'models' 列表提取（如果存在）
        models_list = original_results_raw.get("models")
        if isinstance(models_list, list):
            for m in models_list:
                if isinstance(m, dict):
                    name = m.get("name")
                    if name:
                        original_models_map[name] = {
                            "cv_r2_mean": m.get("cv_r2_mean"),
                            "cv_rmse_mean": m.get("cv_rmse_mean", m.get("cv_rmse")),
                            "cv_mae_mean": m.get("cv_mae_mean", m.get("cv_mae")),
                        }
        # 直接键：过滤掉元信息键，仅保留包含度量的模型块
        for key, val in original_results_raw.items():
            if key in ("run_id", "timestamp", "baseline", "note", "models"):
                continue
            if isinstance(val, dict) and ("cv_r2_mean" in val or "cv_rmse_mean" in val or "cv_mae_mean" in val):
                original_models_map[key] = {
                    "cv_r2_mean": val.get("cv_r2_mean"),
                    "cv_rmse_mean": val.get("cv_rmse_mean"),
                    "cv_mae_mean": val.get("cv_mae_mean"),
                }

    if not original_models_map:
        comparison_content += "Original metrics not found or unrecognized schema.\n"

    # 归一化增强结果
    augmented_models_map = {}
    for k, v in results.items():
        if isinstance(v, dict) and ("cv_r2_mean" in v):
            augmented_models_map[k] = {
                "cv_r2_mean": v.get("cv_r2_mean"),
                "cv_rmse_mean": v.get("cv_rmse_mean"),
                "cv_mae_mean": v.get("cv_mae_mean"),
            }

    # 统一模型命名（例如 random_forest vs randomforest）
    def normalize_name(name: str) -> str:
        if name is None:
            return name
        return name.replace("random_forest", "randomforest")

    normalized_original = {normalize_name(n): m for n, m in original_models_map.items()}
    normalized_augmented = {normalize_name(n): m for n, m in augmented_models_map.items()}

    # 汇总所有模型名（仅模型）
    all_models = sorted(set(list(normalized_original.keys()) + list(normalized_augmented.keys())))

    comparison_content += "| Model | R2 (Original) | R2 (Augmented) | ΔR2 | RMSE (Original) | RMSE (Augmented) | ΔRMSE | MAE (Original) | MAE (Augmented) | ΔMAE |\n"
    comparison_content += "|-------|----------------|----------------|-----|------------------|------------------|-------|----------------|------------------|------|\n"

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else ("N/A" if v is None else str(v))

    def diff(a, b, higher_is_better=True):
        if isinstance(a, (float, int)) and isinstance(b, (float, int)):
            d = float(a) - float(b)
            s = f"{d:+.4f}"
            if higher_is_better:
                if d > 0.01:
                    s += " ↑"
                elif d < -0.01:
                    s += " ↓"
            else:
                if d < -0.01:
                    s += " ↑"
                elif d > 0.01:
                    s += " ↓"
            return s
        return "N/A"

    for model_name in all_models:
        aug = normalized_augmented.get(model_name)
        orig = normalized_original.get(model_name)

        aug_r2 = aug.get("cv_r2_mean") if isinstance(aug, dict) else None
        aug_rmse = aug.get("cv_rmse_mean") if isinstance(aug, dict) else None
        aug_mae = aug.get("cv_mae_mean") if isinstance(aug, dict) else None

        orig_r2 = orig.get("cv_r2_mean") if isinstance(orig, dict) else None
        orig_rmse = orig.get("cv_rmse_mean") if isinstance(orig, dict) else None
        orig_mae = orig.get("cv_mae_mean") if isinstance(orig, dict) else None

        comparison_content += (
            f"| {model_name} | {fmt(orig_r2)} | {fmt(aug_r2)} | {diff(aug_r2, orig_r2, True)} | "
            f"{fmt(orig_rmse)} | {fmt(aug_rmse)} | {diff(aug_rmse, orig_rmse, False)} | "
            f"{fmt(orig_mae)} | {fmt(aug_mae)} | {diff(aug_mae, orig_mae, False)} |\n"
        )

    comparison_content += "\n## Overall Conclusion\n\n"
    comparison_content += "See per-model Δ metrics above.\n"

    comparison_file = os.path.join(CONFIG["OUTDIR"], "comparison_original_vs_augmented.md")
    try:
        with open(comparison_file, "w", encoding="utf-8") as f:
            f.write(comparison_content)
        logging.info(f"Comparison table generated: {comparison_file}")
        return comparison_file
    except Exception as e:
        logging.error(f"Failed to generate comparison table: {e}")
        return None

# Obsolete main() removed; see new main() below using leakage-guard splits.

def write_normalization_report():
    """
    写入 normalization_report.txt.
    Includes MAF drift, PCA variance ratio, Sample_ID uniqueness, SNP subset Hamming distance,
    and y alignment and basic stats for real vs synthetic.
    """
    try:
        x_real_path = os.path.join(CONFIG['PROCESSED'], 'X.csv')
        pca_real_path = os.path.join(CONFIG['PROCESSED'], 'pca_covariates.csv')
        x_syn_path = os.path.join(CONFIG['AUGMENTED'], 'synthetic_snps.csv')
        y_real_path = os.path.join(CONFIG['PROCESSED'], 'y.csv')
        y_syn_path_default = os.path.join(CONFIG['AUGMENTED'], 'synthetic_y.csv')
        y_syn_path = CONFIG.get('AUGMENT_FILE') or y_syn_path_default

        X_real = pd.read_csv(x_real_path)
        PCA_real = pd.read_csv(pca_real_path)
        X_syn = pd.read_csv(x_syn_path)

        # Ensure Sample_ID column
        if 'Sample_ID' not in X_syn.columns:
            X_syn.insert(0, 'Sample_ID', [f"SYNTHETIC_{i}" for i in range(len(X_syn))])

        # SNP columns (exclude Sample_ID)
        real_cols = [c for c in X_real.columns if c != 'Sample_ID']
        syn_cols = [c for c in X_syn.columns if c != 'Sample_ID']

        # Align synthetic to real SNP columns
        syn_aligned = pd.DataFrame(0, index=range(X_syn.shape[0]), columns=real_cols)
        for col in syn_cols:
            if col in syn_aligned.columns:
                syn_aligned[col] = X_syn[col]

        def maf_from_genotypes(df: pd.DataFrame) -> pd.Series:
            n = df.shape[0]
            if n == 0:
                return pd.Series(0, index=df.columns, dtype=float)
            denom = 2.0 * n
            maf = df.sum(axis=0) / denom
            maf = np.minimum(maf, 1 - maf)
            return pd.Series(maf, index=df.columns, dtype=float)

        mafs_real = maf_from_genotypes(X_real[real_cols])
        mafs_syn = maf_from_genotypes(syn_aligned[real_cols])
        maf_drift = (mafs_syn - mafs_real).abs()
        maf_drift_mean = float(maf_drift.mean())
        maf_drift_p95 = float(np.percentile(maf_drift.dropna(), 95)) if maf_drift.dropna().shape[0] > 0 else 0.0

        # PCA variance captured ratio (synthetic/real). Synthetic PCA is placeholder; try file, else 0.
        pc_cols = [c for c in PCA_real.columns if c != 'Sample_ID']
        var_real_pca = float(PCA_real[pc_cols].var().sum()) if len(pc_cols) > 0 else 0.0
        var_syn_pca = 0.0
        syn_pca_path = os.path.join(CONFIG['AUGMENTED'], 'synthetic_pca.csv')
        if os.path.exists(syn_pca_path):
            try:
                PCA_syn = pd.read_csv(syn_pca_path)
                syn_pc_cols = [c for c in PCA_syn.columns if c in pc_cols]
                var_syn_pca = float(PCA_syn[syn_pc_cols].var().sum()) if len(syn_pc_cols) > 0 else 0.0
            except Exception:
                var_syn_pca = 0.0
        pca_variance_ratio = float(var_syn_pca / var_real_pca) if var_real_pca > 0 else 0.0

        # Sample_ID uniqueness in synthetic
        syn_ids = X_syn['Sample_ID'].tolist()
        duplicates = len(syn_ids) - len(set(syn_ids))

        # Hamming distance synthetic↔real on subset
        subset_cols = real_cols[:min(1000, len(real_cols))]
        real_subset = X_real[subset_cols].values
        syn_subset = syn_aligned[subset_cols].values
        if real_subset.shape[0] > 0 and syn_subset.shape[0] > 0 and len(subset_cols) > 0:
            r_size = min(100, real_subset.shape[0])
            s_size = min(100, syn_subset.shape[0])
            r_idx = np.random.choice(real_subset.shape[0], size=r_size, replace=False)
            s_idx = np.random.choice(syn_subset.shape[0], size=s_size, replace=False)
            real_sample = real_subset[r_idx]
            syn_sample = syn_subset[s_idx]
            distances = []
            for i in range(syn_sample.shape[0]):
                a = syn_sample[i]
                b = real_sample[i % real_sample.shape[0]]
                distances.append(float(np.mean(a != b)))
            hamming_mean = float(np.mean(distances))
            hamming_p95 = float(np.percentile(distances, 95))
        else:
            hamming_mean = 0.0
            hamming_p95 = 0.0

        # y stats and alignment
        tgt = CONFIG.get('TARGET_COLUMN')
        y_real_numeric_stats = None
        y_syn_numeric_stats = None
        align_note = ""
        try:
            y_real_df = pd.read_csv(y_real_path)
            if tgt and tgt in y_real_df.columns:
                yr = pd.to_numeric(y_real_df[tgt], errors='coerce')
                y_real_numeric_stats = {
                    "count": int(yr.count()),
                    "mean": float(yr.mean()),
                    "std": float(yr.std()),
                    "min": float(yr.min()),
                    "max": float(yr.max())
                }
            if os.path.exists(y_syn_path):
                y_syn_df = pd.read_csv(y_syn_path)
                if 'Sample_ID' in y_syn_df.columns and 'Sample_ID' in X_syn.columns:
                    mismatch = set(X_syn['Sample_ID']) ^ set(y_syn_df['Sample_ID'])
                    align_note = f"synthetic_y aligned with synthetic_snps: {len(mismatch)==0}; mismatch_count={len(mismatch)}"
                else:
                    align_note = "synthetic_y missing Sample_ID column"
                if tgt:
                    ys = pd.to_numeric(y_syn_df[tgt] if tgt in y_syn_df.columns else y_syn_df.iloc[:, 1] if y_syn_df.shape[1] > 1 else y_syn_df.iloc[:, 0], errors='coerce')
                    y_syn_numeric_stats = {
                        "count": int(ys.count()),
                        "mean": float(ys.mean()),
                        "std": float(ys.std()),
                        "min": float(ys.min()),
                        "max": float(ys.max())
                    }
        except Exception:
            pass

        report_path = os.path.join(CONFIG["AUGMENTED"], "normalization_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Normalization Report\n")
            f.write(f"Dataset: {CONFIG['DATASET']}\n")
            eff = CONFIG.get('AUGMENT_SIZE_EFFECTIVE')
            aug_file = CONFIG.get('AUGMENT_FILE')
            eff_note = f" (effective={eff})" if eff is not None else ""
            file_note = f" file={aug_file}" if aug_file else ""
            f.write(f"Augment mode: {CONFIG.get('AUGMENT_MODE')} size={CONFIG.get('AUGMENT_SIZE')}{eff_note} seed={CONFIG.get('AUGMENT_SEED')}{file_note}\n\n")
            # NEW: include sigma metadata if present
            if CONFIG.get('SIGMA_RESID_FACTOR') is not None:
                f.write(f"Sigma residual factor (metadata): {CONFIG.get('SIGMA_RESID_FACTOR')}\n")
            f.write(f"MAF drift (mean): {maf_drift_mean:.6f}\n")
            f.write(f"MAF drift (p95): {maf_drift_p95:.6f}\n")
            f.write(f"MAF drift thresholds: mean<0.02 => {'PASS' if maf_drift_mean < 0.02 else 'FAIL'}, p95<0.05 => {'PASS' if maf_drift_p95 < 0.05 else 'FAIL'}\n")
            f.write(f"PCA variance captured (synthetic/real ratio): {pca_variance_ratio:.4f}\n")
            f.write(f"Synthetic Sample_ID duplicates: {duplicates}\n")
            f.write(f"Hamming distance synthetic↔real (mean over subset): {hamming_mean:.6f}\n")
            f.write(f"Hamming distance synthetic↔real (p95 over subset): {hamming_p95:.6f}\n")
            if y_real_numeric_stats:
                f.write(f"Real y stats {tgt}: {y_real_numeric_stats}\n")
            if y_syn_numeric_stats:
                f.write(f"Synthetic y stats {tgt}: {y_syn_numeric_stats}\n")
            if align_note:
                f.write(f"{align_note}\n")
            f.write("Leakage guard: synthetic excluded from validation/test; real-only holdout used.\n")

        logging.info(f"Normalization report written: {report_path}")
    except Exception as e:
        logging.warning(f"Failed to write normalization report: {e}")

def main():
    """主函数"""
    logging.info(f"Starting augmented modeling pipeline for {CONFIG['DATASET']}")

    try:
        log_environment()

        if not preflight_checks():
            logging.error("Pre-flight checks failed, exiting")
            sys.exit(1)

        X, y, sample_ids = load_and_align_data()
        try:
            logging.info(f"Loaded data arrays: X={getattr(X, 'shape', None)}, y={getattr(y, 'shape', None)}, sample_ids={len(sample_ids) if sample_ids is not None else 0}")
        except Exception:
            pass

        # Build leakage-guarded splits and real-only holdout
        outer_splits, holdout_indices, _ = build_leakage_guard_splits(sample_ids)
        try:
            logging.info(f"Constructed outer splits: {len(outer_splits)} folds; holdout size: {len(holdout_indices)}")
        except Exception:
            pass

        # Write normalization diagnostics
        write_normalization_report()

        # Train models with leak guard and holdout evaluation
        results = train_models(X, y, sample_ids, outer_splits, holdout_indices)

        update_summary(results)

        comparison_file = generate_comparison_table(results)

        # Post-run audit/backfill to ensure per-model metrics JSONs exist
        try:
            audit_script = os.path.join('scripts', 'audit_and_backfill_model_metrics.py')
            metrics_dir = os.path.join(CONFIG['OUTDIR'], 'metrics')
            cmd = [
                sys.executable,
                audit_script,
                '--dataset', CONFIG['DATASET'],
                '--metrics_dir', metrics_dir,
                '--run_id', CONFIG['RUN_ID'],
                '--backfill'
            ]
            logging.info(f"Running post-run audit/backfill: {' '.join(cmd)}")
            subprocess.run(cmd, check=False)
        except Exception as e:
            logging.warning(f"Post-run audit/backfill skipped due to error: {e}")

        logging.info(f"Augmented pipeline completed successfully for {CONFIG['DATASET']}")
        logging.info(f"Comparison table available at {comparison_file}")
    except BaseException as e:
        # Ensure unexpected errors are captured in the pipeline log and a simple error file
        logging.exception(f"Pipeline failed with an unexpected error: {e}")
        try:
            err_path = os.path.join(CONFIG["OUTDIR"], "logs", f"error_{CONFIG['RUN_ID']}.log")
            with open(err_path, 'w', encoding='utf-8') as ef:
                ef.write(f"Pipeline failed: {str(e)}\n")
            logging.info(f"Wrote error details to {err_path}")
        except Exception:
            pass
        # Exit with non-zero code to signal failure to callers
        sys.exit(1)

if __name__ == "__main__":
    main()
