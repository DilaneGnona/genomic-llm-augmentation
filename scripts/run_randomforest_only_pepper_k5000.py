import os
import logging
import importlib.util
import sys
import argparse

# Ensure working directory is repo root when running
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# Load the augmented pipeline module directly
MODULE_PATH = os.path.join(ROOT, "scripts", "unified_modeling_pipeline_augmented.py")
spec = importlib.util.spec_from_file_location("augmented_pipeline", MODULE_PATH)
aug = importlib.util.module_from_spec(spec)
# Prevent the augmented pipeline from parsing our CLI args on import
_saved_argv = sys.argv[:]
try:
    sys.argv = [MODULE_PATH]
    spec.loader.exec_module(aug)
finally:
    sys.argv = _saved_argv

def parse_args():
    p = argparse.ArgumentParser(description="Run RandomForest-only augmented pipeline for pepper with optional grid overrides")
    p.add_argument("--dataset", choices=["pepper", "pepper_10611831", "ipk_out_raw"], default="pepper")
    p.add_argument("--use_synthetic", type=str, default="true", help="Use synthetic data (true/false)")
    p.add_argument("--synthetic_only", type=str, default="false", help="Train on synthetic-only (true/false)")
    p.add_argument("--selected_k", type=int, default=5000)
    p.add_argument("--sigma_resid_factor", type=float, default=0.10)
    p.add_argument("--models", type=str, default="randomforest", help="Comma-separated models to train (default randomforest)")
    p.add_argument("--cross_validation_outer", type=int, default=3)
    p.add_argument("--cross_validation_inner", type=int, default=2)
    p.add_argument("--holdout_size", type=float, default=0.2)
    p.add_argument("--augment_file", type=str, default=os.path.join("04_augmentation", "pepper", "model_sources", "llama3", "synthetic_y_filtered_f5_s42_k5000.csv"))
    p.add_argument("--overwrite_previous", action="store_true")
    # RandomForest grid overrides
    p.add_argument("--n_estimators", nargs="*", type=int)
    p.add_argument("--max_depth", nargs="*", type=str)
    p.add_argument("--min_samples_split", nargs="*", type=int)
    p.add_argument("--min_samples_leaf", nargs="*", type=int)
    p.add_argument("--max_features", nargs="*", type=str)
    return p.parse_args()

def to_bool(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y"}

def normalize_depth(values):
    if values is None:
        return None
    norm = []
    for v in values:
        if v is None:
            norm.append(None)
            continue
        vs = str(v).strip().lower()
        if vs in {"none", "null"}:
            norm.append(None)
        else:
            norm.append(int(v))
    return norm

def normalize_max_features(values):
    if values is None:
        return None
    norm = []
    for v in values:
        if v is None:
            norm.append(None)
            continue
        vs = str(v).strip().lower()
        if vs in {"none", "null"}:
            norm.append(None)
        else:
            norm.append(vs)
    return norm

args = parse_args()

# Configure for RandomForest on pepper with synthetic augmentation
aug.CONFIG["DATASET"] = args.dataset
aug.CONFIG["AUGMENTED_DATASET"] = f"{aug.CONFIG['DATASET']}_augmented"
aug.CONFIG["PROCESSED"] = os.path.join("02_processed_data", aug.CONFIG["DATASET"]) 
aug.CONFIG["AUGMENTED"] = os.path.join("04_augmentation", aug.CONFIG["DATASET"]) 
aug.CONFIG["OUTDIR"] = os.path.join("03_modeling_results", aug.CONFIG["AUGMENTED_DATASET"]) 

# Ensure output subdirs exist
for subdir in ["logs", "models", "metrics", "plots", "plots_and_tables"]:
    os.makedirs(os.path.join(aug.CONFIG["OUTDIR"], subdir), exist_ok=True)

# Reconfigure logging to point to this run's OUTDIR after import-time defaults
try:
    run_log = os.path.join(aug.CONFIG["OUTDIR"], "logs", f"pipeline_{aug.CONFIG['RUN_ID']}.log")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Remove any existing FileHandlers to avoid writing to a wrong dataset path
    new_handlers = []
    for h in root_logger.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                h.close()
            except Exception:
                pass
            continue
        new_handlers.append(h)
    root_logger.handlers = new_handlers
    # Attach the correct file handler for this run
    fh = logging.FileHandler(run_log)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(fh)
    # Ensure a stream handler exists
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(sh)
    logging.info(f"Runner logging redirected to: {run_log}")
except Exception:
    pass

# Core run settings from CLI
aug.CONFIG["USE_SYNTHETIC"] = to_bool(args.use_synthetic)
aug.CONFIG["SYNTHETIC_ONLY"] = to_bool(args.synthetic_only)
aug.CONFIG["TARGET_COLUMN"] = "Yield_BV" if args.dataset.startswith("pepper") else None
aug.CONFIG["AUGMENT_MODE"] = "llama3"
aug.CONFIG["AUGMENT_SEED"] = 42
aug.CONFIG["AUGMENT_FILE"] = args.augment_file
aug.CONFIG["SELECTED_K"] = args.selected_k
aug.CONFIG["SIGMA_RESID_FACTOR"] = args.sigma_resid_factor
aug.CONFIG["OVERWRITE_PREVIOUS"] = bool(args.overwrite_previous)
aug.CONFIG["MODELS"] = [m.strip() for m in args.models.split(",") if m.strip()]
aug.CONFIG["OUTER_CV_FOLDS"] = int(args.cross_validation_outer)
aug.CONFIG["INNER_CV_FOLDS"] = int(args.cross_validation_inner)
aug.CONFIG["HOLDOUT_FRACTION"] = float(args.holdout_size)

# RandomForest grid overrides propagated into augmented pipeline CONFIG
rf_override = {}
if args.n_estimators:
    rf_override["n_estimators"] = [int(n) for n in args.n_estimators]
if args.max_depth:
    rf_override["max_depth"] = normalize_depth(args.max_depth)
if args.min_samples_split:
    rf_override["min_samples_split"] = [int(x) for x in args.min_samples_split]
if args.min_samples_leaf:
    rf_override["min_samples_leaf"] = [int(x) for x in args.min_samples_leaf]
if args.max_features:
    rf_override["max_features"] = normalize_max_features(args.max_features)
if rf_override:
    aug.CONFIG["RF_OVERRIDE"] = rf_override

logging.info(f"Using RUN_ID={aug.CONFIG['RUN_ID']}")

# Execute the augmented pipeline end-to-end
if __name__ == "__main__":
    aug.main()