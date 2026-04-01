import os
import sys
import logging
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

MODULE_PATH = os.path.join(ROOT, "scripts", "unified_modeling_pipeline_augmented.py")
spec = importlib.util.spec_from_file_location("augmented_pipeline", MODULE_PATH)
aug = importlib.util.module_from_spec(spec)
_saved = sys.argv[:]
try:
    sys.argv = [MODULE_PATH]
    spec.loader.exec_module(aug)
finally:
    sys.argv = _saved

# Override CONFIG for pepper
aug.CONFIG["DATASET"] = "pepper"
aug.CONFIG["AUGMENTED_DATASET"] = f"{aug.CONFIG['DATASET']}_augmented"
aug.CONFIG["PROCESSED"] = os.path.join("02_processed_data", aug.CONFIG["DATASET"]) 
aug.CONFIG["AUGMENTED"] = os.path.join("04_augmentation", aug.CONFIG["DATASET"]) 
aug.CONFIG["OUTDIR"] = os.path.join("03_modeling_results", aug.CONFIG["AUGMENTED_DATASET"]) 
aug.CONFIG["USE_SYNTHETIC"] = True
aug.CONFIG["SYNTHETIC_ONLY"] = False
aug.CONFIG["TARGET_COLUMN"] = "Yield_BV"
aug.CONFIG["AUGMENT_MODE"] = "llama3"
aug.CONFIG["AUGMENT_SEED"] = 42
aug.CONFIG["AUGMENT_FILE"] = os.path.join("04_augmentation", "pepper", "model_sources", "llama3", "synthetic_y_filtered_f5_s42_k5000.csv")
aug.CONFIG["SELECTED_K"] = 5000
aug.CONFIG["MODELS"] = ["randomforest"]

# Simple console logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("Running diagnostics for augmented pipeline (pepper)...")
    ok = aug.preflight_checks()
    print(f"Preflight checks: {ok}")
    try:
        X, y, sample_ids = aug.load_and_align_data()
        print(f"Loaded X shape: {getattr(X, 'shape', None)}; y length: {len(y) if y is not None else None}; samples: {len(sample_ids) if sample_ids is not None else None}")
    except BaseException as e:
        print(f"Error in load_and_align_data: {e}")
        raise

if __name__ == "__main__":
    main()