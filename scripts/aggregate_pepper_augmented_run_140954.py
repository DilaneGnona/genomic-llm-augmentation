import os
import json
import logging
import sys
import importlib.util

RUN_ID = "20251026_140954"
OUTDIR = "03_modeling_results/pepper_augmented"
METRICS_DIR = os.path.join(OUTDIR, "metrics")
SUMMARY_PATH = os.path.join(OUTDIR, "summary.md")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Collect per-model metrics for the specified run_id
results = {}
for fname in os.listdir(METRICS_DIR):
    if fname.endswith(f"_{RUN_ID}.json") and fname.endswith("_metrics_" + RUN_ID + ".json"):
        path = os.path.join(METRICS_DIR, fname)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            model_name = data.get('model_name')
            if not model_name:
                # infer from filename
                model_name = fname.split("_metrics_")[0]
            results[model_name] = data
            logging.info(f"Collected metrics: {fname}")
        except Exception as e:
            logging.warning(f"Failed to read {fname}: {e}")

# Write aggregated metrics
all_metrics_path = os.path.join(METRICS_DIR, "all_models_metrics.json")
with open(all_metrics_path, 'w') as f:
    json.dump(results, f, indent=2)
logging.info(f"Wrote aggregated metrics to {all_metrics_path}")

# Update summary using the augmented pipeline's helper to ensure consistent formatting
spec = importlib.util.spec_from_file_location(
    "aug", os.path.join("scripts", "unified_modeling_pipeline_augmented.py")
)
aug = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aug)
aug.CONFIG["RUN_ID"] = RUN_ID
aug.CONFIG["OUTDIR"] = OUTDIR
aug.update_summary(results)

print(f"Aggregated metrics refreshed at: {all_metrics_path}")
print(f"Updated summary at: {SUMMARY_PATH}")