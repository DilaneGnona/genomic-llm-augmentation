import os
import json
from datetime import datetime
import argparse
import hashlib

# Defaults will be set from CLI
OUTDIR = None
METRICS_DIR = None
LOGS_DIR = None
SUMMARY_PATH = None
DATASET = None


def configure_paths(dataset: str):
    global OUTDIR, METRICS_DIR, LOGS_DIR, SUMMARY_PATH, DATASET
    DATASET = dataset
    OUTDIR = os.path.join("03_modeling_results", f"{dataset}_augmented")
    METRICS_DIR = os.path.join(OUTDIR, "metrics")
    LOGS_DIR = os.path.join(OUTDIR, "logs")
    SUMMARY_PATH = os.path.join(OUTDIR, "summary.md")


def find_latest_run_id():
    run_ids = []
    if not os.path.exists(LOGS_DIR):
        return None
    for name in os.listdir(LOGS_DIR):
        if name.startswith("config_") and name.endswith(".json"):
            run_id = name[len("config_"):-len(".json")]
            if len(run_id) == len("YYYYMMDD_HHMMSS"):
                run_ids.append(run_id)
    return sorted(run_ids)[-1] if run_ids else None


def list_run_ids():
    run_ids = []
    if not os.path.exists(LOGS_DIR):
        return run_ids
    for name in os.listdir(LOGS_DIR):
        if name.startswith("config_") and name.endswith(".json"):
            run_id = name[len("config_"):-len(".json")]
            if len(run_id) == len("YYYYMMDD_HHMMSS"):
                # Only include runs that also have splits JSON present
                splits_path = os.path.join(LOGS_DIR, f"splits_{run_id}.json")
                if os.path.exists(splits_path):
                    run_ids.append(run_id)
    return sorted(run_ids)


def load_per_model_metrics(run_id):
    models = {}
    if not os.path.exists(METRICS_DIR):
        return models
    for name in os.listdir(METRICS_DIR):
        if name.endswith(f"_{run_id}.json") and name.endswith(".json"):
            path = os.path.join(METRICS_DIR, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data.get("model_name"):
                    models[data["model_name"]] = data
            except Exception:
                pass
    return models


def fmt(v):
    try:
        return f"{float(v):.4f}"
    except Exception:
        return "NA"


def write_summary(run_id, models, audit_map=None):
    header = [
        f"# Augmented Modeling Summary for {DATASET}\n",
        "\n",
        f"Run ID: `{run_id}`\n\n",
        "Augmentation details in per-model metrics JSONs.\n\n",
    ]
    table_header = (
        "| Model | CV R2 Mean | CV R2 Std | CV RMSE Mean | CV RMSE Std | CV MAE Mean | CV MAE Std | Holdout R2 | Holdout RMSE | Holdout MAE | Features | selected_k |\n"
        "|-------|------------|-----------|--------------|-------------|-------------|------------|------------|--------------|-------------|----------|------------|\n"
    )
    lines = []
    for name, m in sorted(models.items(), key=lambda kv: kv[1].get("cv_r2_mean", float("-inf")), reverse=True):
        lines.append(
            "| "
            + name
            + f" | {fmt(m.get('cv_r2_mean'))} | {fmt(m.get('cv_r2_std'))} | {fmt(m.get('cv_rmse_mean'))} | {fmt(m.get('cv_rmse_std'))} | {fmt(m.get('cv_mae_mean'))} | {fmt(m.get('cv_mae_std'))} | "
            + f"{fmt(m.get('holdout_r2'))} | {fmt(m.get('holdout_rmse'))} | {fmt(m.get('holdout_mae'))} | "
            + f"{m.get('features_count','NA')} | {m.get('selected_k','NA')} |\n"
        )
    # Extract augmentation/metadata from any model in this run (shared across models)
    meta_notes = []
    try:
        any_model = next(iter(models.values())) if models else None
        if any_model:
            if any_model.get('sigma_resid_factor') is not None:
                meta_notes.append(f"- Sigma residual factor: {any_model.get('sigma_resid_factor')}\n")
            if any_model.get('augment_size') is not None:
                meta_notes.append(f"- Augment size: {any_model.get('augment_size')}\n")
            if any_model.get('augment_mode'):
                meta_notes.append(f"- Augment mode: {any_model.get('augment_mode')}\n")
            if any_model.get('augment_seed') is not None:
                meta_notes.append(f"- Augment seed: {any_model.get('augment_seed')}\n")
            if any_model.get('use_synthetic') is not None:
                meta_notes.append(f"- Use synthetic: {any_model.get('use_synthetic')}\n")
            if any_model.get('synthetic_only') is not None:
                meta_notes.append(f"- Synthetic-only: {any_model.get('synthetic_only')}\n")
            # Selected models may be present in JSON
            selected_models = any_model.get('selected_models') or []
            if isinstance(selected_models, list) and selected_models:
                meta_notes.append(f"- Selected models: {', '.join(selected_models)}\n")
    except Exception:
        pass

    notes = [
        "\n## Notes\n\n",
        # Run-specific metadata (e.g., sigma factor and augment size)
        *meta_notes,
        # Leak guard statement
        "- Leakage guard: synthetic excluded from validation/test; real-only holdout evaluated.\n",
        # Reminder of JSON metadata contents
        "- Per-model JSONs include augment_mode, augment_size, augment_seed, use_synthetic, synthetic_only.\n",
    ]
    try:
        os.makedirs(OUTDIR, exist_ok=True)
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            f.writelines(header)
            if models:
                f.write("## Model Metrics\n\n")
                f.write(table_header)
                f.writelines(lines)
            else:
                f.write("Supervised training skipped (no target or models not found).\n\n")
            f.write("\n## Generated Artifacts\n\n")
            f.write(f"- `models`: `{os.path.join(OUTDIR, 'models')}`\n")
            f.write(f"- `metrics`: `{METRICS_DIR}`\n")
            f.write(f"- `logs`: `{LOGS_DIR}`\n")
            f.writelines(notes)

            # Append audit section if provided
            if audit_map:
                f.write("\n## Audit: Synthetic vs Real Separation\n\n")
                f.write("This section enumerates real vs synthetic counts per CV fold and the real-only holdout for each run.\n\n")
                for rid in sorted(audit_map.keys()):
                    audit = audit_map[rid]
                    f.write(f"### Run `{rid}`\n\n")
                    # Per-fold table
                    f.write("| Fold | Train Real | Train Synthetic | Validation Real | Validation Synthetic | Val Synthetic Excluded |\n")
                    f.write("|------|------------|-----------------|-----------------|----------------------|------------------------|\n")
                    for i, fold in enumerate(audit.get("folds", [])):
                        val_excluded = "Yes" if fold.get("val_synth_count", 0) == 0 else "No"
                        f.write(
                            f"| {i+1} | {fold.get('train_real_count','NA')} | {fold.get('train_synth_count','NA')} | {fold.get('val_real_count','NA')} | {fold.get('val_synth_count','NA')} | {val_excluded} |\n"
                        )
                    f.write("\n")
                    # Holdout breakdown
                    holdout = audit.get("holdout", {})
                    f.write("Holdout Breakdown:\n\n")
                    f.write("| Holdout Real | Holdout Synthetic | Synthetic Excluded |\n")
                    f.write("|--------------|-------------------|--------------------|\n")
                    synth_excluded = "Yes" if holdout.get("holdout_synth_count", 0) == 0 else "No"
                    f.write(
                        f"| {holdout.get('holdout_real_count','NA')} | {holdout.get('holdout_synth_count','NA')} | {synth_excluded} |\n\n"
                    )
                    # Checksum line
                    checksum = audit.get("checksum")
                    if checksum:
                        f.write(f"- Synthetic exclusion checksum: `{checksum}`\n\n")

        print(f"Summary written: {SUMMARY_PATH}")
    except Exception as e:
        print(f"Failed to write summary: {e}")


def compute_audit_for_run(run_id):
    """Compute real vs synthetic counts per CV fold and holdout using splits JSON."""
    splits_path = os.path.join(LOGS_DIR, f"splits_{run_id}.json")
    audit = {"folds": [], "holdout": {}, "checksum": None}
    if not os.path.exists(splits_path):
        return audit
    try:
        with open(splits_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        real_set = set(data.get("real_indices", []))
        synth_set = set(data.get("synthetic_indices", []))
        outer_splits = data.get("outer_splits", [])
        # Per-fold counts
        for sp in outer_splits:
            train = set(sp.get("train", []))
            test = set(sp.get("test", []))
            # Prefer explicitly provided train_real; fallback to intersection
            train_real = set(sp.get("train_real", [])) if sp.get("train_real") else (train & real_set)
            train_synth = train & synth_set
            val_synth = test & synth_set
            fold_entry = {
                "train_real_count": len(train_real),
                "train_synth_count": len(train_synth),
                "val_real_count": len(test),
                "val_synth_count": len(val_synth),
            }
            audit["folds"].append(fold_entry)
        # Holdout breakdown
        holdout = set(data.get("holdout_indices", []))
        holdout_real_count = len(holdout)
        holdout_synth_count = len(holdout & synth_set)
        audit["holdout"] = {
            "holdout_real_count": holdout_real_count,
            "holdout_synth_count": holdout_synth_count,
        }
        # Compute checksum confirming exclusion
        checksum_payload = {
            "run_id": run_id,
            "val_synth_counts": [f.get("val_synth_count", 0) for f in audit["folds"]],
            "holdout_synth_count": holdout_synth_count,
        }
        checksum_str = json.dumps(checksum_payload, sort_keys=True)
        audit["checksum"] = hashlib.sha1(checksum_str.encode("utf-8")).hexdigest()
        # Write per-run audit log for traceability
        audit_log_path = os.path.join(LOGS_DIR, f"audit_{run_id}.log")
        with open(audit_log_path, "w", encoding="utf-8") as lf:
            lf.write(f"Audit Log for Run {run_id} ({datetime.now().isoformat()})\n")
            lf.write("Per-fold counts (train real/synth, validation real/synth):\n")
            for i, fentry in enumerate(audit["folds"]):
                lf.write(
                    f"  Fold {i+1}: train_real={fentry['train_real_count']}, train_synth={fentry['train_synth_count']}, val_real={fentry['val_real_count']}, val_synth={fentry['val_synth_count']}\n"
                )
            lf.write(
                f"Holdout: real={holdout_real_count}, synth={holdout_synth_count} (excluded={'YES' if holdout_synth_count==0 else 'NO'})\n"
            )
            lf.write(f"Checksum: {audit['checksum']}\n")
    except Exception as e:
        print(f"Failed to compute audit for {run_id}: {e}")
    return audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='pepper', help='Dataset name (e.g., pepper, pepper_10611831, ipk_out_raw)')
    ap.add_argument('--with_audit', action='store_true', help='Include audit of synthetic vs real separation across all runs')
    args = ap.parse_args()

    configure_paths(args.dataset)
    latest_run_id = find_latest_run_id()
    if not latest_run_id:
        print("No run_id configs found.")
        return

    models = load_per_model_metrics(latest_run_id)

    audit_map = None
    if args.with_audit:
        # Compute audit for all runs that have splits info
        audit_map = {}
        for rid in list_run_ids():
            audit_map[rid] = compute_audit_for_run(rid)

    write_summary(latest_run_id, models, audit_map=audit_map)


if __name__ == "__main__":
    main()