import os
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE = "c:/Users/OMEN/Desktop/experiment_snp"
PEPPER_AUG_DIR = os.path.join(BASE, "03_modeling_results/pepper_augmented")
PEPPER_AUG_METRICS = os.path.join(PEPPER_AUG_DIR, "metrics/all_models_metrics.json")
PEPPER_AUG_MULTI_SEED_SIGMA025_CSV = os.path.join(PEPPER_AUG_DIR, "plots_and_tables/multi_model_multi_seed_summary_pepper_sigma025.csv")
PEPPER_AUG_MULTI_SEED_SIGMA05_CSV = os.path.join(PEPPER_AUG_DIR, "plots_and_tables/multi_model_multi_seed_summary_pepper_sigma05.csv")

IPK_METRICS = os.path.join(BASE, "03_modeling_results/ipk_out_raw/metrics/all_models_metrics.json")
IPK_MAPPING = os.path.join(BASE, "02_processed_data/ipk_out_raw/mapping_report.json")
OUT_PATH = os.path.join(BASE, "03_modeling_results/final_summary_report.md")
IPK_AUG_DIR = os.path.join(BASE, "03_modeling_results/ipk_out_raw_augmented")
IPK_AUG_MULTI_SEED_CSV = os.path.join(IPK_AUG_DIR, "plots_and_tables/multi_model_multi_seed_summary_ipk_out_raw.csv")
IPK_AUG_SINGLE_JSON = os.path.join(IPK_AUG_DIR, "metrics/all_models_metrics.json")


def load_json(path):
    if not os.path.exists(path):
        logging.warning(f"Missing file: {path}")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def sort_models(metrics_dict):
    items = [(name, m) for name, m in metrics_dict.items() if isinstance(m, dict)]
    return sorted(items, key=lambda x: x[1].get('cv_r2_mean', float('-inf')), reverse=True)


def write_table(f, items):
    f.write("| Model | CV R2 Mean | CV RMSE Mean | CV MAE Mean | Holdout R2 | Holdout RMSE | Holdout MAE |\n")
    f.write("|-------|------------|--------------|-------------|------------|--------------|-------------|\n")
    for name, m in items:
        if 'cv_r2_mean' in m:
            f.write(
                f"| {name} | {m['cv_r2_mean']:.4f} | {m['cv_rmse_mean']:.4f} | {m['cv_mae_mean']:.4f} | "
                f"{m.get('holdout_r2', float('nan')):.4f} | {m.get('holdout_rmse', float('nan')):.4f} | {m.get('holdout_mae', float('nan')):.4f} |\n"
            )
        else:
            f.write(f"| {name} | ERROR | ERROR | ERROR | NA | NA | NA |\n")


def write_hparams(f, items):
    for name, m in items:
        if 'final_best_params' in m:
            f.write(f"### {name}\n")
            for p, v in m['final_best_params'].items():
                f.write(f"- `{p}`: {v}\n")
            if 'features_count' in m:
                f.write(f"- `features_count`: {m['features_count']}\n")
            f.write("\n")


def load_augmented_multi_seed(csv_path):
    """Load aggregated multi-seed augmented metrics from CSV into a dict like the JSON structure."""
    import csv
    results = {}
    if not os.path.exists(csv_path):
        logging.warning(f"Missing multi-seed CSV: {csv_path}")
        return results
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('Model') or row.get('model') or row.get('Model ')
            if not name:
                continue
            try:
                results[name] = {
                    # Map mean columns to the keys used by write_table
                    'cv_r2_mean': float(row.get('CV R² Mean', row.get('cv_r2_mean', 'nan'))),
                    'cv_rmse_mean': float(row.get('CV RMSE Mean', row.get('cv_rmse_mean', 'nan'))),
                    'cv_mae_mean': float(row.get('CV MAE Mean', row.get('cv_mae_mean', 'nan'))),
                    'holdout_r2': float(row.get('Holdout R² Mean', row.get('holdout_r2_mean', 'nan'))),
                    'holdout_rmse': float(row.get('Holdout RMSE Mean', row.get('holdout_rmse_mean', 'nan'))),
                    'holdout_mae': float(row.get('Holdout MAE Mean', row.get('holdout_mae_mean', 'nan'))),
                }
            except Exception:
                # Skip malformed rows
                continue
    return results


def attach_latest_hparams_for_augmented(models_dict, metrics_dir):
    """Attach final_best_params and features_count for augmented models (any dataset) using latest per-model metrics JSONs."""
    if not os.path.isdir(metrics_dir):
        return models_dict
    # For each model, find the latest metrics JSON and attach params
    for model_name in list(models_dict.keys()):
        # Find files like <model>_metrics_*.json
        try:
            candidates = [
                os.path.join(metrics_dir, name)
                for name in os.listdir(metrics_dir)
                if name.startswith(f"{model_name}_metrics_") and name.endswith('.json')
            ]
            # Choose the latest by timestamp suffix
            if not candidates:
                continue
            latest = sorted(candidates)[-1]
            with open(latest, 'r', encoding='utf-8') as f:
                mjson = json.load(f)
            # Attach available fields
            if isinstance(mjson, dict):
                if 'final_best_params' in mjson:
                    models_dict[model_name]['final_best_params'] = mjson['final_best_params']
                if 'features_count' in mjson:
                    models_dict[model_name]['features_count'] = mjson['features_count']
        except Exception:
            continue
    return models_dict


def main():
    logging.info("Generating final summary report across pipelines")
    pepper_metrics = load_json(PEPPER_AUG_METRICS)
    ipk_metrics = load_json(IPK_METRICS)
    mapping = load_json(IPK_MAPPING)
    ipk_aug_multi_seed = load_augmented_multi_seed(IPK_AUG_MULTI_SEED_CSV)
    ipk_aug_multi_seed = attach_latest_hparams_for_augmented(ipk_aug_multi_seed, os.path.join(IPK_AUG_DIR, 'metrics'))
    ipk_aug_single = load_json(IPK_AUG_SINGLE_JSON)

    # Pepper augmented multi-seed summaries for σ=0.25 and σ=0.5
    pepper_aug_multi_seed_s025 = load_augmented_multi_seed(PEPPER_AUG_MULTI_SEED_SIGMA025_CSV)
    pepper_aug_multi_seed_s025 = attach_latest_hparams_for_augmented(pepper_aug_multi_seed_s025, os.path.join(PEPPER_AUG_DIR, 'metrics'))
    pepper_aug_multi_seed_s05 = load_augmented_multi_seed(PEPPER_AUG_MULTI_SEED_SIGMA05_CSV)
    pepper_aug_multi_seed_s05 = attach_latest_hparams_for_augmented(pepper_aug_multi_seed_s05, os.path.join(PEPPER_AUG_DIR, 'metrics'))

    pepper_sorted = sort_models(pepper_metrics)
    ipk_sorted = sort_models(ipk_metrics)
    ipk_aug_sorted = sort_models(ipk_aug_multi_seed)
    ipk_aug_single_sorted = sort_models(ipk_aug_single)

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(f"# Final Summary Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Pepper Augmented (latest single run)
        f.write("## Pepper Augmented (latest single run)\n\n")
        if pepper_sorted:
            write_table(f, pepper_sorted)
            f.write("\n### Best Hyperparameters\n\n")
            write_hparams(f, pepper_sorted[:3])
        else:
            f.write("No metrics available.\n\n")

        # Pepper Augmented (multi-seed, σ=0.25)
        f.write("## Pepper Augmented (multi-seed, σ=0.25, selected_k=1000)\n\n")
        pepper_s025_sorted = sort_models(pepper_aug_multi_seed_s025)
        if pepper_s025_sorted:
            write_table(f, pepper_s025_sorted)
            f.write("\n### Best Hyperparameters (latest runs)\n\n")
            write_hparams(f, pepper_s025_sorted[:3])
        else:
            f.write("No aggregated metrics available for σ=0.25.\n\n")

        # Pepper Augmented (multi-seed, σ=0.5)
        f.write("## Pepper Augmented (multi-seed, σ=0.5, selected_k=1000)\n\n")
        pepper_s05_sorted = sort_models(pepper_aug_multi_seed_s05)
        if pepper_s05_sorted:
            write_table(f, pepper_s05_sorted)
            f.write("\n### Best Hyperparameters (latest runs)\n\n")
            write_hparams(f, pepper_s05_sorted[:3])
        else:
            f.write("No aggregated metrics available for σ=0.5.\n\n")

        # IPK Out Raw
        f.write("## IPK Out Raw\n\n")
        if ipk_sorted:
            write_table(f, ipk_sorted)
            f.write("\n### Best Hyperparameters\n\n")
            write_hparams(f, ipk_sorted[:3])
        else:
            f.write("No metrics available.\n\n")

        # IPK Out Raw Augmented (latest single run)
        f.write("## IPK Out Raw Augmented (latest single run)\n\n")
        if ipk_aug_single_sorted:
            # Try to show the run_id from any model entry
            try:
                any_model = ipk_aug_single_sorted[0][1]
                run_id = any_model.get('run_id')
                if run_id:
                    f.write(f"Run ID: `{run_id}`\n\n")
            except Exception:
                pass
            write_table(f, ipk_aug_single_sorted)
            f.write("\n### Best Hyperparameters (this run)\n\n")
            write_hparams(f, ipk_aug_single_sorted[:3])
        else:
            f.write("No latest single-run augmented metrics available.\n\n")

        # IPK Out Raw Augmented (multi-seed)
        f.write("## IPK Out Raw Augmented (multi-seed)\n\n")
        if ipk_aug_sorted:
            write_table(f, ipk_aug_sorted)
            f.write("\n### Best Hyperparameters (latest runs)\n\n")
            write_hparams(f, ipk_aug_sorted[:3])
        else:
            f.write("No augmented metrics available.\n\n")

        # Mapping coverage
        f.write("## Mapping Coverage (IPK)\n\n")
        if mapping:
            f.write(f"- `total_X_samples`: {mapping.get('total_X_samples', 'NA')}\n")
            f.write(f"- `total_Y_rows`: {mapping.get('total_Y_rows', 'NA')}\n")
            f.write(f"- `mapped_pairs`: {mapping.get('mapped_pairs', 'NA')}\n")
            f.write(f"- `aligned_samples_in_y`: {mapping.get('aligned_samples_in_y', 'NA')}\n")
        else:
            f.write("Mapping report not found.\n")

        # Deepseek vs GLM46 Comparison
        f.write("\n## Deepseek vs GLM46 Comparison\n\n")
        deepseek_vs_glm46_path = os.path.join(BASE, "03_modeling_results/comparative_analysis/deepseek_vs_glm46/deepseek_vs_glm46_summary.md")
        if os.path.exists(deepseek_vs_glm46_path):
            with open(deepseek_vs_glm46_path, 'r', encoding='utf-8') as df:
                # Read the entire content and write it, skipping the duplicate title
                content = df.read()
                # Skip the first line (main title) and keep the rest
                content_lines = content.split('\n')
                if len(content_lines) > 1:
                    # Join from the second line onwards
                    f.write('\n'.join(content_lines[1:]))
        else:
            f.write("No comparison results available.\n")

        # Notes
        f.write("\n## Notes\n\n")
        f.write("- Leakage guard: synthetic samples excluded from validation/test; real-only holdout evaluated.\n")
        f.write("- Augmented sections include the latest single-run metrics and multi-seed aggregates.\n")

    logging.info(f"Wrote final summary: {OUT_PATH}")


if __name__ == "__main__":
    main()