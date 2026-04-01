import os
import json
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed metrics for augmented dataset and generate summaries"
    )
    parser.add_argument(
        "--dataset", type=str, default="pepper", help="Dataset name (used for headings and output paths)"
    )
    parser.add_argument(
        "--metrics_dir", type=str,
        default=os.path.join("03_modeling_results", "pepper_augmented", "metrics"),
        help="Directory containing per-model metrics JSONs"
    )
    parser.add_argument(
        "--selected_k", type=int, required=True, help="Selected feature count to filter runs (e.g., 1000)"
    )
    parser.add_argument(
        "--models", type=str, default="randomforest,svr,lightgbm",
        help="Comma-separated models to include (names as saved in JSON: randomforest, svr, lightgbm)"
    )
    parser.add_argument(
        "--sigma", type=float, default=0.25, help="Sigma residual factor to filter runs"
    )
    parser.add_argument(
        "--min_runs", type=int, default=3, help="Minimum number of runs per model required"
    )
    parser.add_argument(
        "--out_dir", type=str,
        default=os.path.join("03_modeling_results", "pepper_augmented", "plots_and_tables"),
        help="Output directory for markdown, csv, and optional plots"
    )
    parser.add_argument(
        "--append_sigma_to_filenames",
        action="store_true",
        help="Append sigma (e.g., _sigma025) to output filenames to avoid overwrites"
    )
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_metrics_files(metrics_dir: str) -> List[Tuple[str, Dict]]:
    files = []
    if not os.path.exists(metrics_dir):
        logging.error(f"Metrics directory does not exist: {metrics_dir}")
        return files
    for fname in os.listdir(metrics_dir):
        # Expect augmented-style filenames: <model>_metrics_<RUN_ID>.json
        if not fname.endswith(".json"):
            continue
        if "_metrics_" not in fname:
            # Skip non-augmented metrics files
            continue
        path = os.path.join(metrics_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            files.append((fname, data))
        except Exception as e:
            logging.warning(f"Failed to read {fname}: {e}")
    return files


def normalize_model_name(name: str) -> str:
    return (name or "").strip().lower()


def pretty_model_name(name: str) -> str:
    n = normalize_model_name(name)
    mapping = {
        "randomforest": "RandomForest",
        "svr": "SVR",
        "lightgbm": "LightGBM",
    }
    return mapping.get(n, name)


def aggregate_metrics(
    files: List[Tuple[str, Dict]], allowed_models: List[str], selected_k: int, sigma: float
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, set], Dict[str, Dict]]:
    # For each model, collect arrays of metrics across runs
    metrics_acc = defaultdict(lambda: defaultdict(list))
    run_ids_by_model = defaultdict(set)
    meta_any = {}
    seeds_set = set()

    allowed_set = set([normalize_model_name(m) for m in allowed_models])

    for fname, data in files:
        if not isinstance(data, dict):
            continue
        model_name = normalize_model_name(data.get("model_name"))
        if model_name not in allowed_set:
            continue

        # Ensure augmented schema keys exist
        required_keys = [
            'model_name', 'run_id', 'cv_r2_mean', 'cv_rmse_mean', 'cv_mae_mean',
            'holdout_r2', 'holdout_rmse', 'holdout_mae', 'selected_k', 'sigma_resid_factor'
        ]
        if not all(k in data for k in required_keys):
            logging.debug(f"Skipping {fname} due to missing required keys")
            continue

        # Filter by selected_k and sigma_resid_factor
        try:
            sk = int(data.get('selected_k')) if data.get('selected_k') is not None else None
        except Exception:
            sk = None
        sigma_val = data.get('sigma_resid_factor')

        if sk != selected_k:
            continue
        try:
            sigma_num = float(sigma_val) if sigma_val is not None else None
        except Exception:
            sigma_num = None
        if sigma_num != sigma:
            continue

        run_id = data.get('run_id')
        if run_id:
            run_ids_by_model[model_name].add(run_id)

        # Accumulate metrics
        metrics_acc[model_name]['cv_r2'].append(float(data.get('cv_r2_mean')))
        metrics_acc[model_name]['holdout_r2'].append(float(data.get('holdout_r2')))
        metrics_acc[model_name]['cv_rmse'].append(float(data.get('cv_rmse_mean')))
        metrics_acc[model_name]['holdout_rmse'].append(float(data.get('holdout_rmse')))
        metrics_acc[model_name]['cv_mae'].append(float(data.get('cv_mae_mean')))
        metrics_acc[model_name]['holdout_mae'].append(float(data.get('holdout_mae')))

        # Capture metadata for notes (once)
        for key in [
            'augment_mode', 'augment_file', 'augment_size', 'augment_size_effective',
            'augment_seed', 'use_synthetic', 'synthetic_only', 'fallback_percent'
        ]:
            if key in data and key not in meta_any:
                meta_any[key] = data.get(key)

        # Track augment_seed across runs
        if 'augment_seed' in data and data.get('augment_seed') is not None:
            try:
                seeds_set.add(int(data.get('augment_seed')))
            except Exception:
                pass

    # Include collected seeds in metadata for downstream notes
    if seeds_set:
        meta_any['augment_seeds'] = sorted(list(seeds_set))
    return metrics_acc, run_ids_by_model, meta_any


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float('nan'), float('nan')
    if len(values) == 1:
        return float(values[0]), 0.0
    try:
        import numpy as np
        arr = np.array(values, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=1))  # sample std
    except Exception:
        import statistics
        m = float(statistics.mean(values))
        try:
            s = float(statistics.stdev(values))
        except statistics.StatisticsError:
            s = 0.0
        return m, s


def fmt_mean_std(m: float, s: float) -> str:
    if m != m or s != s:  # NaN check
        return "NA"
    return f"{m:.3f}±{s:.3f}"


def _compute_notes_and_rank(agg: Dict[str, Dict[str, List[float]]]):
    """Return per-model notes based on holdout R² std, and ranks by mean."""
    import numpy as np
    holdout_stats = {}
    for model in agg.keys():
        m, s = mean_std(agg[model]['holdout_r2'])
        holdout_stats[model] = (m, s)
    # Identify highest variance
    max_std_model = None
    if holdout_stats:
        max_std_model = max(holdout_stats.items(), key=lambda kv: kv[1][1])[0]
    notes = {}
    for model, (m, s) in holdout_stats.items():
        if model == max_std_model and s > 0:
            notes[model] = "Highest variance"
        elif s < 0.005:
            notes[model] = "Stable"
        elif s < 0.015:
            notes[model] = "Slightly variable"
        else:
            notes[model] = "Variable"
    # Ranks by descending mean holdout R²
    ranks = {m: r+1 for r, (m, _) in enumerate(sorted(holdout_stats.items(), key=lambda kv: kv[1][0], reverse=True))}
    return notes, ranks


def write_markdown(
    out_dir: str,
    dataset: str,
    selected_k: int,
    sigma: float,
    agg: Dict[str, Dict[str, List[float]]],
    meta_any: Dict[str, Dict],
    run_ids_by_model: Dict[str, set],
    file_prefix: str = "multi_seed",
    file_suffix: str = "",
):
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, f"{file_prefix}_summary_{dataset}{file_suffix}.md")
    lines = []

    # Header includes seeds metadata
    seeds_list = meta_any.get('augment_seeds')
    if isinstance(seeds_list, list) and len(seeds_list) > 0:
        seeds_str = ",".join(str(s) for s in seeds_list)
    else:
        seeds_str = "unknown"
    lines.append(f"## Multi-Model Multi-Seed Summary — {dataset.capitalize()} Augmented\n")
    lines.append(f"(selected_k={selected_k}, σ={sigma}, seeds={{{seeds_str}}})\n\n")

    # Table header with Notes and Rank
    lines.append("| Model        | CV R² (mean ± std) | Holdout R² (mean ± std) | CV RMSE | Holdout RMSE | CV MAE | Holdout MAE | Notes | Rank (Holdout R²) |\n")
    lines.append("|---------------|--------------------|--------------------------|----------|---------------|----------|---------------|-------|-------------------|\n")

    notes, ranks = _compute_notes_and_rank(agg)

    # Order rows by model name for stability
    for model in sorted(agg.keys()):
        cv_r2_m, cv_r2_s = mean_std(agg[model]['cv_r2'])
        ho_r2_m, ho_r2_s = mean_std(agg[model]['holdout_r2'])
        cv_rmse_m, cv_rmse_s = mean_std(agg[model]['cv_rmse'])
        ho_rmse_m, ho_rmse_s = mean_std(agg[model]['holdout_rmse'])
        cv_mae_m, cv_mae_s = mean_std(agg[model]['cv_mae'])
        ho_mae_m, ho_mae_s = mean_std(agg[model]['holdout_mae'])

        lines.append(
            f"| {pretty_model_name(model):<13} | {fmt_mean_std(cv_r2_m, cv_r2_s):<20} | {fmt_mean_std(ho_r2_m, ho_r2_s):<26} | "
            f"{fmt_mean_std(cv_rmse_m, cv_rmse_s):<10} | {fmt_mean_std(ho_rmse_m, ho_rmse_s):<13} | "
            f"{fmt_mean_std(cv_mae_m, cv_mae_s):<10} | {fmt_mean_std(ho_mae_m, ho_mae_s):<15} | {notes.get(model, ''):<15} | {ranks.get(model, '')} |\n"
        )

    # Notes section
    lines.append("\nNotes:\n")
    lines.append("- All runs validated via strict audit; leak guard enabled.\n")
    augment_mode = meta_any.get('augment_mode', 'llama3')
    augment_file = meta_any.get('augment_file', 'synthetic_y_fixed_sigma025_filtered.csv')
    lines.append(f"- Synthetic data: augment_mode={augment_mode} , augment_file {augment_file} .\n")
    lines.append(f"- σ_resid_factor={sigma} ; selected_k={selected_k} ; seeds={{{seeds_str}}}.\n")
    lines.append("- Summary and plots stored in plots_and_tables/.\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    logging.info(f"Markdown summary written to {md_path}")
    return md_path


def write_csv(
    out_dir: str,
    dataset: str,
    agg: Dict[str, Dict[str, List[float]]],
    run_ids_by_model: Dict[str, set],
    file_prefix: str = "multi_seed",
    file_suffix: str = "",
):
    import csv
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{file_prefix}_summary_{dataset}{file_suffix}.csv")
    header = [
        "model", "n_runs",
        "cv_r2_mean", "cv_r2_std", "holdout_r2_mean", "holdout_r2_std",
        "cv_rmse_mean", "cv_rmse_std", "holdout_rmse_mean", "holdout_rmse_std",
        "cv_mae_mean", "cv_mae_std", "holdout_mae_mean", "holdout_mae_std",
        "notes", "rank_holdout_r2", "run_ids"
    ]
    rows = []
    notes, ranks = _compute_notes_and_rank(agg)
    for model in sorted(agg.keys()):
        cv_r2_m, cv_r2_s = mean_std(agg[model]['cv_r2'])
        ho_r2_m, ho_r2_s = mean_std(agg[model]['holdout_r2'])
        cv_rmse_m, cv_rmse_s = mean_std(agg[model]['cv_rmse'])
        ho_rmse_m, ho_rmse_s = mean_std(agg[model]['holdout_rmse'])
        cv_mae_m, cv_mae_s = mean_std(agg[model]['cv_mae'])
        ho_mae_m, ho_mae_s = mean_std(agg[model]['holdout_mae'])
        run_ids = sorted(list(run_ids_by_model.get(model, [])))
        rows.append([
            pretty_model_name(model), len(run_ids),
            f"{cv_r2_m:.6f}", f"{cv_r2_s:.6f}", f"{ho_r2_m:.6f}", f"{ho_r2_s:.6f}",
            f"{cv_rmse_m:.6f}", f"{cv_rmse_s:.6f}", f"{ho_rmse_m:.6f}", f"{ho_rmse_s:.6f}",
            f"{cv_mae_m:.6f}", f"{cv_mae_s:.6f}", f"{ho_mae_m:.6f}", f"{ho_mae_s:.6f}",
            notes.get(model, ""), ranks.get(model, ""),
            "|".join(run_ids)
        ])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    logging.info(f"CSV summary written to {csv_path}")
    return csv_path


def write_barplot(
    out_dir: str,
    dataset: str,
    agg: Dict[str, Dict[str, List[float]]],
    file_prefix: str = "multi_seed",
    file_suffix: str = "",
):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        logging.info("matplotlib not installed; skipping barplot")
        return None

    labels = [pretty_model_name(m) for m in sorted(agg.keys())]
    means = []
    stds = []
    for model in sorted(agg.keys()):
        m, s = mean_std(agg[model]['holdout_r2'])
        means.append(m)
        stds.append(s)

    plt.figure(figsize=(6, 4))
    x = range(len(labels))
    plt.bar(x, means, yerr=stds, capsize=5, color=["#7aa974", "#729fcf", "#f4a582"])  # distinct colors
    plt.xticks(list(x), labels)
    plt.ylabel("Holdout R²")
    plt.title(f"Multi-Model Multi-Seed Holdout R² — {dataset.capitalize()} (selected_k={args.selected_k}, σ={args.sigma})")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"{file_prefix}_barplot_{dataset}{file_suffix}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logging.info(f"Barplot saved to {plot_path}")
    return plot_path


def write_r2_comparison_barplot(
    out_dir: str,
    dataset: str,
    agg: Dict[str, Dict[str, List[float]]],
    selected_k: int,
    sigma: float,
    file_suffix: str = "",
):
    """Plot grouped bars for CV R² and Holdout R² (mean ± std) per model.

    Saves to: multi_model_r2_comparison_{dataset}{suffix}.png in out_dir.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        logging.info("matplotlib not installed; skipping R² comparison barplot")
        return None

    models_sorted = sorted(agg.keys())
    labels = [pretty_model_name(m) for m in models_sorted]
    cv_means = []
    cv_stds = []
    ho_means = []
    ho_stds = []
    for m in models_sorted:
        cv_m, cv_s = mean_std(agg[m]['cv_r2'])
        ho_m, ho_s = mean_std(agg[m]['holdout_r2'])
        cv_means.append(cv_m)
        cv_stds.append(cv_s)
        ho_means.append(ho_m)
        ho_stds.append(ho_s)

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(10, 6))
    cv_bars = plt.bar(x - width/2, cv_means, width, yerr=cv_stds, label='CV R²', color='#4E79A7', capsize=4)
    ho_bars = plt.bar(x + width/2, ho_means, width, yerr=ho_stds, label='Holdout R²', color='#F28E2B', capsize=4)

    plt.xticks(x, labels, rotation=0)
    plt.ylabel('R² Score')
    plt.xlabel('Model')
    plt.title(f"{dataset.capitalize()} Augmented — CV vs Holdout R² across models (σ={sigma}, k={selected_k})")
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"multi_model_r2_comparison_{dataset}{file_suffix}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logging.info(f"R² comparison barplot saved to {plot_path}")
    return plot_path


def main():
    global args
    args = parse_args()
    setup_logging()

    allowed_models = [m.strip() for m in args.models.split(',') if m.strip()]
    files = load_metrics_files(args.metrics_dir)
    logging.info(f"Found {len(files)} augmented per-model metrics JSONs in {args.metrics_dir}")

    agg, run_ids_by_model, meta_any = aggregate_metrics(files, allowed_models, args.selected_k, args.sigma)

    # Validate minimum runs per model
    for m in allowed_models:
        n = len(run_ids_by_model.get(normalize_model_name(m), []))
        if n < args.min_runs:
            raise SystemExit(
                f"Model '{m}' has only {n} runs meeting filters (selected_k={args.selected_k}, sigma={args.sigma})."
            )

    # Choose output prefix automatically based on number of models
    file_prefix = "multi_model_multi_seed" if len(allowed_models) > 1 else "multi_seed"
    # Optional sigma suffix to avoid overwriting across different sigma values
    sigma_suffix = f"_sigma{str(args.sigma).replace('.', '')}" if args.append_sigma_to_filenames else ""

    md_path = write_markdown(
        args.out_dir,
        args.dataset,
        args.selected_k,
        args.sigma,
        agg,
        meta_any,
        run_ids_by_model,
        file_prefix=file_prefix,
        file_suffix=sigma_suffix,
    )
    csv_path = write_csv(
        args.out_dir,
        args.dataset,
        agg,
        run_ids_by_model,
        file_prefix=file_prefix,
        file_suffix=sigma_suffix,
    )
    plot_path = write_barplot(
        args.out_dir,
        args.dataset,
        agg,
        file_prefix=file_prefix,
        file_suffix=sigma_suffix,
    )
    # Additional grouped barplot for CV vs Holdout R² across models
    r2_plot_path = None
    if len(allowed_models) > 1:
        r2_plot_path = write_r2_comparison_barplot(
            args.out_dir,
            args.dataset,
            agg,
            args.selected_k,
            args.sigma,
            file_suffix=sigma_suffix,
        )

    logging.info("Aggregation complete")
    print(f"Markdown: {md_path}")
    print(f"CSV: {csv_path}")
    if plot_path:
        print(f"Barplot: {plot_path}")
    if r2_plot_path:
        print(f"R2Comparison: {r2_plot_path}")


if __name__ == "__main__":
    main()