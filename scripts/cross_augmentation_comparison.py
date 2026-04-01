import argparse
import json
import os
import sys
import csv
from typing import List, Dict, Any, Tuple


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    """Safely get nested key from dict"""
    current = d
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k)
        if current is None:
            return default
    return current


def normalize_model_name(name: str) -> str:
    """Normalize model name for consistent lookup"""
    if name is None:
        return ""
    name = name.lower()
    if "randomforest" in name or "rf" in name:
        return "randomforest"
    if "svr" in name:
        return "svr"
    if "lightgbm" in name or "lgbm" in name:
        return "lightgbm"
    if "xgboost" in name or "xgb" in name:
        return "xgboost"
    if "elasticnet" in name:
        return "elasticnet"
    if "lasso" in name:
        return "lasso"
    if "ridge" in name:
        return "ridge"
    return name


def infer_model_from_filename(filename: str) -> str:
    """Infer model name from filename"""
    filename = filename.lower()
    if "randomforest" in filename or "rf" in filename:
        return "randomforest"
    if "svr" in filename:
        return "svr"
    if "lightgbm" in filename:
        return "lightgbm"
    if "xgboost" in filename or "xgb" in filename:
        return "xgboost"
    if "elasticnet" in filename:
        return "elasticnet"
    if "lasso" in filename:
        return "lasso"
    if "ridge" in filename:
        return "ridge"
    return ""


def parse_metric_file(filepath: str) -> Dict[str, Any]:
    """Parse a metric JSON file"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {}

    model_name = infer_model_from_filename(os.path.basename(filepath)) or safe_get(data, ["model_name"])
    if not model_name:
        model_name = safe_get(data, ["model", "name"])
    model_name = normalize_model_name(model_name)

    # CV metrics
    cv_r2_mean = data.get("cv_r2_mean")
    cv_r2_std = data.get("cv_r2_std")
    if cv_r2_mean is None:
        cv_r2_mean = safe_get(data, ["cv", "r2_mean"])
    if cv_r2_std is None:
        cv_r2_std = safe_get(data, ["cv", "r2_std"])

    # Handle GLM46 format
    if cv_r2_mean is None and "average_scores" in data:
        # GLM46 uses average_scores instead of cv_r2_mean
        cv_r2_mean = data["average_scores"].get("r2")
        if "outer_scores" in data:
            # Calculate std from outer scores
            r2_scores = [fold["r2"] for fold in data["outer_scores"]]
            if r2_scores:
                mu = sum(r2_scores) / len(r2_scores)
                cv_r2_std = (sum((x - mu)**2 for x in r2_scores) / len(r2_scores))**0.5

    # Holdout metrics
    holdout_r2 = data.get("holdout_r2")
    if holdout_r2 is None:
        holdout_r2 = safe_get(data, ["holdout", "r2"]) or safe_get(data, ["holdout_metrics", "r2"]) or safe_get(data, ["holdout", "r2_score"]) or safe_get(data, ["holdout_metrics", "r2_score"])

    # Common metadata
    selected_k = data.get("selected_k")
    # For GLM46, extract selected_k from best_params if not present
    if selected_k is None and "outer_scores" in data:
        for fold in data["outer_scores"]:
            if "best_params" in fold and "max_features" in fold["best_params"]:
                selected_k = fold["best_params"]["max_features"]
                break

    sigma_resid = data.get("sigma_resid_factor")
    augment_seed = data.get("augment_seed")
    augment_file = data.get("augment_file")
    augment_mode = data.get("augment_mode")
    run_id = data.get("run_id") or data.get("timestamp")

    return {
        "model": model_name,
        "cv_r2_mean": cv_r2_mean,
        "cv_r2_std": cv_r2_std,
        "holdout_r2": holdout_r2,
        "selected_k": selected_k,
        "sigma_resid_factor": sigma_resid,
        "augment_seed": augment_seed,
        "augment_file": augment_file,
        "augment_mode": augment_mode,
        "run_id": run_id,
        "_path": filepath,
    }


def collect_augmented_metrics(dataset_dir: str, models: List[str], selected_k: int, sigma: float, allowed_seeds: List[int]=None, augment_mode: str=None) -> Dict[str, List[Dict[str, Any]]]:
    """Collect metrics from augmented datasets"""
    metrics_dir = os.path.join(dataset_dir, "metrics")
    out: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models}
    if not os.path.isdir(metrics_dir):
        return out

    for fname in os.listdir(metrics_dir):
        if not fname.endswith(".json"):
            continue
        # explicitly check for rf_metrics.json (GLM46) or model-specific metrics files
        path = os.path.join(metrics_dir, fname)
        print(f"Processing file: {path}")
        rec = parse_metric_file(path)
        print(f"Parsed record: {json.dumps(rec, indent=2)}")
        m = rec.get("model")
        if m not in models:
            continue
        # Filter by augmentation mode if specified
        if augment_mode is not None:
            # For GLM46, we use directory name as proxy
            if augment_mode == "glm46":
                # GLM46 files are in their own directory
                pass  # already filtered by directory
            else:
                # Check augment_mode field in JSON
                json_aug_mode = rec.get("augment_mode")
                if json_aug_mode != augment_mode:
                    continue
        # filter by selected_k and sigma
        if rec.get("selected_k") != selected_k:
            continue
        # Allow records without sigma_resid_factor (like some LLaMA3 files)
        sigma_ok = (rec.get("sigma_resid_factor") is None or rec.get("sigma_resid_factor") == sigma)
        if not sigma_ok:
            continue
        # filter by seeds if provided
        seed = rec.get("augment_seed")
        if allowed_seeds is not None and seed not in allowed_seeds:
            continue
        out[m].append(rec)
    return out


def compute_aggregates(records: List[Dict[str, Any]]) -> Tuple[float, float, float, float, List[str], List[int]]:
    """Return (cv_mean, cv_std_mean, holdout_mean, holdout_std, run_ids, seeds)"""
    cv_means = [r["cv_r2_mean"] for r in records if r.get("cv_r2_mean") is not None]
    cv_stds = [r["cv_r2_std"] for r in records if r.get("cv_r2_std") is not None]
    holdouts = [r["holdout_r2"] for r in records if r.get("holdout_r2") is not None]
    run_ids = [str(r.get("run_id")) for r in records if r.get("run_id")]
    seeds = [int(r.get("augment_seed")) for r in records if r.get("augment_seed") is not None]

    def mean(xs):
        return sum(xs)/len(xs) if xs else None
    def std(xs):
        if not xs:
            return None
        mu = mean(xs)
        return (sum((x-mu)**2 for x in xs)/len(xs))**0.5

    cv_mean = mean(cv_means)
    cv_std_mean = mean(cv_stds) if cv_stds else std(cv_means)
    holdout_mean = mean(holdouts)
    holdout_std = std(holdouts)
    return cv_mean, cv_std_mean, holdout_mean, holdout_std, run_ids, seeds


def ensure_out_dir(path: str):
    """Ensure output directory exists"""
    if not os.path.isdir(path):
        os.makedirs(path)


def write_csv(path: str, rows: List[Dict[str, Any]]):
    """Write rows to CSV file"""
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Deepseek vs GLM46: Cross-Augmentation Comparison")
    parser.add_argument("--models", required=True, help="Comma-separated models: randomforest,svr,lightgbm,xgboost")
    parser.add_argument("--sigma", type=float, required=True, help="sigma_resid_factor filter")
    parser.add_argument("--selected_k", type=int, required=True, help="selected_k filter")
    parser.add_argument("--out_dir", required=True, help="Output directory for summary and plots")
    args = parser.parse_args()

    models = [normalize_model_name(m.strip()) for m in args.models.split(",") if m.strip()]
    sigma = args.sigma
    selected_k = args.selected_k
    out_dir = args.out_dir

    # Resolve augmentation directories
    augmentation_map = {
        "deepseek": os.path.join("03_modeling_results", "pepper_augmented"),
        "glm46": os.path.join("03_modeling_results", "pepper_augmented", "glm46"),
        "llama3": os.path.join("03_modeling_results", "pepper_augmented"),
    }
    augmentations = list(augmentation_map.keys())

    ensure_out_dir(out_dir)

    # Collect augmented metrics
    augmentation_records = {}
    for aug_name, aug_dir in augmentation_map.items():
        augmentation_records[aug_name] = collect_augmented_metrics(aug_dir, models, selected_k, sigma, allowed_seeds=None, augment_mode=aug_name)

    # Aggregate per-model and per-augmentation
    rows = []
    per_model_stats = {}
    for m in models:
        # Get records for each augmentation
        recs_per_aug = {}
        for aug_name in augmentations:
            recs_per_aug[aug_name] = augmentation_records[aug_name].get(m, [])

        # Compute aggregates for each augmentation
        agg_per_aug = {}
        for aug_name, recs in recs_per_aug.items():
            cv_mean, cv_std, hold_mean, hold_std, run_ids, seeds = compute_aggregates(recs)
            agg_per_aug[aug_name] = {
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "hold_mean": hold_mean,
                "hold_std": hold_std,
                "run_ids": run_ids,
                "seeds": seeds,
                "n_runs": len(recs),
            }

        # Build rows for CSV
        for aug_name, agg in agg_per_aug.items():
            rows.append({
                "augmentation": aug_name,
                "model": m,
                "cv_r2_mean": agg["cv_mean"],
                "cv_r2_std": agg["cv_std"],
                "holdout_r2_mean": agg["hold_mean"],
                "holdout_r2_std": agg["hold_std"],
                "n_runs": agg["n_runs"],
                "run_ids": ";".join(agg["run_ids"]),
                "augment_seeds": ";".join(map(str, agg["seeds"])),
            })

        # Build per-model stats for comparison
        deepseek_agg = agg_per_aug["deepseek"]
        glm46_agg = agg_per_aug["glm46"]
        llama3_agg = agg_per_aug["llama3"]
        per_model_stats[m] = {
            "deepseek_holdout_mean": deepseek_agg["hold_mean"] or 0.0,
            "glm46_holdout_mean": glm46_agg["hold_mean"] or 0.0,
            "llama3_holdout_mean": llama3_agg["hold_mean"] or 0.0,
            "deepseek_cv_std": deepseek_agg["cv_std"] or 0.0,
            "glm46_cv_std": glm46_agg["cv_std"] or 0.0,
            "llama3_cv_std": llama3_agg["cv_std"] or 0.0,
            "delta_r2_deepseek_glm46": (deepseek_agg["hold_mean"] or 0.0) - (glm46_agg["hold_mean"] or 0.0),
            "delta_r2_deepseek_llama3": (deepseek_agg["hold_mean"] or 0.0) - (llama3_agg["hold_mean"] or 0.0),
            "delta_r2_glm46_llama3": (glm46_agg["hold_mean"] or 0.0) - (llama3_agg["hold_mean"] or 0.0),
            "deepseek_n": deepseek_agg["n_runs"],
            "glm46_n": glm46_agg["n_runs"],
            "llama3_n": llama3_agg["n_runs"],
        }

    # Write CSV
    csv_path = os.path.join(out_dir, "deepseek_vs_glm46_summary.csv")
    write_csv(csv_path, rows)

    # Markdown summary
    md_path = os.path.join(out_dir, "deepseek_vs_glm46_summary.md")
    lines = []
    lines.append("## Deepseek vs GLM46: Cross-Augmentation Comparison")
    lines.append("")
    lines.append(f"- Filters: selected_k = {selected_k} | sigma_resid_factor = {sigma}")
    lines.append(f"- Models: {', '.join(m.upper() for m in models)}")
    lines.append(f"- Outputs: CSV saved to {out_dir}")
    lines.append("")
    lines.append("### Metrics Table")
    lines.append("")
    # Build a compact table
    lines.append("Model | Deepseek Holdout R² | GLM46 Holdout R² | LLaMA3 Holdout R² | ΔR² (Deepseek - GLM46) | ΔR² (Deepseek - LLaMA3) | ΔR² (GLM46 - LLaMA3) | Deepseek CV std | GLM46 CV std | LLaMA3 CV std")
    lines.append("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
    for m in models:
        s = per_model_stats[m]
        lines.append("{} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}".format(
            m.upper(),
            s["deepseek_holdout_mean"],
            s["glm46_holdout_mean"],
            s["llama3_holdout_mean"],
            s["delta_r2_deepseek_glm46"],
            s["delta_r2_deepseek_llama3"],
            s["delta_r2_glm46_llama3"],
            s["deepseek_cv_std"],
            s["glm46_cv_std"],
            s["llama3_cv_std"],
        ))
    lines.append("")
    lines.append("### Notes")
    lines.append("- Each model appears once per augmentation method; statistics aggregate across available runs.")
    lines.append("- ΔR² positive indicates Deepseek performed better than GLM46.")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Emit success info for CLI
    print(json.dumps({
        "csv": csv_path,
        "markdown": md_path,
        "per_model_stats": per_model_stats,
    }, indent=2))


if __name__ == "__main__":
    main()