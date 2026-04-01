import os
import re
import json
import glob
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_ROOT = os.path.join(ROOT, "03_modeling_results", "summary_for_supervisor")
AUGMENTED_ROOT = os.path.join(SUMMARY_ROOT, "augmented")
OUTDIR = os.path.join(SUMMARY_ROOT, "summaries")

os.makedirs(OUTDIR, exist_ok=True)

def detect_k_from_metrics(meta: dict) -> int | None:
    # Prefer explicit selected_k field if present
    for key in ("selected_k", "SELECTED_K", "selected_features_cap"):
        if key in meta and isinstance(meta[key], (int, float)):
            try:
                return int(meta[key])
            except Exception:
                pass
    # Fallback: parse from augment_file path (e.g., ..._k5000.csv)
    augf = meta.get("augment_file") or meta.get("AUGMENT_FILE")
    if isinstance(augf, str):
        m = re.search(r"_k(\d+)", augf)
        if m:
            return int(m.group(1))
    return None

def load_augmented_metrics():
    datasets = {}
    # Iterate summary_for_supervisor/augmented/*_augmented/general/metrics/**/*.json (recursive)
    if not os.path.isdir(AUGMENTED_ROOT):
        return datasets
    for dname in os.listdir(AUGMENTED_ROOT):
        dpath = os.path.join(AUGMENTED_ROOT, dname)
        if not os.path.isdir(dpath) or not dname.endswith("_augmented"):
            continue
        metrics_dir = os.path.join(dpath, "general", "metrics")
        if not os.path.isdir(metrics_dir):
            datasets[dname] = []
            continue
        rows = []
        # include nested JSONs if any
        files = glob.glob(os.path.join(metrics_dir, "*.json"))
        files += glob.glob(os.path.join(metrics_dir, "**", "*.json"), recursive=True)
        debug_lines = []
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Some files are aggregate (all_models_metrics.json); handle both
                is_aggregate = False
                base = os.path.basename(fp)
                if base == "all_models_metrics.json":
                    is_aggregate = True
                elif isinstance(data, dict) and "model_name" not in data:
                    # Heuristic: any top-level value dict contains model_name
                    for v in data.values():
                        if isinstance(v, dict) and "model_name" in v:
                            is_aggregate = True
                            break
                if is_aggregate:
                    # Aggregate file with per-model entries
                    for mk, md in data.items():
                        if not isinstance(md, dict):
                            continue
                        model = md.get("model_name") or mk.replace("_metrics", "")
                        k_val = detect_k_from_metrics(md)
                        r2 = md.get("cv_r2_mean")
                        rmse = md.get("cv_rmse_mean")
                        mae = md.get("cv_mae_mean")
                        if k_val is not None and r2 is not None:
                            rows.append({"model": model, "k": k_val, "cv_r2_mean": r2, "cv_rmse_mean": rmse, "cv_mae_mean": mae})
                        debug_lines.append(f"AGG {os.path.basename(fp)} :: model={model} k={k_val} r2={r2}")
                elif isinstance(data, dict):
                    # Single-model metrics file
                    model = data.get("model_name")
                    if not model:
                        # derive from filename
                        m = re.match(r"(\w+)_metrics_", base)
                        model = m.group(1) if m else "unknown"
                    k_val = detect_k_from_metrics(data)
                    r2 = data.get("cv_r2_mean")
                    rmse = data.get("cv_rmse_mean")
                    mae = data.get("cv_mae_mean")
                    if k_val is not None and r2 is not None:
                        rows.append({"model": model, "k": k_val, "cv_r2_mean": r2, "cv_rmse_mean": rmse, "cv_mae_mean": mae})
                    debug_lines.append(f"SINGLE {os.path.basename(fp)} :: model={model} k={k_val} r2={r2}")
            except Exception:
                # Skip files that aren't parseable
                pass
        # write debug file per dataset
        try:
            dbg = os.path.join(OUTDIR, f"aug_debug_{dname}.txt")
            with open(dbg, "w", encoding="utf-8") as f:
                f.write("\n".join(debug_lines))
        except Exception:
            pass
        datasets[dname] = rows
    return datasets

def save_csv(dataset_name: str, rows: list[dict]):
    if not rows:
        return None
    out_csv = os.path.join(OUTDIR, f"aug_results_by_k_{dataset_name}.csv")
    cols = ["model", "k", "cv_r2_mean", "cv_rmse_mean", "cv_mae_mean"]
    try:
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
        return out_csv
    except Exception:
        return None

def plot_by_k(dataset_name: str, rows: list[dict]):
    if not rows:
        return None
    # Group by k, then by model
    by_k = defaultdict(list)
    models = set()
    for r in rows:
        by_k[r["k"]].append(r)
        models.add(r["model"])
    ks = sorted(by_k.keys())
    models = sorted(models)
    # Build matrix of r2 means: rows=models, cols=ks
    import numpy as np
    r2_matrix = np.zeros((len(models), len(ks)))
    for j, k in enumerate(ks):
        # average per model at this k (in case of multiple runs)
        bucket = by_k[k]
        for i, m in enumerate(models):
            vals = [x["cv_r2_mean"] for x in bucket if x["model"] == m and isinstance(x.get("cv_r2_mean"), (int, float))]
            if vals:
                r2_matrix[i, j] = float(sum(vals) / len(vals))
            else:
                r2_matrix[i, j] = np.nan

    # Plot grouped bars: x-axis k, bars per model
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ks))
    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        ax.bar(x + i * width, r2_matrix[i], width, label=m)
    ax.set_xlabel("selected_k")
    ax.set_ylabel("CV R^2 (mean)")
    ax.set_title(f"Augmented training results by k — {dataset_name}")
    ax.set_xticks(x + (len(models)-1)*width/2)
    ax.set_xticklabels([str(k) for k in ks])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    out_png = os.path.join(OUTDIR, f"aug_results_by_k_{dataset_name}.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png

def main():
    datasets = load_augmented_metrics()
    outputs = []
    for dname, rows in datasets.items():
        csv_path = save_csv(dname, rows)
        png_path = plot_by_k(dname, rows)
        outputs.append({"dataset": dname, "csv": csv_path, "png": png_path, "rows": len(rows)})
    # Write a small index file for convenience
    index_path = os.path.join(OUTDIR, "aug_results_by_k_index.json")
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2)
    except Exception:
        pass

if __name__ == "__main__":
    main()