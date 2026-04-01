import os
import json
import argparse
from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple

"""
Aggregate augmented RandomForest runs across fallback rates and seeds for a dataset.
Outputs a markdown summary with required columns and ΔHoldout vs specified baseline run.
Optionally plots Holdout R² vs fallback percent per seed.
"""

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def normalize_model_name(name: str) -> str:
    return name.strip().lower().replace(' ', '')


def gather_augmented_runs(outdir: str):
    metrics_dir = os.path.join(outdir, 'metrics')
    runs = defaultdict(dict)  # runs[run_id][model_name] = metrics
    if not os.path.isdir(metrics_dir):
        return runs
    for fname in os.listdir(metrics_dir):
        if not fname.endswith('.json'):
            continue
        # per-model metrics are like: <model>_metrics_<RUN_ID>.json
        parts = fname.split('_metrics_')
        if len(parts) != 2:
            continue
        run_id = parts[1].replace('.json', '')
        data = load_json(os.path.join(metrics_dir, fname))
        if not isinstance(data, dict):
            continue
        model = data.get('model_name') or parts[0]
        runs[run_id][normalize_model_name(model)] = data
    return runs


def gather_baseline_models(original_results_dir: str):
    baseline_path = os.path.join(original_results_dir, 'metrics', 'all_models_metrics.json')
    base = load_json(baseline_path) or {}
    models = {}
    if isinstance(base, dict):
        for k, v in base.items():
            if isinstance(v, dict) and ('cv_r2_mean' in v or 'cv_rmse_mean' in v or 'cv_mae_mean' in v):
                models[normalize_model_name(k)] = v
    return models


def load_baseline_augmented(outdir: str, baseline_run_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not baseline_run_id:
        return None
    path = os.path.join(outdir, 'metrics', f'randomforest_metrics_{baseline_run_id}.json')
    data = load_json(path)
    return data if isinstance(data, dict) else None


def format_row_simple(rate, seed, m, baseline_holdout: Optional[float], highlight: bool = False) -> str:
    cv_r2 = m.get('cv_r2_mean')
    holdout_r2 = m.get('holdout_r2')
    aug_size_eff = m.get('augment_size_effective')
    selected_k = m.get('selected_k')
    delta_holdout_r2 = None
    if isinstance(baseline_holdout, (int, float)) and isinstance(holdout_r2, (int, float)):
        delta_holdout_r2 = holdout_r2 - baseline_holdout
    cells = [
        str(rate) if rate is not None else 'NA',
        str(seed) if seed is not None else 'NA',
        str(selected_k) if selected_k is not None else 'NA',
        str(aug_size_eff) if aug_size_eff is not None else 'NA',
        f"{cv_r2:.4f}" if isinstance(cv_r2, (int, float)) else 'NA',
        f"{holdout_r2:.4f}" if isinstance(holdout_r2, (int, float)) else 'NA',
        f"{delta_holdout_r2:.4f}" if isinstance(delta_holdout_r2, (int, float)) else 'NA'
    ]
    if highlight:
        cells = [f"**{c}**" for c in cells]
    return f"| {' | '.join(cells)} |\n"


def aggregate(dataset: str, root: str, baseline_run_id: Optional[str] = None, compute_deltas: bool = False, plot_holdout: bool = False):
    outdir = os.path.join(root, '03_modeling_results', f'{dataset}_augmented')
    plots_dir = os.path.join(outdir, 'plots_and_tables')
    os.makedirs(plots_dir, exist_ok=True)

    runs = gather_augmented_runs(outdir)
    baseline_aug = load_baseline_augmented(outdir, baseline_run_id)
    baseline_holdout = baseline_aug.get('holdout_r2') if isinstance(baseline_aug, dict) else None

    # Collect RandomForest entries
    entries: List[Tuple[Optional[int], Optional[int], Dict[str, Any]]] = []
    for run_id, model_map in runs.items():
        m = model_map.get('randomforest')
        if not isinstance(m, dict):
            continue
        rate = m.get('fallback_percent')
        seed = m.get('augment_seed')
        entries.append((rate, seed, m))

    # Sort by rate then seed
    entries.sort(key=lambda x: (x[0] if x[0] is not None else 9999, x[1] if x[1] is not None else 9999))

    # Determine best (highest Holdout R²)
    best_idx = None
    best_val = None
    for i, (_, _, m) in enumerate(entries):
        hv = m.get('holdout_r2')
        if isinstance(hv, (int, float)) and (best_val is None or hv > best_val):
            best_val = hv
            best_idx = i

    # Build markdown table per spec
    md = []
    md.append(f"# Sweep Summary for {dataset}\n")
    md.append("\n")
    md.append("| rate | seed | selected_k | augment_size_effective | CV R2 | Holdout R2 | ΔHoldout R2 vs baseline |\n")
    md.append("|------|------|------------|------------------------|-------|------------|--------------------------|\n")

    for i, (rate, seed, m) in enumerate(entries):
        highlight = (best_idx == i)
        md.append(format_row_simple(rate, seed, m, baseline_holdout if compute_deltas else None, highlight=highlight))

    md.append("\n")
    md.append("Notes: synthetic excluded from validation/test; real-only holdout evaluated.\n")
    if compute_deltas and baseline_run_id:
        md.append(f"Baseline run_id: `{baseline_run_id}` (RandomForest).\n")
    if best_idx is not None and isinstance(best_val, (int, float)):
        rate, seed, _ = entries[best_idx]
        md.append(f"\nBest setting: rate={rate}, seed={seed}, Holdout R²={best_val:.4f}.\n")

    out_path = os.path.join(plots_dir, f'sweep_summary_{dataset}.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(''.join(md))
    print(f"Sweep summary written to: {out_path}")

    # Optional plot
    if plot_holdout:
        try:
            import matplotlib.pyplot as plt
            # Group by seed
            data_by_seed: Dict[Optional[int], List[Tuple[Optional[int], float]]] = defaultdict(list)
            for rate, seed, m in entries:
                hv = m.get('holdout_r2')
                if isinstance(hv, (int, float)):
                    data_by_seed[seed].append((rate, hv))
            plt.figure(figsize=(6, 4))
            for seed, pts in data_by_seed.items():
                pts_sorted = sorted(pts, key=lambda x: x[0] if x[0] is not None else 9999)
                xs = [p[0] for p in pts_sorted]
                ys = [p[1] for p in pts_sorted]
                plt.plot(xs, ys, marker='o', label=f'seed {seed}')
            plt.xlabel('fallback_percent')
            plt.ylabel('Holdout R²')
            plt.title(f'Holdout R² vs fallback_percent ({dataset}, RandomForest)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plot_path = os.path.join(outdir, 'plots', 'holdout_vs_fallback.png')
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Plot saved to: {plot_path}")
        except Exception as e:
            print(f"Skipping plot generation due to error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate augmented RandomForest sweep results for a dataset')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., pepper or ipk_out_raw)')
    parser.add_argument('--root', default='.', help='Repository root path')
    parser.add_argument('--baseline_run_id', default=None, help='Augmented baseline run_id for delta computation (RandomForest)')
    parser.add_argument('--compute_deltas', action='store_true', help='Compute ΔHoldout vs baseline_run_id')
    parser.add_argument('--plot_holdout_vs_fallback', action='store_true', help='Generate plot of Holdout R² vs fallback percent per seed')
    args = parser.parse_args()
    aggregate(args.dataset, args.root, baseline_run_id=args.baseline_run_id, compute_deltas=args.compute_deltas, plot_holdout=args.plot_holdout_vs_fallback)