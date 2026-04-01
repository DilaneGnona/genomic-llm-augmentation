import os
import json
import argparse
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

def load_json(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def gather_randomforest_metrics(dataset: str, root: str) -> List[Dict]:
    metrics_dir = os.path.join(root, '03_modeling_results', f'{dataset}_augmented', 'metrics')
    items = []
    if not os.path.isdir(metrics_dir):
        return items
    for fname in os.listdir(metrics_dir):
        if not (fname.startswith('randomforest_metrics_') and fname.endswith('.json')):
            continue
        data = load_json(os.path.join(metrics_dir, fname))
        if isinstance(data, dict):
            items.append(data)
    return items

def compute_series(metrics: List[Dict], only_selected_k: Optional[List[int]] = None):
    # Map: selected_k -> { fallback_percent -> [holdout_r2 values across seeds] }
    series: Dict[Optional[int], Dict[Optional[int], List[float]]] = defaultdict(lambda: defaultdict(list))
    for m in metrics:
        sel_k = m.get('selected_k')
        rate = m.get('fallback_percent')
        hv = m.get('holdout_r2')
        if only_selected_k is not None and sel_k not in only_selected_k:
            continue
        if isinstance(hv, (int, float)):
            series[sel_k][rate].append(hv)
    # Reduce to mean per (k, rate)
    reduced: Dict[Optional[int], Dict[Optional[int], float]] = {}
    for k, rate_map in series.items():
        reduced[k] = {}
        for rate, vals in rate_map.items():
            if vals:
                reduced[k][rate] = sum(vals) / len(vals)
    return reduced

def plot_series(dataset: str, reduced: Dict[Optional[int], Dict[Optional[int], float]], output_file: str):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.figure(figsize=(7, 4))
    # Sort k values for consistent legend
    for k in sorted([kv for kv in reduced.keys() if kv is not None]):
        pts = reduced.get(k, {})
        pts_sorted = sorted([(rate, val) for rate, val in pts.items() if rate is not None], key=lambda x: x[0])
        if not pts_sorted:
            continue
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        plt.plot(xs, ys, marker='o', label=f'selected_k={k}')
    plt.xlabel('fallback_percent')
    plt.ylabel('Holdout R²')
    plt.title(f'Holdout R² vs fallback_percent ({dataset}, RandomForest)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Plot saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate plot of Holdout R² vs fallback percent for RandomForest, grouped by selected_k')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., pepper)')
    parser.add_argument('--root', default='.', help='Repository root path')
    parser.add_argument('--output_file', required=True, help='Output PNG path')
    parser.add_argument('--selected_k', nargs='*', type=int, help='Optional list of selected_k values to include')
    args = parser.parse_args()

    metrics = gather_randomforest_metrics(args.dataset, args.root)
    reduced = compute_series(metrics, only_selected_k=args.selected_k)
    if not reduced:
        print('No RandomForest metrics found to plot.')
        return
    plot_series(args.dataset, reduced, args.output_file)

if __name__ == '__main__':
    main()