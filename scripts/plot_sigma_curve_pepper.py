import os
import json
import argparse
import logging
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot LightGBM performance vs sigma for Pepper")
    p.add_argument("--metrics_dir", type=str, default=os.path.join("03_modeling_results", "pepper_augmented", "metrics"))
    p.add_argument("--selected_k", type=int, default=1000)
    p.add_argument("--model", type=str, default="lightgbm")
    p.add_argument("--out_dir", type=str, default=os.path.join("03_modeling_results", "pepper_augmented", "plots_and_tables"))
    return p.parse_args()


def load_metrics(metrics_dir, model, selected_k):
    records = []
    for name in os.listdir(metrics_dir):
        if not name.endswith('.json'):
            continue
        if not name.startswith(f"{model}_metrics_"):
            continue
        path = os.path.join(metrics_dir, name)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            if data.get('selected_k') != selected_k:
                continue
            sigma = data.get('sigma_resid_factor')
            if sigma is None:
                continue
            ho_r2 = data.get('holdout_r2')
            cv_r2 = data.get('cv_r2_mean')
            run_id = data.get('run_id')
            records.append({
                'sigma': float(sigma),
                'holdout_r2': float(ho_r2) if ho_r2 is not None else None,
                'cv_r2': float(cv_r2) if cv_r2 is not None else None,
                'run_id': run_id,
            })
        except Exception:
            continue
    return records


def aggregate_by_sigma(records):
    agg = defaultdict(list)
    for r in records:
        agg[r['sigma']].append(r)
    summary = []
    for sigma, items in sorted(agg.items()):
        ho_vals = [x['holdout_r2'] for x in items if x['holdout_r2'] is not None]
        cv_vals = [x['cv_r2'] for x in items if x['cv_r2'] is not None]
        summary.append({
            'sigma': sigma,
            'n_runs': len(items),
            'holdout_r2_mean': sum(ho_vals) / len(ho_vals) if ho_vals else None,
            'cv_r2_mean': sum(cv_vals) / len(cv_vals) if cv_vals else None,
        })
    return summary


def write_csv(summary, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("sigma,n_runs,holdout_r2_mean,cv_r2_mean\n")
        for row in summary:
            f.write(f"{row['sigma']},{row['n_runs']},{row['holdout_r2_mean']},{row['cv_r2_mean']}\n")


def plot(summary, out_path):
    sigmas = [row['sigma'] for row in summary]
    ho = [row['holdout_r2_mean'] for row in summary]
    cv = [row['cv_r2_mean'] for row in summary]

    plt.figure(figsize=(6, 4))
    plt.plot(sigmas, ho, marker='o', label='Holdout R2')
    plt.plot(sigmas, cv, marker='s', label='CV R2')
    plt.xlabel('sigma_resid_factor')
    plt.ylabel('R2')
    plt.title('Pepper LightGBM Performance vs Sigma (k=1000)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Scanning {args.metrics_dir} for model={args.model}, selected_k={args.selected_k}")
    recs = load_metrics(args.metrics_dir, args.model, args.selected_k)
    if not recs:
        logging.warning("No records found. Ensure metrics exist with sigma_resid_factor and selected_k.")
        return
    summary = aggregate_by_sigma(recs)
    csv_path = os.path.join(args.out_dir, 'sigma_curve_lightgbm_pepper.csv')
    png_path = os.path.join(args.out_dir, 'sigma_curve_lightgbm_pepper.png')
    write_csv(summary, csv_path)
    plot(summary, png_path)
    logging.info(f"Wrote CSV: {csv_path}")
    logging.info(f"Wrote plot: {png_path}")


if __name__ == '__main__':
    main()