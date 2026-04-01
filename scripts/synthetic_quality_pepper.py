import os
import glob
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


PROC_DIR = os.path.join('02_processed_data', 'pepper')
AUG_DIR = os.path.join('04_augmentation', 'pepper')
DIAG_DIR = os.path.join(AUG_DIR, 'diagnostics')
X_REAL_PATH = os.path.join(PROC_DIR, 'X.csv')
Y_REAL_PATH = os.path.join(PROC_DIR, 'y.csv')
X_SYN_PATH = os.path.join(AUG_DIR, 'synthetic_snps.csv')


def log_message(fp, message):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {message}"
    print(line)
    with open(fp, 'a', encoding='utf-8') as f:
        f.write(line + "\n")


def detect_id_column(df: pd.DataFrame) -> str:
    if 'Sample_ID' in df.columns:
        return 'Sample_ID'
    if 'IID' in df.columns:
        return 'IID'
    return df.columns[0]


def normalize_id_column(df: pd.DataFrame) -> pd.DataFrame:
    id_col = detect_id_column(df)
    if id_col != 'Sample_ID':
        df = df.rename(columns={id_col: 'Sample_ID'})
    df['Sample_ID'] = df['Sample_ID'].astype(str)
    return df


def find_synthetic_y_path() -> str:
    candidates = glob.glob(os.path.join(AUG_DIR, 'synthetic_y_filtered*.csv'))
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    fallback = os.path.join(AUG_DIR, 'synthetic_y.csv')
    return fallback if os.path.exists(fallback) else ''


def remove_pos_ref_row_if_present(df: pd.DataFrame) -> pd.DataFrame:
    # Many pipelines store a POS/REF metadata row at index 0; drop if non-numeric pattern is present
    if df.shape[0] > 0:
        first_row = df.iloc[0]
        # Heuristic: if more than 30% of SNP columns are non-numeric-like in first row, drop it
        non_num = 0
        total = 0
        for c in df.columns[1:]:  # exclude Sample_ID
            total += 1
            val = first_row[c]
            try:
                float(val)
            except Exception:
                non_num += 1
        if total > 0 and (non_num / total) > 0.3:
            return df.iloc[1:].reset_index(drop=True)
    return df


def align_features(real_X: pd.DataFrame, syn_X: pd.DataFrame):
    real_cols = set(real_X.columns) - {'Sample_ID'}
    syn_cols = set(syn_X.columns) - {'Sample_ID'}
    common = sorted(list(real_cols.intersection(syn_cols)))
    real_aligned = real_X[['Sample_ID'] + common].copy()
    syn_aligned = syn_X[['Sample_ID'] + common].copy()
    return real_aligned, syn_aligned, common


def compute_hist_overlap(y_real: np.ndarray, y_syn: np.ndarray, bins: int = 50):
    # Use common range
    y_min = float(np.nanmin([np.nanmin(y_real), np.nanmin(y_syn)]))
    y_max = float(np.nanmax([np.nanmax(y_real), np.nanmax(y_syn)]))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        return 0.0
    hist_r, edges = np.histogram(y_real, bins=bins, range=(y_min, y_max), density=True)
    hist_s, _ = np.histogram(y_syn, bins=bins, range=(y_min, y_max), density=True)
    bin_widths = np.diff(edges)
    overlap_area = np.sum(np.minimum(hist_r, hist_s) * bin_widths)
    # Since densities integrate to 1 across range, overlap_area in [0,1]
    return float(overlap_area)


def pca_metrics(real_mat: np.ndarray, syn_mat: np.ndarray, n_components: int = 10):
    n_components = max(2, min(n_components, real_mat.shape[0], real_mat.shape[1]))
    pca = PCA(n_components=n_components, svd_solver='auto', random_state=42)
    real_proj = pca.fit_transform(real_mat)
    syn_proj = pca.transform(syn_mat)
    real_ev_ratio_sum = float(np.sum(pca.explained_variance_ratio_))

    # Synthetic variance captured by real PCs: variance of projections relative to total syn variance
    syn_proj_var = float(np.sum(np.var(syn_proj, axis=0, ddof=1)))
    syn_total_var = float(np.sum(np.var(syn_mat, axis=0, ddof=1)))
    syn_capture_ratio = float(syn_proj_var / syn_total_var) if syn_total_var > 0 else 0.0
    pca_ratio = float(syn_capture_ratio / real_ev_ratio_sum) if real_ev_ratio_sum > 0 else 0.0

    # Nearest neighbor distances in PC space
    # Compute Euclidean distances from each synthetic to nearest real
    # For efficiency, use brute-force pairwise then min
    dists = []
    cos_sims = []
    for i in range(syn_proj.shape[0]):
        syn_vec = syn_proj[i:i+1, :]
        # Euclidean
        diff = real_proj - syn_vec
        eu = np.sqrt(np.sum(diff * diff, axis=1))
        dists.append(float(np.min(eu)))
        # Cosine similarity
        cs = cosine_similarity(real_proj, syn_vec).flatten()
        cos_sims.append(float(np.max(cs)))

    metrics = {
        'real_ev_ratio_sum': real_ev_ratio_sum,
        'syn_capture_ratio': syn_capture_ratio,
        'pca_capture_ratio_synthetic_over_real': pca_ratio,
        'euclid_nn_mean': float(np.mean(dists)) if dists else math.nan,
        'euclid_nn_p95': float(np.percentile(dists, 95)) if dists else math.nan,
        'cosine_nn_mean': float(np.mean(cos_sims)) if cos_sims else math.nan,
        'cosine_nn_p95': float(np.percentile(cos_sims, 95)) if cos_sims else math.nan,
    }
    return metrics, real_proj, syn_proj


def hamming_distance_summary(real_mat: np.ndarray, syn_mat: np.ndarray, subset_cols: int = 1000):
    # Use integer-coded SNPs (0/1/2) and compute mismatch proportion over subset
    n_features = real_mat.shape[1]
    k = min(subset_cols, n_features)
    rng = np.random.default_rng(42)
    cols = rng.choice(n_features, size=k, replace=False)
    real_sub = real_mat[:, cols]
    syn_sub = syn_mat[:, cols]

    # For each synthetic row, nearest real by Hamming (mismatch count)
    means = []
    p95s = []
    for i in range(syn_sub.shape[0]):
        syn_row = syn_sub[i, :]
        mismatches = np.not_equal(real_sub, syn_row)
        # proportion mismatches per real sample
        prop = np.mean(mismatches, axis=1)
        nn = float(np.min(prop))
        means.append(nn)
    # Summary
    return {
        'hamming_nn_mean': float(np.mean(means)) if means else math.nan,
        'hamming_nn_p95': float(np.percentile(means, 95)) if means else math.nan,
        'subset_features': int(k)
    }


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    ensure_dir(DIAG_DIR)
    log_fp = os.path.join(DIAG_DIR, 'quality_log.txt')
    # Initialize log
    with open(log_fp, 'w', encoding='utf-8') as f:
        f.write('Synthetic Quality Diagnostics Log\n')
        f.write(f'Timestamp: {datetime.now().isoformat()}\n')
        f.write('='*60 + '\n')

    try:
        log_message(log_fp, 'Loading datasets')
        if not os.path.exists(X_REAL_PATH) or not os.path.exists(Y_REAL_PATH):
            log_message(log_fp, 'ERROR: Real dataset files not found')
            return 1
        if not os.path.exists(X_SYN_PATH):
            log_message(log_fp, 'ERROR: Synthetic SNPs file not found')
            return 1
        y_syn_path = find_synthetic_y_path()
        if not y_syn_path:
            log_message(log_fp, 'ERROR: Synthetic y file not found')
            return 1
        log_message(log_fp, f"Using synthetic y file: {y_syn_path}")

        log_message(log_fp, f"Reading synthetic X from {X_SYN_PATH}")
        X_syn = pd.read_csv(X_SYN_PATH)
        X_syn = normalize_id_column(X_syn)

        log_message(log_fp, f"Reading synthetic y from {y_syn_path}")
        y_syn_df = pd.read_csv(y_syn_path)
        y_syn_df = normalize_id_column(y_syn_df)

        # Determine common SNP columns to minimize real X read
        syn_cols = [c for c in X_syn.columns if c != 'Sample_ID']
        log_message(log_fp, f"Synthetic SNP columns detected: {len(syn_cols)}")

        log_message(log_fp, f"Reading real X from {X_REAL_PATH} (header)")
        real_cols = pd.read_csv(X_REAL_PATH, nrows=0).columns.tolist()
        real_cols = [c for c in real_cols if c != 'Sample_ID']
        common_cols_prefetch = sorted(list(set(real_cols).intersection(set(syn_cols))))
        usecols = ['Sample_ID'] + common_cols_prefetch
        log_message(log_fp, f"Reading real X limited to {len(common_cols_prefetch)} intersecting SNPs")
        # Robust CSV parsing: fallback to python engine if C engine fails (e.g., due to malformed lines)
        try:
            X_real = pd.read_csv(X_REAL_PATH, usecols=usecols, low_memory=False)
        except Exception as e:
            # Some pandas versions do not support 'low_memory' with the python engine.
            # Fallback by switching to the python engine and reading in chunks with bad lines skipped.
            log_message(log_fp, f"WARN: Failed reading real X with default engine ({repr(e)}); retrying chunked with engine='python' and bad lines skipped")
            chunks = []
            try:
                for chunk in pd.read_csv(X_REAL_PATH, usecols=usecols, engine='python', on_bad_lines='skip', chunksize=20000):
                    chunks.append(chunk)
            except TypeError:
                # Fallback for older pandas versions without on_bad_lines; use deprecated flags
                try:
                    for chunk in pd.read_csv(X_REAL_PATH, usecols=usecols, engine='python', chunksize=20000, error_bad_lines=False, warn_bad_lines=True):
                        chunks.append(chunk)
                except TypeError:
                    # Ultimate fallback without skipping; may still fail on malformed rows
                    for chunk in pd.read_csv(X_REAL_PATH, usecols=usecols, engine='python', chunksize=20000):
                        chunks.append(chunk)
            X_real = pd.concat(chunks, axis=0, ignore_index=True)
            log_message(log_fp, f"Read real X in {len(chunks)} chunks; combined shape={X_real.shape}")
        log_message(log_fp, f"Read real X fallback result: shape={X_real.shape}, first_cols={list(X_real.columns[:5])}")
        X_real = normalize_id_column(X_real)
        X_real = remove_pos_ref_row_if_present(X_real)

        log_message(log_fp, f"Reading real y from {Y_REAL_PATH}")
        y_real_df = pd.read_csv(Y_REAL_PATH)
        y_real_df = normalize_id_column(y_real_df)

        log_message(log_fp, f'Real X shape: {X_real.shape}; Synthetic X shape: {X_syn.shape}')
        log_message(log_fp, f'Real y shape: {y_real_df.shape}; Synthetic y shape: {y_syn_df.shape} ({os.path.basename(y_syn_path)})')

        # Choose target column in y_real
        tgt_col = None
        for cand in ['Yield_BV', 'target', 'y', 'Yield']:
            if cand in y_real_df.columns:
                tgt_col = cand
                break
        if tgt_col is None:
            tgt_candidates = [c for c in y_real_df.columns if c != 'Sample_ID']
            tgt_col = tgt_candidates[0] if tgt_candidates else None
        if tgt_col is None:
            log_message(log_fp, 'ERROR: Could not determine target column in real y.csv')
            return 1

        # Align real y with real X by Sample_ID
        real_rows = X_real[['Sample_ID']].merge(y_real_df[['Sample_ID', tgt_col]], on='Sample_ID', how='left')
        y_real = pd.to_numeric(real_rows[tgt_col], errors='coerce').astype(float).values

        # Align features (intersection)
        real_aligned, syn_aligned, common_cols = align_features(X_real, X_syn)
        if not common_cols:
            log_message(log_fp, 'ERROR: No intersecting SNP features between real and synthetic')
            return 1
        log_message(log_fp, f'Intersecting SNP features: {len(common_cols)}')

        # Ensure numeric encoding for SNPs
        def to_numeric_df(df):
            out = df.copy()
            for c in common_cols:
                out[c] = pd.to_numeric(out[c], errors='coerce')
            return out
        real_num = to_numeric_df(real_aligned)
        syn_num = to_numeric_df(syn_aligned)

        # Compute per-feature stats
        mean_real = real_num[common_cols].mean(axis=0)
        mean_syn = syn_num[common_cols].mean(axis=0)
        var_real = real_num[common_cols].var(axis=0, ddof=1)
        var_syn = syn_num[common_cols].var(axis=0, ddof=1)
        miss_real = real_num[common_cols].isna().mean(axis=0)
        miss_syn = syn_num[common_cols].isna().mean(axis=0)
        diff_mean = (mean_syn - mean_real).abs().sort_values(ascending=False)
        top10 = list(diff_mean.head(10).index)

        # Prepare matrices for PCA/Hamming (drop NaNs)
        real_mat = real_num[common_cols].values
        syn_mat = syn_num[common_cols].values
        # Impute NaNs with column means from real for consistency
        col_means = np.nanmean(real_mat, axis=0)
        inds_real = np.where(np.isnan(real_mat))
        real_mat[inds_real] = np.take(col_means, inds_real[1])
        inds_syn = np.where(np.isnan(syn_mat))
        syn_mat[inds_syn] = np.take(col_means, inds_syn[1])

        # PCA metrics
        pca_info, real_proj, syn_proj = pca_metrics(real_mat, syn_mat, n_components=10)
        log_message(log_fp, f"PCA capture ratio (synthetic/real): {pca_info['pca_capture_ratio_synthetic_over_real']:.4f}")
        log_message(log_fp, f"Nearest-neighbor Euclidean (PC space) mean/p95: {pca_info['euclid_nn_mean']:.4f}/{pca_info['euclid_nn_p95']:.4f}")

        # Hamming summary
        ham = hamming_distance_summary(real_mat=np.rint(real_mat).astype(int), syn_mat=np.rint(syn_mat).astype(int), subset_cols=1000)
        log_message(log_fp, f"Hamming NN mismatch mean/p95 (over {ham['subset_features']} SNPs): {ham['hamming_nn_mean']:.4f}/{ham['hamming_nn_p95']:.4f}")

        # Target stats and overlap
        y_syn_col_candidates = [c for c in y_syn_df.columns if c != 'Sample_ID']
        if not y_syn_col_candidates:
            log_message(log_fp, 'ERROR: No target column found in synthetic y')
            return 1
        y_syn_col = y_syn_col_candidates[0]
        y_syn = pd.to_numeric(y_syn_df[y_syn_col], errors='coerce').astype(float).values
        y_real_stats = {
            'count': int(np.sum(np.isfinite(y_real))),
            'mean': float(np.nanmean(y_real)),
            'std': float(np.nanstd(y_real, ddof=1)),
            'min': float(np.nanmin(y_real)),
            'max': float(np.nanmax(y_real)),
        }
        y_syn_stats = {
            'count': int(np.sum(np.isfinite(y_syn))),
            'mean': float(np.nanmean(y_syn)),
            'std': float(np.nanstd(y_syn, ddof=1)),
            'min': float(np.nanmin(y_syn)),
            'max': float(np.nanmax(y_syn)),
        }
        overlap_coeff = compute_hist_overlap(y_real, y_syn, bins=50)
        log_message(log_fp, f"Yield_BV histogram overlap coefficient: {overlap_coeff:.4f}")

        # Visualizations
        sns.set(style='whitegrid')

        # PCA scatter
        plt.figure(figsize=(8, 6))
        plt.scatter(real_proj[:, 0], real_proj[:, 1], s=12, alpha=0.6, label='Real', color='tab:blue')
        plt.scatter(syn_proj[:, 0], syn_proj[:, 1], s=12, alpha=0.6, label='Synthetic', color='tab:orange')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA: Real vs Synthetic (fit on real)')
        plt.legend()
        pca_plot_fp = os.path.join(DIAG_DIR, 'pca_real_vs_synthetic.png')
        plt.tight_layout()
        plt.savefig(pca_plot_fp, dpi=150)
        plt.close()
        log_message(log_fp, f"Saved {pca_plot_fp}")

        # Yield distribution
        plt.figure(figsize=(8, 6))
        sns.kdeplot(y_real, label='Real', color='tab:blue', fill=True, alpha=0.3)
        sns.kdeplot(y_syn, label='Synthetic', color='tab:orange', fill=True, alpha=0.3)
        plt.title('Yield_BV Distribution: Real vs Synthetic')
        plt.legend()
        yield_plot_fp = os.path.join(DIAG_DIR, 'yield_distribution.png')
        plt.tight_layout()
        plt.savefig(yield_plot_fp, dpi=150)
        plt.close()
        log_message(log_fp, f"Saved {yield_plot_fp}")

        # SNP boxplots for top 10 mean-drift features
        top_cols = top10
        df_box = pd.DataFrame({
            'Sample_ID': list(real_num['Sample_ID']) + list(syn_num['Sample_ID']),
            'kind': ['real'] * real_num.shape[0] + ['synthetic'] * syn_num.shape[0]
        })
        for c in top_cols:
            df_box[c] = list(real_num[c].values) + list(syn_num[c].values)
        melt = df_box.melt(id_vars=['Sample_ID', 'kind'], value_vars=top_cols, var_name='SNP', value_name='Genotype')
        plt.figure(figsize=(max(10, len(top_cols) * 1.2), 6))
        sns.boxplot(data=melt, x='SNP', y='Genotype', hue='kind')
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 10 SNPs by mean drift: Real vs Synthetic')
        plt.legend()
        boxplot_fp = os.path.join(DIAG_DIR, 'snp_boxplots.png')
        plt.tight_layout()
        plt.savefig(boxplot_fp, dpi=150)
        plt.close()
        log_message(log_fp, f"Saved {boxplot_fp}")

        # Optional correlation heatmap comparison on top 10
        corr_real = pd.DataFrame(real_mat, columns=common_cols)[top_cols].corr()
        corr_syn = pd.DataFrame(syn_mat, columns=common_cols)[top_cols].corr()
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.heatmap(corr_real, vmin=-1, vmax=1, cmap='vlag')
        plt.title('Correlation (Real)')
        plt.subplot(1, 2, 2)
        sns.heatmap(corr_syn, vmin=-1, vmax=1, cmap='vlag')
        plt.title('Correlation (Synthetic)')
        corr_fp = os.path.join(DIAG_DIR, 'corr_comparison.png')
        plt.tight_layout()
        plt.savefig(corr_fp, dpi=150)
        plt.close()
        log_message(log_fp, f"Saved {corr_fp}")

        # Report summary
        report_fp = os.path.join(DIAG_DIR, 'synthetic_quality_report.md')
        with open(report_fp, 'w', encoding='utf-8') as f:
            f.write('# Synthetic Quality Report — Pepper\n')
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write('## Summary Statistics\n')
            f.write(f"- Intersecting SNP features: `{len(common_cols)}`\n")
            f.write(f"- PCA capture ratio (synthetic/real): `{pca_info['pca_capture_ratio_synthetic_over_real']:.4f}`\n")
            f.write(f"- Euclidean NN distance in PC space (mean/p95): `{pca_info['euclid_nn_mean']:.4f}` / `{pca_info['euclid_nn_p95']:.4f}`\n")
            f.write(f"- Cosine NN similarity in PC space (mean/p95): `{pca_info['cosine_nn_mean']:.4f}` / `{pca_info['cosine_nn_p95']:.4f}`\n")
            f.write(f"- Hamming NN mismatch (mean/p95 over {ham['subset_features']} SNPs): `{ham['hamming_nn_mean']:.4f}` / `{ham['hamming_nn_p95']:.4f}`\n")
            f.write(f"- Yield_BV overlap coefficient: `{overlap_coeff:.4f}`\n\n")

            f.write('## Target Statistics\n')
            f.write(f"- Real Yield_BV: `{y_real_stats}`\n")
            f.write(f"- Synthetic Yield_BV: `{y_syn_stats}`\n\n")

            f.write('## Per-Feature Drift (Top 10 by |mean diff|)\n')
            for c in top_cols:
                f.write(f"- `{c}`: mean_real=`{mean_real[c]:.4f}`, mean_syn=`{mean_syn[c]:.4f}`, var_real=`{var_real[c]:.4f}`, var_syn=`{var_syn[c]:.4f}`, miss_rate_diff=`{(miss_syn[c]-miss_real[c]):.4f}`\n")
            f.write('\n')

            f.write('## Visual Diagnostics\n')
            f.write(f"- PCA scatter: `{pca_plot_fp}`\n")
            f.write(f"- Yield_BV density: `{yield_plot_fp}`\n")
            f.write(f"- SNP boxplots: `{boxplot_fp}`\n")
            f.write(f"- Correlation heatmaps: `{corr_fp}`\n\n")

            f.write('## Interpretation\n')
            interp = []
            if overlap_coeff >= 0.7:
                interp.append('Label distribution alignment is good (overlap ≥ 0.7).')
            elif overlap_coeff < 0.5:
                interp.append('Label distribution misalignment detected (overlap < 0.5).')
            else:
                interp.append('Label distribution moderately aligned (0.5 ≤ overlap < 0.7).')

            if pca_info['pca_capture_ratio_synthetic_over_real'] < 0.6:
                interp.append('Synthetic variance lies outside dominant real PCA subspace (low capture ratio).')
            else:
                interp.append('Synthetic variance reasonably captured by real PCA subspace.')

            if ham['hamming_nn_mean'] > 0.3:
                interp.append('Genotype drift is substantial (high nearest-neighbor mismatch).')
            else:
                interp.append('Genotype mismatch to nearest real sample is moderate or low.')

            f.write('- ' + '\n- '.join(interp) + '\n\n')
            f.write('Overall, these diagnostics indicate whether synthetic data is close enough to real to be useful for augmentation. If PCA capture is low and genotype drift is high, consider re-tuning augmentation to better match real genotype structure and target distribution.')

        log_message(log_fp, f"Wrote report {report_fp}")
        log_message(log_fp, 'Diagnostics completed successfully')
        return 0
    except Exception as e:
        log_message(log_fp, f'ERROR: Exception during diagnostics: {repr(e)}')
        return 1


if __name__ == '__main__':
    exit(main())