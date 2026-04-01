import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

PROC_DIR = os.path.join('02_processed_data', 'pepper')
AUG_DIR = os.path.join('04_augmentation', 'pepper')
X_REAL_PATH = os.path.join(PROC_DIR, 'X.csv')
X_SYN_PATH = os.path.join(AUG_DIR, 'synthetic_snps.csv')


def normalize_id_column(df: pd.DataFrame) -> pd.DataFrame:
    id_col = 'Sample_ID' if 'Sample_ID' in df.columns else ( 'IID' if 'IID' in df.columns else df.columns[0] )
    if id_col != 'Sample_ID':
        df = df.rename(columns={id_col: 'Sample_ID'})
    df['Sample_ID'] = df['Sample_ID'].astype(str)
    return df


def align_features(real_X: pd.DataFrame, syn_X: pd.DataFrame):
    real_cols = set(real_X.columns) - {'Sample_ID'}
    syn_cols = set(syn_X.columns) - {'Sample_ID'}
    common = sorted(list(real_cols.intersection(syn_cols)))
    real_aligned = real_X[['Sample_ID'] + common].copy()
    syn_aligned = syn_X[['Sample_ID'] + common].copy()
    return real_aligned, syn_aligned, common


def pca_metrics(real_mat: np.ndarray, syn_mat: np.ndarray, n_components: int = 10):
    n_components = max(2, min(n_components, real_mat.shape[0], real_mat.shape[1]))
    pca = PCA(n_components=n_components, svd_solver='auto', random_state=42)
    real_proj = pca.fit_transform(real_mat)
    syn_proj = pca.transform(syn_mat)
    real_ev_ratio_sum = float(np.sum(pca.explained_variance_ratio_))
    syn_proj_var = float(np.sum(np.var(syn_proj, axis=0, ddof=1)))
    syn_total_var = float(np.sum(np.var(syn_mat, axis=0, ddof=1)))
    syn_capture_ratio = float(syn_proj_var / syn_total_var) if syn_total_var > 0 else 0.0
    pca_ratio = float(syn_capture_ratio / real_ev_ratio_sum) if real_ev_ratio_sum > 0 else 0.0
    return {
        'real_ev_ratio_sum': real_ev_ratio_sum,
        'syn_capture_ratio': syn_capture_ratio,
        'pca_capture_ratio_synthetic_over_real': pca_ratio,
    }, real_proj, syn_proj


def hamming_distance_summary(real_mat: np.ndarray, syn_mat: np.ndarray, subset_cols: int = 1000):
    n_features = real_mat.shape[1]
    k = min(subset_cols, n_features)
    rng = np.random.default_rng(42)
    cols = rng.choice(n_features, size=k, replace=False)
    real_sub = real_mat[:, cols]
    syn_sub = syn_mat[:, cols]
    means = []
    for i in range(syn_sub.shape[0]):
        syn_row = syn_sub[i, :]
        mismatches = np.not_equal(real_sub, syn_row)
        prop = np.mean(mismatches, axis=1)
        means.append(float(np.min(prop)))
    return {
        'hamming_nn_mean': float(np.mean(means)) if means else np.nan,
        'subset_features': int(k),
    }


def robust_read_real_subset(path: str, usecols: list, max_rows: int = 50000) -> pd.DataFrame:
    # Try fast path first
    try:
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        return df if max_rows is None or len(df) <= max_rows else df.head(max_rows)
    except Exception:
        # Chunked fallback with python engine
        chunks = []
        try:
            total = 0
            for chunk in pd.read_csv(path, usecols=usecols, engine='python', on_bad_lines='skip', chunksize=10000):
                if max_rows is not None and total >= max_rows:
                    break
                take = chunk
                if max_rows is not None and total + len(take) > max_rows:
                    take = take.head(max_rows - total)
                chunks.append(take)
                total += len(take)
        except TypeError:
            total = 0
            for chunk in pd.read_csv(path, usecols=usecols, engine='python', chunksize=10000):
                if max_rows is not None and total >= max_rows:
                    break
                take = chunk
                if max_rows is not None and total + len(take) > max_rows:
                    take = take.head(max_rows - total)
                chunks.append(take)
                total += len(take)
        return pd.concat(chunks, axis=0, ignore_index=True)


def main():
    diag_dir = os.path.join(AUG_DIR, 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)
    out_fp = os.path.join(diag_dir, 'quick_quality_results.txt')
    def write(line: str):
        print(line)
        with open(out_fp, 'a', encoding='utf-8') as f:
            f.write(line + "\n")

    # Reset file
    with open(out_fp, 'w', encoding='utf-8') as f:
        f.write('Quick Quality Results\n')
        f.write('======================\n')

    write("[quick] Loading synthetic and real headers")
    X_syn = pd.read_csv(X_SYN_PATH)
    X_syn = normalize_id_column(X_syn)
    syn_cols = [c for c in X_syn.columns if c != 'Sample_ID']
    real_cols = pd.read_csv(X_REAL_PATH, nrows=0).columns.tolist()
    real_cols = [c for c in real_cols if c != 'Sample_ID']
    common = sorted(list(set(syn_cols).intersection(set(real_cols))))
    usecols = ['Sample_ID'] + common
    write(f"[quick] Intersecting SNPs: {len(common)}")
    try:
        X_real = robust_read_real_subset(X_REAL_PATH, usecols, max_rows=50000)
    except Exception as e:
        write(f"[quick] ERROR reading real X subset: {repr(e)}")
        return
    write(f"[quick] Real subset shape: {X_real.shape}")
    X_real = normalize_id_column(X_real)
    # Align
    real_aligned, syn_aligned, common_cols = align_features(X_real, X_syn)
    write(f"[quick] Aligned shapes real/syn: {real_aligned.shape}/{syn_aligned.shape}")
    # Numeric and impute
    real_mat = real_aligned[common_cols].apply(pd.to_numeric, errors='coerce').values
    syn_mat = syn_aligned[common_cols].apply(pd.to_numeric, errors='coerce').values
    col_means = np.nanmean(real_mat, axis=0)
    real_inds = np.where(np.isnan(real_mat))
    real_mat[real_inds] = np.take(col_means, real_inds[1])
    syn_inds = np.where(np.isnan(syn_mat))
    syn_mat[syn_inds] = np.take(col_means, syn_inds[1])
    try:
        # PCA metrics
        pca_info, _, _ = pca_metrics(real_mat, syn_mat, n_components=10)
        write(f"[quick] PCA capture ratio (synthetic/real): {pca_info['pca_capture_ratio_synthetic_over_real']:.4f}")
    except Exception as e:
        write(f"[quick] ERROR computing PCA metrics: {repr(e)}")
        pca_info = {'pca_capture_ratio_synthetic_over_real': np.nan}
    try:
        # Hamming
        ham = hamming_distance_summary(np.rint(real_mat).astype(int), np.rint(syn_mat).astype(int), subset_cols=1000)
        write(f"[quick] Hamming NN mismatch mean over {ham['subset_features']} SNPs: {ham['hamming_nn_mean']:.4f}")
    except Exception as e:
        write(f"[quick] ERROR computing Hamming mismatch: {repr(e)}")


if __name__ == '__main__':
    main()