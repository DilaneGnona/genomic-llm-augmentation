import os
import json
import sys
import numpy as np
import pandas as pd

def normalize_id(df: pd.DataFrame) -> pd.DataFrame:
    if 'Sample_ID' in df.columns:
        return df
    if 'IID' in df.columns:
        return df.rename(columns={'IID': 'Sample_ID'})
    return df.rename(columns={df.columns[0]: 'Sample_ID'})

def compute_maf(geno: pd.DataFrame) -> pd.Series:
    n = geno.shape[0]
    denom = 2.0 * n
    p = geno.sum(axis=0) / denom
    maf = np.minimum(p, 1.0 - p)
    return pd.Series(maf, index=geno.columns, dtype=float)

def build_ld_blocks(geno: pd.DataFrame, r2_threshold: float, min_block_size: int):
    cols = list(geno.columns)
    X = geno.values.astype(float)
    X = X - np.nanmean(X, axis=0, keepdims=True)
    std = np.nanstd(X, axis=0, ddof=0)
    std[std == 0] = 1.0
    X = X / std
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    R2 = C * C
    n = len(cols)
    visited = np.zeros(n, dtype=bool)
    blocks = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp_idx = []
        visited[i] = True
        while stack:
            k = stack.pop()
            comp_idx.append(k)
            neighbors = np.where((R2[k] >= r2_threshold) & (np.arange(n) != k))[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(comp_idx) >= min_block_size:
            blocks.append([cols[j] for j in sorted(set(comp_idx))])
        else:
            for j in comp_idx:
                blocks.append([cols[j]])
    return blocks

def main():
    root = os.getcwd()
    x_real = os.path.join(root, '02_processed_data', 'pepper', 'X.csv')
    x_syn = os.path.join(root, '04_augmentation', 'pepper', 'synthetic_snps.csv')
    out_dir = os.path.join(root, '04_augmentation', 'pepper')
    maf_csv = os.path.join(out_dir, 'real_maf.csv')
    ld_json = os.path.join(out_dir, 'real_ld_blocks.json')
    stats_json = os.path.join(out_dir, 'real_genetics_stats.json')

    Xr = pd.read_csv(x_real)
    Xr = normalize_id(Xr)
    cols_r = [c for c in Xr.columns if c != 'Sample_ID']

    if os.path.exists(x_syn):
        Xs = pd.read_csv(x_syn)
        Xs = normalize_id(Xs)
        cols_s = [c for c in Xs.columns if c != 'Sample_ID']
        cols = [c for c in cols_r if c in cols_s]
    else:
        cols = cols_r

    geno = Xr[cols].copy()
    mafs = compute_maf(geno)
    pd.DataFrame({'SNP': mafs.index, 'MAF': mafs.values}).to_csv(maf_csv, index=False)

    blocks = build_ld_blocks(geno, r2_threshold=0.2, min_block_size=5)
    with open(ld_json, 'w', encoding='utf-8') as f:
        json.dump({'method': 'corr_threshold', 'r2_threshold': 0.2, 'min_block_size': 5, 'blocks': blocks}, f, indent=2)

    block_sizes = [len(b) for b in blocks]
    summary = {
        'num_snps': int(len(cols)),
        'num_samples': int(Xr.shape[0]),
        'maf_mean': float(mafs.mean()),
        'maf_p95': float(np.percentile(mafs.dropna(), 95)) if mafs.dropna().shape[0] > 0 else 0.0,
        'ld_method': 'corr_threshold',
        'r2_threshold': 0.2,
        'min_block_size': 5,
        'block_count': int(len(blocks)),
        'block_size_mean': float(np.mean(block_sizes)) if block_sizes else 0.0,
        'block_size_p95': float(np.percentile(block_sizes, 95)) if block_sizes else 0.0
    }
    with open(stats_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote: {maf_csv}, {ld_json}, {stats_json}")

if __name__ == '__main__':
    main()