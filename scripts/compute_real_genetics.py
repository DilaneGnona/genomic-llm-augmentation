import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any


def setup_logger(log_path: str | None):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path) if log_path else logging.StreamHandler()
        ]
    )


def detect_id_column(df: pd.DataFrame) -> str:
    if 'Sample_ID' in df.columns:
        return 'Sample_ID'
    if 'IID' in df.columns:
        return 'IID'
    return df.columns[0]


def normalize_id(df: pd.DataFrame) -> pd.DataFrame:
    cid = detect_id_column(df)
    if cid != 'Sample_ID':
        df = df.rename(columns={cid: 'Sample_ID'})
    return df


def compute_maf(geno: pd.DataFrame) -> pd.Series:
    n = geno.shape[0]
    if n == 0:
        return pd.Series(0.0, index=geno.columns, dtype=float)
    denom = 2.0 * n
    # assume genotypes encoded {0,1,2}
    p = geno.sum(axis=0) / denom
    maf = np.minimum(p, 1.0 - p)
    return pd.Series(maf, index=geno.columns, dtype=float)


def build_ld_blocks_by_corr_threshold(geno: pd.DataFrame, r2_threshold: float, min_block_size: int) -> List[List[str]]:
    cols = list(geno.columns)
    X = geno.values.astype(float)
    # standardize columns to mean 0, std 1 to approximate correlation
    X = X - np.nanmean(X, axis=0, keepdims=True)
    std = np.nanstd(X, axis=0, ddof=0)
    std[std == 0] = 1.0
    X = X / std

    # Compute correlation matrix; for modest SNP counts (intersection), this is feasible
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    R2 = C * C

    # Build adjacency based on r^2 threshold, ignoring self
    n = len(cols)
    visited = np.zeros(n, dtype=bool)
    blocks: List[List[str]] = []

    for i in range(n):
        if visited[i]:
            continue
        # BFS/DFS to collect connected component
        stack = [i]
        comp_idx = []
        visited[i] = True
        while stack:
            k = stack.pop()
            comp_idx.append(k)
            # neighbors above threshold
            neighbors = np.where((R2[k] >= r2_threshold) & (np.arange(n) != k))[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        # enforce minimum block size: if too small, keep as singleton
        if len(comp_idx) >= min_block_size:
            blocks.append([cols[j] for j in sorted(set(comp_idx))])
        else:
            for j in comp_idx:
                blocks.append([cols[j]])

    return blocks


def main():
    ap = argparse.ArgumentParser(description="Compute real genotype statistics (MAF, LD blocks)")
    ap.add_argument('--x_real', required=True, type=str, help='Path to real X.csv')
    ap.add_argument('--out', required=True, type=str, help='Output JSON summary path')
    ap.add_argument('--maf_report', required=True, type=str, help='Output CSV path for per-SNP MAF')
    ap.add_argument('--ld_blocks', required=True, type=str, help='Output JSON path for LD blocks')
    ap.add_argument('--ld_method', type=str, default='corr_threshold', choices=['corr_threshold'], help='LD block method')
    ap.add_argument('--r2_threshold', type=float, default=0.2, help='r^2 threshold to form edges')
    ap.add_argument('--min_block_size', type=int, default=5, help='Minimum LD block size; smaller become singletons')
    ap.add_argument('--x_syn', type=str, default=None, help='Optional synthetic_snps.csv to restrict to intersecting SNPs')
    ap.add_argument('--log', type=str, default=None, help='Optional log file path')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.maf_report), exist_ok=True)
    os.makedirs(os.path.dirname(args.ld_blocks), exist_ok=True)

    setup_logger(args.log)
    logging.info("Loading real X and optional synthetic for intersection")

    X_real = pd.read_csv(args.x_real)
    X_real = normalize_id(X_real)
    real_cols = [c for c in X_real.columns if c != 'Sample_ID']

    intersect_cols = real_cols
    if args.x_syn and os.path.exists(args.x_syn):
        X_syn = pd.read_csv(args.x_syn)
        X_syn = normalize_id(X_syn)
        syn_cols = [c for c in X_syn.columns if c != 'Sample_ID']
        intersect_cols = [c for c in real_cols if c in syn_cols]
        logging.info(f"Intersecting SNPs: {len(intersect_cols)} of {len(real_cols)} real SNPs")
        if len(intersect_cols) == 0:
            logging.error("No intersecting SNPs between real and synthetic")
            raise SystemExit(1)

    geno = X_real[intersect_cols].copy()
    logging.info(f"Computing MAF for {geno.shape[1]} SNPs across {geno.shape[0]} samples")
    mafs = compute_maf(geno)

    maf_df = pd.DataFrame({'SNP': mafs.index, 'MAF': mafs.values})
    maf_df.to_csv(args.maf_report, index=False)
    logging.info(f"Wrote MAF report: {args.maf_report}")

    logging.info(f"Building LD blocks using method={args.ld_method}, r2_threshold={args.r2_threshold}, min_block_size={args.min_block_size}")
    blocks = build_ld_blocks_by_corr_threshold(geno, args.r2_threshold, args.min_block_size)

    with open(args.ld_blocks, 'w', encoding='utf-8') as f:
        json.dump({'method': args.ld_method, 'r2_threshold': args.r2_threshold, 'min_block_size': args.min_block_size, 'blocks': blocks}, f, indent=2)
    logging.info(f"Wrote LD blocks: {args.ld_blocks} (count={len(blocks)})")

    # Summary JSON
    block_sizes = [len(b) for b in blocks]
    summary = {
        'num_snps': int(len(intersect_cols)),
        'num_samples': int(X_real.shape[0]),
        'maf_mean': float(mafs.mean()),
        'maf_p95': float(np.percentile(mafs.dropna(), 95)) if mafs.dropna().shape[0] > 0 else 0.0,
        'ld_method': args.ld_method,
        'r2_threshold': float(args.r2_threshold),
        'min_block_size': int(args.min_block_size),
        'block_count': int(len(blocks)),
        'block_size_mean': float(np.mean(block_sizes)) if block_sizes else 0.0,
        'block_size_p95': float(np.percentile(block_sizes, 95)) if block_sizes else 0.0
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Wrote genetics summary: {args.out}")


if __name__ == '__main__':
    main()