import argparse
import os
import sys
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def setup_logger(log_path: str):
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def parse_args():
    ap = argparse.ArgumentParser(description="Filter synthetic samples by PCA proximity and clamp targets to real percentiles")
    ap.add_argument('--dataset', required=True, type=str, help='Dataset name (e.g., pepper)')
    ap.add_argument('--synthetic_file', required=True, type=str, help='Path to synthetic_y.csv')
    ap.add_argument('--real_file', required=True, type=str, help='Path to real y.csv')
    ap.add_argument('--pca_file', required=True, type=str, help='Path to real pca_covariates.csv (for reporting)')
    ap.add_argument('--target_column', required=True, type=str, help='Target column name (e.g., Yield_BV)')
    ap.add_argument('--pca_threshold', required=True, type=float, help='Percentile threshold (0–1) for PCA proximity (e.g., 0.95)')
    ap.add_argument('--target_percentile_low', required=True, type=float, help='Low percentile for clamping (e.g., 5)')
    ap.add_argument('--target_percentile_high', required=True, type=float, help='High percentile for clamping (e.g., 95)')
    ap.add_argument('--fallback_percent', type=float, default=None, help='Minimal retained proportion (0–100) if PCA threshold drops all samples')
    ap.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (e.g., PCA)')
    ap.add_argument('--whiten_pca', action='store_true', help='Use PCA whitening for balanced component scales')
    ap.add_argument('--output_file', required=True, type=str, help='Output path for filtered synthetic y')
    ap.add_argument('--log_file', type=str, default=None, help='Optional log file path; defaults next to output_file')
    return ap.parse_args()


def load_data(dataset: str, synthetic_y_path: str, real_y_path: str):
    # Derive X paths based on dataset; these are required to compute PCA embeddings
    real_x_path = os.path.join('02_processed_data', dataset, 'X.csv')
    synthetic_snps_path = os.path.join(os.path.dirname(synthetic_y_path), 'synthetic_snps.csv')

    if not os.path.exists(real_x_path):
        raise FileNotFoundError(f"Real X file not found: {real_x_path}")
    if not os.path.exists(synthetic_snps_path):
        raise FileNotFoundError(f"Synthetic SNPs file not found: {synthetic_snps_path}")
    if not os.path.exists(synthetic_y_path):
        raise FileNotFoundError(f"Synthetic y file not found: {synthetic_y_path}")
    if not os.path.exists(real_y_path):
        raise FileNotFoundError(f"Real y file not found: {real_y_path}")

    # First read synthetic SNPs to get the column names we need
    logging.info(f"Reading synthetic SNPs file: {synthetic_snps_path}")
    X_syn = pd.read_csv(synthetic_snps_path, low_memory=False)
    logging.info(f"Synthetic X shape: {X_syn.shape}")
    
    # Get the list of SNP columns from synthetic data
    snp_columns = X_syn.columns.tolist()
    
    # Read only the necessary SNP columns from real X file (much faster and memory efficient)
    logging.info(f"Reading real X file (only SNP columns: {len(snp_columns)} columns): {real_x_path}")
    X_real = pd.read_csv(real_x_path, usecols=snp_columns, low_memory=False)
    logging.info(f"Real X shape: {X_real.shape}")
    
    # Ensure columns are in the same order
    X_real = X_real[snp_columns]
    
    logging.info(f"Reading synthetic y file: {synthetic_y_path}")
    y_syn = pd.read_csv(synthetic_y_path, low_memory=False)
    logging.info(f"Synthetic y shape: {y_syn.shape}")
    
    logging.info(f"Reading real y file: {real_y_path}")
    y_real = pd.read_csv(real_y_path, low_memory=False)
    logging.info(f"Real y shape: {y_real.shape}")

    return X_real, X_syn, y_syn, y_real


def align_snp_columns(X_real: pd.DataFrame, X_syn: pd.DataFrame):
    # Expect first column to be Sample_ID
    real_has_id = 'Sample_ID' in X_real.columns
    syn_has_id = 'Sample_ID' in X_syn.columns
    if not real_has_id:
        raise RuntimeError('Real X.csv must include Sample_ID column')
    if not syn_has_id:
        # Insert synthetic Sample_IDs if missing
        X_syn = X_syn.copy()
        X_syn.insert(0, 'Sample_ID', [f'SYNTHETIC_{i}' for i in range(len(X_syn))])

    real_cols = [c for c in X_real.columns if c != 'Sample_ID']
    syn_cols = [c for c in X_syn.columns if c != 'Sample_ID']
    intersect = [c for c in real_cols if c in syn_cols]
    if len(intersect) == 0:
        raise RuntimeError('No overlapping SNP columns between real and synthetic')

    # Ensure numeric and fill missing
    Xr = X_real[['Sample_ID'] + intersect].copy()
    Xs = X_syn[['Sample_ID'] + intersect].copy()
    for df in (Xr, Xs):
        df[intersect] = df[intersect].apply(pd.to_numeric, errors='coerce').fillna(0)

    return Xr, Xs, intersect


def fit_pca_on_real(X_real_matrix: np.ndarray, n_components: int = 20, seed: int = 42, whiten: bool = False):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xr_scaled = scaler.fit_transform(X_real_matrix)
    n_comp_eff = max(1, min(n_components, Xr_scaled.shape[1]))
    pca = PCA(n_components=n_comp_eff, random_state=seed, whiten=whiten)
    Xr_pca = pca.fit_transform(Xr_scaled)
    return scaler, pca, Xr_pca


def transform_to_pca(scaler: StandardScaler, pca: PCA, X_matrix: np.ndarray):
    X_scaled = scaler.transform(X_matrix)
    return pca.transform(X_scaled)


def compute_distances_to_centroid(X_pca: np.ndarray):
    mu = X_pca.mean(axis=0)
    dists = np.linalg.norm(X_pca - mu, axis=1)
    return dists, mu


def main():
    args = parse_args()

    # Seed all random paths for reproducibility
    try:
        np.random.seed(args.seed)
    except Exception:
        pass

    # Derive log path next to output_file unless explicitly provided
    out_dir = os.path.dirname(args.output_file)
    os.makedirs(out_dir, exist_ok=True)
    log_path = args.log_file if args.log_file else os.path.join(out_dir, 'filter_log.txt')
    setup_logger(log_path)

    logging.info('Starting synthetic confidence filtering')
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Synthetic y: {args.synthetic_file}")
    logging.info(f"Real y: {args.real_file}")
    logging.info(f"Real PCA covariates (for reporting): {args.pca_file}")
    logging.info(f"Target column: {args.target_column}")
    logging.info(f"PCA proximity percentile threshold: {args.pca_threshold}")
    logging.info(f"Clamp percentiles: low={args.target_percentile_low}, high={args.target_percentile_high}")
    if args.fallback_percent is not None:
        logging.info(f"Fallback retention percent if empty: {args.fallback_percent}")
    logging.info(f"Seed: {args.seed}")
    logging.info(f"PCA whitening: {'ON' if args.whiten_pca else 'OFF'}")
    logging.info(f"Output file: {args.output_file}")
    logging.info(f"Log file: {log_path}")
    
    # Load data
    logging.info("Loading data...")
    X_real, X_syn, y_syn, y_real = load_data(args.dataset, args.synthetic_file, args.real_file)
    logging.info(f"Loaded real X: {X_real.shape}")
    logging.info(f"Loaded synthetic X: {X_syn.shape}")
    logging.info(f"Loaded synthetic y: {y_syn.shape}")
    logging.info(f"Loaded real y: {y_real.shape}")
    
    # Align SNP columns
    logging.info("Aligning SNP columns...")
    Xr_aligned, Xs_aligned, shared_cols = align_snp_columns(X_real, X_syn)
    logging.info(f"Aligned real X: {Xr_aligned.shape}, synthetic X: {Xs_aligned.shape}")
    logging.info(f"Shared columns: {len(shared_cols)}")
    
    # Prepare matrices for PCA
    logging.info("Preparing matrices for PCA...")
    Xr_matrix = Xr_aligned[shared_cols].values
    Xs_matrix = Xs_aligned[shared_cols].values
    logging.info(f"Real X matrix: {Xr_matrix.shape}, synthetic X matrix: {Xs_matrix.shape}")
    
    # Fit PCA on real data
    logging.info("Fitting PCA on real data...")
    scaler, pca, Xr_pca = fit_pca_on_real(Xr_matrix, seed=args.seed, whiten=args.whiten_pca)
    logging.info(f"PCA fitted with {Xr_pca.shape[1]} components")
    
    # Transform synthetic data to PCA space
    logging.info("Transforming synthetic data to PCA space...")
    Xs_pca = transform_to_pca(scaler, pca, Xs_matrix)
    logging.info(f"Synthetic data transformed to PCA: {Xs_pca.shape}")
    
    # Compute distances
    logging.info("Computing distances...")
    dists_real, mu_real = compute_distances_to_centroid(Xr_pca)
    # Compute synthetic distances relative to the real centroid (not synthetic centroid)
    dists_syn = np.linalg.norm(Xs_pca - mu_real, axis=1)
    # Log distance statistics to debug filtering
    logging.info(f"Real distances - min: {np.min(dists_real)}, max: {np.max(dists_real)}, mean: {np.mean(dists_real)}")
    logging.info(f"Synthetic distances - min: {np.min(dists_syn)}, max: {np.max(dists_syn)}, mean: {np.mean(dists_syn)}")
    logging.info(f"Distances computed: real {len(dists_real)}, synthetic {len(dists_syn)}")
    
    # Filter synthetic samples
    logging.info("Filtering synthetic samples...")
    # Convert threshold from proportion (0-1) to percentile (0-100) if needed
    pca_percentile = args.pca_threshold * 100 if args.pca_threshold <= 1.0 else args.pca_threshold
    threshold = np.percentile(dists_real, pca_percentile)
    logging.info(f"PCA percentile used: {pca_percentile}")
    mask = dists_syn <= threshold
    logging.info(f"Threshold: {threshold}, samples passing: {mask.sum()}")
    
    # Clamp targets
    logging.info("Clamping targets...")
    y_clamp = y_syn.copy()
    y_clamp[args.target_column] = np.clip(
        y_clamp[args.target_column],
        np.percentile(y_real[args.target_column], args.target_percentile_low),
        np.percentile(y_real[args.target_column], args.target_percentile_high)
    )
    
    # Apply filter mask
    logging.info("Applying filter mask...")
    y_filtered = y_clamp[mask].copy()
    logging.info(f"Filtered y shape: {y_filtered.shape}")
    
    # Apply fallback if too few samples passed
    if args.fallback_percent is not None:
        if len(y_filtered) < (len(y_syn) * args.fallback_percent / 100):
            logging.info(f"Too few samples passed ({len(y_filtered)}). Applying fallback to retain {args.fallback_percent}%...")
            # Calculate the number of samples to retain
            n_retain = max(1, int(len(y_syn) * args.fallback_percent / 100))
            # Sort synthetic samples by distance and retain the closest ones
            sorted_indices = np.argsort(dists_syn)
            fallback_mask = np.zeros(len(y_syn), dtype=bool)
            fallback_mask[sorted_indices[:n_retain]] = True
            y_filtered = y_clamp[fallback_mask].copy()
            logging.info(f"Fallback applied. Retained {len(y_filtered)} samples.")
    
    # Save results
    logging.info("Saving results...")
    y_filtered.to_csv(args.output_file, index=False)
    logging.info(f"Saved filtered synthetic y to {args.output_file}")
    
    logging.info("Synthetic confidence filtering completed successfully!")

if __name__ == '__main__':
    main()