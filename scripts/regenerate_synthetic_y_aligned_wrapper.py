import os
import argparse
import numpy as np
import pandas as pd

# Reuse logic from the existing generator
from regenerate_synthetic_y_pepper import (
    normalize_id_column,
    choose_target_column,
    align_columns_for_training,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def main():
    ap = argparse.ArgumentParser(description="Wrapper to regenerate synthetic labels aligned to fixed SNPs")
    ap.add_argument('--x_syn', type=str, required=True, help='Path to synthetic SNPs (fixed)')
    ap.add_argument('--x_real', type=str, required=True, help='Path to real X.csv')
    ap.add_argument('--y_real', type=str, required=True, help='Path to real y.csv')
    ap.add_argument('--target_column', type=str, default=None, help='Target column in real y.csv (default auto)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sigma_resid_factor', type=float, default=None, help='Scale factor for CV residual sigma')
    ap.add_argument('--clamp_percentiles', type=float, nargs=2, default=(10.0, 90.0), help='Clamp synthetic targets to real y percentiles [low, high]')
    ap.add_argument('--output_y', type=str, required=True, help='Output path for synthetic y')
    args = ap.parse_args()

    # Load data
    print(f"Reading synthetic SNPs: {args.x_syn}")
    syn_snps = pd.read_csv(args.x_syn)
    print(f"Reading real X: {args.x_real}")
    real_X = pd.read_csv(args.x_real)
    print(f"Reading real y: {args.y_real}")
    real_y_df = pd.read_csv(args.y_real)

    syn_snps = normalize_id_column(syn_snps)
    real_X = normalize_id_column(real_X)
    real_y_df = normalize_id_column(real_y_df)

    sample_ids = syn_snps['Sample_ID'].astype(str)

    tgt_col = args.target_column if args.target_column else choose_target_column(real_y_df)
    if not tgt_col:
        raise RuntimeError('Could not determine target column in real y.csv')
    print(f"Using target column: {tgt_col}")

    # Align rows by Sample_ID to get target vector
    real_rows = real_X[['Sample_ID']].merge(real_y_df[['Sample_ID', tgt_col]], on='Sample_ID', how='left')
    real_y_series = pd.to_numeric(real_rows[tgt_col], errors='coerce').astype(float)

    # Align genotype columns (intersection)
    real_aligned, syn_aligned, common_cols = align_columns_for_training(real_X, syn_snps)
    print(f"Intersecting SNP columns: {len(common_cols)}")
    if len(common_cols) == 0:
        raise RuntimeError('No overlapping SNP columns between real and synthetic data')

    # Filter to non-NaN target rows for training
    valid_mask = np.isfinite(real_y_series.values)
    real_aligned_valid = real_aligned.loc[valid_mask].reset_index(drop=True)
    real_y_valid = real_y_series.values[valid_mask]

    # Train teacher (Ridge) and generate synthetic targets
    rng = np.random.default_rng(args.seed)
    rid = Ridge(alpha=1.0)
    Xf = real_aligned_valid.drop('Sample_ID', axis=1).values
    n_samples = Xf.shape[0]
    if n_samples >= 2:
        n_splits = max(2, min(5, n_samples))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        resid = []
        for tr, te in kf.split(Xf):
            rid.fit(Xf[tr], real_y_valid[tr])
            yh = rid.predict(Xf[te])
            resid.append(real_y_valid[te] - yh)
        sigma = float(np.sqrt(np.mean(np.concatenate(resid) ** 2)))
    else:
        rid.fit(Xf, real_y_valid)
        yh_all = rid.predict(Xf)
        sigma = float(np.sqrt(np.mean((real_y_valid - yh_all) ** 2)))
    if args.sigma_resid_factor is not None and np.isfinite(args.sigma_resid_factor) and args.sigma_resid_factor > 0:
        sigma *= float(args.sigma_resid_factor)
    # Fit on all real, predict synthetic
    rid.fit(Xf, real_y_valid)
    yhat = rid.predict(syn_aligned.drop('Sample_ID', axis=1).values)
    noise = rng.normal(0, sigma, size=len(yhat))
    y_syn_vals = yhat + noise
    print(f"Teacher: Ridge, CV residual sigma={sigma:.6f}, sigma_factor={args.sigma_resid_factor}")

    # Clamp to reasonable range based on real y percentiles
    valid_real = real_y_series.dropna().values
    p_low, p_high = np.percentile(valid_real, [args.clamp_percentiles[0], args.clamp_percentiles[1]])
    y_syn_vals = np.clip(y_syn_vals, p_low, p_high)
    print(f"Clamp range: [{p_low:.6f}, {p_high:.6f}] from real percentiles {args.clamp_percentiles}")

    # Write out synthetic targets
    out_df = pd.DataFrame({
        'Sample_ID': sample_ids,
        tgt_col: y_syn_vals.astype(float)
    })
    os.makedirs(os.path.dirname(args.output_y), exist_ok=True)
    out_df.to_csv(args.output_y, index=False)

    print(f"Wrote {len(out_df)} synthetic targets to {args.output_y}")


if __name__ == '__main__':
    main()