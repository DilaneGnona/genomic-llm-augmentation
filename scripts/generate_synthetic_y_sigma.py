import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


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


def align_columns_for_training(real_X: pd.DataFrame, syn_X: pd.DataFrame):
    real_X = normalize_id_column(real_X)
    syn_X = normalize_id_column(syn_X)
    real_cols = [c for c in real_X.columns if c != 'Sample_ID']
    syn_cols = [c for c in syn_X.columns if c != 'Sample_ID']
    common = [c for c in real_cols if c in syn_cols]
    if len(common) == 0:
        raise RuntimeError('No overlapping SNP columns between real and synthetic data')
    return real_X[['Sample_ID'] + common].copy(), syn_X[['Sample_ID'] + common].copy(), common


def main():
    ap = argparse.ArgumentParser(description='Generate synthetic y with controlled sigma (robust)')
    ap.add_argument('--x_syn', type=str, required=True)
    ap.add_argument('--x_real', type=str, required=True)
    ap.add_argument('--y_real', type=str, required=True)
    ap.add_argument('--target_column', type=str, required=True)
    ap.add_argument('--sigma_resid_factor', type=float, default=0.5)
    ap.add_argument('--clamp_percentiles', type=float, nargs=2, default=(5.0, 95.0))
    ap.add_argument('--output_y', type=str, required=True)
    args = ap.parse_args()

    syn_snps = pd.read_csv(args.x_syn)
    real_X = pd.read_csv(args.x_real)
    real_y_df = pd.read_csv(args.y_real)

    syn_snps = normalize_id_column(syn_snps)
    real_X = normalize_id_column(real_X)
    real_y_df = normalize_id_column(real_y_df)

    tgt_col = args.target_column
    sample_ids = syn_snps['Sample_ID'].astype(str)

    # Align target to real_X IDs
    real_rows = real_X[['Sample_ID']].merge(real_y_df[['Sample_ID', tgt_col]], on='Sample_ID', how='left')
    real_y_series = pd.to_numeric(real_rows[tgt_col], errors='coerce').astype(float)

    real_aligned, syn_aligned, common_cols = align_columns_for_training(real_X, syn_snps)

    valid_mask = np.isfinite(real_y_series.values)
    real_aligned_valid = real_aligned.loc[valid_mask].reset_index(drop=True)
    real_y_valid = real_y_series.values[valid_mask]

    if real_aligned_valid.shape[0] == 0:
        # Degenerate case: fallback to sampling within real y percentiles
        valid_real = real_y_series.dropna().values
        if valid_real.size == 0:
            # No real values at all; use zero-centered noise
            mu, sigma_base = 0.0, 1.0
        else:
            mu = float(np.mean(valid_real))
            sigma_base = float(np.std(valid_real))
        sigma = sigma_base * (args.sigma_resid_factor if args.sigma_resid_factor and args.sigma_resid_factor > 0 else 1.0)
        rng = np.random.default_rng(42)
        y_syn_vals = rng.normal(mu, sigma, size=len(sample_ids))
    else:
        # Fit Ridge teacher and estimate residual sigma robustly
        Xf = real_aligned_valid.drop('Sample_ID', axis=1).values
        rid = Ridge(alpha=1.0)
        n_samples = Xf.shape[0]
        if n_samples >= 2:
            n_splits = max(2, min(5, n_samples))
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            resid_parts = []
            for tr, te in kf.split(Xf):
                rid.fit(Xf[tr], real_y_valid[tr])
                yh = rid.predict(Xf[te])
                resid_parts.append(real_y_valid[te] - yh)
            sigma_base = float(np.sqrt(np.mean(np.concatenate(resid_parts) ** 2)))
        else:
            rid.fit(Xf, real_y_valid)
            yh_all = rid.predict(Xf)
            sigma_base = float(np.sqrt(np.mean((real_y_valid - yh_all) ** 2)))
        sigma = sigma_base * (args.sigma_resid_factor if args.sigma_resid_factor and args.sigma_resid_factor > 0 else 1.0)
        rid.fit(Xf, real_y_valid)
        yhat = rid.predict(syn_aligned.drop('Sample_ID', axis=1).values)
        rng = np.random.default_rng(42)
        y_syn_vals = yhat + rng.normal(0, sigma, size=len(yhat))

    # Clamp
    valid_real = real_y_series.dropna().values
    if valid_real.size > 0:
        p_low, p_high = np.percentile(valid_real, [args.clamp_percentiles[0], args.clamp_percentiles[1]])
        y_syn_vals = np.clip(y_syn_vals, p_low, p_high)

    out_df = pd.DataFrame({'Sample_ID': sample_ids, tgt_col: y_syn_vals.astype(float)})
    os.makedirs(os.path.dirname(args.output_y), exist_ok=True)
    out_df.to_csv(args.output_y, index=False)
    print(f"Wrote {len(out_df)} synthetic targets to {args.output_y} (common SNPs={len(common_cols)})")


if __name__ == '__main__':
    main()