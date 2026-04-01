import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import argparse
import logging

PROC_DIR_DEFAULT = os.path.join('02_processed_data', 'pepper')
AUG_DIR_DEFAULT = os.path.join('04_augmentation', 'pepper')
SNPS_PATH_DEFAULT = os.path.join(AUG_DIR_DEFAULT, 'synthetic_snps.csv')
REAL_X_PATH_DEFAULT = os.path.join(PROC_DIR_DEFAULT, 'X.csv')
REAL_Y_PATH_DEFAULT = os.path.join(PROC_DIR_DEFAULT, 'y.csv')
OUT_PATH_DEFAULT = os.path.join(AUG_DIR_DEFAULT, 'synthetic_y.csv')


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
    return df


def choose_target_column(y_df: pd.DataFrame) -> str | None:
    if 'Yield_BV' in y_df.columns:
        return 'Yield_BV'
    if 'YIELD' in y_df.columns:
        return 'YIELD'
    numeric_cols = [c for c in y_df.columns if c != 'Sample_ID' and pd.api.types.is_numeric_dtype(y_df[c])]
    return numeric_cols[0] if numeric_cols else None


def align_columns_for_training(real_X: pd.DataFrame, syn_X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    real_X = normalize_id_column(real_X)
    syn_X = normalize_id_column(syn_X)
    real_cols = [c for c in real_X.columns if c != 'Sample_ID']
    syn_cols = [c for c in syn_X.columns if c != 'Sample_ID']
    common = [c for c in real_cols if c in syn_cols]
    if len(common) == 0:
        raise RuntimeError('No overlapping SNP columns between real and synthetic data')
    real_aligned = real_X[['Sample_ID'] + common].copy()
    syn_aligned = syn_X[['Sample_ID'] + common].copy()
    return real_aligned, syn_aligned, common


def ridge_teacher_generate(real_X: pd.DataFrame, real_y: np.ndarray, syn_X: pd.DataFrame, seed: int, sigma_factor: float | None):
    # Estimate residual sigma via CV when sufficient samples exist; fallback otherwise
    rng = np.random.default_rng(seed)
    rid = Ridge(alpha=1.0)
    Xf = real_X.drop('Sample_ID', axis=1).values
    n_samples = Xf.shape[0]
    if n_samples >= 2:
        n_splits = max(2, min(5, n_samples))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        resid_parts = []
        for tr, te in kf.split(Xf):
            rid.fit(Xf[tr], real_y[tr])
            yh = rid.predict(Xf[te])
            resid_parts.append(real_y[te] - yh)
        sigma = float(np.sqrt(np.mean(np.concatenate(resid_parts) ** 2)))
    else:
        # Fallback: fit on all and compute residual on training
        rid.fit(Xf, real_y)
        yh_all = rid.predict(Xf)
        sigma = float(np.sqrt(np.mean((real_y - yh_all) ** 2)))
    if sigma_factor is not None and np.isfinite(sigma_factor) and sigma_factor > 0:
        sigma *= float(sigma_factor)
    # Fit on all real, predict synthetic
    rid.fit(Xf, real_y)
    yhat = rid.predict(syn_X.drop('Sample_ID', axis=1).values)
    noise = rng.normal(0, sigma, size=len(yhat))
    ysyn = yhat + noise
    return ysyn, sigma


def main():
    ap = argparse.ArgumentParser(description='Regenerate synthetic labels (pepper) with teacher and clamping')
    ap.add_argument('--teacher', type=str, default='ridge', choices=['ridge'], help='Teacher model')
    ap.add_argument('--x_syn', type=str, default=SNPS_PATH_DEFAULT, help='Path to synthetic SNPs')
    ap.add_argument('--x_real', type=str, default=REAL_X_PATH_DEFAULT, help='Path to real X.csv')
    ap.add_argument('--y_real', type=str, default=REAL_Y_PATH_DEFAULT, help='Path to real y.csv')
    ap.add_argument('--target_column', type=str, default=None, help='Target column in real y.csv (default auto)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sigma_resid_factor', type=float, default=None, help='Scale factor for CV residual sigma')
    ap.add_argument('--clamp_percentiles', type=float, nargs=2, default=(1.0, 99.0), help='Clamp synthetic targets to real y percentiles [low, high]')
    ap.add_argument('--output_y', type=str, default=OUT_PATH_DEFAULT, help='Output path for synthetic y')
    ap.add_argument('--log', type=str, default=None, help='Optional log file path')
    args = ap.parse_args()

    if args.log:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(args.log), logging.StreamHandler()]
        )

    # Load data
    logging.info(f"Reading synthetic SNPs: {args.x_syn}")
    syn_snps = pd.read_csv(args.x_syn)
    logging.info(f"Reading real X: {args.x_real}")
    real_X = pd.read_csv(args.x_real)
    logging.info(f"Reading real y: {args.y_real}")
    real_y_df = pd.read_csv(args.y_real)

    syn_snps = normalize_id_column(syn_snps)
    real_X = normalize_id_column(real_X)
    real_y_df = normalize_id_column(real_y_df)

    sample_ids = syn_snps['Sample_ID'].astype(str)

    tgt_col = args.target_column if args.target_column else choose_target_column(real_y_df)
    if not tgt_col:
        raise RuntimeError('Could not determine target column in real y.csv')
    logging.info(f"Using target column: {tgt_col}")

    # Align rows by Sample_ID (ensure y length equals X rows)
    real_rows = real_X[['Sample_ID']].merge(real_y_df[['Sample_ID', tgt_col]], on='Sample_ID', how='left')
    real_y_series = pd.to_numeric(real_rows[tgt_col], errors='coerce').astype(float)

    # Align genotype columns (intersection)
    real_aligned, syn_aligned, common_cols = align_columns_for_training(real_X, syn_snps)
    logging.info(f"Intersecting SNP columns: {len(common_cols)}")

    # Filter to non-NaN target rows for training
    valid_mask = np.isfinite(real_y_series.values)
    real_aligned_valid = real_aligned.loc[valid_mask].reset_index(drop=True)
    real_y_valid = real_y_series.values[valid_mask]

    # Train teacher and generate synthetic targets
    y_syn_vals, eff_sigma = ridge_teacher_generate(real_aligned_valid, real_y_valid, syn_aligned, args.seed, args.sigma_resid_factor)
    logging.info(f"Teacher: {args.teacher}, CV residual sigma={eff_sigma:.6f}, sigma_factor={args.sigma_resid_factor}")

    # Clamp to reasonable range based on real y percentiles
    valid_real = real_y_series.dropna().values
    p_low, p_high = np.percentile(valid_real, [args.clamp_percentiles[0], args.clamp_percentiles[1]])
    y_syn_vals = np.clip(y_syn_vals, p_low, p_high)
    logging.info(f"Clamp range: [{p_low:.6f}, {p_high:.6f}] from real percentiles {args.clamp_percentiles}")

    # Write out synthetic targets
    out_df = pd.DataFrame({
        'Sample_ID': sample_ids,
        tgt_col: y_syn_vals.astype(float)
    })
    os.makedirs(os.path.dirname(args.output_y), exist_ok=True)
    out_df.to_csv(args.output_y, index=False)

    print(f"Wrote {len(out_df)} synthetic targets to {args.output_y}")
    print(f"Teacher: Ridge, CV residual sigma={eff_sigma:.6f}, sigma_factor={args.sigma_resid_factor}")
    print(f"Columns intersected: {len(common_cols)} of real SNPs; clamp range [{p_low:.4f}, {p_high:.4f}] from real y percentiles")


if __name__ == '__main__':
    main()