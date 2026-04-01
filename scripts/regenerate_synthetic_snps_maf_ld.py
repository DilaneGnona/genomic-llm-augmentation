import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from typing import List, Dict


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


def load_maf_map(maf_csv: str) -> Dict[str, float]:
    m = {}
    df = pd.read_csv(maf_csv)
    # support columns named SNP, MAF
    snp_col = 'SNP' if 'SNP' in df.columns else df.columns[0]
    maf_col = 'MAF' if 'MAF' in df.columns else df.columns[1]
    for _, row in df.iterrows():
        m[str(row[snp_col])] = float(row[maf_col])
    return m


def hw_genotype_sample(p: float, n: int, rng: np.random.Generator) -> np.ndarray:
    # Hardy-Weinberg equilibrium: AA:(1-p)^2, Aa:2p(1-p), aa:p^2; encode minor allele count
    probs = np.array([(1-p)**2, 2*p*(1-p), p**2], dtype=float)
    probs = probs / probs.sum()
    # sample 0,1,2 counts directly
    return rng.choice([0, 1, 2], size=n, p=probs)


def gaussian_copula_block(p_list: List[float], n: int, rng: np.random.Generator, noise_sigma: float = 0.5) -> np.ndarray:
    # Construct correlated Bernoulli via shared latent Z; two draws per SNP to produce 0/1/2 genotype
    Z = rng.normal(0.0, 1.0, size=n)
    g_mat = np.zeros((n, len(p_list)), dtype=int)
    # thresholds per SNP
    from scipy.stats import norm
    t_list = [norm.ppf(1.0 - min(max(p, 1e-6), 1 - 1e-6)) for p in p_list]
    for j, t in enumerate(t_list):
        eps1 = rng.normal(0.0, noise_sigma, size=n)
        eps2 = rng.normal(0.0, noise_sigma, size=n)
        a1 = (Z + eps1) > t
        a2 = (Z + eps2) > t
        g_mat[:, j] = a1.astype(int) + a2.astype(int)  # 0/1/2
    return g_mat


def adjust_maf_towards_target(col: np.ndarray, target_maf: float, tol: float, rng: np.random.Generator) -> np.ndarray:
    n = len(col)
    denom = 2.0 * n
    cur_p = col.sum() / denom
    cur_maf = min(cur_p, 1.0 - cur_p)
    if abs(cur_maf - target_maf) <= tol:
        return col
    # flip minor allele in random positions to move towards target
    target_p = target_maf
    # choose direction: increase or decrease minor allele count
    if cur_maf < target_maf:
        # need to increase minor allele count: convert 0->1 or 1->2 randomly
        zero_idx = np.where(col == 0)[0]
        one_idx = np.where(col == 1)[0]
        deficit = int(round(denom * (target_p - cur_p)))
        flip_from_zero = min(len(zero_idx), max(0, deficit // 2))
        flip_from_one = min(len(one_idx), max(0, deficit - 2 * flip_from_zero))
        if flip_from_zero > 0:
            chosen = rng.choice(zero_idx, size=flip_from_zero, replace=False)
            col[chosen] = 1
        if flip_from_one > 0:
            chosen = rng.choice(one_idx, size=flip_from_one, replace=False)
            col[chosen] = 2
    else:
        # need to decrease minor allele count: convert 2->1 or 1->0 randomly
        two_idx = np.where(col == 2)[0]
        one_idx = np.where(col == 1)[0]
        surplus = int(round(denom * (cur_p - target_p)))
        flip_from_two = min(len(two_idx), max(0, surplus // 2))
        flip_from_one = min(len(one_idx), max(0, surplus - 2 * flip_from_two))
        if flip_from_two > 0:
            chosen = rng.choice(two_idx, size=flip_from_two, replace=False)
            col[chosen] = 1
        if flip_from_one > 0:
            chosen = rng.choice(one_idx, size=flip_from_one, replace=False)
            col[chosen] = 0
    return col


def main():
    ap = argparse.ArgumentParser(description="Regenerate synthetic SNPs with MAF+LD constraints")
    ap.add_argument('--x_template', required=True, type=str, help='Path to synthetic_snps.csv (template for Sample_ID count)')
    ap.add_argument('--real_maf', required=True, type=str, help='Path to real_maf.csv from compute_real_genetics.py')
    ap.add_argument('--ld_blocks', required=True, type=str, help='Path to real_ld_blocks.json')
    ap.add_argument('--intersect_with_real', type=str, default='True', help='Use intersection of template and real MAF SNPs (True/False)')
    ap.add_argument('--output_snps', required=True, type=str, help='Output path for regenerated synthetic_snps_matched.csv')
    ap.add_argument('--log', type=str, default=None, help='Log file path')
    ap.add_argument('--maf_tolerance', type=float, default=0.02, help='Allowed absolute MAF deviation')
    ap.add_argument('--ld_within_block', type=str, default='preserve_pairwise_freq', choices=['preserve_pairwise_freq', 'independent_hw'], help='Block generation strategy')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    args = ap.parse_args()

    use_intersect = str(args.intersect_with_real).strip().lower() in ('true', '1', 'yes', 'y', 't')
    os.makedirs(os.path.dirname(args.output_snps), exist_ok=True)
    setup_logger(args.log)

    logging.info("Loading template synthetic SNPs and real MAF/LD")
    X_template = pd.read_csv(args.x_template)
    X_template = normalize_id(X_template)
    sample_ids = X_template['Sample_ID'].astype(str).tolist()
    maf_map = load_maf_map(args.real_maf)
    with open(args.ld_blocks, 'r', encoding='utf-8') as f:
        ld_info = json.load(f)
    blocks: List[List[str]] = ld_info.get('blocks', [])

    # Determine SNP set
    template_cols = [c for c in X_template.columns if c != 'Sample_ID']
    real_cols = list(maf_map.keys())
    if use_intersect:
        snp_set = [c for c in template_cols if c in real_cols]
    else:
        snp_set = template_cols
    logging.info(f"SNP columns selected: {len(snp_set)}")
    if len(snp_set) == 0:
        logging.error("No SNPs selected for regeneration")
        raise SystemExit(1)

    # Filter blocks to selected SNPs
    blocks_filtered = []
    in_set = set(snp_set)
    for b in blocks:
        bf = [c for c in b if c in in_set]
        if len(bf) > 0:
            blocks_filtered.append(bf)

    # Any SNPs not covered by blocks become singletons
    covered = set([c for b in blocks_filtered for c in b])
    singletons = [c for c in snp_set if c not in covered]
    for s in singletons:
        blocks_filtered.append([s])

    rng = np.random.default_rng(args.seed)
    n = len(sample_ids)
    out_arr = np.zeros((n, len(snp_set)), dtype=int)
    col_index = {c: i for i, c in enumerate(snp_set)}

    logging.info(f"Generating genotypes: {len(blocks_filtered)} blocks, {len(singletons)} singletons, n_samples={n}")
    # Generate per block
    for b in blocks_filtered:
        p_list = [maf_map.get(c, 0.1) for c in b]
        # clamp p to [0.001, 0.499]
        p_list = [float(np.clip(p, 1e-3, 0.499)) for p in p_list]
        if args.ld_within_block == 'preserve_pairwise_freq' and len(b) > 1:
            try:
                g_block = gaussian_copula_block(p_list, n, rng)
            except Exception:
                g_block = np.stack([hw_genotype_sample(p, n, rng) for p in p_list], axis=1)
        else:
            g_block = np.stack([hw_genotype_sample(p, n, rng) for p in p_list], axis=1)
        # place into out_arr
        for j, col in enumerate(b):
            out_arr[:, col_index[col]] = g_block[:, j]

    # MAF adjustment pass
    logging.info("Adjusting MAF to within tolerance where needed")
    for col in snp_set:
        idx = col_index[col]
        target_maf = float(np.clip(maf_map.get(col, 0.1), 1e-3, 0.499))
        out_arr[:, idx] = adjust_maf_towards_target(out_arr[:, idx], target_maf, args.maf_tolerance, rng)

    # Build output dataframe
    out_df = pd.DataFrame(out_arr, columns=snp_set)
    out_df.insert(0, 'Sample_ID', sample_ids)
    out_df.to_csv(args.output_snps, index=False)
    logging.info(f"Wrote regenerated synthetic SNPs: {args.output_snps}")

    # Report compliance
    denom = 2.0 * n
    cur_p = out_df[snp_set].sum(axis=0).values / denom
    cur_maf = np.minimum(cur_p, 1.0 - cur_p)
    tgt_maf = np.array([float(np.clip(maf_map.get(c, 0.1), 1e-3, 0.499)) for c in snp_set])
    drift = np.abs(cur_maf - tgt_maf)
    mean_drift = float(np.mean(drift))
    p95_drift = float(np.percentile(drift, 95))
    pass_rate = float(np.mean(drift <= args.maf_tolerance))
    logging.info(f"MAF drift mean={mean_drift:.6f}, p95={p95_drift:.6f}, within_tol_rate={pass_rate:.3f}")


if __name__ == '__main__':
    main()