import pandas as pd
import numpy as np
from scipy.stats import chisquare
import os
import logging

def check_hwe(genotypes):
    """
    Checks Hardy-Weinberg Equilibrium for a set of genotypes (0, 1, 2).
    HWE: p^2 + 2pq + q^2 = 1
    """
    n = len(genotypes)
    if n == 0: return np.nan
    
    # Counts
    n0 = np.sum(genotypes == 0)
    n1 = np.sum(genotypes == 1)
    n2 = np.sum(genotypes == 2)
    
    # Allele frequencies
    p = (2*n0 + n1) / (2*n)
    q = 1 - p
    
    # Expected counts
    e0 = (p**2) * n
    e1 = (2*p*q) * n
    e2 = (q**2) * n
    
    # Chi-square test
    obs = [n0, n1, n2]
    exp = [e0, e1, e2]
    
    # Avoid division by zero
    if any(e < 5 for e in exp): return np.nan
    
    chi, p_val = chisquare(obs, exp)
    return p_val

def biological_validation(dataset_path, output_report):
    logging.info(f"Starting Biological Validation for {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Filter only SNP columns (integers 0, 1, 2)
    snp_cols = [c for c in df.columns if df[c].dtype in [np.int64, np.float64] and df[c].isin([0, 1, 2, 0.0, 1.0, 2.0]).all()]
    
    if not snp_cols:
        logging.warning("No SNP columns found for validation.")
        return
    
    hwe_results = {}
    for col in snp_cols:
        hwe_results[col] = check_hwe(df[col].dropna())
    
    hwe_df = pd.DataFrame.from_dict(hwe_results, orient='index', columns=['p_value_HWE'])
    hwe_df['is_significant'] = hwe_df['p_value_HWE'] < 0.05
    
    # Summary
    n_total = len(hwe_df)
    n_valid = hwe_df['p_value_HWE'].notna().sum()
    n_passed = (hwe_df['p_value_HWE'] >= 0.05).sum()
    
    report = f"""
# Biological Validation Report: {os.path.basename(dataset_path)}
- Total SNPs analyzed: {n_total}
- Validated for HWE (min sample size): {n_valid}
- Passed HWE (p >= 0.05): {n_passed} ({ (n_passed/n_valid*100) if n_valid > 0 else 0 :.2f}%)
    
Note: A high percentage of SNPs passing HWE indicates that the synthetic data respects natural genetic inheritance patterns.
    """
    
    with open(output_report, "w") as f:
        f.write(report)
    
    logging.info(f"Validation report saved to {output_report}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example for pepper
    base = r"c:\Users\OMEN\Desktop\experiment_snp"
    pepper_synth = os.path.join(base, "02_processed_data", "pepper_augmented", "unified_dataset.csv")
    if os.path.exists(pepper_synth):
        biological_validation(pepper_synth, os.path.join(base, "article", "biological_validation_pepper.md"))
    
    ipk_synth = os.path.join(base, "02_processed_data", "ipk_out_raw_augmented", "unified_dataset.csv")
    if os.path.exists(ipk_synth):
        biological_validation(ipk_synth, os.path.join(base, "article", "biological_validation_ipk.md"))
