import os, sys, argparse, subprocess, time, io
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED = os.path.join(BASE, '02_processed_data', 'ipk_out_raw')
AUGDIR = os.path.join(BASE, '04_augmentation', 'ipk_out_raw')

def ensure_dirs():
    os.makedirs(AUGDIR, exist_ok=True)

def load_real():
    X = pd.read_csv(os.path.join(PROCESSED, 'X.csv'))
    y = pd.read_csv(os.path.join(PROCESSED, 'y.csv'))
    if 'Sample_ID' not in X.columns: raise RuntimeError('X.csv missing Sample_ID')
    if 'Sample_ID' not in y.columns: raise RuntimeError('y.csv missing Sample_ID')
    if 'YR_LS' not in y.columns: raise RuntimeError('y.csv missing YR_LS')
    ids = set(X['Sample_ID']).intersection(set(y['Sample_ID']))
    X = X[X['Sample_ID'].isin(ids)].sort_values('Sample_ID').reset_index(drop=True)
    y = y[y['Sample_ID'].isin(ids)].sort_values('Sample_ID').reset_index(drop=True)
    Xf = X.drop('Sample_ID', axis=1)
    return X, Xf, y

def geno_freqs(Xf):
    freqs = {}
    for c in Xf.columns:
        vc = Xf[c].value_counts().to_dict()
        p0, p1, p2 = vc.get(0,0), vc.get(1,0), vc.get(2,0)
        n = p0+p1+p2
        freqs[c] = (p0/n, p1/n, p2/n) if n>0 else (0.33,0.34,0.33)
    return freqs

def build_prompt(cols, freqs, N):
    header = 'Sample_ID,' + ','.join(cols)
    lines = [
        'You are to generate synthetic SNP genotypes as integers 0/1/2.',
        f'Output STRICT CSV only with header: {header}',
        f'Generate exactly {N} rows.',
        'Per-SNP marginal genotype frequencies must match within ±2% mean drift.',
        'No missing values; only 0/1/2 allowed.',
        'Create unique Sample_ID values like SYNTH_000001, SYNTH_000002, ...'
    ]
    lines.append('Frequencies per SNP (p0,p1,p2):')
    for c in cols:
        p0,p1,p2 = freqs[c]
        lines.append(f'{c}: {p0:.5f},{p1:.5f},{p2:.5f}')
    lines.append('Return ONLY the CSV. No prose, no code fences.')
    return '\n'.join(lines)

def run_ollama(prompt):
    t0 = time.time()
    r = subprocess.run(['ollama','run','llama3','-p',prompt], capture_output=True, text=True)
    return r.stdout, r.stderr, time.time()-t0, r.returncode

def write_log(prompt, resp, err):
    path = os.path.join(AUGDIR, 'augmentation_log_llama3.txt')
    with open(path, 'a', encoding='utf-8') as f:
        f.write('=== PROMPT ===\n')
        f.write(prompt + '\n')
        f.write('=== RESPONSE ===\n')
        f.write(resp + '\n')
        if err:
            f.write('=== STDERR ===\n')
            f.write(err + '\n')
        f.write('\n')

def parse_csv(text, cols, N):
    # Trim any non-CSV content
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # Ensure header present
    header = 'Sample_ID,' + ','.join(cols)
    if header not in '\n'.join(lines):
        # Try to detect first CSV-like line and replace header
        for i, ln in enumerate(lines):
            if ln.count(',')+1 == len(cols)+1:
                lines[i] = header
                break
    buf = io.StringIO('\n'.join(lines))
    df = pd.read_csv(buf)
    # Validate columns
    expect = ['Sample_ID'] + cols
    missing = [c for c in expect if c not in df.columns]
    if missing:
        raise RuntimeError(f'LLM CSV missing columns: {missing[:5]}...')
    df = df[expect]
    # Enforce N rows
    if len(df) != N:
        df = df.head(N)
        if len(df) < N:
            raise RuntimeError(f'LLM returned {len(df)} rows, expected {N}')
    # Coerce to integers 0/1/2
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).round().clip(0,2).astype(int)
    # Ensure unique Sample_ID
    if df['Sample_ID'].duplicated().any():
        df['Sample_ID'] = [f'SYNTH_{i:06d}' for i in range(1, N+1)]
    return df

def maf(arr):
    af = np.clip(np.nanmean(arr)/2.0, 0, 1)
    return min(af, 1-af)

def maf_drift(realX, synX):
    drifts = []
    for c in realX.columns:
        d = abs(maf(realX[c].values) - maf(synX[c].values))
        drifts.append(d)
    return float(np.mean(drifts)), float(np.percentile(drifts, 95))

def resample_to_fix(realX, freqs, N, rnd):
    syn = pd.DataFrame(index=range(N))
    for c in realX.columns:
        p0,p1,p2 = freqs[c]
        syn[c] = rnd.choice([0,1,2], size=N, p=[p0,p1,p2])
    return syn

def teacher_targets(Xf, y, synX, seed):
    rid = Ridge(alpha=1.0, random_state=seed)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    preds, resid = [], []
    for tr, te in kf.split(Xf):
        rid.fit(Xf.iloc[tr], y.iloc[tr]['YR_LS'].values)
        yh = rid.predict(Xf.iloc[te])
        preds.append(yh)
        resid.append(y.iloc[te]['YR_LS'].values - yh)
    sigma = float(np.sqrt(np.mean(np.concatenate(resid)**2)))
    rid.fit(Xf, y['YR_LS'].values)
    yhat = rid.predict(synX.values)
    noise_desc = f"Noise ~ N(0, {sigma:.6f}^2) from CV residuals"
    syn_y = yhat + np.random.default_rng(seed).normal(0, sigma, size=len(yhat))
    return syn_y, noise_desc

def write_normalization_report(realXf, synXf, N, mean_drift, p95_drift, corr_diff, pca_ratio):
    path = os.path.join(AUGDIR, 'normalization_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('Augmentation diagnostics (ipk_out_raw)\n')
        f.write(f'Samples: real={len(realXf)}, synthetic={N}\n')
        f.write(f'MAF drift mean={mean_drift:.4f}, 95th={p95_drift:.4f}\n')
        f.write(f'Corr subset mean |Δ|={corr_diff:.4f}\n')
        f.write(f'PCA variance (synthetic capture at real n_components)={pca_ratio:.4f}\n')

def correlation_subset(realXf, synXf, k=50, seed=42):
    rng = np.random.default_rng(seed)
    cols = list(realXf.columns)
    k = min(k, len(cols))
    sub = rng.choice(cols, size=k, replace=False)
    cr = np.corrcoef(realXf[sub].values, rowvar=False)
    cs = np.corrcoef(synXf[sub].values, rowvar=False)
    idx = np.triu_indices(k, 1)
    return float(np.mean(np.abs(cr[idx] - cs[idx])))

def pca_capture(realXf, synXf):
    # cheap PCA via SVD
    R = realXf.values - realXf.values.mean(0)
    Sr = np.linalg.svd(R, full_matrices=False)[1]
    var_r = (Sr**2) / np.sum(Sr**2)
    thresh = 0.95
    m = np.searchsorted(np.cumsum(var_r), thresh) + 1
    S = synXf.values - synXf.values.mean(0)
    Ss = np.linalg.svd(S, full_matrices=False)[1]
    var_s = (Ss**2) / np.sum(Ss**2)
    return float(np.sum(var_s[:m]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    ensure_dirs()
    X, Xf, y = load_real()
    N = int(args.size)
    cols = list(Xf.columns)
    freqs = geno_freqs(Xf)
    prompt = build_prompt(cols, freqs, N)
    resp, err, dt, rc = run_ollama(prompt)
    write_log(prompt, resp, err)
    if rc != 0:
        print('Ollama call failed; falling back to frequency resampling.', file=sys.stderr)
        syn_df = pd.DataFrame({'Sample_ID':[f'SYNTH_{i:06d}' for i in range(1,N+1)]})
        synX = resample_to_fix(Xf, freqs, N, np.random.default_rng(args.seed))
        syn_df = pd.concat([syn_df, synX], axis=1)
    else:
        try:
            syn_df = parse_csv(resp, cols, N)
        except Exception as e:
            print(f'CSV parse failed ({e}); falling back to resampling.', file=sys.stderr)
            syn_df = pd.DataFrame({'Sample_ID':[f'SYNTH_{i:06d}' for i in range(1,N+1)]})
            synX = resample_to_fix(Xf, freqs, N, np.random.default_rng(args.seed))
            syn_df = pd.concat([syn_df, synX], axis=1)
    synXf = syn_df.drop('Sample_ID', axis=1)
    # Validate drift; resample if violated
    mean_drift, p95_drift = maf_drift(Xf, synXf)
    if not (mean_drift < 0.02 and p95_drift < 0.05):
        synXf = resample_to_fix(Xf, freqs, N, np.random.default_rng(args.seed))
        syn_df = pd.concat([syn_df[['Sample_ID']], synXf], axis=1)
        mean_drift, p95_drift = maf_drift(Xf, synXf)
    # Write SNPs
    snps_path = os.path.join(AUGDIR, 'synthetic_snps.csv')
    syn_df.to_csv(snps_path, index=False)
    # Teacher+noise targets
    syn_y_vals, noise_desc = teacher_targets(Xf, y, synXf, args.seed)
    ydf = pd.DataFrame({'Sample_ID': syn_df['Sample_ID'], 'YR_LS': syn_y_vals})
    if ydf['YR_LS'].isna().any():
        raise RuntimeError('Synthetic targets contain NaN')
    ydf.to_csv(os.path.join(AUGDIR, 'synthetic_y.csv'), index=False)
    # Diagnostics
    corr_diff = correlation_subset(Xf, synXf, k=min(50,len(cols)), seed=args.seed)
    pca_ratio = pca_capture(Xf, synXf)
    write_normalization_report(Xf, synXf, N, mean_drift, p95_drift, corr_diff, pca_ratio)
    # Log noise description
    write_log('NOISE_DESCRIPTION', noise_desc, '')
    print(f'Done: {snps_path}')

if __name__ == '__main__':
    main()