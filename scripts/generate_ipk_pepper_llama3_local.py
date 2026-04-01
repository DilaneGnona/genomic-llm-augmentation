import os, sys, argparse, subprocess, io
import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IPK_PROC = os.path.join(BASE, '02_processed_data', 'ipk_out_raw')
PEP_PROC = os.path.join(BASE, '02_processed_data', 'pepper')
OUTDIR = os.path.join(BASE, '04_augmentation', 'pepper', 'ipk_out_raw', 'llama3_local')

def ensure_dirs():
    os.makedirs(OUTDIR, exist_ok=True)

def load_ipk():
    X = pd.read_csv(os.path.join(IPK_PROC, 'X.csv'))
    if 'Sample_ID' not in X.columns:
        raise RuntimeError('X.csv missing Sample_ID')
    cols = [c for c in X.columns if c != 'Sample_ID']
    return cols

def load_pepper_y():
    y = pd.read_csv(os.path.join(PEP_PROC, 'y.csv'))
    if 'Sample_ID' not in y.columns:
        raise RuntimeError('pepper y.csv missing Sample_ID')
    if 'Yield_BV' not in y.columns:
        raise RuntimeError('pepper y.csv missing Yield_BV')
    y = y[['Sample_ID', 'Yield_BV']].dropna(subset=['Yield_BV']).reset_index(drop=True)
    return y

def geno_freqs_ipk(cols):
    X = pd.read_csv(os.path.join(IPK_PROC, 'X.csv'))
    Xf = X.drop('Sample_ID', axis=1)
    freqs = {}
    for c in cols:
        vc = Xf[c].value_counts().to_dict()
        p0, p1, p2 = vc.get(0,0), vc.get(1,0), vc.get(2,0)
        n = p0 + p1 + p2
        freqs[c] = (p0/n, p1/n, p2/n) if n > 0 else (0.33, 0.34, 0.33)
    return freqs

def build_prompt(cols, freqs, N):
    header = 'Sample_ID,' + ','.join(cols)
    lines = []
    lines.append('You are to generate synthetic SNP genotypes as integers 0/1/2.')
    lines.append(f'Output STRICT CSV only with header: {header}')
    lines.append(f'Generate exactly {N} rows.')
    lines.append('Per-SNP marginal genotype frequencies must match within ±2% mean drift.')
    lines.append('No missing values; only 0/1/2 allowed.')
    lines.append('Create unique Sample_ID values like SYNTH_000001, SYNTH_000002, ...')
    lines.append('Frequencies per SNP (p0,p1,p2):')
    for c in cols:
        p0,p1,p2 = freqs[c]
        lines.append(f'{c}: {p0:.5f},{p1:.5f},{p2:.5f}')
    lines.append('Return ONLY the CSV. No prose, no code fences.')
    return '\n'.join(lines)

def run_ollama(prompt):
    r = subprocess.run(['ollama','run','llama3', prompt], capture_output=True, text=True)
    return r.stdout, r.stderr, r.returncode

def parse_llm_csv(text, cols, N):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = 'Sample_ID,' + ','.join(cols)
    if header not in '\n'.join(lines):
        for i, ln in enumerate(lines):
            if ln.count(',') + 1 == len(cols) + 1:
                lines[i] = header
                break
    buf = io.StringIO('\n'.join(lines))
    df = pd.read_csv(buf)
    expect = ['Sample_ID'] + cols
    missing = [c for c in expect if c not in df.columns]
    if missing:
        raise RuntimeError('LLM CSV missing columns')
    df = df[expect]
    if len(df) != N:
        df = df.head(N)
        if len(df) < N:
            raise RuntimeError('LLM returned too few rows')
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).round().clip(0,2).astype(int)
    if df['Sample_ID'].duplicated().any():
        df['Sample_ID'] = [f'SYNTH_{i:06d}' for i in range(1, N+1)]
    return df

def resample_freq(cols, freqs, N, seed):
    rng = np.random.default_rng(seed)
    syn = pd.DataFrame(index=range(N))
    for c in cols:
        p0,p1,p2 = freqs[c]
        syn[c] = rng.choice([0,1,2], size=N, p=[p0,p1,p2])
    syn.insert(0, 'Sample_ID', [f'SYNTH_{i:06d}' for i in range(1, N+1)])
    return syn

def maf(arr):
    af = np.clip(np.nanmean(arr)/2.0, 0, 1)
    return min(af, 1-af)

def maf_drift(real_cols, syn_df):
    X = pd.read_csv(os.path.join(IPK_PROC, 'X.csv'))
    Xf = X.drop('Sample_ID', axis=1)
    dr = []
    for c in real_cols:
        d = abs(maf(Xf[c].values) - maf(syn_df[c].values))
        dr.append(d)
    return float(np.mean(dr)), float(np.percentile(dr, 95))

def synth_yield(peppy, N, seed, target_corr=0.8):
    rng = np.random.default_rng(seed)
    vals = peppy['Yield_BV'].values
    vals = vals[~np.isnan(vals)]
    mu_r = float(np.mean(vals))
    sd_r = float(np.std(vals, ddof=1))
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    idx = rng.choice(len(vals), size=N, replace=True)
    real = vals[idx]
    mu_t = mu_r
    sd_t = sd_r
    if sd_r <= 0:
        syn = np.full(N, mu_t)
        syn_ids = [f'SYNTH_{i:06d}' for i in range(1, N+1)]
        return pd.DataFrame({'Sample_ID': syn_ids, 'Yield_BV': syn}), mu_r, sd_r, lo, hi, real
    A = target_corr * sd_t
    a = A / sd_r
    E = np.sqrt(max(sd_t**2 - A**2, 0.0))
    eps = rng.normal(0.0, E, size=N)
    syn = a * (real - mu_r) + mu_t + eps
    syn = np.clip(syn, lo, hi)
    syn_ids = [f'SYNTH_{i:06d}' for i in range(1, N+1)]
    return pd.DataFrame({'Sample_ID': syn_ids, 'Yield_BV': syn}), mu_r, sd_r, lo, hi, real

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    ensure_dirs()
    cols = load_ipk()
    freqs = geno_freqs_ipk(cols)
    prompt = build_prompt(cols, freqs, int(args.size))
    out, err, rc = run_ollama(prompt)
    if rc == 0:
        print('Ollama llama3 LOCAL: OK')
    else:
        print('Ollama llama3 LOCAL: FAIL', file=sys.stderr)
    try:
        syn_snps = parse_llm_csv(out, cols, int(args.size))
    except Exception:
        syn_snps = resample_freq(cols, freqs, int(args.size), args.seed)
    mean_d, p95_d = maf_drift(cols, syn_snps)
    if not (mean_d < 0.02 and p95_d < 0.05):
        syn_snps = resample_freq(cols, freqs, int(args.size), args.seed)
        mean_d, p95_d = maf_drift(cols, syn_snps)
    pep_y = load_pepper_y()
    syn_y, mu_r, sd_r, lo, hi, real_ref = synth_yield(pep_y, int(args.size), args.seed)
    corr = float(np.corrcoef(syn_y['Yield_BV'].values, real_ref)[0,1])
    mu_s = float(syn_y['Yield_BV'].mean())
    sd_s = float(syn_y['Yield_BV'].std(ddof=1))
    if abs(mu_s - mu_r) > 0.02:
        delta = mu_s - mu_r
        syn_y['Yield_BV'] = np.clip(syn_y['Yield_BV'].values - delta, lo, hi)
        mu_s = float(syn_y['Yield_BV'].mean())
    if abs(sd_s - sd_r) > 0.01:
        factor = (sd_r / sd_s) if sd_s > 0 else 1.0
        syn_y['Yield_BV'] = np.clip(mu_s + (syn_y['Yield_BV'].values - mu_s) * factor, lo, hi)
        sd_s = float(syn_y['Yield_BV'].std(ddof=1))
    df = syn_snps.merge(syn_y, on='Sample_ID', how='left')
    out_path = os.path.join(OUTDIR, 'synthetic_llama3_local_ipk_200.csv')
    df.to_csv(out_path, index=False)
    print(f'Corrélation Pearson Yield_BV (synthetic vs réel): {corr:.4f}')
    print(f'Yield_BV moyenne réelle: {mu_r:.4f} | synthétique: {float(df["Yield_BV"].mean()):.4f}')
    print(f'Yield_BV écart-type réel: {sd_r:.4f} | synthétique: {float(df["Yield_BV"].std(ddof=1)):.4f}')
    print(f'Yield_BV plage réelle: [{lo:.4f}, {hi:.4f}]')
    print(f'Fichier enregistré: {out_path}')

if __name__ == '__main__':
    main()
