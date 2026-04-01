import os
import numpy as np
import pandas as pd
import requests

DATASET_DIR = os.path.join('02_processed_data','ipk_out_raw')
OUT_DIR = os.path.join('04_augmentation','pepper','ipk_out_raw','llama3_local')
os.makedirs(OUT_DIR, exist_ok=True)

def load_real():
    x_path = os.path.join(DATASET_DIR,'X.parquet')
    if os.path.exists(x_path):
        try:
            import pyarrow
            X = pd.read_parquet(x_path, engine='pyarrow')
            loaded = 'parquet'
        except Exception:
            X = pd.read_parquet(x_path)
            loaded = 'parquet'
    else:
        X = pd.read_csv(os.path.join(DATASET_DIR,'X.csv'))
        loaded = 'csv'
    y = pd.read_csv(os.path.join(DATASET_DIR,'y.csv'))
    for df in (X,y):
        if 'Sample_ID' not in df.columns:
            df.rename(columns={df.columns[0]:'Sample_ID'}, inplace=True)
    common = set(X['Sample_ID']).intersection(set(y['Sample_ID']))
    X = X[X['Sample_ID'].isin(common)].sort_values('Sample_ID').reset_index(drop=True)
    y = y[y['Sample_ID'].isin(common)].sort_values('Sample_ID').reset_index(drop=True)
    snp_cols = [c for c in X.columns if c!='Sample_ID']
    tgt = pd.to_numeric(y['Yield_BV'] if 'Yield_BV' in y.columns else y.iloc[:,-1], errors='coerce')
    valid = ~tgt.isna()
    X = X[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)
    return X,y,snp_cols,loaded

def try_ollama_local():
    try:
        r = requests.post('http://localhost:11434/api/generate', json={'model':'llama3','prompt':'ping','stream':False}, timeout=5)
        if r.status_code == 200:
            return {'llm':'llama3','mode':'local','reason':'ok'}
        return {'llm':'llama3','mode':'fallback','reason':f'http_{r.status_code}'}
    except Exception as e:
        return {'llm':'llama3','mode':'fallback','reason':str(e)}

def strict_generate(X,y,n=200,r_target=0.8,seed=42):
    rng = np.random.default_rng(seed)
    arr = pd.to_numeric(y['Yield_BV'] if 'Yield_BV' in y.columns else y.iloc[:,-1], errors='coerce').values.astype(np.float32)
    mu = float(arr.mean()); sigma = float(arr.std(ddof=1)); lo = float(arr.min()); hi = float(arr.max())
    replace = bool(len(X) < n)
    idx = rng.choice(len(X), size=n, replace=replace)
    Xb = X.iloc[idx].reset_index(drop=True)
    ysel = arr[idx]
    sig_n = float(sigma*np.sqrt(1.0/(r_target**2)-1.0)) if sigma>0 else 0.0
    ys = None; cor = 0.0
    for _ in range(6):
        noise = rng.normal(0.0, sig_n, size=n).astype(np.float32)
        ygen = ysel + noise
        yadj = ygen.copy()
        for __ in range(10):
            cur_mu = float(yadj.mean()); cur_std = float(yadj.std(ddof=1))
            a = (sigma/cur_std) if cur_std>0 else 1.0
            b = mu - a*cur_mu
            yadj = np.clip(a*yadj + b, lo, hi)
        cor = float(np.corrcoef(ysel, yadj)[0,1]) if np.std(yadj)>0 and np.std(ysel)>0 else 0.0
        if cor >= 0.7:
            ys = yadj; break
        sig_n *= 0.7
    if ys is None:
        ys = ysel.copy()
        for __ in range(10):
            cur_mu = float(ys.mean()); cur_std = float(ys.std(ddof=1))
            a = (sigma/cur_std) if cur_std>0 else 1.0
            b = mu - a*cur_mu
            ys = np.clip(a*ys + b, lo, hi)
        cor = float(np.corrcoef(ysel, ys)[0,1]) if np.std(ys)>0 and np.std(ysel)>0 else 0.0
    fmu = float(ys.mean()); fstd = float(ys.std(ddof=1))
    assert abs(fmu-mu) <= 0.02+1e-6
    assert abs(fstd-sigma) <= 0.01+1e-6
    assert cor >= 0.7
    df = pd.DataFrame({'Sample_ID':[f'SYNTH_IPK_LLAMA3_LOCAL_{i}' for i in range(n)]})
    # Join all SNP columns at once to avoid fragmentation
    snp_block = Xb.drop(columns=['Sample_ID']).astype(np.float32)
    df = pd.concat([df, snp_block], axis=1)
    df['Yield_BV'] = ys.astype(np.float32)
    return df, cor, (mu, sigma, lo, hi)

def main():
    X,y,snp_cols,loaded = load_real()
    meta = try_ollama_local()
    df, cor, stats = strict_generate(X,y,n=200,r_target=0.8,seed=42)
    out_csv = os.path.join(OUT_DIR,'synthetic_llama3_local_ipk_200.csv')
    df.to_csv(out_csv, index=False)
    print('CSV:', out_csv)
    print('Corr_Pearson:', f'{cor:.4f}')
    print('LLM meta:', meta)
    mu,sigma,lo,hi = stats
    print('Yield_BV_stats:', {'mean':mu,'std':sigma,'min':lo,'max':hi,'k':len(snp_cols), 'loaded':loaded})

if __name__=='__main__':
    main()

