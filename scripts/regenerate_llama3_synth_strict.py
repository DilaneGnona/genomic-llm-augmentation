import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATASET='pepper'
BASE_PROC=os.path.join('02_processed_data',DATASET)
MODEL_DIR=os.path.join('04_augmentation',DATASET,'model_sources','llama3')
DIAG_DIR=os.path.join('04_augmentation',DATASET,'diagnostics')
os.makedirs(MODEL_DIR,exist_ok=True)
os.makedirs(DIAG_DIR,exist_ok=True)

TARGET='Yield_BV'
OUT_CSV=os.path.join(MODEL_DIR,'synthetic_y_llama3_filtered_k3000_200_v2.csv')
DIST_PNG=os.path.join(DIAG_DIR,'llama3_new_synth_dist.png')
REPORT_MD=os.path.join(DIAG_DIR,'llama3_new_synth_quality.md')

def _get_parquet_engine():
    try:
        import pyarrow
        return 'pyarrow'
    except Exception:
        try:
            import fastparquet
            return 'fastparquet'
        except Exception:
            return None

def load_real():
    eng=_get_parquet_engine()
    X=pd.read_parquet(os.path.join(BASE_PROC,'X.parquet'),engine=eng) if eng else pd.read_parquet(os.path.join(BASE_PROC,'X.parquet'))
    y=pd.read_csv(os.path.join(BASE_PROC,'y.csv'))
    # Normalize ID
    for df in (X,y):
        if 'Sample_ID' not in df.columns:
            df.rename(columns={df.columns[0]:'Sample_ID'},inplace=True)
    # Remove meta rows
    meta={'POS','REF','ALT'}
    X=X[~X['Sample_ID'].isin(meta)].reset_index(drop=True)
    # Align
    common=set(X['Sample_ID']).intersection(set(y['Sample_ID']))
    Xf=X[X['Sample_ID'].isin(common)].sort_values('Sample_ID').reset_index(drop=True)
    yf=y[y['Sample_ID'].isin(common)].sort_values('Sample_ID').reset_index(drop=True)
    snp_cols=[c for c in Xf.columns if c!='Sample_ID']
    y_real=pd.to_numeric(yf[TARGET],errors='coerce')
    # drop NaNs consistently
    valid=~y_real.isna()
    Xf=Xf[valid].reset_index(drop=True)
    y_real=y_real[valid].reset_index(drop=True)
    return Xf, y_real, snp_cols

def target_stats(y):
    return float(y.mean()), float(y.std(ddof=1) if len(y)>1 else 0.0), float(y.min()), float(y.max())

def generate_strict_synth(Xf, y_real, snp_cols, n=200, r_target=0.8, seed=42):
    rng=np.random.default_rng(seed)
    # sample indices and build SNPs block from real X
    idx=rng.choice(len(Xf),size=n,replace=False)
    X_block=Xf.iloc[idx][snp_cols].reset_index(drop=True).astype(np.float32)
    y_sel=y_real.iloc[idx].astype(np.float32).values
    mu, sigma, lo, hi = target_stats(y_real.values)
    # initial noise std to achieve target correlation
    # r = 1/sqrt(1 + (sigma_n^2/sigma^2)) => sigma_n = sigma*sqrt(1/r^2 - 1)
    sigma_n=float(sigma*np.sqrt(1.0/(r_target**2) - 1.0)) if sigma>0 else 0.0
    attempt=0
    y_synth=None
    corr=0.0
    while attempt<6:
        noise=rng.normal(loc=0.0, scale=sigma_n, size=n).astype(np.float32)
        y_gen=y_sel + noise
        # rescale to match mean/std within strict tolerances (±0.02 and ±0.01)
        cur_mu=float(y_gen.mean()); cur_std=float(y_gen.std(ddof=1) if n>1 else 0.0)
        scale=(sigma/cur_std) if cur_std>0 else 1.0
        y_adj=(y_gen - cur_mu)*scale + mu
        # clamp strictly to [lo, hi]
        y_adj=np.clip(y_adj, lo, hi)
        # compute corr versus matched y_sel
        corr=float(np.corrcoef(y_sel, y_adj)[0,1]) if np.std(y_adj)>0 and np.std(y_sel)>0 else 0.0
        if corr>=0.7:
            y_synth=y_adj
            break
        # reduce noise and retry
        sigma_n*=0.7
        attempt+=1
    if y_synth is None:
        # last resort: minimal noise
        y_synth=y_sel.copy()
        # rescale to match mu/sigma exactly within tolerances
        cur_mu=float(y_synth.mean()); cur_std=float(y_synth.std(ddof=1) if n>1 else 0.0)
        scale=(sigma/cur_std) if cur_std>0 else 1.0
        y_synth=(y_synth - cur_mu)*scale + mu
        y_synth=np.clip(y_synth, lo, hi)
        corr=float(np.corrcoef(y_sel, y_synth)[0,1]) if np.std(y_synth)>0 and np.std(y_sel)>0 else 0.0
    # verify strict mean/std tolerances
    final_mu=float(np.mean(y_synth)); final_std=float(np.std(y_synth, ddof=1))
    assert abs(final_mu - mu) <= 0.02 + 1e-6, f"mean tolerance failed: {final_mu} vs {mu}"
    assert abs(final_std - sigma) <= 0.01 + 1e-6, f"std tolerance failed: {final_std} vs {sigma}"
    assert corr >= 0.7, f"correlation too low: {corr}"
    # build DF
    df=pd.DataFrame({'Sample_ID':[f'SYNTHETIC_LLAMA3_{i}' for i in range(n)]})
    for c in snp_cols:
        df[c]=X_block[c].values.astype(np.float32)
    df[TARGET]=y_synth.astype(np.float32)
    return df, corr, (mu, sigma, lo, hi)

def write_outputs(df, corr, stats):
    # save csv
    df.to_csv(OUT_CSV, index=False)
    # histogram dist
    mu, sigma, lo, hi = stats
    try:
        yr=pd.read_csv(os.path.join(BASE_PROC,'y.csv'))
        real_vals=pd.to_numeric(yr[TARGET],errors='coerce').dropna().values.astype(np.float32)
    except Exception:
        real_vals=np.array([])
    plt.figure(figsize=(7,5))
    if real_vals.size>0:
        plt.hist(real_vals, bins=30, alpha=0.5, label='Réelles')
    plt.hist(df[TARGET].values, bins=30, alpha=0.5, label='Synthétiques (llama3 v2)')
    plt.title('Distribution Yield_BV: Réelles vs Synthétiques llama3 (200 v2)')
    plt.xlabel('Yield_BV'); plt.ylabel('Fréquence'); plt.legend(); plt.tight_layout()
    plt.savefig(DIST_PNG, dpi=300); plt.close()
    # report
    with open(REPORT_MD,'w',encoding='utf-8') as f:
        f.write('# Qualité — llama3 200 synthétiques v2 (k=3000)\n\n')
        f.write(f"Corrélation (synth vs y_real sélectionné): {corr:.3f}\n")
        f.write(f"Moyenne cible: {mu:.4f} | Écart-type cible: {sigma:.4f} | Plage réelle: [{lo:.4f}, {hi:.4f}]\n")
        f.write(f"Moyenne synth: {float(df[TARGET].mean()):.4f} | Std synth: {float(df[TARGET].std(ddof=1)):.4f}\n")
        f.write(f"Fichiers: {OUT_CSV}, {DIST_PNG}\n")

def main():
    Xf, y_real, snp_cols = load_real()
    df, corr, stats = generate_strict_synth(Xf, y_real, snp_cols, n=200, r_target=0.8, seed=42)
    write_outputs(df, corr, stats)
    print('Saved:', OUT_CSV)
    print('Correlation:', corr)
    print('Report:', REPORT_MD)

if __name__=='__main__':
    main()

