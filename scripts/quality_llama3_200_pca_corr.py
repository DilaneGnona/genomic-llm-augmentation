import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATASET='pepper'
BASE_PROC=os.path.join('02_processed_data',DATASET)
QUAL_DIR=os.path.join('03_modeling_results','quality_analysis')
os.makedirs(QUAL_DIR,exist_ok=True)

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
    # Align IDs
    common=set(X['Sample_ID']).intersection(set(y['Sample_ID']))
    Xf=X[X['Sample_ID'].isin(common)].sort_values('Sample_ID').reset_index(drop=True)
    yf=y[y['Sample_ID'].isin(common)].sort_values('Sample_ID').reset_index(drop=True)
    snp_cols=[c for c in Xf.columns if c!='Sample_ID']
    Xs=Xf[snp_cols].astype(np.float32).values
    yr=pd.to_numeric(yf['Yield_BV'],errors='coerce').values.astype(np.float32)
    return Xs, yr, snp_cols

def load_llama3_synth_200(snp_cols):
    p=os.path.join('04_augmentation','pepper','model_sources','llama3','synthetic_y_llama3_filtered_k3000_200.csv')
    df=pd.read_csv(p)
    if 'Sample_ID' not in df.columns:
        df.rename(columns={df.columns[0]:'Sample_ID'},inplace=True)
    # Ensure columns match
    cols=[c for c in df.columns if c in snp_cols]
    snps=df[cols].copy()
    for c in snp_cols:
        if c not in snps.columns:
            snps[c]=0.0
    snps=snps[snp_cols]
    Xs=snps.astype(np.float32).values
    yt=pd.to_numeric(df['Yield_BV'] if 'Yield_BV' in df.columns else df.iloc[:,-1],errors='coerce').values.astype(np.float32)
    return Xs, yt

def save_pca_plot(real_X, synth_X):
    X=np.vstack([real_X,synth_X])
    labels=np.array([0]*len(real_X)+[1]*len(synth_X))
    scaler=StandardScaler(with_mean=True,with_std=True)
    Xs=scaler.fit_transform(X)
    pca=PCA(n_components=2,random_state=42)
    comps=pca.fit_transform(Xs)
    plt.figure(figsize=(7,6))
    plt.scatter(comps[labels==0,0],comps[labels==0,1],c='blue',s=10,label='Réelles')
    plt.scatter(comps[labels==1,0],comps[labels==1,1],c='green',s=12,label='llama3 (200)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA des SNPs (k=3000) : Données Réelles vs Synthétiques llama3 (200)')
    plt.legend()
    out=os.path.join(QUAL_DIR,'llama3_200synth_pca.png')
    plt.tight_layout()
    plt.savefig(out,dpi=300)
    plt.close()
    return out

def save_corr_heatmap(y_real,y_synth):
    # Single-cell heatmap
    corr=np.corrcoef(y_real[np.isfinite(y_real)],y_synth[np.isfinite(y_synth)])[0,1]
    val=np.array([[corr]],dtype=np.float32)
    plt.figure(figsize=(3,3))
    plt.imshow(val,cmap='viridis',vmin=-1,vmax=1)
    plt.colorbar()
    plt.xticks([0],["Synthétiques llama3 (200)"])
    plt.yticks([0],["Données Réelles (Yield_BV)"])
    plt.title('Corrélation Yield_BV (k=3000) : Réelles vs Synthétiques llama3 (200)')
    for i in range(1):
        for j in range(1):
            plt.text(j,i,f"{corr:.3f}",ha='center',va='center',color='white')
    out=os.path.join(QUAL_DIR,'llama3_200synth_correlation.png')
    plt.tight_layout()
    plt.savefig(out,dpi=300)
    plt.close()
    return out,corr

def write_short_report(pca_path,corr):
    md=os.path.join(QUAL_DIR,'llama3_200synth_quality_short.md')
    with open(md,'w',encoding='utf-8') as f:
        f.write('# Qualité synthétiques llama3 (200, k=3000)\n\n')
        f.write('1. PCA: voir la figure `llama3_200synth_pca.png` — inspection visuelle du regroupement des points synthétiques par rapport aux réelles.\n')
        f.write(f'2. Corrélation Yield_BV (réelles vs synthétiques llama3 200): {corr:.3f}\n')
        concl = 'cohérents' if corr >= 0.3 else 'modérément cohérents'
        f.write(f'3. Conclusion: les synthétiques sont {concl} avec les réelles selon la corrélation univariée et la PCA.\n')
    return md

def main():
    real_X, y_real, snp_cols = load_real()
    synth_X, y_synth = load_llama3_synth_200(snp_cols)
    pca_path = save_pca_plot(real_X, synth_X)
    corr_path, corr = save_corr_heatmap(y_real, y_synth)
    md = write_short_report(pca_path, corr)
    print('PCA figure:', pca_path)
    print('Correlation figure:', corr_path)
    print('Report:', md)
    # Persist a simple log
    with open(os.path.join(QUAL_DIR,'llama3_200synth_quality_log.txt'),'w',encoding='utf-8') as f:
        f.write(f'PCA: {pca_path}\n')
        f.write(f'Correlation fig: {corr_path}\n')
        f.write(f'Correlation value: {corr:.4f}\n')
        f.write(f'Report: {md}\n')

if __name__=='__main__':
    main()
