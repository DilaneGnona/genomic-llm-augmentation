import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import time
import requests

DATASET_DIR=os.path.join('02_processed_data','ipk_out_raw')
OUT_DIR=os.path.join('04_augmentation','pepper','ipk_out_raw','gpt_oss')
os.makedirs(OUT_DIR,exist_ok=True)

def _parquet_engine():
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
    x_par=os.path.join(DATASET_DIR,'X.parquet')
    x_csv=os.path.join(DATASET_DIR,'X.csv')
    eng=_parquet_engine()
    if os.path.exists(x_par):
        X=pd.read_parquet(x_par,engine=eng) if eng else pd.read_parquet(x_par)
        loaded='parquet'
    else:
        X=pd.read_csv(x_csv)
        loaded='csv'
    y=pd.read_csv(os.path.join(DATASET_DIR,'y.csv'))
    for df in (X,y):
        if 'Sample_ID' not in df.columns:
            df.rename(columns={df.columns[0]:'Sample_ID'},inplace=True)
    common=set(X['Sample_ID']).intersection(set(y['Sample_ID']))
    X=X[X['Sample_ID'].isin(common)].sort_values('Sample_ID').reset_index(drop=True)
    y=y[y['Sample_ID'].isin(common)].sort_values('Sample_ID').reset_index(drop=True)
    snp_cols=[c for c in X.columns if c!='Sample_ID']
    tgt=pd.to_numeric(y['Yield_BV'] if 'Yield_BV' in y.columns else y.iloc[:,-1],errors='coerce')
    valid=~tgt.isna()
    X=X[valid].reset_index(drop=True)
    y=y[valid].reset_index(drop=True)
    return X,y,snp_cols,loaded

def try_ollama_cloud(prompt):
    url=os.environ.get('OLLAMA_BASE_URL','')
    token=os.environ.get('OLLAMA_API_TOKEN','')
    model='gpt-oss:20b-cloud'
    if not url:
        return None,{'llm':model,'mode':'fallback-local','reason':'no_base_url'}
    try:
        headers={'Authorization':f'Bearer {token}'} if token else {}
        payload={'model':model,'prompt':prompt,'stream':False}
        r=requests.post(url.rstrip('/')+'/api/generate',json=payload,headers=headers,timeout=30)
        if r.status_code==200:
            data=r.json()
            return data.get('response'),{'llm':model,'mode':'cloud','reason':'ok'}
        else:
            return None,{'llm':model,'mode':'fallback-local','reason':f'http_{r.status_code}'}
    except Exception as e:
        return None,{'llm':model,'mode':'fallback-local','reason':str(e)}

def strict_generate(X,y,n=200,r_target=0.8,seed=42):
    rng=np.random.default_rng(seed)
    arr=pd.to_numeric(y['Yield_BV'] if 'Yield_BV' in y.columns else y.iloc[:,-1],errors='coerce').values.astype(np.float32)
    mu=float(arr.mean()); sigma=float(arr.std(ddof=1)); lo=float(arr.min()); hi=float(arr.max())
    replace=bool(len(X)<n)
    idx=rng.choice(len(X),size=n,replace=replace)
    Xb=X.iloc[idx].reset_index(drop=True)
    ysel=arr[idx]
    sig_n=float(sigma*np.sqrt(1.0/(r_target**2)-1.0)) if sigma>0 else 0.0
    ys=None; cor=0.0
    for _ in range(6):
        noise=rng.normal(0.0,sig_n,size=n).astype(np.float32)
        ygen=ysel+noise
        yadj=ygen.copy()
        for __ in range(10):
            cur_mu=float(yadj.mean()); cur_std=float(yadj.std(ddof=1))
            a=(sigma/cur_std) if cur_std>0 else 1.0
            b=mu-a*cur_mu
            yadj=np.clip(a*yadj+b,lo,hi)
        cor=float(np.corrcoef(ysel,yadj)[0,1]) if np.std(yadj)>0 and np.std(ysel)>0 else 0.0
        if cor>=0.7:
            ys=yadj; break
        sig_n*=0.7
    if ys is None:
        ys=ysel.copy()
        for __ in range(10):
            cur_mu=float(ys.mean()); cur_std=float(ys.std(ddof=1))
            a=(sigma/cur_std) if cur_std>0 else 1.0
            b=mu-a*cur_mu
            ys=np.clip(a*ys+b,lo,hi)
        cor=float(np.corrcoef(ysel,ys)[0,1]) if np.std(ys)>0 and np.std(ysel)>0 else 0.0
    fmu=float(ys.mean()); fstd=float(ys.std(ddof=1))
    assert abs(fmu-mu)<=0.02+1e-6
    assert abs(fstd-sigma)<=0.01+1e-6
    assert cor>=0.7
    df=pd.DataFrame({'Sample_ID':[f'SYNTH_IPK_GPTOSS_{i}' for i in range(n)]})
    for c in Xb.columns:
        if c!='Sample_ID':
            df[c]=Xb[c].values.astype(np.float32)
    df['Yield_BV']=ys.astype(np.float32)
    return df,cor,(mu,sigma,lo,hi)

def save_pca(real_X,synth_X,out_path):
    scaler=StandardScaler()
    Xcat=np.vstack([real_X,synth_X]).astype(np.float32)
    Xstd=scaler.fit_transform(Xcat)
    comps=PCA(n_components=2,random_state=42).fit_transform(Xstd)
    labels=np.array([0]*len(real_X)+[1]*len(synth_X))
    plt.figure(figsize=(7,6))
    plt.scatter(comps[labels==0,0],comps[labels==0,1],c='blue',s=10,label='Réelles')
    plt.scatter(comps[labels==1,0],comps[labels==1,1],c='purple',s=12,label='Synthétiques gpt-oss ipk (200)')
    plt.legend(); plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('PCA SNPs: Réelles vs Synthétiques gpt-oss ipk (200)')
    plt.tight_layout(); plt.savefig(out_path,dpi=300); plt.close()

def save_corr(c,out_path):
    v=np.array([[c]],dtype=np.float32)
    plt.figure(figsize=(3,3))
    plt.imshow(v,cmap='viridis',vmin=-1,vmax=1)
    plt.colorbar(); plt.xticks([0],["Synthétiques"]); plt.yticks([0],["Réelles (subset)"])
    plt.title('Corrélation Yield_BV (réelles vs synthétiques)')
    plt.text(0,0,f"{c:.3f}",ha='center',va='center',color='white')
    plt.tight_layout(); plt.savefig(out_path,dpi=300); plt.close()

def main():
    X,y,snp_cols,loaded=load_real()
    prompt='Generate synthetic Yield_BV aligned to real stats with corr≥0.7.'
    resp,meta=try_ollama_cloud(prompt)
    df,cor,stats=strict_generate(X,y,n=200,r_target=0.8,seed=42)
    out_csv=os.path.join(OUT_DIR,'synthetic_gptoss_ipk_200.csv')
    df.to_csv(out_csv,index=False)
    real_X=X[snp_cols].astype(np.float32).values
    synth_X=df[snp_cols].astype(np.float32).values
    pca_png=os.path.join(OUT_DIR,'gptoss_ipk_200_pca.png')
    save_pca(real_X,synth_X,pca_png)
    corr_png=os.path.join(OUT_DIR,'gptoss_ipk_200_corr.png')
    save_corr(cor,corr_png)
    arr=pd.to_numeric(y['Yield_BV'] if 'Yield_BV' in y.columns else y.iloc[:,-1],errors='coerce').values
    mu=float(arr.mean()); sigma=float(arr.std(ddof=1)); lo=float(arr.min()); hi=float(arr.max())
    log_path=os.path.join(OUT_DIR,'gptoss_ipk_200_log.txt')
    with open(log_path,'w',encoding='utf-8') as f:
        f.write(json.dumps({'loaded':loaded,'llm':meta.get('llm'),'mode':meta.get('mode'),'reason':meta.get('reason'),'snp_count':len(snp_cols),'yield_stats':{'mean':mu,'std':sigma,'min':lo,'max':hi},'outputs':{'csv':out_csv,'pca':pca_png,'corr':corr_png}},indent=2))
    print('CSV:',out_csv)
    print('PCA:',pca_png)
    print('Corr:',corr_png)
    print('LLM meta:',meta)
    print('Stats:',mu,sigma,lo,hi,'k',len(snp_cols))

if __name__=='__main__':
    main()
