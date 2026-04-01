import os
import numpy as np
import pandas as pd

DATASET='pepper'
BASE_PROC=os.path.join('02_processed_data',DATASET)
BASE_AUG=os.path.join('04_augmentation',DATASET)
MODEL_SOURCES=os.path.join(BASE_AUG,'model_sources')
TARGET='Yield_BV'

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

def load_real_x():
    p=os.path.join(BASE_PROC,'X.parquet')
    eng=_get_parquet_engine()
    df=pd.read_parquet(p,engine=eng) if eng else pd.read_parquet(p)
    snp=[c for c in df.columns if c!='Sample_ID'] if 'Sample_ID' in df.columns else df.columns.tolist()
    return df,snp

def build_200(llm):
    src=os.path.join(MODEL_SOURCES,llm,f'synthetic_y_{llm}_filtered_k3000.csv')
    if not os.path.exists(src):
        print(f'[SKIP] {llm}: source not found')
        return
    df=pd.read_csv(src)
    # Detect target column
    tcol=TARGET if TARGET in df.columns else (df.columns[1] if df.columns[0]=='Sample_ID' and len(df.columns)>1 else df.columns[0])
    df=df[[c for c in df.columns if c in ['Sample_ID',tcol]]].copy()
    np.random.seed(42)
    if len(df)>=200:
        idx=np.random.choice(len(df),size=200,replace=False)
        df=df.iloc[idx].reset_index(drop=True)
    else:
        add=np.random.choice(len(df),size=200-len(df),replace=True)
        df=pd.concat([df,df.iloc[add].reset_index(drop=True)],ignore_index=True)
    real_X,snp=load_real_x()
    ridx=np.random.choice(len(real_X),size=200,replace=True)
    Xs=real_X.iloc[ridx][snp].reset_index(drop=True)
    out=pd.concat([df[['Sample_ID']],Xs.astype('float32')],axis=1)
    out[TARGET]=pd.to_numeric(df[tcol],errors='coerce').astype('float32')
    path=os.path.join(MODEL_SOURCES,llm,f'synthetic_y_{llm}_filtered_k3000_200.csv')
    out.to_csv(path,index=False)
    print(f'[OK] {llm} → {path}')

def main():
    import sys
    llms=sys.argv[1:] or ['glm46','qwen','minimax']
    for llm in llms:
        build_200(llm)

if __name__=='__main__':
    main()
