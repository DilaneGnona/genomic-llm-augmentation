import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from pytorch_tabnet.tab_model import TabNetRegressor
import torch

DATASET='pepper'
BASE_PROC=os.path.join('02_processed_data',DATASET)
QUAL_DIR=os.path.join('03_modeling_results','quality_analysis')
DL_OUT=os.path.join('03_modeling_results','dl_results')
COMP_OUT=os.path.join('03_modeling_results','comparative_analysis')
os.makedirs(DL_OUT,exist_ok=True)
os.makedirs(QUAL_DIR,exist_ok=True)
os.makedirs(COMP_OUT,exist_ok=True)

TABNET_PARAMS={
    'n_d':8,'n_a':8,'n_steps':3,'gamma':1.3,
    'mask_type':'sparsemax',
}

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
    pca=pd.read_csv(os.path.join(BASE_PROC,'pca_covariates.csv'))
    for df in (X,y,pca):
        if 'Sample_ID' not in df.columns:
            df.rename(columns={df.columns[0]:'Sample_ID'},inplace=True)
    meta={'POS','REF','ALT'}
    X=X[~X['Sample_ID'].isin(meta)].reset_index(drop=True)
    pca=pca[~pca['Sample_ID'].isin(meta)].reset_index(drop=True)
    common=set(X['Sample_ID']).intersection(set(y['Sample_ID'])).intersection(set(pca['Sample_ID']))
    Xf=X[X['Sample_ID'].isin(common)].sort_values('Sample_ID').reset_index(drop=True)
    yf=y[y['Sample_ID'].isin(common)].sort_values('Sample_ID').reset_index(drop=True)
    pcaf=pca[pca['Sample_ID'].isin(common)].sort_values('Sample_ID').reset_index(drop=True)
    Xs=Xf.drop(columns=['Sample_ID']).astype(np.float32)
    Xp=pcaf.drop(columns=['Sample_ID']).astype(np.float32)
    Xc=pd.concat([Xs,Xp],axis=1)
    tgt=pd.to_numeric(yf['Yield_BV'],errors='coerce')
    valid=~tgt.isna()
    return Xc[valid].reset_index(drop=True), tgt[valid].reset_index(drop=True), yf['Sample_ID'][valid].tolist(), Xs.columns.tolist(), Xp.columns.tolist()

def load_synth(path, snp_cols, pca_cols):
    df=pd.read_csv(path)
    if 'Sample_ID' not in df.columns:
        df.rename(columns={df.columns[0]:'Sample_ID'},inplace=True)
    snps=df[[c for c in df.columns if c in snp_cols]].copy()
    for c in snp_cols:
        if c not in snps.columns: snps[c]=0.0
    snps=snps[snp_cols].astype(np.float32)
    pz=pd.DataFrame(0.0,index=np.arange(len(df)),columns=pca_cols).astype(np.float32)
    X=pd.concat([snps,pz],axis=1)
    y=pd.to_numeric(df['Yield_BV'] if 'Yield_BV' in df.columns else df.iloc[:,-1],errors='coerce')
    sids=[f'SYNTHETIC_{sid}' for sid in df['Sample_ID'].astype(str).tolist()]
    return X.reset_index(drop=True), y.reset_index(drop=True), sids

def build_splits(ids, holdout_frac=0.2):
    real_idx=[i for i,sid in enumerate(ids) if not str(sid).startswith('SYNTHETIC_')]
    syn_idx=[i for i,sid in enumerate(ids) if str(sid).startswith('SYNTHETIC_')]
    real_arr=np.array(real_idx)
    real_train,real_hold=train_test_split(real_arr,test_size=holdout_frac,random_state=42)
    holdout=sorted(real_hold.tolist())
    real_cv=sorted(real_train.tolist())
    outer=[]
    kf=KFold(n_splits=2,shuffle=True,random_state=42)
    for tr,te in kf.split(np.arange(len(real_cv))):
        tr_real=np.array(real_cv)[tr].tolist()
        te_real=np.array(real_cv)[te].tolist()
        train_idx=sorted(list(set(tr_real)|set(syn_idx)))
        outer.append({'train':train_idx,'test':te_real,'train_real':tr_real})
    return outer,holdout

def gpu_zscore(X):
    t=torch.tensor(X.values.astype(np.float32), device='cuda')
    m=t.mean(dim=0)
    s=t.std(dim=0)
    s=torch.where(s==0, torch.ones_like(s), s)
    z=(t-m)/s
    return z.detach().cpu().numpy()

def tabnet_train(X, y, ids, label, log_path, metrics_path, loss_png, imp_png):
    with open(log_path,'w',encoding='utf-8') as lg:
        if not torch.cuda.is_available():
            lg.write('GPU not available. Aborting.\n')
            print('GPU not available. Aborting.')
            return None
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        device_name = 'cuda'
        msg1='Using optimized Parquet loader for X features.'
        msg2=f'Device: {device_name}, threads={torch.get_num_threads()}'
        msg3=f'Start TabNet: {label}'
        lg.write(msg1+'\n'); lg.write(msg2+'\n'); lg.write(msg3+'\n'); lg.flush()
        print(msg1); print(msg2); print(msg3)
        Xn=gpu_zscore(X)
        print('GPU z-score normalization done')
        outer,holdout=build_splits(ids,0.2)
        r2s=[]; rmses=[]; losses=[]
        t0=time.time()
        for i, sp in enumerate(outer):
            tr=np.array(sp['train']); te=np.array(sp['test'])
            model=TabNetRegressor(**TABNET_PARAMS, device_name=device_name, optimizer_fn=torch.optim.Adam, optimizer_params={'lr':0.001})
            start=time.time()
            ytr=y.values[tr].reshape(-1,1)
            yte=y.values[te].reshape(-1,1)
            model.fit(Xn[tr], ytr, eval_set=[(Xn[te], yte)], max_epochs=50, patience=5, batch_size=32, loss_fn='mse')
            yp=model.predict(Xn[te]).ravel()
            r2=float(r2_score(y.values[te], yp)); rmse=float(np.sqrt(mean_squared_error(y.values[te], yp)))
            r2s.append(r2); rmses.append(rmse)
            line=f'Fold {i+1}/2: R2={r2:.4f}, RMSE={rmse:.4f}, time={time.time()-start:.2f}s'
            lg.write(line+'\n'); lg.flush()
            print(line)
            try:
                losses.append(model.history['loss'])
            except Exception:
                pass
        real_idx=[i for i,sid in enumerate(ids) if not str(sid).startswith('SYNTHETIC_')]
        syn_idx=[i for i,sid in enumerate(ids) if str(sid).startswith('SYNTHETIC_')]
        real_train_final=sorted(list(set(real_idx)-set(holdout)))
        final_idx=np.array(sorted(list(set(real_train_final)|set(syn_idx))))
        fm=TabNetRegressor(**TABNET_PARAMS, device_name=device_name, optimizer_fn=torch.optim.Adam, optimizer_params={'lr':0.001})
        fm.fit(Xn[final_idx], y.values[final_idx].reshape(-1,1), max_epochs=50, patience=5, batch_size=32, loss_fn='mse')
        hold=np.array(holdout)
        yph=fm.predict(Xn[hold]).ravel()
        hold_r2=float(r2_score(y.values[hold], yph)); hold_rmse=float(np.sqrt(mean_squared_error(y.values[hold], yph)))
        met={'model':'tabnet','label':label,'cv_r2_mean':float(np.mean(r2s)),'cv_r2_std':float(np.std(r2s)),'cv_rmse_mean':float(np.mean(rmses)),'cv_rmse_std':float(np.std(rmses)),'holdout_r2':hold_r2,'holdout_rmse':hold_rmse,'features_count':int(X.shape[1]),'device':'GPU','train_time_s':float(time.time()-t0)}
        with open(metrics_path,'w',encoding='utf-8') as f: json.dump(met,f,indent=2)
        print(f'Holdout: R2={hold_r2:.4f}, RMSE={hold_rmse:.4f}')
        # loss plot
        try:
            if losses:
                plt.figure(figsize=(8,4))
                for i,l in enumerate(losses): plt.plot(l,label=f'fold{i+1}')
                plt.legend(); plt.title(f'TabNet Loss ({label})'); plt.tight_layout(); plt.savefig(loss_png,dpi=300); plt.close()
        except Exception: pass
        # importance
        try:
            imps=fm.feature_importances_
            idxs=np.argsort(imps)[-20:]; vals=imps[idxs]
            plt.figure(figsize=(8,6)); plt.bar(range(len(vals)),vals); plt.title(f'Top 20 SNP importances (TabNet {label})'); plt.tight_layout(); plt.savefig(imp_png,dpi=300); plt.close()
        except Exception: pass
        return met

def write_comparison_tabnet_vs_ml():
    rows=[]
    ml_dir=os.path.join('03_modeling_results','ml_results')
    rf_v2=os.path.join(ml_dir,'rf_llama3_200synth_v2_metrics.json')
    lgb_v2=os.path.join(ml_dir,'lgbm_llama3_200synth_v2_metrics.json')
    if os.path.exists(rf_v2):
        rfj=json.load(open(rf_v2)); rows.append({'Modèle':'RF','Dataset':'llama3_v2','Holdout_R²':rfj.get('holdout_r2'),'Holdout_RMSE':rfj.get('holdout_rmse'),'CV_R²':rfj.get('cv_r2_mean'),'Temps_entraînement':None,'Device':'CPU'})
    if os.path.exists(lgb_v2):
        lbj=json.load(open(lgb_v2)); rows.append({'Modèle':'LGBM','Dataset':'llama3_v2','Holdout_R²':lbj.get('holdout_r2'),'Holdout_RMSE':lbj.get('holdout_rmse'),'CV_R²':lbj.get('cv_r2_mean'),'Temps_entraînement':None,'Device':'CPU'})
    tb_base=os.path.join(DL_OUT,'tabnet_baseline_gpu_metrics.json')
    tb_v2=os.path.join(DL_OUT,'tabnet_llama3_200synth_v2_gpu_metrics.json')
    if os.path.exists(tb_base):
        tb=json.load(open(tb_base)); rows.append({'Modèle':'TabNet','Dataset':'baseline','Holdout_R²':tb.get('holdout_r2'),'Holdout_RMSE':tb.get('holdout_rmse'),'CV_R²':tb.get('cv_r2_mean'),'Temps_entraînement':tb.get('train_time_s'),'Device':'GPU'})
    if os.path.exists(tb_v2):
        tv=json.load(open(tb_v2)); rows.append({'Modèle':'TabNet','Dataset':'llama3_v2','Holdout_R²':tv.get('holdout_r2'),'Holdout_RMSE':tv.get('holdout_rmse'),'CV_R²':tv.get('cv_r2_mean'),'Temps_entraînement':tv.get('train_time_s'),'Device':'GPU'})
    out_csv=os.path.join('03_modeling_results','comparative_analysis','ml_vs_dl_final_gpu_comparison.csv')
    pd.DataFrame(rows)[['Modèle','Dataset','Holdout_R²','Holdout_RMSE','CV_R²','Temps_entraînement','Device']].to_csv(out_csv,index=False)
    # Plot
    try:
        plt.figure(figsize=(8,4))
        labels=[f"{r['Modèle']}-{r['Dataset']}" for r in rows]
        vals=[r['Holdout_R²'] for r in rows]
        colors=['steelblue' if 'baseline' in l else 'seagreen' for l in labels]
        plt.bar(labels,vals,color=colors); plt.ylabel('Holdout R²'); plt.title('ML vs DL (baseline / llama3 v2, GPU)'); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(os.path.join('03_modeling_results','comparative_analysis','ml_vs_dl_final_gpu_r2.png'),dpi=300); plt.close()
    except Exception: pass
    # Report
    md=os.path.join('03_modeling_results','comparative_analysis','ml_vs_dl_final_gpu_report.md')
    with open(md,'w',encoding='utf-8') as f:
        f.write('# ML vs DL final GPU report\n\n'); f.write(f'Comparatif: {out_csv}\n'); f.write('Voir ml_vs_dl_final_gpu_r2.png pour le graphique.\n')

def main():
    if not torch.cuda.is_available():
        print('GPU not available. Aborting.')
        return
    Xr, yr, sids, snp_cols, pca_cols = load_real()
    mb = tabnet_train(Xr, yr, sids, 'baseline_gpu', os.path.join(DL_OUT,'tabnet_baseline_gpu_logs.txt'), os.path.join(DL_OUT,'tabnet_baseline_gpu_metrics.json'), os.path.join(DL_OUT,'tabnet_baseline_gpu_loss.png'), os.path.join(DL_OUT,'tabnet_baseline_gpu_snp_importance.png'))
    Xs, ys, ssids = load_synth(os.path.join('04_augmentation',DATASET,'model_sources','llama3','synthetic_y_llama3_filtered_k3000_200_v2.csv'), snp_cols, pca_cols)
    Xc=pd.concat([pd.DataFrame(Xr), pd.DataFrame(Xs)], axis=0).reset_index(drop=True)
    yc=np.concatenate([yr.values, ys.values])
    ids=sids+ssids
    mv2 = tabnet_train(Xc, pd.Series(yc), ids, 'llama3_200synth_v2_gpu', os.path.join(DL_OUT,'tabnet_llama3_200synth_v2_gpu_logs.txt'), os.path.join(DL_OUT,'tabnet_llama3_200synth_v2_gpu_metrics.json'), os.path.join(DL_OUT,'tabnet_llama3_200synth_v2_gpu_loss.png'), os.path.join(DL_OUT,'tabnet_llama3_v2_gpu_snp_importance.png'))
    write_comparison_tabnet_vs_ml()
    print('Done TabNet baseline + llama3 v2 (GPU)')

if __name__=='__main__':
    main()
