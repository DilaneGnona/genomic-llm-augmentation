import os
import sys
import json
import glob
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error

DATASET='pepper'
PROCESSED=os.path.join('02_processed_data',DATASET)
SYN_V1=os.path.join('04_augmentation',DATASET,'model_sources','llama3','synthetic_y_llama3_filtered_k3000_200.csv')
SYN_V2=os.path.join('04_augmentation',DATASET,'model_sources','llama3','synthetic_y_llama3_filtered_k3000_200_v2.csv')
ML_OUT=os.path.join('03_modeling_results','ml_results')
os.makedirs(ML_OUT,exist_ok=True)

RF_PARAMS={'n_estimators':75,'max_depth':15,'max_features':'sqrt','max_samples':0.8}
LGBM_PARAMS={'n_estimators':100,'max_depth':10,'learning_rate':0.1,'subsample':0.8,'colsample_bytree':0.8,'random_state':42,'verbosity':-1}

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
    X=pd.read_parquet(os.path.join(PROCESSED,'X.parquet'),engine=eng) if eng else pd.read_parquet(os.path.join(PROCESSED,'X.parquet'))
    y=pd.read_csv(os.path.join(PROCESSED,'y.csv'))
    pca=pd.read_csv(os.path.join(PROCESSED,'pca_covariates.csv'))
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
    Xs=Xf.drop(columns=['Sample_ID']).astype(np.float32).values
    Xp=pcaf.drop(columns=['Sample_ID']).astype(np.float32).values
    Xc=np.concatenate([Xs,Xp],axis=1)
    tgt=pd.to_numeric(yf['Yield_BV'],errors='coerce')
    valid=~tgt.isna()
    return Xc[valid].astype(np.float32), tgt[valid].values.astype(np.float32), yf['Sample_ID'][valid].tolist(), Xf.drop(columns=['Sample_ID']).columns.tolist(), pcaf.drop(columns=['Sample_ID']).columns.tolist()

def load_synth(path, snp_cols, pca_cols):
    df=pd.read_csv(path)
    if 'Sample_ID' not in df.columns:
        df.rename(columns={df.columns[0]:'Sample_ID'},inplace=True)
    snps=df[[c for c in df.columns if c in snp_cols]].copy()
    for c in snp_cols:
        if c not in snps.columns:
            snps[c]=0.0
    snps=snps[snp_cols]
    pz=pd.DataFrame(0.0,index=np.arange(len(df)),columns=pca_cols)
    X=np.concatenate([snps.astype(np.float32).values,pz.astype(np.float32).values],axis=1)
    y=pd.to_numeric(df['Yield_BV'] if 'Yield_BV' in df.columns else df.iloc[:,-1],errors='coerce').astype(np.float32).values
    sids=[f'SYNTHETIC_{sid}' for sid in df['Sample_ID'].astype(str).tolist()]
    return X,y,sids

def build_splits(sample_ids, holdout_frac=0.2):
    real_idx=[i for i,sid in enumerate(sample_ids) if not str(sid).startswith('SYNTHETIC_')]
    syn_idx=[i for i,sid in enumerate(sample_ids) if str(sid).startswith('SYNTHETIC_')]
    from sklearn.model_selection import train_test_split,KFold
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

def train_lgbm_combined(X,y,ids,label,log_path,metrics_path,imp_png):
    with open(log_path,'w',encoding='utf-8') as lg:
        lg.write(f'Start LGBM: {label}\n'); lg.flush()
        outer,holdout=build_splits(ids,0.2)
        r2s=[]; rmses=[]
        for i,sp in enumerate(outer):
            tr=np.array(sp['train']); te=np.array(sp['test'])
            m=LGBMRegressor(**LGBM_PARAMS)
            m.fit(X[tr],y[tr])
            yp=m.predict(X[te])
            r2=float(r2_score(y[te],yp)); rmse=float(np.sqrt(mean_squared_error(y[te],yp)))
            r2s.append(r2); rmses.append(rmse)
            lg.write(f'Fold {i+1}/2: R2={r2:.4f}, RMSE={rmse:.4f}\n'); lg.flush()
        real_idx=[i for i,sid in enumerate(ids) if not str(sid).startswith('SYNTHETIC_')]
        syn_idx=[i for i,sid in enumerate(ids) if str(sid).startswith('SYNTHETIC_')]
        real_train_final=sorted(list(set(real_idx)-set(holdout)))
        final_idx=np.array(sorted(list(set(real_train_final)|set(syn_idx))))
        fm=LGBMRegressor(**LGBM_PARAMS)
        fm.fit(X[final_idx],y[final_idx])
        hold=np.array(holdout)
        yph=fm.predict(X[hold])
        hold_r2=float(r2_score(y[hold],yph)); hold_rmse=float(np.sqrt(mean_squared_error(y[hold],yph)))
        met={'model':'lightgbm','label':label,'cv_r2_mean':float(np.mean(r2s)),'cv_r2_std':float(np.std(r2s)),'cv_rmse_mean':float(np.mean(rmses)),'cv_rmse_std':float(np.std(rmses)),'holdout_r2':hold_r2,'holdout_rmse':hold_rmse,'features_count':int(X.shape[1])}
        with open(metrics_path,'w',encoding='utf-8') as f: json.dump(met,f,indent=2)
        try:
            imps=fm.feature_importances_
            idxs=np.argsort(imps)[-20:]; vals=imps[idxs]
            plt.figure(figsize=(8,6)); plt.bar(range(len(vals)),vals); plt.title(f'Top 20 SNP importances ({label})'); plt.tight_layout(); plt.savefig(imp_png,dpi=300); plt.close()
        except Exception: pass
        return met

def run_rf_with_unified(augment_file, out_metrics, out_logs, out_imp_png):
    llm_log_dir=os.path.dirname(out_logs); os.makedirs(llm_log_dir,exist_ok=True)
    cmd=[sys.executable, os.path.join('scripts','unified_modeling_pipeline_augmented.py'), '--dataset', DATASET, '--use_synthetic', 'True', '--selected_k','3000','--cross_validation_outer','2','--cross_validation_inner','2','--holdout_size','0.2','--overwrite_previous','--models','randomforest','--augment_file', augment_file,'--rf_n_estimators','75','--rf_max_depth','15','--rf_max_features','sqrt','--rf_max_samples','0.8']
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
    with open(out_logs,'w',encoding='utf-8') as vf:
        for line in proc.stdout: vf.write(line); sys.stdout.write(line)
    code=proc.wait()
    # copy last RF metrics
    metrics_dir=os.path.join('03_modeling_results',f'{DATASET}_augmented','metrics')
    cand=sorted(glob.glob(os.path.join(metrics_dir,'randomforest_metrics_*.json')),key=os.path.getmtime)
    if not cand: return None
    last=cand[-1]
    with open(last,'r',encoding='utf-8') as f: m=json.load(f)
    with open(out_metrics,'w',encoding='utf-8') as f: json.dump(m,f,indent=2)
    # load final model for importance
    run_id=m.get('run_id'); model_dir=os.path.join('03_modeling_results',f'{DATASET}_augmented','models',f'randomforest_{run_id}')
    try:
        import joblib
        fm=joblib.load(os.path.join(model_dir,'randomforest_final_model.joblib'))
        imps=getattr(fm,'feature_importances_',None)
        if imps is not None:
            idxs=np.argsort(imps)[-20:]; vals=imps[idxs]
            plt.figure(figsize=(8,6)); plt.bar(range(len(vals)),vals); plt.title('Top 20 SNP importances (RF llama3 v2)'); plt.tight_layout(); plt.savefig(out_imp_png,dpi=300); plt.close()
    except Exception: pass
    return m

def corr_from_files(synth_path):
    yr=pd.read_csv(os.path.join(PROCESSED,'y.csv'))
    r=np.float32(pd.to_numeric(yr['Yield_BV'],errors='coerce').dropna().values)
    ys=pd.read_csv(synth_path)
    s=np.float32(pd.to_numeric(ys['Yield_BV'] if 'Yield_BV' in ys.columns else ys.iloc[:,-1],errors='coerce').dropna().values)
    n=min(len(r),len(s))
    return float(np.corrcoef(r[:n],s[:n])[0,1])

def write_full_comparison(rf_v2,lgbm_v2):
    rows=[]
    # LGBM baseline/v1
    lb=os.path.join(ML_OUT,'lgbm_baseline_metrics.json'); lv1=os.path.join(ML_OUT,'lgbm_llama3_200synth_metrics.json'); lv2=os.path.join(ML_OUT,'lgbm_llama3_200synth_v2_metrics.json')
    if os.path.exists(lb): rows.append({'Modèle':'LGBM','Dataset':'baseline','Holdout_R²':json.load(open(lb))['holdout_r2'],'Holdout_RMSE':json.load(open(lb))['holdout_rmse'],'CV_R²':json.load(open(lb))['cv_r2_mean'],'Corrélation_Synth':None})
    if os.path.exists(lv1): rows.append({'Modèle':'LGBM','Dataset':'llama3_v1','Holdout_R²':json.load(open(lv1))['holdout_r2'],'Holdout_RMSE':json.load(open(lv1))['holdout_rmse'],'CV_R²':json.load(open(lv1))['cv_r2_mean'],'Corrélation_Synth':corr_from_files(SYN_V1)})
    if os.path.exists(lv2): rows.append({'Modèle':'LGBM','Dataset':'llama3_v2','Holdout_R²':json.load(open(lv2))['holdout_r2'],'Holdout_RMSE':json.load(open(lv2))['holdout_rmse'],'CV_R²':json.load(open(lv2))['cv_r2_mean'],'Corrélation_Synth':corr_from_files(SYN_V2)})
    # RF baseline/v1/v2
    rf_base=os.path.join('03_modeling_results','baseline','randomforest_baseline_v2_metrics.json')
    rf_v1=os.path.join('03_modeling_results','pepper_augmented_v2','metrics','llama3_randomforest_200synth_metrics.json')
    if os.path.exists(rf_base):
        mb=json.load(open(rf_base)); rows.append({'Modèle':'RF','Dataset':'baseline','Holdout_R²':mb['holdout_r2'],'Holdout_RMSE':mb['holdout_rmse'],'CV_R²':mb['cv_r2_mean'],'Corrélation_Synth':None})
    if os.path.exists(rf_v1):
        mv1=json.load(open(rf_v1)); rows.append({'Modèle':'RF','Dataset':'llama3_v1','Holdout_R²':mv1['holdout_r2'],'Holdout_RMSE':mv1['holdout_rmse'],'CV_R²':mv1['cv_r2_mean'],'Corrélation_Synth':corr_from_files(SYN_V1)})
    if rf_v2 is not None:
        rows.append({'Modèle':'RF','Dataset':'llama3_v2','Holdout_R²':rf_v2.get('holdout_r2'),'Holdout_RMSE':rf_v2.get('holdout_rmse'),'CV_R²':rf_v2.get('cv_r2_mean'),'Corrélation_Synth':corr_from_files(SYN_V2)})
    out_csv=os.path.join(ML_OUT,'full_ml_comparison.csv')
    pd.DataFrame(rows)[['Modèle','Dataset','Holdout_R²','Holdout_RMSE','CV_R²','Corrélation_Synth']].to_csv(out_csv,index=False)
    # R2 plot
    try:
        plt.figure(figsize=(8,4))
        labels=[f"{r['Modèle']}-{r['Dataset']}" for r in rows]
        vals=[r['Holdout_R²'] for r in rows]
        colors=[]
        for r in rows:
            if r['Dataset']=='baseline': colors.append('steelblue')
            elif r['Dataset']=='llama3_v1': colors.append('orange')
            else: colors.append('seagreen')
        plt.bar(labels,vals,color=colors)
        plt.ylabel('Holdout R²'); plt.title('Comparaison Holdout R² (RF/LGBM, baseline vs llama3 v1/v2)')
        plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(os.path.join(ML_OUT,'full_ml_r2_comparison.png'),dpi=300); plt.close()
    except Exception: pass
    # Analysis md
    md=os.path.join(ML_OUT,'full_ml_analysis.md')
    with open(md,'w',encoding='utf-8') as f:
        f.write('# Analyse complète RF/LGBM (baseline vs llama3 v1/v2)\n\n')
        f.write(f"Fichier comparatif: {out_csv}\n")
        f.write('Voir full_ml_r2_comparison.png pour la synthèse graphique.\n')

def pca_v2(snp_cols):
    # reuse quality script logic for v2 PCA
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    eng=_get_parquet_engine()
    X=pd.read_parquet(os.path.join(PROCESSED,'X.parquet'),engine=eng) if eng else pd.read_parquet(os.path.join(PROCESSED,'X.parquet'))
    if 'Sample_ID' not in X.columns:
        X.rename(columns={X.columns[0]:'Sample_ID'},inplace=True)
    meta={'POS','REF','ALT'}
    X=X[~X['Sample_ID'].isin(meta)].reset_index(drop=True)
    df=pd.read_csv(SYN_V2)
    cols=[c for c in df.columns if c in snp_cols]
    snps=df[cols].copy()
    for c in snp_cols:
        if c not in snps.columns: snps[c]=0.0
    snps=snps[snp_cols]
    real_X=X[snp_cols].astype(np.float32).values
    synth_X=snps.astype(np.float32).values
    Xcat=np.vstack([real_X,synth_X])
    labels=np.array([0]*len(real_X)+[1]*len(synth_X))
    Xstd=StandardScaler().fit_transform(Xcat)
    comps=PCA(n_components=2,random_state=42).fit_transform(Xstd)
    plt.figure(figsize=(7,6))
    plt.scatter(comps[labels==0,0],comps[labels==0,1],c='blue',s=10,label='Réelles')
    plt.scatter(comps[labels==1,0],comps[labels==1,1],c='green',s=12,label='llama3 v2 (200)')
    plt.legend(); plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('PCA SNPs: Réelles vs llama3 v2 (200, k=3000)')
    out=os.path.join('03_modeling_results','quality_analysis','llama3_200synth_v2_pca.png')
    plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()
    return out

def write_pca_compare_report():
    md=os.path.join('03_modeling_results','quality_analysis','llama3_synth_quality_v1_vs_v2.md')
    with open(md,'w',encoding='utf-8') as f:
        f.write('# Qualité synthétiques llama3 v1 vs v2 (k=3000)\n\n')
        f.write('- PCA v1: `llama3_200synth_pca.png`\n')
        f.write('- PCA v2: `llama3_200synth_v2_pca.png`\n')
        f.write(f"- Corrélation v1: {corr_from_files(SYN_V1):.4f}\n")
        f.write(f"- Corrélation v2: {corr_from_files(SYN_V2):.4f}\n")

def main():
    # RF with v2
    rf_m=run_rf_with_unified(SYN_V2, os.path.join(ML_OUT,'rf_llama3_200synth_v2_metrics.json'), os.path.join(ML_OUT,'rf_llama3_200synth_v2_logs.txt'), os.path.join(ML_OUT,'rf_llama3_200synth_v2_snp_importance.png'))
    # LGBM with v2
    Xr, yr, sids, snp_cols, pca_cols = load_real()
    Xs, ys, ssids = load_synth(SYN_V2, snp_cols, pca_cols)
    Xc=np.vstack([Xr,Xs]); yc=np.concatenate([yr,ys]); ids=sids+ssids
    lgbm_m=train_lgbm_combined(Xc,yc,ids,'llama3_200synth_v2', os.path.join(ML_OUT,'lgbm_llama3_200synth_v2_logs.txt'), os.path.join(ML_OUT,'lgbm_llama3_200synth_v2_metrics.json'), os.path.join(ML_OUT,'lgbm_llama3_200synth_v2_snp_importance.png'))
    # Comparison
    write_full_comparison(rf_m,lgbm_m)
    # PCA v2 and report
    pca_v2(snp_cols)
    write_pca_compare_report()
    print('Done.')

if __name__=='__main__':
    main()

