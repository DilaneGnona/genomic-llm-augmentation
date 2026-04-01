import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor

DATASET = 'pepper'
BASE_PROC = os.path.join('02_processed_data', DATASET)
ML_OUT = os.path.join('03_modeling_results', 'ml_results')
os.makedirs(ML_OUT, exist_ok=True)

PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': -1
}

def _get_parquet_engine():
    try:
        import pyarrow  # noqa: F401
        return 'pyarrow'
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return 'fastparquet'
        except Exception:
            return None

def load_real():
    eng = _get_parquet_engine()
    X = pd.read_parquet(os.path.join(BASE_PROC, 'X.parquet'), engine=eng)
    y = pd.read_csv(os.path.join(BASE_PROC, 'y.csv'))
    pca = pd.read_csv(os.path.join(BASE_PROC, 'pca_covariates.csv'))
    # Normalize ID column
    for df in (X, y, pca):
        if 'Sample_ID' not in df.columns:
            df.rename(columns={df.columns[0]: 'Sample_ID'}, inplace=True)
    # Remove meta rows
    meta = {'POS','REF','ALT'}
    X = X[~X['Sample_ID'].isin(meta)].reset_index(drop=True)
    pca = pca[~pca['Sample_ID'].isin(meta)].reset_index(drop=True)
    # Align common Sample_IDs
    common = set(X['Sample_ID']).intersection(set(y['Sample_ID'])).intersection(set(pca['Sample_ID']))
    Xf = X[X['Sample_ID'].isin(common)].sort_values('Sample_ID').reset_index(drop=True)
    yf = y[y['Sample_ID'].isin(common)].sort_values('Sample_ID').reset_index(drop=True)
    pcaf = pca[pca['Sample_ID'].isin(common)].sort_values('Sample_ID').reset_index(drop=True)
    # Features
    X_snp = Xf.drop(columns=['Sample_ID'])
    X_pca = pcaf.drop(columns=['Sample_ID'])
    X_comb = pd.concat([X_snp.astype(np.float32), X_pca.astype(np.float32)], axis=1)
    # Target
    tgt = pd.to_numeric(yf['Yield_BV'], errors='coerce')
    # Drop NaNs
    valid = ~tgt.isna()
    X_comb = X_comb[valid].reset_index(drop=True)
    tgt = tgt[valid].reset_index(drop=True)
    sample_ids = yf['Sample_ID'][valid].tolist()
    return X_comb.values.astype(np.float32), tgt.values.astype(np.float32), sample_ids, X_snp.columns.tolist(), X_pca.columns.tolist()

def load_llama3_synth_200(snp_cols, pca_cols):
    synth_path = os.path.join('04_augmentation','pepper','model_sources','llama3','synthetic_y_llama3_filtered_k3000_200.csv')
    df = pd.read_csv(synth_path)
    if 'Sample_ID' not in df.columns:
        df.rename(columns={df.columns[0]: 'Sample_ID'}, inplace=True)
    # Ensure SNP columns match order
    snps = df[[c for c in df.columns if c in snp_cols]].copy()
    # Reindex to full snp_cols
    for c in snp_cols:
        if c not in snps.columns:
            snps[c] = 0.0
    snps = snps[snp_cols]
    # PCA zeros
    pca_zero = pd.DataFrame(0.0, index=np.arange(len(df)), columns=pca_cols)
    Xs = pd.concat([snps.astype(np.float32), pca_zero.astype(np.float32)], axis=1)
    ys = pd.to_numeric(df['Yield_BV'] if 'Yield_BV' in df.columns else df.iloc[:, -1], errors='coerce').astype(np.float32)
    sids = [f"SYNTHETIC_{sid}" for sid in df['Sample_ID'].astype(str).tolist()]
    return Xs.values.astype(np.float32), ys.values.astype(np.float32), sids

def build_splits(sample_ids, holdout_frac=0.2):
    real_idx = [i for i,sid in enumerate(sample_ids) if not str(sid).startswith('SYNTHETIC_')]
    syn_idx = [i for i,sid in enumerate(sample_ids) if str(sid).startswith('SYNTHETIC_')]
    from sklearn.model_selection import train_test_split, KFold
    real_arr = np.array(real_idx)
    real_train, real_hold = train_test_split(real_arr, test_size=holdout_frac, random_state=42)
    holdout = sorted(real_hold.tolist())
    real_for_cv = sorted(real_train.tolist())
    outer = []
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    for tr, te in kf.split(np.arange(len(real_for_cv))):
        tr_real = np.array(real_for_cv)[tr].tolist()
        te_real = np.array(real_for_cv)[te].tolist()
        train_idx = sorted(list(set(tr_real) | set(syn_idx)))
        outer.append({'train': train_idx, 'test': te_real, 'train_real': tr_real})
    return outer, holdout

def train_lgbm(X, y, sample_ids, label, log_file, metrics_file, importance_png):
    with open(log_file, 'w', encoding='utf-8') as lg:
        lg.write(f"Start LGBM training: {label}\n")
        lg.flush()
        outer, holdout = build_splits(sample_ids, 0.2)
        r2s, rmses = [], []
        start = time.time()
        for i, sp in enumerate(outer):
            tr = np.array(sp['train'])
            te = np.array(sp['test'])
            model = LGBMRegressor(**PARAMS)
            model.fit(X[tr], y[tr])
            yp = model.predict(X[te])
            r2 = r2_score(y[te], yp)
            rmse = float(np.sqrt(mean_squared_error(y[te], yp)))
            r2s.append(float(r2))
            rmses.append(rmse)
            lg.write(f"Fold {i+1}/2: R2={r2:.4f}, RMSE={rmse:.4f}\n")
            lg.flush()
        # Final model (train = all real non-holdout + synthetic)
        real_idx = [i for i,sid in enumerate(sample_ids) if not str(sid).startswith('SYNTHETIC_')]
        syn_idx = [i for i,sid in enumerate(sample_ids) if str(sid).startswith('SYNTHETIC_')]
        real_train_final = sorted(list(set(real_idx) - set(holdout)))
        final_idx = np.array(sorted(list(set(real_train_final) | set(syn_idx))))
        final_model = LGBMRegressor(**PARAMS)
        final_model.fit(X[final_idx], y[final_idx])
        # Holdout eval
        hold = np.array(holdout)
        yph = final_model.predict(X[hold])
        hold_r2 = float(r2_score(y[hold], yph))
        hold_rmse = float(np.sqrt(mean_squared_error(y[hold], yph)))
        lg.write(f"Holdout: R2={hold_r2:.4f}, RMSE={hold_rmse:.4f}\n")
        lg.write(f"Done in {time.time()-start:.2f}s\n")
        lg.flush()
        # Metrics JSON
        m = {
            'model': 'lightgbm',
            'label': label,
            'cv_r2_mean': float(np.mean(r2s)),
            'cv_r2_std': float(np.std(r2s)),
            'cv_rmse_mean': float(np.mean(rmses)),
            'cv_rmse_std': float(np.std(rmses)),
            'holdout_r2': hold_r2,
            'holdout_rmse': hold_rmse,
            'features_count': int(X.shape[1])
        }
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(m, f, indent=2)
        # Importance plot
        try:
            importances = final_model.feature_importances_
            idxs = np.argsort(importances)[-20:]
            vals = importances[idxs]
            plt.figure(figsize=(8,6))
            plt.bar(range(len(vals)), vals)
            plt.title(f"Top 20 SNP importances ({label})")
            plt.tight_layout()
            plt.savefig(importance_png, dpi=300)
            plt.close()
        except Exception:
            pass
        return m

def main():
    # Baseline
    Xr, yr, sids, snp_cols, pca_cols = load_real()
    sbaseline = sids
    mb = train_lgbm(Xr, yr, sbaseline,
                    'baseline',
                    os.path.join(ML_OUT, 'lgbm_baseline_logs.txt'),
                    os.path.join(ML_OUT, 'lgbm_baseline_metrics.json'),
                    os.path.join(ML_OUT, 'lgbm_baseline_snp_importance.png'))
    # llama3 200 synth
    Xs, ys, ssids = load_llama3_synth_200(snp_cols, pca_cols)
    X_comb = np.vstack([Xr, Xs])
    y_comb = np.concatenate([yr, ys])
    s_comb = sbaseline + ssids
    ml = train_lgbm(X_comb, y_comb, s_comb,
                    'llama3_200synth',
                    os.path.join(ML_OUT, 'lgbm_llama3_200synth_logs.txt'),
                    os.path.join(ML_OUT, 'lgbm_llama3_200synth_metrics.json'),
                    os.path.join(ML_OUT, 'lgbm_llama3_200synth_snp_importance.png'))
    # Comparison CSV vs RF
    comp_csv = os.path.join(ML_OUT, 'lgbm_vs_rf_comparison.csv')
    rows = []
    rows.append({'Model':'LGBM_baseline','Holdout_R2': mb['holdout_r2']})
    rows.append({'Model':'LGBM_llama3_200','Holdout_R2': ml['holdout_r2']})
    # Try RF v2 llama3
    rf_ll_path = os.path.join('03_modeling_results','pepper_augmented_v2','metrics','llama3_randomforest_200synth_metrics.json')
    if os.path.exists(rf_ll_path):
        with open(rf_ll_path,'r',encoding='utf-8') as f:
            rf_ll = json.load(f)
        rows.append({'Model':'RF_llama3_200','Holdout_R2': rf_ll.get('holdout_r2')})
    rf_base_path = os.path.join('03_modeling_results','baseline','randomforest_baseline_v2_metrics.json')
    if os.path.exists(rf_base_path):
        with open(rf_base_path,'r',encoding='utf-8') as f:
            rf_b = json.load(f)
        rows.append({'Model':'RF_baseline','Holdout_R2': rf_b.get('holdout_r2')})
    pd.DataFrame(rows).to_csv(comp_csv, index=False)
    # R2 barplot
    try:
        labels = [r['Model'] for r in rows]
        vals = [r['Holdout_R2'] for r in rows]
        plt.figure(figsize=(8,4))
        bars = plt.bar(labels, vals, color=['steelblue','seagreen','gray','red'][:len(labels)])
        plt.ylabel('Holdout R²')
        plt.title('LGBM vs RF (baseline + llama3, k=3000)')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(ML_OUT,'lgbm_vs_rf_r2.png'), dpi=300)
        plt.close()
    except Exception:
        pass
    # Analysis MD
    md = os.path.join(ML_OUT, 'lgbm_analysis.md')
    with open(md,'w',encoding='utf-8') as f:
        f.write('# Analyse LightGBM vs RandomForest (baseline + llama3)\n\n')
        f.write(f"Baseline LGBM Holdout R²: {mb['holdout_r2']:.4f}\n")
        f.write(f"llama3 200 LGBM Holdout R²: {ml['holdout_r2']:.4f}\n")
        if os.path.exists(rf_ll_path):
            f.write(f"llama3 200 RF Holdout R²: {rf_ll.get('holdout_r2'):.4f}\n")
        if os.path.exists(rf_base_path):
            f.write(f"Baseline RF Holdout R²: {rf_b.get('holdout_r2'):.4f}\n")
        f.write('\nVoir lgbm_vs_rf_comparison.csv et lgbm_vs_rf_r2.png pour la synthèse graphique.\n')

if __name__ == '__main__':
    main()

