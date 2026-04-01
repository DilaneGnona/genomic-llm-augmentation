import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC_IPK = os.path.join(BASE, '02_processed_data', 'ipk_out_raw')
AUG_BASE = os.path.join(BASE, '04_augmentation', 'pepper', 'ipk_out_raw')
OUTDIR = os.path.join(BASE, '03_modeling', 'pepper', 'ipk_out_raw')

def ensure_dirs():
    os.makedirs(OUTDIR, exist_ok=True)

def choose_target_column(y_df: pd.DataFrame) -> str | None:
    if 'Yield_BV' in y_df.columns:
        return 'Yield_BV'
    if 'YR_LS' in y_df.columns:
        return 'YR_LS'
    numeric_cols = [c for c in y_df.columns if c != 'Sample_ID' and pd.api.types.is_numeric_dtype(y_df[c])]
    return numeric_cols[0] if numeric_cols else None

def load_real():
    xp = os.path.join(PROC_IPK, 'X.parquet')
    xc = os.path.join(PROC_IPK, 'X.csv')
    X = pd.read_parquet(xp) if os.path.exists(xp) else pd.read_csv(xc)
    y = pd.read_csv(os.path.join(PROC_IPK, 'y.csv'))
    if 'Sample_ID' not in X.columns or 'Sample_ID' not in y.columns:
        raise RuntimeError('Sample_ID manquant')
    tgt = choose_target_column(y)
    if tgt is None:
        raise RuntimeError('Aucune colonne cible disponible')
    y = y[['Sample_ID', tgt]].dropna().reset_index(drop=True)
    ids = set(X['Sample_ID']).intersection(set(y['Sample_ID']))
    X = X[X['Sample_ID'].isin(ids)].sort_values('Sample_ID').reset_index(drop=True)
    y = y[y['Sample_ID'].isin(ids)].sort_values('Sample_ID').reset_index(drop=True)
    return X, y, tgt

def load_synthetic(path):
    df = pd.read_csv(path)
    if 'Sample_ID' not in df.columns:
        raise RuntimeError('Synthetic missing Sample_ID')
    return df

def align_features(real_cols, df):
    expect = ['Sample_ID'] + real_cols
    for c in expect:
        if c not in df.columns:
            df[c] = 0
    keep = [c for c in df.columns if c in expect or c == 'Yield_BV']
    return df[keep]

def get_lgbm():
    try:
        from lightgbm import LGBMRegressor
        return 'lightgbm', LGBMRegressor
    except Exception:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            return 'sklearn_hgb', HistGradientBoostingRegressor
        except Exception:
            from sklearn.ensemble import GradientBoostingRegressor
            return 'sklearn_gb', GradientBoostingRegressor

def train_eval(Xdf: pd.DataFrame, ydf: pd.DataFrame, target_col: str):
    df = Xdf.merge(ydf, on='Sample_ID', how='inner')
    Xf = df.drop(['Sample_ID', target_col], axis=1).values
    yv = df[target_col].values
    X_tr, X_te, y_tr, y_te = train_test_split(Xf, yv, test_size=0.2, random_state=42)
    impl, Cls = get_lgbm()
    if impl == 'lightgbm':
        model = Cls(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
    elif impl == 'sklearn_hgb':
        model = Cls(max_depth=5, learning_rate=0.1, max_iter=50, random_state=42)
    else:
        model = Cls(max_depth=5, learning_rate=0.1, n_estimators=50, random_state=42)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0
    yh = model.predict(X_te)
    r2 = float(r2_score(y_te, yh))
    rmse = float(np.sqrt(mean_squared_error(y_te, yh)))
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    scores = []
    for tr, te in kf.split(Xf):
        if impl == 'lightgbm':
            m2 = Cls(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
        elif impl == 'sklearn_hgb':
            m2 = Cls(max_depth=5, learning_rate=0.1, max_iter=50, random_state=42)
        else:
            m2 = Cls(max_depth=5, learning_rate=0.1, n_estimators=50, random_state=42)
        m2.fit(Xf[tr], yv[tr])
        yh_cv = m2.predict(Xf[te])
        scores.append(r2_score(yv[te], yh_cv))
    cv_r2 = float(np.mean(scores))
    return {'r2_holdout': r2, 'rmse_holdout': rmse, 'cv_r2_mean': cv_r2, 'train_time_s': train_time, 'n_samples': int(len(df)), 'impl': impl}

def main():
    ensure_dirs()
    X_real, y_real, tgt = load_real()
    real_cols = [c for c in X_real.columns if c != 'Sample_ID']
    y_real_named = y_real.rename(columns={tgt: 'Yield_BV'})
    syn_glm = load_synthetic(os.path.join(AUG_BASE, 'glm46', 'synthetic_glm46_ipk_200.csv'))
    syn_llama = load_synthetic(os.path.join(AUG_BASE, 'llama3_local', 'synthetic_llama3_local_ipk_200.csv'))
    syn_deep = load_synthetic(os.path.join(AUG_BASE, 'deepseek', 'synthetic_deepseek_ipk_200.csv'))
    syn_glm = align_features(real_cols, syn_glm)
    syn_llama = align_features(real_cols, syn_llama)
    syn_deep = align_features(real_cols, syn_deep)
    scen_baseline_X = X_real.copy()
    scen_baseline_y = y_real_named[['Sample_ID', 'Yield_BV']]
    scen_glm_X = pd.concat([X_real, syn_glm.drop('Yield_BV', axis=1)], axis=0, ignore_index=True)
    scen_glm_y = pd.concat([y_real_named[['Sample_ID', 'Yield_BV']], syn_glm[['Sample_ID', 'Yield_BV']]], axis=0, ignore_index=True)
    scen_llama_X = pd.concat([X_real, syn_llama.drop('Yield_BV', axis=1)], axis=0, ignore_index=True)
    scen_llama_y = pd.concat([y_real_named[['Sample_ID', 'Yield_BV']], syn_llama[['Sample_ID', 'Yield_BV']]], axis=0, ignore_index=True)
    scen_deep_X = pd.concat([X_real, syn_deep.drop('Yield_BV', axis=1)], axis=0, ignore_index=True)
    scen_deep_y = pd.concat([y_real_named[['Sample_ID', 'Yield_BV']], syn_deep[['Sample_ID', 'Yield_BV']]], axis=0, ignore_index=True)
    results = []
    for name, Xdf, ydf in [
        ('baseline', scen_baseline_X, scen_baseline_y),
        ('glm46', scen_glm_X, scen_glm_y),
        ('llama3', scen_llama_X, scen_llama_y),
        ('deepseek', scen_deep_X, scen_deep_y),
    ]:
        res = train_eval(Xdf, ydf, 'Yield_BV')
        res['scenario'] = name
        results.append(res)
    df_res = pd.DataFrame(results)[['scenario','n_samples','r2_holdout','rmse_holdout','cv_r2_mean','train_time_s','impl']]
    out_csv = os.path.join(OUTDIR, 'lgbm_ipk_parquet_llms_results.csv')
    df_res.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(df_res['scenario'], df_res['r2_holdout'], color=['#4e79a7','#59a14f','#f28e2b','#e15759'])
    ax.set_ylabel('R² (holdout)')
    ax.set_title('LightGBM R² (Parquet)')
    fig.tight_layout()
    fig_path_r2 = os.path.join(OUTDIR, 'lgbm_ipk_r2_parquet.png')
    fig.savefig(fig_path_r2)
    plt.close(fig)
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.bar(df_res['scenario'], df_res['rmse_holdout'], color=['#4e79a7','#59a14f','#f28e2b','#e15759'])
    ax2.set_ylabel('RMSE (holdout)')
    ax2.set_title('LightGBM RMSE (Parquet)')
    fig2.tight_layout()
    fig_path_rmse = os.path.join(OUTDIR, 'lgbm_ipk_rmse_parquet.png')
    fig2.savefig(fig_path_rmse)
    plt.close(fig2)
    t_parq_start = time.time()
    _ = train_eval(scen_baseline_X, scen_baseline_y, 'Yield_BV')
    t_parq = time.time() - t_parq_start
    X_csv = pd.read_csv(os.path.join(PROC_IPK, 'X.csv'))
    ids = set(X_csv['Sample_ID']).intersection(set(scen_baseline_y['Sample_ID']))
    X_csv = X_csv[X_csv['Sample_ID'].isin(ids)].sort_values('Sample_ID').reset_index(drop=True)
    t_csv_start = time.time()
    _ = train_eval(X_csv, scen_baseline_y, 'Yield_BV')
    t_csv = time.time() - t_csv_start
    log_path = os.path.join(OUTDIR, 'lgbm_training_time_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f'baseline_parquet_train_time_s={t_parq:.4f}\n')
        f.write(f'baseline_csv_train_time_s={t_csv:.4f}\n')
        f.write(f'impl={results[0]["impl"]}\n')
        f.write(f'results_csv={out_csv}\n')
        f.write(f'r2_png={fig_path_r2}\n')
        f.write(f'rmse_png={fig_path_rmse}\n')
    print(out_csv)
    print(fig_path_r2)
    print(fig_path_rmse)
    print(log_path)

if __name__ == '__main__':
    main()

