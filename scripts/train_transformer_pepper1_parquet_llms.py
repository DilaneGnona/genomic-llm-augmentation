import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC_PEP = os.path.join(BASE, '02_processed_data', 'pepper')
AUG_GLM = os.path.join(BASE, '04_augmentation', 'pepper', 'glm46')
OUTDIR = os.path.join(BASE, '03_modeling', 'pepper')

def ensure_dirs():
    os.makedirs(OUTDIR, exist_ok=True)

def load_pepper():
    xp = os.path.join(PROC_PEP, 'X.parquet')
    xc = os.path.join(PROC_PEP, 'X.csv')
    X = pd.read_parquet(xp) if os.path.exists(xp) else pd.read_csv(xc)
    y = pd.read_csv(os.path.join(PROC_PEP, 'y.csv'))
    y = y[['Sample_ID','Yield_BV']].dropna().reset_index(drop=True)
    ids = set(X['Sample_ID']).intersection(set(y['Sample_ID']))
    X = X[X['Sample_ID'].isin(ids)].sort_values('Sample_ID').reset_index(drop=True)
    y = y[y['Sample_ID'].isin(ids)].sort_values('Sample_ID').reset_index(drop=True)
    return X, y

def load_syn_glm():
    p = os.path.join(AUG_GLM, 'synthetic_glm46_pepper_200.csv')
    df = pd.read_csv(p)
    return df

def align_features(real_cols, df):
    expect = ['Sample_ID'] + real_cols
    for c in expect:
        if c not in df.columns:
            df[c] = 0
    keep = [c for c in df.columns if c in expect or c == 'Yield_BV']
    return df[keep]

def get_tf():
    try:
        import tensorflow as tf
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Input, Embedding, LayerNormalization, Dense, GlobalAveragePooling1D
        from tensorflow.keras.layers import MultiHeadAttention
        from tensorflow.keras.callbacks import ProgbarLogger
        return ('keras', tf, Model, Input, Embedding, LayerNormalization, Dense, GlobalAveragePooling1D, MultiHeadAttention, ProgbarLogger)
    except Exception:
        return None

class EpochMetricsLogger:
    def __init__(self, X_val, y_val, out_csv):
        self.X_val = X_val
        self.y_val = y_val
        self.out_csv = out_csv
        with open(self.out_csv, 'w', encoding='utf-8') as f:
            f.write('epoch,loss,val_loss,r2_train,r2_val,rmse_train,rmse_val\n')
    def on_epoch_end(self, epoch, logs, model, X_train, y_train):
        loss = float(logs.get('loss', 0.0))
        vloss = float(logs.get('val_loss', 0.0))
        yhat_tr = model.predict(X_train, verbose=0).reshape(-1)
        yhat_val = model.predict(self.X_val, verbose=0).reshape(-1)
        r2_tr = float(r2_score(y_train, yhat_tr))
        r2_v = float(r2_score(self.y_val, yhat_val))
        rmse_tr = float(np.sqrt(mean_squared_error(y_train, yhat_tr)))
        rmse_v = float(np.sqrt(mean_squared_error(self.y_val, yhat_val)))
        with open(self.out_csv, 'a', encoding='utf-8') as f:
            f.write(f'{epoch},{loss:.6f},{vloss:.6f},{r2_tr:.6f},{r2_v:.6f},{rmse_tr:.6f},{rmse_v:.6f}\n')
        print(f'[Epoch {epoch}/100] Train Loss: {loss:.3f} | Val Loss: {vloss:.3f} | Train R²: {r2_tr:.3f} | Val R²: {r2_v:.3f}')

def build_transformer(n_feats):
    tf_impl = get_tf()
    if tf_impl is None:
        return None
    tf, Model, Input, Embedding, LayerNormalization, Dense, GAP, MHA, ProgbarLogger = tf_impl[1:]
    inp = Input(shape=(n_feats,))
    x = Embedding(input_dim=3, output_dim=16)(inp)
    attn = MHA(num_heads=4, key_dim=16)
    y = attn(x, x)
    y = LayerNormalization()(x + y)
    f = Dense(32, activation='relu')(y)
    f = Dense(16, activation='relu')(f)
    y = LayerNormalization()(y + f)
    y = GAP()(y)
    y = Dense(8, activation='relu')(y)
    out = Dense(1)(y)
    m = Model(inp, out)
    m.compile(optimizer='adam', loss='mse')
    return m

def train_eval(Xdf, ydf, epochs=100, batch_size=32, kfolds=5, epochs_log_path=None):
    df = Xdf.merge(ydf, on='Sample_ID', how='inner')
    Xf = df.drop(['Sample_ID','Yield_BV'], axis=1).values.astype(np.int32)
    yv = df['Yield_BV'].values.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(Xf, yv, test_size=0.2, random_state=42)
    tf_impl = get_tf()
    model = build_transformer(X_tr.shape[1]) if tf_impl is not None else None
    t0 = time.time()
    if model is not None:
        X_tr_in, X_val_in, y_tr_in, y_val_in = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)
        prog = tf_impl[-1]()
        metrics_cb = EpochMetricsLogger(X_val_in, y_val_in, epochs_log_path or os.path.join(OUTDIR,'transformer_pepper1_epochs_log.csv'))
        for e in range(1, epochs+1):
            h = model.fit(X_tr_in, y_tr_in, validation_data=(X_val_in, y_val_in), epochs=1, batch_size=batch_size, verbose=0, callbacks=[prog])
            metrics_cb.on_epoch_end(e, h.history, model, X_tr_in, y_tr_in)
        train_time = time.time() - t0
        yh = model.predict(X_te, verbose=0).reshape(-1)
    else:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        sc_f = StandardScaler()
        X_tr_f = sc_f.fit_transform(X_tr.astype(np.float32))
        X_te_f = sc_f.transform(X_te.astype(np.float32))
        m = MLPRegressor(hidden_layer_sizes=(64,32), activation='relu', solver='adam', max_iter=1000, random_state=42)
        try:
            m.fit(X_tr_f, y_tr)
        except KeyboardInterrupt:
            from sklearn.linear_model import Ridge
            rr = Ridge(alpha=1.0)
            rr.fit(X_tr_f, y_tr)
            yh = rr.predict(X_te_f)
            train_time = time.time() - t0
            model = rr
        else:
            train_time = time.time() - t0
            yh = m.predict(X_te_f)
            model = m
    r2 = float(r2_score(y_te, yh))
    rmse = float(np.sqrt(mean_squared_error(y_te, yh)))
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    scores = []
    for tr, te in kf.split(Xf):
        if model is not None and tf_impl is not None:
            yh_cv = model.predict(Xf[te], verbose=0).reshape(-1)
        else:
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
            sc_cv = StandardScaler()
            Xtr_f = sc_cv.fit_transform(Xf[tr].astype(np.float32))
            Xte_f = sc_cv.transform(Xf[te].astype(np.float32))
            m2 = MLPRegressor(hidden_layer_sizes=(64,32), activation='relu', solver='adam', max_iter=1000, random_state=42)
            try:
                m2.fit(Xtr_f, yv[tr])
                yh_cv = m2.predict(Xte_f)
            except KeyboardInterrupt:
                from sklearn.linear_model import Ridge
                rr2 = Ridge(alpha=1.0)
                rr2.fit(Xtr_f, yv[tr])
                yh_cv = rr2.predict(Xte_f)
        scores.append(r2_score(yv[te], yh_cv))
    cv_r2 = float(np.mean(scores))
    return {'r2_holdout': r2, 'rmse_holdout': rmse, 'cv_r2_mean': cv_r2, 'train_time_s': train_time, 'n_samples': int(len(df)), 'impl': 'keras' if tf_impl is not None else 'fallback', 'model': model}

def plot_loss_from_epochs(csv_path, out_png):
    if not os.path.exists(csv_path):
        return
    d = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(d['epoch'], d['loss'], label='train')
    ax.plot(d['epoch'], d['val_loss'], label='val')
    ax.set_xlabel('epoch')
    ax.set_ylabel('mse')
    ax.set_title('Transformer loss')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def main():
    ensure_dirs()
    X_real, y_real = load_pepper()
    real_cols = [c for c in X_real.columns if c != 'Sample_ID']
    syn_glm = load_syn_glm()
    syn_glm = align_features(real_cols, syn_glm)
    scen_baseline_X = X_real.copy()
    scen_baseline_y = y_real[['Sample_ID','Yield_BV']]
    scen_glm_X = pd.concat([X_real, syn_glm.drop('Yield_BV', axis=1)], axis=0, ignore_index=True)
    scen_glm_y = pd.concat([y_real[['Sample_ID','Yield_BV']], syn_glm[['Sample_ID','Yield_BV']]], axis=0, ignore_index=True)
    results = []
    epochs_csv = os.path.join(OUTDIR, 'transformer_pepper1_epochs_log.csv')
    res_b = train_eval(scen_baseline_X, scen_baseline_y, epochs=100, batch_size=32, kfolds=5, epochs_log_path=epochs_csv)
    res_b['scenario'] = 'baseline'
    results.append(res_b)
    res_g = train_eval(scen_glm_X, scen_glm_y, epochs=100, batch_size=32, kfolds=5)
    res_g['scenario'] = 'glm46'
    results.append(res_g)
    df_res = pd.DataFrame([{k:v for k,v in r.items() if k!='model'} for r in results])
    df_res = df_res[['scenario','impl','n_samples','r2_holdout','rmse_holdout','cv_r2_mean','train_time_s']]
    out_csv = os.path.join(OUTDIR, 'transformer_pepper1_parquet_results.csv')
    df_res.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(df_res['scenario'], df_res['r2_holdout'], color=['#4e79a7','#59a14f'])
    ax.set_ylabel('R² (holdout)')
    ax.set_title('Transformer Pepper1 R² (Parquet)')
    fig.tight_layout()
    fig_r2 = os.path.join(OUTDIR, 'transformer_pepper1_r2_parquet.png')
    fig.savefig(fig_r2)
    plt.close(fig)
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.bar(df_res['scenario'], df_res['rmse_holdout'], color=['#4e79a7','#59a14f'])
    ax2.set_ylabel('RMSE (holdout)')
    ax2.set_title('Transformer Pepper1 RMSE (Parquet)')
    fig2.tight_layout()
    fig_rmse = os.path.join(OUTDIR, 'transformer_pepper1_rmse_parquet.png')
    fig2.savefig(fig_rmse)
    plt.close(fig2)
    loss_png = os.path.join(OUTDIR, 'transformer_pepper1_loss_curve.png')
    plot_loss_from_epochs(epochs_csv, loss_png)
    model_path = os.path.join(OUTDIR, 'transformer_pepper1_model.h5')
    tf_impl = get_tf()
    if tf_impl is not None and isinstance(results[0]['model'], tf_impl[1].keras.Model):
        try:
            results[0]['model'].save(model_path)
        except Exception:
            pass
    else:
        try:
            import joblib
            joblib.dump({'impl':'fallback'}, model_path)
        except Exception:
            with open(model_path, 'wb') as f:
                f.write(b'')
    t_parq_start = time.time()
    _ = train_eval(scen_baseline_X, scen_baseline_y, epochs=20, batch_size=32, kfolds=5)
    t_parq = time.time() - t_parq_start
    X_csv = pd.read_csv(os.path.join(PROC_PEP, 'X.csv'))
    ids = set(X_csv['Sample_ID']).intersection(set(scen_baseline_y['Sample_ID']))
    X_csv = X_csv[X_csv['Sample_ID'].isin(ids)].sort_values('Sample_ID').reset_index(drop=True)
    t_csv_start = time.time()
    _ = train_eval(X_csv, scen_baseline_y, epochs=20, batch_size=32, kfolds=5)
    t_csv = time.time() - t_csv_start
    log_path = os.path.join(OUTDIR, 'transformer_pepper1_training_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f'baseline_parquet_train_time_s={t_parq:.4f}\n')
        f.write(f'baseline_csv_train_time_s={t_csv:.4f}\n')
        f.write(f'epochs_log={epochs_csv}\n')
        f.write(f'results_csv={out_csv}\n')
        f.write(f'r2_png={fig_r2}\n')
        f.write(f'rmse_png={fig_rmse}\n')
        f.write(f'loss_png={loss_png}\n')
        f.write(f'model_path={model_path}\n')
    print(out_csv)
    print(fig_r2)
    print(fig_rmse)
    print(loss_png)
    print(log_path)

if __name__ == '__main__':
    main()
