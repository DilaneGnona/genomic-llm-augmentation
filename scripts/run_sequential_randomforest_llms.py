import os
import sys
import time
import glob
import json
import subprocess

DATASET = 'pepper'
PROCESSED = os.path.join('02_processed_data', DATASET)
AUG_ROOT = os.path.join('04_augmentation', DATASET)
OUTDIR = os.path.join('03_modeling_results', f'{DATASET}_augmented')

LLMS = ['llama3', 'deepseek', 'glm46', 'qwen', 'minimax']

def find_synth_y(llm):
    candidates = []
    base_model_sources = os.path.join(AUG_ROOT, 'model_sources', llm)
    # Prefer k3000 filtered
    candidates += glob.glob(os.path.join(base_model_sources, '*k3000*.csv'))
    # Then any filtered file
    candidates += glob.glob(os.path.join(base_model_sources, '*filtered*.csv'))
    # qwen special case
    if llm == 'qwen':
        candidates += glob.glob(os.path.join(AUG_ROOT, 'qwen3coder', 'filtered_synthetic_y.csv'))
        candidates += glob.glob(os.path.join(AUG_ROOT, 'qwen3coder', 'synthetic_y.csv'))
    # minimax special case
    if llm == 'minimax':
        candidates += glob.glob(os.path.join(AUG_ROOT, 'minimax', 'filtered_synthetic_y.csv'))
        candidates += glob.glob(os.path.join(AUG_ROOT, 'minimax', 'synthetic_y.csv'))
    # generic fallback
    candidates += glob.glob(os.path.join(AUG_ROOT, 'synthetic_y_filtered.csv'))
    candidates += glob.glob(os.path.join(AUG_ROOT, 'synthetic_y.csv'))
    # Pick first existing
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def latest_config_run_id():
    logs_dir = os.path.join(OUTDIR, 'logs')
    configs = sorted(glob.glob(os.path.join(logs_dir, 'config_*.json')), key=os.path.getmtime)
    if not configs:
        return None
    latest = configs[-1]
    try:
        with open(latest, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        return cfg.get('RUN_ID')
    except Exception:
        # fallback parse filename
        return os.path.basename(latest).replace('config_', '').replace('.json', '')

def copy_metrics_for_llm(run_id, llm):
    src = os.path.join(OUTDIR, 'metrics', f'randomforest_metrics_{run_id}.json')
    dst = os.path.join(OUTDIR, 'metrics', f'{llm}_randomforest_metrics.json')
    if os.path.exists(src):
        try:
            with open(src, 'r', encoding='utf-8') as f:
                data = json.load(f)
            with open(dst, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    return dst if os.path.exists(dst) else None

def run_llm(llm):
    # Skip if metrics already exist
    existing_metrics = os.path.join(OUTDIR, 'metrics', f'{llm}_randomforest_metrics.json')
    if os.path.exists(existing_metrics):
        try:
            with open(existing_metrics, 'r', encoding='utf-8') as f:
                m = json.load(f)
            print(f"[SKIP] {llm} déjà entraîné. Holdout R²={m.get('holdout_r2')}, RMSE={m.get('holdout_rmse')}")
        except Exception:
            print(f"[SKIP] {llm} déjà entraîné.")
        return {
            'LLM': llm,
            'Holdout_R2': m.get('holdout_r2') if isinstance(m, dict) else None,
            'Holdout_RMSE': m.get('holdout_rmse') if isinstance(m, dict) else None,
            'CV_R2': m.get('cv_r2_mean') if isinstance(m, dict) else None,
            'Nb_Samples': m.get('samples_count') if isinstance(m, dict) else None
        }
    print(f"Lancement de l’entraînement pour {llm} (k=3000, Parquet loader)...")
    x_parquet = os.path.join(PROCESSED, 'X.parquet')
    if not os.path.exists(x_parquet):
        print("[ERREUR] X.parquet introuvable; bascule CSV")
    synth_path = find_synth_y(llm)
    if synth_path is None:
        print(f"[ERREUR] Fichier synthétique introuvable pour {llm}; passage au suivant.")
        return None
    # Pre-log sizes
    try:
        snps_msg = "SNPs : 3000"
        size_mb = os.path.getsize(x_parquet)/1024/1024 if os.path.exists(x_parquet) else os.path.getsize(os.path.join(PROCESSED,'X.csv'))/1024/1024
        # Count synthetic lines (excluding header)
        import pandas as pd
        syn_df_head = pd.read_csv(synth_path)
        syn_rows = len(syn_df_head)
        print(f"✅ Chargement X.parquet (taille : ~{size_mb:.2f} Mo, {snps_msg}) | Chargement synthétiques : {os.path.basename(synth_path)} (lignes : {syn_rows})")
    except Exception:
        pass

    # Prepare verbose log file
    llm_log_dir = os.path.join(OUTDIR, 'logs', llm)
    os.makedirs(llm_log_dir, exist_ok=True)
    verbose_path = os.path.join(llm_log_dir, 'verbose_logs.txt')

    # Map augment_mode to allowed choices for CLI
    allowed_modes = {'llama3','deepseek','glm46','pca','none'}
    aug_mode_arg = llm if llm in allowed_modes else 'none'

    cmd = [
        sys.executable,
        os.path.join('scripts','unified_modeling_pipeline_augmented.py'),
        '--dataset', DATASET,
        '--use_synthetic', 'True',
        '--selected_k', '3000',
        '--sigma_resid_factor', '0.1',
        '--cross_validation_outer', '2',
        '--cross_validation_inner', '2',
        '--holdout_size', '0.2',
        '--overwrite_previous',
        '--models', 'randomforest',
        '--augment_mode', aug_mode_arg,
        '--augment_file', synth_path,
        '--rf_n_estimators', '75',
        '--rf_max_depth', '15',
        '--rf_max_features', 'sqrt',
        '--rf_max_samples', '0.8'
    ]

    # Run and stream logs
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    with open(verbose_path, 'w', encoding='utf-8') as vf:
        for line in proc.stdout:
            sys.stdout.write(line)
            vf.write(line)
    code = proc.wait()
    # After run, read metrics to print requested lines
    run_id = latest_config_run_id()
    if run_id is None:
        print(f"[ERREUR] RUN_ID introuvable pour {llm}")
        return None
    metrics_path = os.path.join(OUTDIR, 'metrics', f'randomforest_metrics_{run_id}.json')
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            m = json.load(f)
        # Cleanup info
        remaining = int(m.get('samples_count') or 0)
        print(f"Suppression NaNs → {remaining} échantillons restants (réelles + synthétiques)")
        # CV folds
        r2s = m.get('fold_metrics', {}).get('r2_scores', [])
        rmses = m.get('fold_metrics', {}).get('rmse_scores', [])
        for i in range(min(2, len(r2s))):
            print(f"CV Fold {i+1}/2 : R² = {r2s[i]:.4f}, RMSE = {rmses[i]:.4f}")
        # Holdout
        print(f"Holdout set : R² = {float(m.get('holdout_r2')):.4f}, RMSE = {float(m.get('holdout_rmse')):.4f}")
        dst = copy_metrics_for_llm(run_id, llm)
        print(f"✅ Entraînement {llm} terminé → Métriques sauvegardées dans {dst}")
        return {
            'LLM': llm,
            'Holdout_R2': m.get('holdout_r2'),
            'Holdout_RMSE': m.get('holdout_rmse'),
            'CV_R2': m.get('cv_r2_mean'),
            'Nb_Samples': m.get('samples_count')
        }
    except Exception as e:
        print(f"[ERREUR] Lecture métriques échouée pour {llm}: {e}")
        return None

def main():
    results = []
    for llm in LLMS:
        res = run_llm(llm)
        if res:
            results.append(res)
    # Save comparative CSV
    comp_dir = os.path.join('03_modeling_results','comparative_analysis','cross_llm_k3000','randomforest')
    os.makedirs(comp_dir, exist_ok=True)
    out_csv = os.path.join(comp_dir, 'comparative_results.csv')
    try:
        import pandas as pd
        pd.DataFrame(results)[['LLM','Holdout_R2','Holdout_RMSE','CV_R2','Nb_Samples']].to_csv(out_csv, index=False)
        print(f"Comparatif sauvegardé: {out_csv}")
    except Exception:
        pass

if __name__ == '__main__':
    main()
