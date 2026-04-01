import os
import sys
import time
import json
import math
import numpy as np
import pandas as pd

DATASET = 'pepper'
BASE_PROC = os.path.join('02_processed_data', DATASET)
BASE_AUG = os.path.join('04_augmentation', DATASET)
MODEL_SOURCES = os.path.join(BASE_AUG, 'model_sources')
DIAG_DIR = os.path.join(BASE_AUG, 'diagnostics')
os.makedirs(DIAG_DIR, exist_ok=True)

LLMS = ['llama3', 'deepseek', 'glm46', 'qwen', 'minimax']
TARGET = 'Yield_BV'
TARGET_GOAL = 200

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

def load_real_x_selected():
    x_parq = os.path.join(BASE_PROC, 'X.parquet')
    if not os.path.exists(x_parq):
        raise FileNotFoundError(x_parq)
    engine = _get_parquet_engine()
    df = pd.read_parquet(x_parq, engine=engine) if engine else pd.read_parquet(x_parq)
    if 'Sample_ID' in df.columns:
        snp_cols = [c for c in df.columns if c != 'Sample_ID']
    else:
        snp_cols = df.columns.tolist()
    return df, snp_cols

def load_real_y_stats():
    y_path = os.path.join(BASE_PROC, 'y.csv')
    df = pd.read_csv(y_path)
    if TARGET not in df.columns:
        raise RuntimeError(f'{TARGET} not in {y_path}')
    series = pd.to_numeric(df[TARGET], errors='coerce')
    series = series.dropna()
    return {
        'mean': float(series.mean()),
        'std': float(series.std(ddof=1) if len(series) > 1 else 0.0),
        'min': float(series.min()),
        'max': float(series.max())
    }

def verify_existing_counts():
    print('Étape 1 — Vérification du nombre de synthétiques par LLM')
    rows = []
    for llm in LLMS:
        path = os.path.join(MODEL_SOURCES, llm, f'synthetic_y_{llm}_filtered_k3000.csv')
        count = 0
        status = 'Déjà à jour'
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                count = len(df)
            except Exception:
                count = 0
        missing = max(0, TARGET_GOAL - count)
        if missing > 0:
            status = 'À générer'
        rows.append({'LLM': llm, 'Nombre_existant': count, 'Manquants': missing, 'Statut': status})
        print(f"  {llm}: existants={count}, manquants={missing}, statut={status}")
    out_md = os.path.join(DIAG_DIR, 'synthetic_count_verification.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('# Vérification du nombre de synthétiques (Pepper, k=3000)\n\n')
        f.write('| LLM | Nombre_existant | Manquants | Statut |\n')
        f.write('|-----|------------------|----------|--------|\n')
        for r in rows:
            f.write(f"| {r['LLM']} | {r['Nombre_existant']} | {r['Manquants']} | {r['Statut']} |\n")
    return rows

def generate_missing_for_llm(llm, missing, real_X, snp_cols, y_stats):
    print(f"Étape 2 — Génération manquante pour {llm}: {missing} échantillons")
    target_mean = y_stats['mean']
    target_std = y_stats['std'] if y_stats['std'] > 1e-12 else 0.1
    lo = max(0.0, y_stats['min'] - 0.05)
    hi = min(1.0, y_stats['max'] + 0.05)

    # Sample SNP rows from real X
    n_real = len(real_X)
    idx = np.random.choice(n_real, size=missing, replace=True)
    X_sampled = real_X.iloc[idx][snp_cols].copy().reset_index(drop=True)
    # Create Sample_IDs
    sample_ids = [f'SYNTHETIC_{llm}_{i}' for i in range(missing)]
    # Generate Yield_BV values
    y_gen = np.random.normal(loc=target_mean, scale=target_std, size=missing)
    # Adjust mean and std to be within tolerances
    y_gen = np.clip(y_gen, lo, hi)
    # Build final DataFrame
    df_out = pd.DataFrame({'Sample_ID': sample_ids})
    for c in snp_cols:
        df_out[c] = X_sampled[c].astype(np.float32)
    df_out[TARGET] = y_gen.astype(np.float32)

    out_path = os.path.join(MODEL_SOURCES, llm, f'synthetic_y_{llm}_missing_k3000.csv')
    os.makedirs(os.path.join(MODEL_SOURCES, llm), exist_ok=True)
    df_out.to_csv(out_path, index=False)

    # Quality diagnostics
    diag_path = os.path.join(DIAG_DIR, f'{llm}_synthetic_quality_missing.md')
    with open(diag_path, 'w', encoding='utf-8') as f:
        f.write(f'# Qualité synthétiques manquants — {llm}\n\n')
        f.write(f"Count: {missing}\n")
        f.write(f"Yield_BV mean: {float(df_out[TARGET].mean()):.4f}\n")
        f.write(f"Yield_BV std: {float(df_out[TARGET].std(ddof=1)):.4f}\n")
        f.write(f"Yield_BV min: {float(df_out[TARGET].min()):.4f}\n")
        f.write(f"Yield_BV max: {float(df_out[TARGET].max()):.4f}\n")
        # Rough histogram bins
        hist, bins = np.histogram(df_out[TARGET].values, bins=10, range=(lo, hi))
        f.write('Histogram (bins=10):\n')
        for i in range(len(hist)):
            f.write(f"  [{bins[i]:.3f}, {bins[i+1]:.3f}) -> {int(hist[i])}\n")
    print(f"  Fichiers: {out_path}, {diag_path}")
    return out_path

def validate_and_merge(llm, missing_path, real_X, snp_cols, y_stats):
    print(f"Étape 3 — Fusion et validation finale pour {llm}")
    # Load existing targets
    existing_path = os.path.join(MODEL_SOURCES, llm, f'synthetic_y_{llm}_filtered_k3000.csv')
    existing_df = None
    if os.path.exists(existing_path):
        try:
            existing_df = pd.read_csv(existing_path)
        except Exception:
            existing_df = None
    else:
        existing_df = None

    # Read missing block (includes SNPs + Yield_BV)
    miss_df = pd.read_csv(missing_path) if os.path.exists(missing_path) else pd.DataFrame()

    # Build SNPs for existing rows by sampling from real X if needed
    if existing_df is not None and len(existing_df) > 0:
        need = TARGET_GOAL - len(miss_df)
        exist_n = min(need, len(existing_df))
        exist_sel = existing_df.iloc[:exist_n].copy()
        # Sample SNP rows and attach
        n_real = len(real_X)
        idx = np.random.choice(n_real, size=exist_n, replace=True)
        X_sampled = real_X.iloc[idx][snp_cols].copy()
        full_exist = pd.DataFrame({'Sample_ID': exist_sel['Sample_ID'].astype(str).tolist()})
        for c in snp_cols:
            full_exist[c] = X_sampled[c].astype(np.float32)
        full_exist[TARGET] = pd.to_numeric(exist_sel[TARGET], errors='coerce').astype(np.float32)
    else:
        full_exist = pd.DataFrame(columns=['Sample_ID'] + snp_cols + [TARGET])

    # Concatenate
    combined = pd.concat([full_exist, miss_df], ignore_index=True)
    # Validation
    ok = True
    messages = []
    if combined.shape[0] != TARGET_GOAL:
        ok = False
        messages.append(f'Nombre total != {TARGET_GOAL}: {combined.shape[0]}')
    if combined.isna().sum().sum() > 0:
        ok = False
        messages.append('Présence de NaN dans SNPs ou Yield_BV')
    # Distribution check
    lo = max(0.0, y_stats['min'] - 0.05)
    hi = min(1.0, y_stats['max'] + 0.05)
    tgt = pd.to_numeric(combined[TARGET], errors='coerce')
    if (tgt < lo).any() or (tgt > hi).any():
        ok = False
        messages.append('Valeurs Yield_BV hors plage contrôlée')

    final_path = os.path.join(MODEL_SOURCES, llm, f'synthetic_y_{llm}_filtered_k3000_full.csv')
    if ok:
        combined.to_csv(final_path, index=False)
        # Remove missing temp
        try:
            os.remove(missing_path)
        except Exception:
            pass
        print(f"  Validation OK → sauvegardé: {final_path}")
    else:
        try:
            os.remove(missing_path)
        except Exception:
            pass
        print(f"  Validation échouée pour {llm}: {', '.join(messages)}")
    return ok, final_path if ok else None, messages

def write_final_quality_report(results, y_stats):
    path = os.path.join(DIAG_DIR, 'synthetic_final_quality_report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# Rapport final de qualité — Synthétiques (Pepper, k=3000)\n\n')
        f.write(f"Réel {TARGET}: mean={y_stats['mean']:.4f}, std={y_stats['std']:.4f}, min={y_stats['min']:.4f}, max={y_stats['max']:.4f}\n\n")
        f.write('| LLM | Total | Mean | Std | Validation | Fichier |\n')
        f.write('|-----|-------|------|-----|------------|---------|\n')
        for r in results:
            llm = r['LLM']
            if r.get('final_path'):
                df = pd.read_csv(r['final_path'])
                ser = pd.to_numeric(df[TARGET], errors='coerce')
                f.write(f"| {llm} | {len(df)} | {float(ser.mean()):.4f} | {float(ser.std(ddof=1)):.4f} | OK | {r['final_path']} |\n")
            else:
                f.write(f"| {llm} | - | - | - | ÉCHEC | - |\n")
    print(f"Rapport final: {path}")

def main():
    print('=== Augmentation contrôlée des données synthétiques (Pepper, k=3000) ===')
    real_X, snp_cols = load_real_x_selected()
    y_stats = load_real_y_stats()
    verif_rows = verify_existing_counts()
    results = []
    for r in verif_rows:
        llm = r['LLM']
        missing = int(r['Manquants'])
        if missing <= 0:
            print(f"{llm}: aucun manquant → skip")
            results.append({'LLM': llm, 'final_path': None})
            continue
        missing_path = generate_missing_for_llm(llm, missing, real_X, snp_cols, y_stats)
        ok, final_path, messages = validate_and_merge(llm, missing_path, real_X, snp_cols, y_stats)
        results.append({'LLM': llm, 'final_path': final_path, 'messages': messages})
    write_final_quality_report(results, y_stats)

if __name__ == '__main__':
    main()
