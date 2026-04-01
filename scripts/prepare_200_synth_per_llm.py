import os
import numpy as np
import pandas as pd

DATASET = 'pepper'
BASE_PROC = os.path.join('02_processed_data', DATASET)
BASE_AUG = os.path.join('04_augmentation', DATASET)
MODEL_SOURCES = os.path.join(BASE_AUG, 'model_sources')
LLMS = ['llama3','deepseek','glm46','qwen','minimax']
TARGET = 'Yield_BV'
OUT_SUFFIX = '_filtered_k3000_200.csv'

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
    engine = _get_parquet_engine()
    df = pd.read_parquet(x_parq, engine=engine) if engine else pd.read_parquet(x_parq)
    if 'Sample_ID' in df.columns:
        snp_cols = [c for c in df.columns if c != 'Sample_ID']
    else:
        snp_cols = df.columns.tolist()
    return df, snp_cols

def prepare_llm(llm, real_X, snp_cols):
    src = os.path.join(MODEL_SOURCES, llm, f'synthetic_y_{llm}_filtered_k3000.csv')
    if not os.path.exists(src):
        # fallback to generic filtered if present
        print(f"[WARN] {src} not found; skipping {llm}")
        return None
    df = pd.read_csv(src)
    # Keep only needed columns
    keep_cols = ['Sample_ID', TARGET]
    df = df[[c for c in df.columns if c in keep_cols]].copy()
    # Sample to 200 rows deterministically
    np.random.seed(42)
    idx = np.random.choice(len(df), size=min(200, len(df)), replace=False)
    df = df.iloc[idx].reset_index(drop=True)
    # If fewer than 200, upsample
    if len(df) < 200:
        add_idx = np.random.choice(len(df), size=200 - len(df), replace=True)
        df = pd.concat([df, df.iloc[add_idx].reset_index(drop=True)], ignore_index=True)
    # Attach SNPs sampled from real_X
    rx_idx = np.random.choice(len(real_X), size=200, replace=True)
    X_sampled = real_X.iloc[rx_idx][snp_cols].reset_index(drop=True)
    out = pd.concat([df[['Sample_ID']], X_sampled.astype('float32')], axis=1)
    out[TARGET] = pd.to_numeric(df[TARGET], errors='coerce').astype('float32')
    out_path = os.path.join(MODEL_SOURCES, llm, f'synthetic_y_{llm}{OUT_SUFFIX}')
    out.to_csv(out_path, index=False)
    print(f"[OK] {llm} → {out_path}")
    return out_path

def main():
    print('=== Préparation des 200 synthétiques par LLM (Pepper, k=3000) ===')
    real_X, snp_cols = load_real_x_selected()
    for llm in LLMS:
        prepare_llm(llm, real_X, snp_cols)

if __name__ == '__main__':
    main()

