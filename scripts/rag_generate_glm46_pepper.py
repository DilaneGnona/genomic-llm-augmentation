import os
import pandas as pd
import numpy as np
import subprocess

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CTX_PATH = os.path.join(BASE, '04_augmentation', 'pepper', 'rag_context_pepper.csv')
OUT_DIR = os.path.join(BASE, '04_augmentation', 'pepper', 'glm46')
OUT_PATH = os.path.join(OUT_DIR, 'synthetic_glm46_pepper_200_rag.csv')

def load_context():
    df = pd.read_csv(CTX_PATH)
    cols = [c for c in df.columns if c != 'Sample_ID']
    return df, cols

def to_raw_csv_string(df: pd.DataFrame):
    return df.to_csv(index=False)

def build_prompt(context_csv: str, cols: list):
    # User-provided prompt template
    prompt_template = """You are an expert in plant genomics and genomic prediction. 
 
 Your task is to generate SYNTHETIC genomic samples (SNP → phenotype) 
 for a supervised learning experiment on the PEPPER dataset. 
 
 IMPORTANT: This is a scientific experiment. 
 You must strictly follow the constraints below. 
 
 -------------------------------------------------- 
 GENERAL CONTEXT 
 -------------------------------------------------- 
 - Task: Genomic prediction (SNP → Yield_BV) 
 - Dataset: Pepper (Gold Standard) 
 - Use case: Data augmentation to improve downstream machine learning 
 - Downstream model: Random Forest (fixed, same hyperparameters as baseline) 
 - Evaluation: Model will be trained on (real train + synthetic) and tested on a fixed real holdout set 
 - Therefore, synthetic data MUST be realistic and diverse 
 
 -------------------------------------------------- 
 INPUT CONTEXT (TRAIN ONLY) 
 -------------------------------------------------- 
 You will be provided with: 
 - A small context extracted ONLY from the training split 
 - This context represents realistic SNP patterns and Yield_BV values 
 - Do NOT assume access to the test or holdout data 
 
 -------------------------------------------------- 
 GENERATION TASK 
 -------------------------------------------------- 
 Generate EXACTLY 200 synthetic samples. 
 
 Each sample must contain: 
 - The same SNP columns as the Pepper dataset 
 - One target column: Yield_BV 
 
 -------------------------------------------------- 
 CRITICAL CONSTRAINTS 
 -------------------------------------------------- 
 1. SNP VALUES 
 - SNPs are discrete numeric values (e.g., 0 / 1 / 2) 
 - Respect realistic allele frequencies 
 - Avoid duplicated or near-duplicated samples 
 - Maintain diversity across samples 
 
 2. PHENOTYPE (Yield_BV) 
 - Yield_BV must be continuous 
 - Yield_BV must fall within the realistic range observed in the training data 
 - Preserve natural variance (DO NOT collapse to a constant value) 
 
 3. STRUCTURE 
 - Preserve plausible relationships between SNPs and Yield_BV 
 - Do NOT generate random noise unrelated to genetics 
 - Do NOT overfit or copy samples from the context 
 
 4. FORMAT (MANDATORY) 
 - Output MUST be valid CSV 
 - Header row included 
 - One row = one sample 
 - No explanations, no comments, no markdown 
 - Output ONLY the CSV 
 
 -------------------------------------------------- 
 EXPERIMENTAL CONTROL 
 -------------------------------------------------- 
 - This generation will be repeated with different random seeds 
 - Do NOT assume this is a one-shot generation 
 - Consistency and reproducibility are critical 
 
 -------------------------------------------------- 
 CONTEXT DATA
 --------------------------------------------------
 Here is the context CSV (use these patterns to inform your generation):
 {context_csv}
 
 -------------------------------------------------- 
 FINAL INSTRUCTION 
 -------------------------------------------------- 
 Generate the synthetic dataset now. 
 
 Output ONLY the CSV file.
 """
    return prompt_template.format(context_csv=context_csv)

def run_ollama_generate(prompt: str):
    models_to_try = ['llama3:latest', 'glm-4.6:cloud', 'glm46', 'glm-4.6']
    
    # Force subprocess for debugging
    # try:
    #     import ollama
    #     for m in models_to_try:
    #         try:
    #             r = ollama.generate(model=m, prompt=prompt)
    #             resp = r.get('response', '')
    #             if resp: return resp
    #         except:
    #             continue
    # except ImportError:
    #     pass

    print("Trying subprocess...", flush=True)
    # Fallback to subprocess
    for m in models_to_try:
        print(f"Trying model: {m}", flush=True)
        try:
            # Pass prompt via stdin to avoid command line length limits and quoting issues
            p = subprocess.run(['ollama', 'run', m], input=prompt, capture_output=True, encoding='utf-8', timeout=600)
            print(f"Subprocess finished. Return code: {p.returncode}", flush=True)
            if p.returncode == 0 and p.stdout.strip():
                return p.stdout
            else:
                print(f"Model {m} failed with code {p.returncode}", flush=True)
                print(f"Stderr: {p.stderr}", flush=True)
        except Exception as e:
            print(f"Model {m} exception: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
            
    return ''

def normalize_csv_text(text: str):
    t = text.strip()
    if '```' in t:
        t = t.split('```')[1].strip() if len(t.split('```')) > 1 else t
    t = t.replace('\r\n', '\n').replace('\r', '\n')
    return t

def parse_llm_csv(text: str, cols: list, N: int):
    t = normalize_csv_text(text)
    df = pd.read_csv(pd.io.common.StringIO(t))
    want = ['Sample_ID'] + cols
    for c in want:
        if c not in df.columns:
            df[c] = 0
    df = df[want + (['Yield_BV'] if 'Yield_BV' in df.columns else [])]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').round().clip(0,2).astype('Int64').fillna(0).astype(np.int32)
    if 'Yield_BV' in df.columns:
        df['Yield_BV'] = pd.to_numeric(df['Yield_BV'], errors='coerce')
    df = df.head(N).copy()
    ids = [f'SYNTH_{i:06d}' for i in range(1, len(df)+1)]
    df['Sample_ID'] = ids
    return df

def enforce_yield_stats(df: pd.DataFrame, mean_target: float = 6.7861, std_target: float = 0.1008, lo: float = 6.58, hi: float = 6.99):
    if 'Yield_BV' not in df.columns:
        df['Yield_BV'] = mean_target
    y = df['Yield_BV'].fillna(mean_target).values.astype(np.float64)
    y = np.clip(y, lo, hi)
    mu = float(y.mean())
    sd = float(y.std(ddof=1)) if len(y) > 1 else 0.0
    if sd > 0:
        y = mu + (y - mu) * (std_target / sd)
    y = np.clip(y, lo, hi)
    delta = mu - mean_target
    y = np.clip(y - delta, lo, hi)
    df['Yield_BV'] = y
    return df

def main():
    print("Starting generation...", flush=True)
    print(f"OUT_PATH: {OUT_PATH}", flush=True)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    ctx_df, cols = load_context()
    print(f"Context loaded. Shape: {ctx_df.shape}", flush=True)
    print(f"Columns: {cols}", flush=True)
    
    ctx_csv = to_raw_csv_string(ctx_df)
    prompt = build_prompt(ctx_csv, cols)
    print("Prompt built. Length:", len(prompt), flush=True)
    
    text = run_ollama_generate(prompt)
    print("LLM response received. Length:", len(text), flush=True)
    # print("First 100 chars:", text[:100])

    if not text.strip():
        raise RuntimeError('Ollama generation failed')
    
    print("Parsing CSV...")
    df = parse_llm_csv(text, cols, 200)
    print(f"Parsed DataFrame shape: {df.shape}")
    
    df = enforce_yield_stats(df, 6.7861, 0.1008, 6.58, 6.99)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved to: {OUT_PATH}")

if __name__ == '__main__':
    main()
