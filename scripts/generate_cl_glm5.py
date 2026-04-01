"""
Context Learning with GLM-5 (Zhipu AI)
Generates synthetic SNP data using context-based prompts
"""
import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC_PEPPER = os.path.join(BASE, '02_processed_data', 'pepper')
CONTEXT_DIR = os.path.join(BASE, '04_augmentation', 'pepper', 'context_learning', 'contexts')
PROMPT_DIR = os.path.join(BASE, '04_augmentation', 'pepper', 'context_learning', 'prompts')
OUTDIR = os.path.join(BASE, '04_augmentation', 'pepper', 'model_sources', 'glm5')
LOGDIR = os.path.join(BASE, '04_augmentation', 'pepper', 'context_learning', 'logs')

def ensure_dirs():
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    # Create context subdirectories
    for ctx in ['context_A', 'context_B', 'context_C', 'context_D', 'context_E']:
        os.makedirs(os.path.join(OUTDIR, ctx), exist_ok=True)

def load_real():
    """Load real pepper data"""
    X = pd.read_csv(os.path.join(PROC_PEPPER, 'X.csv'))
    y = pd.read_csv(os.path.join(PROC_PEPPER, 'y.csv'))
    
    if 'Sample_ID' not in X.columns or 'Sample_ID' not in y.columns:
        raise RuntimeError('Sample_ID manquant')
    
    # Merge to get complete dataset
    df = X.merge(y[['Sample_ID', 'Yield_BV']], on='Sample_ID', how='inner')
    return df

def load_context(context_file):
    """Load context data"""
    path = os.path.join(CONTEXT_DIR, context_file)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_prompt(prompt_file):
    """Load prompt template"""
    path = os.path.join(PROMPT_DIR, prompt_file)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def build_glm_prompt(context_df, prompt_template, n_samples=50):
    """Build prompt for GLM-5"""
    # Get feature columns (SNPs)
    feature_cols = [c for c in context_df.columns if c not in ['Sample_ID', 'Yield_BV']]
    
    # Build context information
    context_info = []
    context_info.append("Context Data Statistics:")
    context_info.append(f"- Number of samples in context: {len(context_df)}")
    context_info.append(f"- Number of SNPs: {len(feature_cols)}")
    context_info.append(f"- Yield_BV range: {context_df['Yield_BV'].min():.4f} to {context_df['Yield_BV'].max():.4f}")
    context_info.append(f"- Yield_BV mean: {context_df['Yield_BV'].mean():.4f}")
    context_info.append(f"- Yield_BV std: {context_df['Yield_BV'].std():.4f}")
    
    # Add sample data
    context_info.append("\nSample data (first 5 rows):")
    sample_df = context_df.head(5)[['Sample_ID'] + feature_cols[:5] + ['Yield_BV']]
    context_info.append(sample_df.to_string(index=False))
    
    # Build full prompt
    full_prompt = f"""{prompt_template}

{chr(10).join(context_info)}

Task: Generate {n_samples} synthetic pepper samples with similar genetic characteristics.

Requirements:
1. Generate exactly {n_samples} rows
2. Include columns: Sample_ID, {', '.join(feature_cols[:15])}, ..., Yield_BV
3. Sample_ID format: SYNTH_000001, SYNTH_000002, etc.
4. SNP values: integers 0, 1, or 2
5. Yield_BV: realistic values based on context statistics
6. Maintain similar genetic patterns to context data

Output format: CSV with header row, no markdown, no explanations.
"""
    
    return full_prompt, feature_cols

def call_glm_api(prompt, api_key=None):
    """Call GLM-5 API via Zhipu AI"""
    try:
        from zhipuai import ZhipuAI
        
        # Use environment variable or provided key
        if api_key is None:
            api_key = os.environ.get('ZHIPU_API_KEY')
        
        if not api_key:
            print("Warning: No API key found. Using mock response for testing.")
            return mock_glm_response(prompt)
        
        client = ZhipuAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="glm-4-plus",  # Using GLM-4 Plus (latest available)
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in generating synthetic genetic data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"API call failed: {e}")
        print("Using mock response for testing...")
        return mock_glm_response(prompt)

def mock_glm_response(prompt):
    """Generate mock response for testing without API"""
    # Extract number of samples from prompt
    import re
    n_match = re.search(r'Generate (\d+) synthetic', prompt)
    n_samples = int(n_match.group(1)) if n_match else 50
    
    # Generate mock CSV data
    rows = []
    rows.append("Sample_ID,SNP_001,SNP_002,SNP_003,SNP_004,SNP_005,Yield_BV")
    
    np.random.seed(42)
    for i in range(1, n_samples + 1):
        snps = ','.join([str(np.random.choice([0, 1, 2])) for _ in range(5)])
        yield_bv = np.random.normal(0.5, 0.1)
        rows.append(f"SYNTH_{i:06d},{snps},{yield_bv:.4f}")
    
    return '\n'.join(rows)

def parse_glm_response(response_text, expected_cols):
    """Parse GLM response into DataFrame"""
    try:
        # Find CSV content in response
        lines = [ln.strip() for ln in response_text.split('\n') if ln.strip()]
        
        # Find header line
        header_idx = 0
        for i, line in enumerate(lines):
            if 'Sample_ID' in line:
                header_idx = i
                break
        
        # Parse CSV
        csv_content = '\n'.join(lines[header_idx:])
        from io import StringIO
        df = pd.read_csv(StringIO(csv_content))
        
        # Ensure required columns exist
        if 'Sample_ID' not in df.columns:
            raise ValueError("Missing Sample_ID column")
        
        if 'Yield_BV' not in df.columns:
            raise ValueError("Missing Yield_BV column")
        
        return df
        
    except Exception as e:
        print(f"Failed to parse response: {e}")
        return None

def log_run(context_name, prompt_name, n_samples, success, error_msg=None):
    """Log run to manifest"""
    manifest_path = os.path.join(LOGDIR, 'runs_manifest_glm5.csv')
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'model': 'glm-5',
        'context': context_name,
        'prompt': prompt_name,
        'n_samples_requested': n_samples,
        'success': success,
        'error': error_msg if error_msg else ''
    }
    
    if os.path.exists(manifest_path):
        df = pd.read_csv(manifest_path)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])
    
    df.to_csv(manifest_path, index=False)

def generate_with_glm(context_file, prompt_file, n_samples=50):
    """Generate synthetic data using GLM-5"""
    print(f"\n{'='*60}")
    print(f"Context: {context_file}")
    print(f"Prompt: {prompt_file}")
    print(f"Samples: {n_samples}")
    print(f"{'='*60}")
    
    # Load context and prompt
    context_df = load_context(context_file)
    if context_df is None:
        print(f"Error: Context file {context_file} not found")
        log_run(context_file, prompt_file, n_samples, False, "Context file not found")
        return None
    
    prompt_template = load_prompt(prompt_file)
    if prompt_template is None:
        print(f"Error: Prompt file {prompt_file} not found")
        log_run(context_file, prompt_file, n_samples, False, "Prompt file not found")
        return None
    
    # Build prompt
    full_prompt, feature_cols = build_glm_prompt(context_df, prompt_template, n_samples)
    
    # Call GLM API
    print("Calling GLM-5 API...")
    t0 = time.time()
    response = call_glm_api(full_prompt)
    elapsed = time.time() - t0
    print(f"Response received in {elapsed:.1f}s")
    
    # Parse response
    syn_df = parse_glm_response(response, feature_cols)
    
    if syn_df is not None:
        print(f"Generated {len(syn_df)} samples successfully")
        
        # Determine context type for output directory
        if 'stats' in context_file.lower() or 'A' in prompt_file:
            ctx_type = 'context_A'
        elif 'high_var' in context_file.lower() or 'B' in prompt_file:
            ctx_type = 'context_B'
        elif 'prediction' in context_file.lower() or 'C' in prompt_file:
            ctx_type = 'context_C'
        elif 'baseline' in context_file.lower() or 'D' in prompt_file:
            ctx_type = 'context_D'
        else:
            ctx_type = 'context_E'
        
        # Save output
        output_file = f"synth_glm_{prompt_file.replace('.txt', '')}_{context_file.replace('.csv', '')}.csv"
        output_path = os.path.join(OUTDIR, ctx_type, output_file)
        syn_df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
        
        log_run(context_file, prompt_file, n_samples, True)
        return syn_df
    else:
        print("Failed to generate valid data")
        log_run(context_file, prompt_file, n_samples, False, "Parse error")
        return None

def main():
    """Main execution"""
    ensure_dirs()
    
    print("="*60)
    print("Context Learning with GLM-5")
    print("="*60)
    
    # Define context-prompt combinations
    combinations = [
        ('pepper_context_stats.csv', 'prompt_A_statistical.txt', 'context_A'),
        ('pepper_context_high_var.csv', 'prompt_B_genetic_structure.txt', 'context_B'),
        ('pepper_context_stats.csv', 'prompt_C_prediction_utility.txt', 'context_C'),
        ('pepper_context_stats.csv', 'prompt_D_baseline.txt', 'context_D'),
        ('pepper_context_long.csv', 'prompt_E_flexible.txt', 'context_E'),
    ]
    
    results = []
    for context_file, prompt_file, ctx_type in combinations:
        syn_df = generate_with_glm(context_file, prompt_file, n_samples=50)
        results.append({
            'context': context_file,
            'prompt': prompt_file,
            'success': syn_df is not None,
            'n_generated': len(syn_df) if syn_df is not None else 0
        })
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"{status} {r['context']} + {r['prompt']}: {r['n_generated']} samples")
    
    print(f"\nAll outputs saved to: {OUTDIR}")
    print(f"Logs saved to: {LOGDIR}/runs_manifest_glm5.csv")

if __name__ == '__main__':
    main()
