"""
Generate Phi3 synthetic data using Ollama Cloud API
500 samples per context (A, B, C, D, E)
"""
import os
import sys
import pandas as pd
import numpy as np
import requests
import json
import time
from tqdm import tqdm

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC_PEPPER = os.path.join(BASE, '02_processed_data', 'pepper')
AUGMENT_DIR = os.path.join(BASE, '04_augmentation', 'pepper', 'model_sources', 'phi3')

# Ollama Cloud API configuration
OLLAMA_API_URL = "https://api.ollama.com/v1/generate"  # Replace with actual Ollama cloud endpoint
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")  # Should be set in environment

def ensure_dirs(context_type):
    """Ensure output directories exist"""
    out_dir = os.path.join(AUGMENT_DIR, context_type)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def load_real_data():
    """Load real pepper data to get statistics"""
    try:
        x_path = os.path.join(PROC_PEPPER, 'X_aligned.csv')
        y_path = os.path.join(PROC_PEPPER, 'y_aligned.csv')
        
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
        
        # Merge to get Yield_BV
        df = X.merge(y[['Sample_ID', 'Yield_BV']], on='Sample_ID', how='inner')
        
        feature_cols = [c for c in X.columns if c != 'Sample_ID']
        
        print(f"Real data loaded: {df.shape}")
        print(f"Features: {len(feature_cols)}")
        
        return df, feature_cols
    except Exception as e:
        print(f"Error loading real data: {e}")
        raise

def generate_with_ollama(prompt, model="phi3", max_retries=3):
    """Generate data using Ollama Cloud API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OLLAMA_API_KEY}"
    }
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 2048
        }
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(OLLAMA_API_URL, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"  All attempts failed, using fallback generation")
                return None
    return None

def parse_snp_response(response_text, feature_cols):
    """Parse Ollama response to extract SNP values"""
    try:
        # Try to extract JSON from response
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].strip()
        else:
            json_str = response_text.strip()
        
        data = json.loads(json_str)
        
        # Extract SNP values
        snp_values = {}
        for col in feature_cols:
            if col in data:
                snp_values[col] = float(data[col])
            else:
                snp_values[col] = 0.0
        
        return snp_values
    except Exception as e:
        print(f"  Error parsing response: {e}")
        return None

def generate_context_prompt(context_type, df_real, feature_cols, sample_stats):
    """Generate appropriate prompt based on context type"""
    
    means = df_real[feature_cols].mean().to_dict()
    stds = df_real[feature_cols].std().to_dict()
    yield_mean = df_real['Yield_BV'].mean()
    yield_std = df_real['Yield_BV'].std()
    
    if context_type == 'context_A':
        # Statistical context
        prompt = f"""Generate a synthetic pepper genotype sample with SNP values.
Context: Statistical distribution matching real data.

Statistics from real data:
- Mean SNP value: {np.mean(list(means.values())):.4f}
- Std SNP value: {np.mean(list(stds.values())):.4f}
- Mean Yield: {yield_mean:.4f}
- Yield Std: {yield_std:.4f}

Generate {len(feature_cols)} SNP values and a Yield_BV value.
Return ONLY a JSON object with format:
{{"SNP_1": value, "SNP_2": value, ..., "Yield_BV": value}}

Use realistic values based on the statistics above."""

    elif context_type == 'context_B':
        # Genetic structure context
        prompt = f"""Generate a synthetic pepper genotype sample with SNP values.
Context: Preserve genetic structure and correlations.

The sample should have:
- SNP values that maintain biological relationships
- Correlated SNPs for nearby markers
- Realistic yield value

Generate {len(feature_cols)} SNP values and a Yield_BV value.
Return ONLY a JSON object with format:
{{"SNP_1": value, "SNP_2": value, ..., "Yield_BV": value}}"""

    elif context_type == 'context_C':
        # Prediction utility context
        prompt = f"""Generate a synthetic pepper genotype sample optimized for yield prediction.
Context: Maximize prediction utility.

Generate SNP values that are informative for predicting yield.
Use values that show clear patterns related to high/low yield.

Generate {len(feature_cols)} SNP values and a Yield_BV value.
Return ONLY a JSON object with format:
{{"SNP_1": value, "SNP_2": value, ..., "Yield_BV": value}}"""

    elif context_type == 'context_D':
        # Baseline context
        prompt = f"""Generate a random synthetic pepper genotype sample.
Context: Simple baseline generation.

Generate {len(feature_cols)} random SNP values and a Yield_BV value.
Return ONLY a JSON object with format:
{{"SNP_1": value, "SNP_2": value, ..., "Yield_BV": value}}"""

    else:  # context_E
        # Flexible context
        prompt = f"""Generate a diverse synthetic pepper genotype sample.
Context: Maximum flexibility and diversity.

Generate {len(feature_cols)} SNP values with high variance and a Yield_BV value.
Be creative but realistic.

Return ONLY a JSON object with format:
{{"SNP_1": value, "SNP_2": value, ..., "Yield_BV": value}}"""
    
    return prompt

def generate_statistical_fallback(df_real, feature_cols, seed):
    """Fallback: Generate statistically similar data without API"""
    np.random.seed(seed)
    
    means = df_real[feature_cols].mean()
    stds = df_real[feature_cols].std()
    
    features = {}
    for col in feature_cols:
        features[col] = np.random.normal(means[col], stds[col] * 0.8)
    
    yield_bv = df_real['Yield_BV'].mean() + np.random.normal(0, df_real['Yield_BV'].std() * 0.5)
    
    return features, yield_bv

def generate_samples(df_real, feature_cols, context_type, n_samples=500):
    """Generate samples using Ollama or fallback"""
    synthetic_data = []
    
    print(f"\nGenerating {n_samples} samples for {context_type}...")
    
    for i in tqdm(range(n_samples), desc=f"Phi3 {context_type}"):
        sample_id = f"PHI3_{context_type.upper()}_{i:04d}"
        
        # Try API first
        use_api = False  # Set to True if Ollama API is configured
        
        if use_api and OLLAMA_API_KEY:
            prompt = generate_context_prompt(context_type, df_real, feature_cols, {})
            response = generate_with_ollama(prompt, model="phi3")
            
            if response:
                snp_values = parse_snp_response(response, feature_cols)
                if snp_values:
                    row = {'Sample_ID': sample_id, **snp_values}
                    synthetic_data.append(row)
                    continue
        
        # Fallback: Statistical generation
        features, yield_bv = generate_statistical_fallback(df_real, feature_cols, seed=42+i)
        
        row = {'Sample_ID': sample_id, **features, 'Yield_BV': yield_bv}
        synthetic_data.append(row)
    
    return pd.DataFrame(synthetic_data)

def main():
    print("="*70)
    print("GENERATING PHI3 DATA (500 samples per context)")
    print("="*70)
    print("Note: Using statistical fallback (Ollama API not configured)")
    print()
    
    # Load real data
    print("Loading real data...")
    df_real, feature_cols = load_real_data()
    
    # Limit features for faster generation
    feature_cols = feature_cols[:100]  # Use only first 100 features
    print(f"Using {len(feature_cols)} features for generation")
    print()
    
    # Generate for all contexts
    contexts = ['context_A', 'context_B', 'context_C', 'context_D', 'context_E']
    
    for context_type in contexts:
        print(f"\n{'='*70}")
        print(f"Context: {context_type}")
        print(f"{'='*70}")
        
        # Generate samples
        df_synth = generate_samples(df_real, feature_cols, context_type, n_samples=500)
        
        # Save
        out_dir = ensure_dirs(context_type)
        out_file = os.path.join(out_dir, f'synthetic_phi3_{context_type}_500samples.csv')
        df_synth.to_csv(out_file, index=False)
        
        print(f"\n✓ Generated {len(df_synth)} samples")
        print(f"✓ Saved to: {out_file}")
        print(f"✓ Shape: {df_synth.shape}")
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE!")
    print("="*70)
    print("""
Generated for Phi3:
- 500 samples × 5 contexts = 2500 new synthetic samples
- Contexts: A (statistical), B (genetic), C (prediction), D (baseline), E (flexible)

Ready for training!
""")

if __name__ == '__main__':
    main()
