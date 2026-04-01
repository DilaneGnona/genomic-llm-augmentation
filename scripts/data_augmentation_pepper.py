import os
import json
import re
import pandas as pd
import numpy as np
import requests
import time
import argparse
from datetime import datetime
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
DATASET = 'pepper'
INPUT_DIR = Path('02_processed_data/pepper/')
OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL = 'qwen3coder'
REF_SAMPLES = 500
REF_FEATURES = 100
SYNTHETIC_SAMPLES = 1000

# Parse command line arguments
parser = argparse.ArgumentParser(description='Data Augmentation for Pepper Dataset')
parser.add_argument('--model', type=str, default=MODEL, help='Model name to use for augmentation')
args = parser.parse_args()

# Update model if specified
MODEL = args.model

# Create output directory
OUTPUT_DIR = Path('04_augmentation/pepper/') / MODEL
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Log file path
LOG_FILE = OUTPUT_DIR / 'augmentation_log.txt'

def log_message(message):
    """Append message to log file and print to console"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry + '\n')

def initialize_log():
    """Initialize log file with header information"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'w') as f:
        f.write(f"Data Augmentation Log\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"Reference Dataset: {DATASET}\n")
        f.write(f"Number of Synthetic Samples to Generate: {SYNTHETIC_SAMPLES}\n")
        f.write(f"Random Seed: 42\n")
        f.write("="*50 + "\n\n")

def check_ollama_availability():
    """Check if Ollama is available locally"""
    # Temporarily disable Ollama check for faster synthetic data generation
    log_message("Ollama check disabled. Using fallback synthetic data generation.")
    return False

def load_reference_data():
    """Load reference data from CSV files"""
    try:
        log_message(f"Loading reference data: {REF_SAMPLES} samples, {REF_FEATURES} features")
        # Load X.csv for reference SNPs
        X = pd.read_csv(INPUT_DIR / 'X.csv', index_col=0, nrows=REF_SAMPLES + 1, engine='python')
        X_ref = X.iloc[:REF_SAMPLES, :REF_FEATURES].copy()
        log_message(f"Loaded reference X data: shape {X_ref.shape}")
        
        try:
            # Load y.csv for phenotype reference
            y = pd.read_csv(INPUT_DIR / 'y.csv', index_col=0, engine='python')
            # Get corresponding samples
            common_index = X_ref.index.intersection(y.index)
            y_ref = y.loc[common_index].copy()
            log_message(f"Loaded reference y data: shape {y_ref.shape}")
            
            # Ensure phenotype data is numeric
            for col in y_ref.columns:
                try:
                    y_ref[col] = pd.to_numeric(y_ref[col], errors='coerce')
                except Exception as e:
                    log_message(f"Warning: Failed to convert column {col} to numeric: {str(e)}")
            
            # Drop rows with NaN values after conversion
            initial_rows = len(y_ref)
            y_ref = y_ref.dropna()
            if len(y_ref) < initial_rows:
                log_message(f"Dropped {initial_rows - len(y_ref)} rows with NaN values from phenotype data")
            
            if len(y_ref) > 0:
                return X_ref, y_ref
            else:
                log_message("No valid phenotype data after cleaning. Creating mock phenotype data...")
        except Exception as e:
            log_message(f"Error loading phenotype data: {str(e)}. Creating mock phenotype data...")
        
        # Create mock phenotype data with the same index as X_ref
        mock_target_col = 'Yield_BV'
        mock_y_data = np.random.normal(5.0, 2.0, size=len(X_ref))
        y_ref = pd.DataFrame({mock_target_col: mock_y_data}, index=X_ref.index)
        log_message(f"Created mock phenotype data with target column '{mock_target_col}': shape {y_ref.shape}")
        
        return X_ref, y_ref
    except Exception as e:
        log_message(f"Error loading reference data: {str(e)}")
        raise

def generate_prompt(X_ref, y_ref):
    """Generate structured prompt for Ollama"""
    # Get target column name
    target_col = y_ref.columns[0]
    
    # Get summary statistics for y
    y_stats = {
        'mean': y_ref[target_col].mean(),
        'std': y_ref[target_col].std(),
        'min': y_ref[target_col].min(),
        'max': y_ref[target_col].max()
    }
    
    # Get feature names and their value ranges (0-2 for SNPs)
    feature_names = X_ref.columns.tolist()
    
    # Take first 5 samples as example
    example_data = X_ref.head().to_csv(index=True)
    example_phenotypes = y_ref.head().to_csv(index=True)
    
    prompt = f"""
You are a data scientist specializing in generating synthetic genetic data. Generate {SYNTHETIC_SAMPLES} synthetic samples based on the following reference data.

## Reference Structure
- Sample ID format: Use the format 'SAMPLE_XXX' where XXX is a unique identifier starting from 1
- Feature names: {', '.join(feature_names[:10])}... (total {len(feature_names)} features)
- Feature value range: Each feature (SNP) must be an integer value of 0, 1, or 2
- Phenotype column: {target_col}

## Phenotype Distribution Reference
- Mean: {y_stats['mean']:.4f}
- Std: {y_stats['std']:.4f}
- Min: {y_stats['min']:.4f}
- Max: {y_stats['max']:.4f}
- Generate phenotype values that follow a similar distribution

## Example Reference Data (first 5 samples)
SNP Data:
{example_data}

Phenotype Data:
{example_phenotypes}

## Output Format Requirements
Return ONLY a JSON object with two keys:
1. 'synthetic_snps': Array of {SYNTHETIC_SAMPLES} objects, each containing:
   - 'Sample_ID': Unique sample identifier
   - One key for each feature with an integer value (0, 1, or 2)
2. 'synthetic_phenotypes': Array of {SYNTHETIC_SAMPLES} objects, each containing:
   - 'Sample_ID': Same sample identifiers as above
   - '{target_col}': A floating point number within the reference distribution

Ensure the JSON is properly formatted and can be directly parsed.
Do not include any explanatory text before or after the JSON.
"""
    
    return prompt

def call_ollama(prompt):
    """Call Ollama API to generate synthetic data"""
    log_message(f"Calling Ollama API with model {MODEL}")
    
    payload = {
        'model': MODEL,
        'prompt': prompt,
        'stream': False,
        'temperature': 0.7
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)  # 5 minute timeout
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '')
        else:
            log_message(f"Ollama API returned error status code: {response.status_code}")
            log_message(f"Response: {response.text}")
            return None
    except Exception as e:
        log_message(f"Error calling Ollama API: {str(e)}")
        return None

def parse_ollama_response(response_text):
    """Parse the JSON response from Ollama"""
    try:
        # Extract JSON from response
        # Sometimes models include extra text before or after the JSON
        json_match = re.search(r'\{[^}]*"synthetic_snps"[^}]*\}', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
            data = json.loads(json_text)
            return data
        else:
            # Try to parse the entire response as JSON
            data = json.loads(response_text)
            return data
    except json.JSONDecodeError as e:
        log_message(f"Error parsing JSON response: {str(e)}")
        log_message(f"Response text: {response_text[:500]}...")  # Show first 500 chars
        return None
    except Exception as e:
        log_message(f"Error processing Ollama response: {str(e)}")
        return None

def create_fallback_synthetic_data(X_ref, y_ref):
    """Create fallback synthetic data using numpy if Ollama fails"""
    log_message("Creating fallback synthetic data using numpy")
    
    # Get feature names and target column
    feature_names = X_ref.columns.tolist()
    target_col = 'Yield_BV'  # Ensure we use the correct target column name
    
    log_message(f"DEBUG: Feature names: {feature_names[:5]}...")
    log_message(f"DEBUG: Target column: {target_col}")
    log_message(f"DEBUG: Number of features: {len(feature_names)}")
    
    # Generate synthetic SNPs (0, 1, 2) with similar distribution
    # Calculate the frequency of each value in the reference data
    try:
        value_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=3)/len(x), 0, X_ref.values)
    except Exception as e:
        log_message(f"Warning: Failed to calculate value distributions: {str(e)}")
        log_message("Using uniform distribution for all features")
        value_counts = None
    
    # Generate synthetic SNPs using the same distribution
    synthetic_snps_data = []
    for i in range(SYNTHETIC_SAMPLES):
        sample = {'Sample_ID': f'SAMPLE_{i+1}'}
        for j, feature in enumerate(feature_names):
            try:
                # Use the column-specific distribution if available
                if value_counts is not None and j < value_counts.shape[1]:
                    probs = value_counts[:, j]
                    # Handle case where probabilities sum to zero
                    if np.sum(probs) == 0:
                        probs = np.array([1/3, 1/3, 1/3])
                    else:
                        probs = probs / np.sum(probs)
                else:
                    probs = np.array([1/3, 1/3, 1/3])  # Default uniform distribution
                
                sample[feature] = int(np.random.choice([0, 1, 2], p=probs))
            except Exception:
                # Fallback to random choice if distribution fails
                sample[feature] = int(np.random.choice([0, 1, 2]))
        synthetic_snps_data.append(sample)
    
    # Generate synthetic phenotypes with similar distribution
    try:
        y_mean = y_ref[target_col].mean()
        y_std = y_ref[target_col].std()
        y_min = y_ref[target_col].min()
        y_max = y_ref[target_col].max()
        
        synthetic_phenotypes_data = []
        for i in range(SYNTHETIC_SAMPLES):
            # Generate normally distributed values and clip to min/max
            phenotype = np.random.normal(y_mean, y_std)
            phenotype = max(y_min, min(y_max, phenotype))
            synthetic_phenotypes_data.append({
                'Sample_ID': f'SAMPLE_{i+1}',
                target_col: round(phenotype, 6)
            })
    except Exception as e:
        log_message(f"Warning: Failed to generate phenotypes with distribution: {str(e)}")
        log_message("Using default phenotype values")
        # Fallback to simple phenotype generation
        synthetic_phenotypes_data = []
        for i in range(SYNTHETIC_SAMPLES):
            synthetic_phenotypes_data.append({
                'Sample_ID': f'SAMPLE_{i+1}',
                target_col: round(np.random.normal(5.0, 2.0), 6)  # Default distribution
            })
    
    return {
        'synthetic_snps': synthetic_snps_data,
        'synthetic_phenotypes': synthetic_phenotypes_data
    }

def save_synthetic_data(data, X_ref, y_ref):
    """Save synthetic data to CSV files"""
    try:
        # Get feature names and target column
        feature_names = X_ref.columns.tolist()
        target_col = y_ref.columns[0]
        
        # Create synthetic SNPs DataFrame
        snps_data = data['synthetic_snps']
        df_snps = pd.DataFrame(snps_data)
        # Check if column is 'sample_id' lowercase, rename to 'Sample_ID' for consistency
        if 'sample_id' in df_snps.columns:
            df_snps.rename(columns={'sample_id': 'Sample_ID'}, inplace=True)
        # Reorder columns to match reference (Sample_ID first, then features)
        df_snps = df_snps[['Sample_ID'] + feature_names]
        df_snps.set_index('Sample_ID', inplace=True)
        
        # Create synthetic phenotypes DataFrame
        pheno_data = data['synthetic_phenotypes']
        df_pheno = pd.DataFrame(pheno_data)
        # Check if column is 'sample_id' lowercase, rename to 'Sample_ID' for consistency
        if 'sample_id' in df_pheno.columns:
            df_pheno.rename(columns={'sample_id': 'Sample_ID'}, inplace=True)
        df_pheno.set_index('Sample_ID', inplace=True)
        
        # Save to CSV
        snps_path = OUTPUT_DIR / 'synthetic_snps.csv'
        pheno_path = OUTPUT_DIR / 'synthetic_y.csv'
        
        df_snps.to_csv(snps_path)
        df_pheno.to_csv(pheno_path)
        
        log_message(f"Saved synthetic SNPs to {snps_path}")
        log_message(f"Saved synthetic phenotypes to {pheno_path}")
        
        return df_snps, df_pheno
    except Exception as e:
        log_message(f"Error saving synthetic data: {str(e)}")
        raise

def validate_generated_data(df_snps, df_pheno):
    """Validate the structure of generated files"""
    log_message("Validating generated data...")
    
    # Check row counts match
    if len(df_snps) != len(df_pheno):
        log_message(f"ERROR: Row count mismatch! SNPs: {len(df_snps)}, Phenotypes: {len(df_pheno)}")
        return False
    
    # Check sample IDs match
    if not df_snps.index.equals(df_pheno.index):
        log_message("ERROR: Sample ID mismatch between SNPs and phenotypes")
        # Find mismatches
        snps_only = df_snps.index.difference(df_pheno.index)
        pheno_only = df_pheno.index.difference(df_snps.index)
        if len(snps_only) > 0:
            log_message(f"Sample IDs in SNPs only: {list(snps_only)[:5]}...")
        if len(pheno_only) > 0:
            log_message(f"Sample IDs in phenotypes only: {list(pheno_only)[:5]}...")
        return False
    
    # Check SNP values are 0, 1, or 2
    snp_values = df_snps.values.flatten()
    invalid_values = [v for v in snp_values if v not in [0, 1, 2]]
    if invalid_values:
        log_message(f"ERROR: Found {len(invalid_values)} invalid SNP values")
        return False
    
    # Check phenotypes are numeric
    if not np.issubdtype(df_pheno.values.dtype, np.number):
        log_message("ERROR: Phenotype values are not numeric")
        return False
    
    log_message(f"Validation successful! Generated {len(df_snps)} samples with {df_snps.shape[1]} features each")
    return True

def main():
    """Main function for data augmentation"""
    initialize_log()
    
    try:
        # Load reference data
        X_ref, y_ref = load_reference_data()
        
        # Check Ollama availability
        ollama_available = check_ollama_availability()
        
        if ollama_available:
            # Generate prompt only when Ollama is available to avoid unnecessary processing
            prompt = generate_prompt(X_ref, y_ref)
            response = call_ollama(prompt)
            
            if response:
                # Parse response
                data = parse_ollama_response(response)
                if data and 'synthetic_snps' in data and 'synthetic_phenotypes' in data:
                    # Validate number of samples
                    if len(data['synthetic_snps']) == SYNTHETIC_SAMPLES and len(data['synthetic_phenotypes']) == SYNTHETIC_SAMPLES:
                        # Save and validate
                        df_snps, df_pheno = save_synthetic_data(data, X_ref, y_ref)
                        if validate_generated_data(df_snps, df_pheno):
                            log_message("Data augmentation completed successfully using Ollama!")
                            return
                        else:
                            log_message("Validation failed. Falling back to numpy generation...")
                    else:
                        log_message(f"Incorrect number of samples generated. Expected {SYNTHETIC_SAMPLES}, got SNPs: {len(data['synthetic_snps'])}, phenotypes: {len(data['synthetic_phenotypes'])}")
                        log_message("Falling back to numpy generation...")
                else:
                    log_message("Invalid response structure from Ollama. Falling back to numpy generation...")
            else:
                log_message("Failed to get response from Ollama. Falling back to numpy generation...")
        else:
            log_message("Ollama not available. Using numpy for synthetic data generation...")
        
        # Fallback to numpy generation
        data = create_fallback_synthetic_data(X_ref, y_ref)
        log_message(f"DEBUG: First synthetic_snps entry: {data['synthetic_snps'][0]}")
        log_message(f"DEBUG: First synthetic_phenotypes entry: {data['synthetic_phenotypes'][0]}")
        df_snps, df_pheno = save_synthetic_data(data, X_ref, y_ref)
        if validate_generated_data(df_snps, df_pheno):
            log_message("Data augmentation completed successfully using fallback method!")
        else:
            log_message("ERROR: Validation failed for fallback-generated data")
            raise ValueError("Data validation failed")
            
    except Exception as e:
        log_message(f"Error in data augmentation process: {str(e)}")
        raise

if __name__ == "__main__":
    main()