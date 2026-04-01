import os
import logging
import time
from datetime import datetime
import csv
import math
import statistics
import json
import sys

# Setup logging with UTF-8 encoding
log_file = r'c:\Users\OMEN\Desktop\experiment_snp\logs\ipk_out_raw_prep.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Clear existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File handler with UTF-8 encoding
file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# Configure root logger
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logger = logging.getLogger()
logger.info("Starting IPK dataset processing")

# Check if required packages are installed
try:
    import numpy as np
    logger.info(f"numpy version: {np.__version__}")
except ImportError:
    logger.error("numpy is not installed. Please install it with 'pip install numpy'")
    sys.exit(1)

try:
    import pandas as pd
    logger.info(f"pandas version: {pd.__version__}")
except ImportError:
    logger.error("pandas is not installed. Please install it with 'pip install pandas'")
    sys.exit(1)

try:
    import pyreadr
    logger.info(f"pyreadr version: {pyreadr.__version__}")
except ImportError:
    logger.error("pyreadr is not installed. This is required to read RDS files.")
    logger.error("NOTE: pyreadr requires 64-bit Python on Windows.")
    sys.exit(1)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# Add handlers to root logger
logging.root.addHandler(file_handler)
logging.root.addHandler(console_handler)
logging.root.setLevel(logging.INFO)
logger = logging.getLogger()

# Redirect stdout to log file as well
class LoggerWriter:
    def __init__(self, level):
        self.level = level
    def write(self, message):
        if message.strip():
            self.level(message.strip())
    def flush(self):
        pass

sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.error)

print(f"Starting preprocessing of ipk_out_raw dataset at {datetime.now()}")

# Input and output directories
input_dir = r'c:\Users\OMEN\Desktop\experiment_snp\ipk_out_raw'
rds_file = os.path.join(input_dir, 'GBS_8070_samples_29846_coded_SNPs_non_imputed.rds')
phenotype_file = os.path.join(input_dir, 'Geno_IDs_and_Phenotypes.txt')

output_dir = r'c:\Users\OMEN\Desktop\experiment_snp\02_processed_data\ipk_out_raw'
os.makedirs(output_dir, exist_ok=True)

# Function to load RDS file
def load_rds_file(file_path):
    """Load RDS file using pyreadr"""
    try:
        result = pyreadr.read_r(file_path)
        # The result is a dictionary with keys as table names
        # Typically, there's only one table
        if result:
            return list(result.values())[0]
        else:
            raise ValueError("No data found in RDS file")
    except Exception as e:
        logger.error(f"Error loading RDS file: {str(e)}")
        raise

# Step A: Convert RDS to Python format and perform QC
print("\nSTEP A: Converting RDS file and performing QC")
print("=" * 50)

# Read RDS file
print(f"Reading RDS file: {rds_file}")
genotype_data = load_rds_file(rds_file)
print(f"Loaded genotype data with shape: {genotype_data.shape}")

# Check column names and structure
print(f"Data columns: {genotype_data.columns.tolist()[:5]}...")
print(f"Data types: {genotype_data.dtypes}")

# Determine which columns are variants and which are sample IDs
# Assuming first column is sample IDs
if isinstance(genotype_data, pd.DataFrame):
    sample_ids = genotype_data.iloc[:, 0].tolist()
    variant_columns = genotype_data.columns[1:].tolist()
    print(f"Found {len(sample_ids)} samples and {len(variant_columns)} variants")
    
    # Transpose the data for easier processing (samples as columns, variants as rows)
    # Keep variant IDs as index
    transposed_data = genotype_data.set_index('Marker').T
    print(f"Transposed data shape: {transposed_data.shape}")
    
    # Initial data statistics
    total_variants = transposed_data.shape[0]
    total_samples = transposed_data.shape[1]
    print(f"Initial data: {total_variants} SNPs and {total_samples} samples")
    
    # Extract variant information (VAR_ID, CHR, POS, REF, ALT) from the original data
    variant_info = []
    for _, row in genotype_data.iterrows():
        var_id = row['Marker']
        chr_name = row['CHROM'] if 'CHROM' in row else "Unknown"
        pos = row['POS'] if 'POS' in row else 0
        ref = row['REF'] if 'REF' in row else "N"
        alt = row['ALT'] if 'ALT' in row else "N"
        
        # Convert pos to integer if possible
        try:
            pos = int(pos)
        except (ValueError, TypeError):
            pos = 0
        
        variant_info.append({
            "VAR_ID": var_id,
            "CHR": chr_name,
            "POS": pos,
            "REF": ref,
            "ALT": alt
        })
    
    # Convert to numpy array with safe numeric conversion
    # First, create a copy of the transposed data
    numeric_data = transposed_data.copy()
    
    # Identify which columns can be safely converted to numeric
    # Track columns with conversion issues
    conversion_issues = []
    
    for col in numeric_data.columns:
        try:
            # Try to convert the entire column with coercion
            converted_values = pd.to_numeric(numeric_data[col], errors='coerce')
            
            # Check if all values were successfully converted
            if converted_values.isna().all():
                conversion_issues.append(col)
            else:
                # Replace the column with converted values
                numeric_data[col] = converted_values
        except Exception as e:
            conversion_issues.append(col)
    
    # Log any conversion issues
    if conversion_issues:
        logger.warning(f"Conversion issues with {len(conversion_issues)} columns, replacing with NaN")
    
    # Create genotype matrix from the numeric data
    genotype_matrix = numeric_data.values
    
    # Check for NaN values
    nan_count = np.isnan(genotype_matrix).sum()
    if nan_count > 0:
        logger.warning(f"Warning: Found {nan_count} NaN values in genotype matrix after conversion")
    else:
        logger.info("Successfully converted all values to numeric without NaNs")
    
    # Step A1: SNP Quality Control
    print("\nPerforming SNP filtering...")
    
    # Initialize arrays to track passing SNPs
    snp_pass_filter = np.ones(total_variants, dtype=bool)
    snp_failure_reasons = {i: [] for i in range(total_variants)}
    
    # 1. MAF filter (>= 0.05)
    maf_threshold = 0.05
    maf_failures = 0
    
    # 2. Missing rate filter (<= 0.05)
    missing_threshold = 0.05
    missing_failures = 0
    
    # Calculate metrics for each SNP
    for i in range(total_variants):
        snp_data = genotype_matrix[i]
        valid_data = snp_data[~np.isnan(snp_data)]
        
        # Missing rate calculation
        missing_rate = 1.0 - (len(valid_data) / total_samples)
        if missing_rate > missing_threshold:
            snp_pass_filter[i] = False
            snp_failure_reasons[i].append(f"missing rate {missing_rate:.4f} > {missing_threshold}")
            missing_failures += 1
            continue
        
        # MAF calculation (assuming 0/1/2 encoding)
        if len(valid_data) > 0:
            # Count number of alleles (each sample contributes 2 alleles)
            total_alleles = len(valid_data) * 2
            alt_count = np.sum(valid_data)
            ref_count = total_alleles - alt_count
            
            # Calculate allele frequencies
            ref_freq = ref_count / total_alleles
            alt_freq = alt_count / total_alleles
            maf = min(ref_freq, alt_freq)
            
            if maf < maf_threshold:
                snp_pass_filter[i] = False
                snp_failure_reasons[i].append(f"MAF {maf:.4f} < {maf_threshold}")
                maf_failures += 1
    
    # Apply SNP filters
    filtered_genotype_matrix = genotype_matrix[snp_pass_filter]
    filtered_variant_info = [variant_info[i] for i in range(total_variants) if snp_pass_filter[i]]
    n_snps = filtered_genotype_matrix.shape[0]
    
    print(f"SNP filtering: {n_snps} passed, {total_variants - n_snps} failed")
    print(f"  - MAF failures: {maf_failures}")
    print(f"  - Missing rate failures: {missing_failures}")
    
    # Step A2: Sample Quality Control
    print("\nPerforming sample filtering...")
    
    # Initialize arrays to track passing samples
    sample_pass_filter = np.ones(total_samples, dtype=bool)
    sample_failure_reasons = {i: [] for i in range(total_samples)}
    
    # 1. Missing rate filter (<= 0.10)
    sample_missing_threshold = 0.10
    sample_missing_failures = 0
    
    # 2. Heterozygosity filter (±3sigma)
    het_rates = []
    
    # Calculate metrics for each sample
    for i in range(total_samples):
        sample_data = filtered_genotype_matrix[:, i]
        valid_data = sample_data[~np.isnan(sample_data)]
        
        # Missing rate calculation
        if len(sample_data) > 0:
            missing_rate = 1.0 - (len(valid_data) / len(sample_data))
            if missing_rate > sample_missing_threshold:
                sample_pass_filter[i] = False
                sample_failure_reasons[i].append(f"missing rate {missing_rate:.4f} > {sample_missing_threshold}")
                sample_missing_failures += 1
                continue
            
            # Heterozygosity calculation (assuming 1 = heterozygous)
            heterozygous_count = np.sum(valid_data == 1)
            het_rate = heterozygous_count / len(valid_data)
            het_rates.append((i, het_rate))
    
    # Apply heterozygosity filter if we have enough samples
    if len(het_rates) > 1:
        rates = [rate for _, rate in het_rates]
        mean_het = statistics.mean(rates)
        std_het = statistics.stdev(rates)
        
        # Calculate 3 sigma bounds
        lower_bound = mean_het - 3 * std_het
        upper_bound = mean_het + 3 * std_het
        
        het_failures = 0
        for i, rate in het_rates:
            if rate < lower_bound or rate > upper_bound:
                sample_pass_filter[i] = False
                sample_failure_reasons[i].append(f"heterozygosity {rate:.4f} outside [{lower_bound:.4f}, {upper_bound:.4f}]")
                het_failures += 1
        
        print(f"  - Heterozygosity failures: {het_failures}")
    else:
        mean_het = 0
        std_het = 0
    
    # Apply sample filters
    filtered_genotype_matrix = filtered_genotype_matrix[:, sample_pass_filter]
    filtered_sample_ids = [sample_ids[i] for i in range(total_samples) if sample_pass_filter[i]]
    n_samples = filtered_genotype_matrix.shape[1]
    
    print(f"Sample filtering: {n_samples} passed, {total_samples - n_samples} failed")
    print(f"  - Missing rate failures: {sample_missing_failures}")
    print(f"After filtering: {n_snps} SNPs and {n_samples} samples")
    
    # Step B: Imputation and Encoding
    print("\nSTEP B: Imputation and Encoding")
    print("=" * 50)
    
    # Count missing values before imputation
    total_values = n_snps * n_samples
    missing_count_before = np.sum(np.isnan(filtered_genotype_matrix))
    
    # Simple mode imputation
    print(f"Imputing missing values...")
    imputed_count = 0
    
    for i in range(n_snps):
        snp_data = filtered_genotype_matrix[i]
        missing_indices = np.isnan(snp_data)
        if np.any(missing_indices):
            # Get valid values
            valid_values = snp_data[~missing_indices]
            
            if len(valid_values) > 0:
                # Calculate mode
                unique_values, counts = np.unique(valid_values, return_counts=True)
                mode_value = unique_values[np.argmax(counts)]
                
                # Impute missing values with mode
                filtered_genotype_matrix[i, missing_indices] = mode_value
                imputed_count += np.sum(missing_indices)
    
    # Calculate imputation rate
    imputation_rate = imputed_count / total_values if total_values > 0 else 0
    print(f"Imputation completed: {imputed_count} out of {total_values} values ({imputation_rate:.6f} rate)")
    print(f"Genotypes encoded in additive format (0,1,2)")
    
    # Step C: PCA for Population Structure
    print("\nSTEP C: PCA for Population Structure")
    print("=" * 50)
    
    # Prepare data for PCA (transpose so samples are rows)
    pca_data = filtered_genotype_matrix.T.copy()
    
    # Check if we need to thin SNPs for memory efficiency
    n_snps_for_pca = n_snps
    if n_snps > 10000:
        thinning_factor = max(1, n_snps // 5000)  # Reduce to ~5000 SNPs
        pca_data = pca_data[:, ::thinning_factor]
        n_snps_for_pca = pca_data.shape[1]
        print(f"Reduced SNPs for PCA: {n_snps} -> {n_snps_for_pca} (thinning factor: {thinning_factor})")
    
    # Center and standardize the data
    mean_vals = np.nanmean(pca_data, axis=0)
    std_vals = np.nanstd(pca_data, axis=0)
    
    # Avoid division by zero
    std_vals[std_vals == 0] = 1
    
    # Standardize
    pca_data_std = (pca_data - mean_vals) / std_vals
    
    # Calculate covariance matrix
    covariance_matrix = np.cov(pca_data_std, rowvar=False)
    
    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate PCs (top 5)
    n_pcs = min(5, len(eigenvalues))
    pcs = pca_data_std @ eigenvectors[:, :n_pcs]
    
    # Calculate explained variance
    explained_variance = eigenvalues / np.sum(eigenvalues)
    explained_variance_ratio = explained_variance[:n_pcs]
    
    print(f"PCA completed. Top {n_pcs} PCs explain {np.sum(explained_variance_ratio):.4f} of variance")
    
    # Save PCA covariates
    pca_covariates_path = os.path.join(output_dir, 'pca_covariates.csv')
    pca_df = pd.DataFrame(
        pcs,
        index=filtered_sample_ids,
        columns=[f'PC{i+1}' for i in range(n_pcs)]
    )
    pca_df.index.name = 'Sample_ID'
    pca_df.to_csv(pca_covariates_path)
    print(f"PCA covariates saved to {pca_covariates_path}")
    
    # Create a simple text-based PCA plot
    pca_plot_path = os.path.join(output_dir, 'pca_plot.txt')
    if n_pcs >= 2:
        with open(pca_plot_path, 'w') as f:
            f.write("PCA Plot (PC1 vs PC2)\n")
            f.write("=" * 50 + "\n")
            f.write(f"PC1: {explained_variance_ratio[0]:.4f} variance\n")
            f.write(f"PC2: {explained_variance_ratio[1]:.4f} variance\n\n")
            
            # Simple text-based scatter plot
            f.write("\nSample positions:\n")
            for i, sample_id in enumerate(filtered_sample_ids[:10]):  # Show first 10 samples
                f.write(f"{sample_id}: PC1={pcs[i, 0]:.4f}, PC2={pcs[i, 1]:.4f}\n")
            if len(filtered_sample_ids) > 10:
                f.write(f"... and {len(filtered_sample_ids) - 10} more samples\n")
    
    # Step D: Phenotype Handling
    print("\nSTEP D: Phenotype Handling")
    print("=" * 50)
    
    # Read phenotype file
    print(f"Reading phenotype file: {phenotype_file}")
    
try:
    # Try to read with different delimiters
    for delim in ['\t', ',', ';']:
        try:
            phenotype_data = pd.read_csv(phenotype_file, delimiter=delim)
            print(f"Successfully read phenotype file with delimiter: {delim}")
            print(f"Phenotype data shape: {phenotype_data.shape}")
            print(f"Phenotype columns: {phenotype_data.columns.tolist()}")
            break
        except:
            continue
    else:
        raise ValueError("Could not read phenotype file with any common delimiter")
    
    # Find common samples between genotype and phenotype data
    # Try different potential ID columns
    genotype_sample_set = set(filtered_sample_ids)
    common_samples = []
    join_column = None
    
    # Try to find a column that contains matching sample IDs
    for col in phenotype_data.columns:
        phenotype_sample_set = set(str(id) for id in phenotype_data[col].dropna())
        intersection = genotype_sample_set.intersection(phenotype_sample_set)
        
        if len(intersection) > 0:
            join_column = col
            common_samples = list(intersection)
            print(f"Found join column '{join_column}' with {len(common_samples)} common samples")
            break
    
    # Create y.csv with aligned phenotypes
    y_path = os.path.join(output_dir, 'y.csv')
    
    if join_column and common_samples:
        # Align samples and create y.csv
        sample_map = {}
        for sample_id in common_samples:
            # Find index in genotype data
            if sample_id in filtered_sample_ids:
                sample_map[sample_id] = 'kept_in_both'
        
        # Create phenotype matrix (simplified - just keep all phenotype columns)
        y_df = phenotype_data[phenotype_data[join_column].astype(str).isin(common_samples)]
        y_df = y_df.set_index(join_column)
        
        # Ensure samples are in the same order as genotype data
        ordered_samples = [s for s in filtered_sample_ids if s in y_df.index]
        y_df = y_df.loc[ordered_samples]
        
        # Save y.csv
        y_df.to_csv(y_path)
        print(f"Phenotype data saved to {y_path}")
        print(f"Aligned {y_df.shape[0]} samples with phenotypes")
    else:
        # Create empty y.csv with header only
        with open(y_path, 'w') as f:
            f.write("Sample_ID\n")
            for sample_id in filtered_sample_ids:
                f.write(f"{sample_id}\n")
        print("Created empty y.csv with sample IDs only")
    
    # Step E: Save Output Files
    print("\nSTEP E: Saving Output Files")
    print("=" * 50)
    
    # 1. Save genotype matrix (X.csv)
    X_path = os.path.join(output_dir, 'X.csv')
    X_df = pd.DataFrame(
        filtered_genotype_matrix.T,
        index=filtered_sample_ids,
        columns=[v['VAR_ID'] for v in filtered_variant_info]
    )
    X_df.index.name = 'Sample_ID'
    X_df.to_csv(X_path)
    print(f"Genotype matrix saved to {X_path}")
    
    # 2. Save variant manifest
    variant_manifest_path = os.path.join(output_dir, 'variant_manifest.csv')
    variant_df = pd.DataFrame(filtered_variant_info)
    variant_df.to_csv(variant_manifest_path, index=False)
    print(f"Variant manifest saved to {variant_manifest_path}")
    
    # 3. Save sample map
    sample_map_path = os.path.join(output_dir, 'sample_map.csv')
    sample_map_data = []
    for i, sample_id in enumerate(sample_ids):
        status = "kept" if sample_pass_filter[i] else "removed"
        reasons = "; ".join(sample_failure_reasons[i]) if i in sample_failure_reasons else ""
        sample_map_data.append({
            "Sample_ID": sample_id,
            "Status": status,
            "Failure_Reason": reasons
        })
    sample_map_df = pd.DataFrame(sample_map_data)
    sample_map_df.to_csv(sample_map_path, index=False)
    print(f"Sample map saved to {sample_map_path}")
    
    # 4. Create QC report
    qc_report_path = os.path.join(output_dir, 'qc_report.txt')
    with open(qc_report_path, 'w', encoding='utf-8') as f:
        f.write(f"=== Quality Control Report for ipk_out_raw ===\n\n")
        f.write(f"Dataset: ipk_out_raw\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Quality Control Filters Applied:\n")
        f.write("  - Minor Allele Frequency (MAF) >= 0.05\n")
        f.write("  - Genotype Call Rate >= 0.95 (missing <= 0.05)\n")
        f.write("  - Sample Call Rate >= 0.90 (missing <= 0.10)\n")
        f.write(f"  - Heterozygosity within +/-3sigma of mean\n\n")
        
        f.write("1. VARIANT QUALITY CONTROL\n")
        f.write("==========================\n")
        f.write(f"Total variants parsed: {total_variants}\n")
        f.write(f"Variants passed filtering: {n_snps}\n")
        f.write(f"Variants failed filtering: {total_variants - n_snps}\n\n")
        
        f.write("2. SAMPLE QUALITY CONTROL\n")
        f.write("==========================\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Samples passed filtering: {n_samples}\n")
        f.write(f"Samples failed filtering: {total_samples - n_samples}\n\n")
        
        f.write("3. IMPUTATION SUMMARY\n")
        f.write("==========================\n")
        f.write(f"Total genotype calls: {total_values}\n")
        f.write(f"Missing calls imputed: {imputed_count}\n")
        f.write(f"Imputation rate: {imputation_rate:.6f}\n\n")
        
        f.write("4. HETEROZYGOSITY SUMMARY\n")
        f.write("==========================\n")
        if mean_het > 0:
            f.write(f"Average sample heterozygosity: {mean_het:.6f}\n")
            f.write(f"Heterozygosity SD: {std_het:.6f}\n")
        else:
            f.write("Heterozygosity statistics not calculated (insufficient data)\n")
        
        f.write("\n5. PRINCIPAL COMPONENT ANALYSIS\n")
        f.write("==============================\n")
        f.write(f"SNPs used for PCA: {n_snps_for_pca}\n")
        f.write(f"Samples used for PCA: {n_samples}\n")
        f.write(f"PCs calculated: {n_pcs}\n")
        for i in range(n_pcs):
            f.write(f"  PC{i+1}: {explained_variance_ratio[i]:.4f} variance\n")
        
        f.write("\n6. DATASET SUMMARY\n")
        f.write("==========================\n")
        f.write(f"Final dataset dimensions: {n_samples} samples x {n_snps} SNPs\n")
        f.write(f"Data transformation: Imputed and encoded in additive format (0,1,2)\n")
        f.write(f"Output files created in: {output_dir}\n")
        
        if join_column and common_samples:
            f.write(f"\n7. PHENOTYPE ALIGNMENT\n")
            f.write("==========================\n")
            f.write(f"Phenotype file: {phenotype_file}\n")
            f.write(f"Join column: {join_column}\n")
            f.write(f"Common samples: {len(common_samples)}\n")
    
    print(f"QC report saved to {qc_report_path}")
    print(f"Text-based PCA plot saved to {pca_plot_path}")
    print("\nPreprocessing completed successfully!")

except Exception as e:
    logger.error(f"Error during preprocessing: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)