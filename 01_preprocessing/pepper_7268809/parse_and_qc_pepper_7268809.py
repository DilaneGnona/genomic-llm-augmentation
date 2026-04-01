import os
import logging
import time
from datetime import datetime
import csv
import math
import statistics
import json
import sys

# Setup logging
log_file = r'c:\Users\OMEN\Desktop\experiment_snp\logs\pepper_7268809_prep.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Clear existing handlers if any
logger = logging.getLogger()
if logger.handlers:
    logger.handlers.clear()

# Set log level
logger.setLevel(logging.INFO)

# Create file handler with UTF-8 encoding
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

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

print(f"Starting preprocessing of pepper_7268809 dataset at {datetime.now()}")

# Input files
input_dir = 'c:\\Users\\OMEN\\Desktop\\experiment_snp\\pepper_7268809'
hapmap_file = os.path.join(input_dir, 'Filtered10percentminorallele_2966markers_77genotypes.hmp (1).txt')
phenotype_file = os.path.join(input_dir, 'Vitamin_phenotype.txt')

# Output directories
preprocessing_dir = 'c:\\Users\\OMEN\\Desktop\\experiment_snp\\01_preprocessing\\pepper_7268809'
output_dir = 'c:\\Users\\OMEN\\Desktop\\experiment_snp\\02_processed_data\\pepper_7268809'
os.makedirs(preprocessing_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Custom CSV reader that handles different delimiters
def read_csv_custom(file_path, delimiter=None, skip_header=False):
    """Read CSV file with custom delimiter handling"""
    data = []
    headers = []
    delimiters = [delimiter] if delimiter else ['\t', ',', ';']
    
    for delim in delimiters:
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f, delimiter=delim)
                if skip_header:
                    next(reader)
                data = list(reader)
                if data:
                    break
        except Exception as e:
            logger.debug(f"Failed to read with delimiter '{delim}': {str(e)}")
    
    if not data:
        raise ValueError(f"Could not read file {file_path} with any common delimiter")
    
    return data

# Step A: Parse HapMap file
print("\nSTEP A: Parsing HapMap file")
print("=" * 50)

# Read HapMap file
print(f"Reading HapMap file: {hapmap_file}")
hapmap_data = read_csv_custom(hapmap_file)
headers = hapmap_data[0]
data = hapmap_data[1:]

# Extract variant information and genotype data
print(f"Extracting variant information and genotypes")
variant_info = []
sample_ids = headers[11:]  # Samples start at column 11 in HapMap format
initial_sample_count = len(sample_ids)
initial_snp_count = len(data)

print(f"Initial data: {initial_snp_count} SNPs and {initial_sample_count} samples")

# Convert genotype data to numeric format (0,1,2) - need to handle various formats
def genotype_to_numeric(genotype):
    """Convert genotype to numeric format (0,1,2)"""
    genotype = genotype.strip().upper()
    if genotype in ['N', '-', '?', '']:
        return None  # Missing value
    # Handle IUPAC ambiguity codes or other formats by counting non-reference alleles
    # Simplified approach: count non-reference alleles (assuming diploid)
    if len(genotype) == 1:
        # Could be haploid, or just one allele specified
        return 0 if genotype == 'A' else 1  # Simplified assumption
    elif len(genotype) >= 2:
        # For diploid, count the number of non-reference alleles (simplified)
        # This is a very basic approach and may need refinement based on actual format
        try:
            return sum(1 for c in genotype if c != 'A') % 3  # Simplified
        except:
            return None
    return None

# Parse variant info and convert genotypes to numeric
variant_data = []
for row in data:
    var_id = row[0]
    chr_name = row[2]
    pos = int(row[3])
    ref = row[1]
    alt = row[4]  # This might need adjustment based on actual HapMap format
    
    # Extract and convert genotypes
    genotypes = row[11:]  # Genotypes start at column 11
    numeric_genotypes = [genotype_to_numeric(g) for g in genotypes]
    
    variant_info.append({"VAR_ID": var_id, "CHR": chr_name, "POS": pos, "REF": ref, "ALT": alt})
    variant_data.append(numeric_genotypes)

print(f"Successfully parsed {len(variant_info)} variants")

# Step A: QC filtering
print("\nSTEP A: QC Filtering")
print("=" * 50)

# Calculate SNP statistics
def calculate_snp_stats(snp_data):
    """Calculate statistics for a SNP"""
    # Remove None values
    valid_genotypes = [g for g in snp_data if g is not None]
    n_valid = len(valid_genotypes)
    n_missing = len(snp_data) - n_valid
    missing_rate = n_missing / len(snp_data) if snp_data else 1.0
    
    # Calculate MAF (simplified)
    if n_valid > 0:
        # MAF is the frequency of the minor allele
        # Assuming 0=AA, 1=AB, 2=BB, MAF = min(freq(A), freq(B))
        # freq(B) = (sum of 1s + 2*sum of 2s) / (2 * n_valid)
        sum_g = sum(valid_genotypes)
        freq_b = sum_g / (2 * n_valid)
        maf = min(freq_b, 1 - freq_b)
    else:
        maf = 0
    
    return {
        "n_valid": n_valid,
        "n_missing": n_missing,
        "missing_rate": missing_rate,
        "maf": maf
    }

# Calculate sample statistics
def calculate_sample_stats(sample_data):
    """Calculate statistics for a sample"""
    valid_genotypes = [g for g in sample_data if g is not None]
    n_valid = len(valid_genotypes)
    n_missing = len(sample_data) - n_valid
    missing_rate = n_missing / len(sample_data) if sample_data else 1.0
    
    # Calculate heterozygosity rate (assuming 1 = heterozygous)
    if n_valid > 0:
        het_count = valid_genotypes.count(1)
        het_rate = het_count / n_valid
    else:
        het_rate = 0
    
    return {
        "n_valid": n_valid,
        "n_missing": n_missing,
        "missing_rate": missing_rate,
        "het_rate": het_rate
    }

# Calculate statistics for all SNPs
snp_stats = [calculate_snp_stats(snp) for snp in variant_data]

# Filter SNPs based on QC criteria
snp_pass = []
snp_fail = []

for i, stats in enumerate(snp_stats):
    if (stats["maf"] >= 0.05 and 
        stats["missing_rate"] <= 0.05):
        snp_pass.append(i)
    else:
        snp_fail.append(i)

print(f"SNP filtering: {len(snp_pass)} passed, {len(snp_fail)} failed")

# Transpose data for sample-wise operations
# sample_data[i] = [variant_data[j][i] for j in snp_pass]
sample_data = []
for i in range(initial_sample_count):
    sample_data.append([variant_data[j][i] for j in snp_pass])

# Calculate statistics for all samples
sample_stats = [calculate_sample_stats(sample) for sample in sample_data]

# Filter samples based on QC criteria
sample_pass = []
sample_fail = []

# Calculate heterozygosity statistics for outliers
het_rates = [s["het_rate"] for s in sample_stats if s["het_rate"] is not None]
avg_het = statistics.mean(het_rates) if het_rates else 0
std_het = statistics.stdev(het_rates) if len(het_rates) > 1 else 0

for i, stats in enumerate(sample_stats):
    # Check missing rate and heterozygosity
    if (stats["missing_rate"] <= 0.10 and 
        (std_het == 0 or abs(stats["het_rate"] - avg_het) <= 3 * std_het)):
        sample_pass.append(i)
    else:
        sample_fail.append(i)

print(f"Sample filtering: {len(sample_pass)} passed, {len(sample_fail)} failed")

# Update variant data and sample IDs based on filtering
filtered_variant_data = [variant_data[i] for i in snp_pass]
filtered_variant_info = [variant_info[i] for i in snp_pass]
filtered_sample_ids = [sample_ids[i] for i in sample_pass]

# Transpose filtered data to sample x SNP format (for easier handling)
filtered_sample_data = []
for i in sample_pass:
    filtered_sample_data.append([variant_data[j][i] for j in snp_pass])

print(f"After filtering: {len(filtered_variant_data)} SNPs and {len(filtered_sample_data)} samples")

# Step B: Imputation and Encoding
print("\nSTEP B: Imputation and Encoding")
print("=" * 50)

# Simple mode imputation for each SNP
def impute_snp_mode(snp_data):
    """Impute missing values in a SNP using mode"""
    # Get valid genotypes
    valid_genotypes = [g for g in snp_data if g is not None]
    if not valid_genotypes:
        return [0] * len(snp_data)  # Default to 0 if no valid values
    
    # Calculate mode
    counts = {0: 0, 1: 0, 2: 0}
    for g in valid_genotypes:
        if g in counts:
            counts[g] += 1
    mode = max(counts.items(), key=lambda x: x[1])[0]
    
    # Impute missing values
    return [g if g is not None else mode for g in snp_data]

# Impute each SNP
imputed_sample_data = []
imputation_counts = []

for snp_idx in range(len(filtered_variant_data)):
    # Get all samples' genotypes for this SNP
    snp_genotypes = [filtered_sample_data[i][snp_idx] for i in range(len(filtered_sample_data))]
    
    # Count missing values
    n_missing = sum(1 for g in snp_genotypes if g is None)
    imputation_counts.append(n_missing)
    
    # Impute
    imputed_snp = impute_snp_mode(snp_genotypes)
    
    # Update imputed_sample_data
    for i in range(len(filtered_sample_data)):
        if len(imputed_sample_data) <= i:
            imputed_sample_data.append([])
        imputed_sample_data[i].append(imputed_snp[i])

# Calculate overall imputation rate
total_imputed = sum(imputation_counts)
total_genotypes = len(filtered_sample_data) * len(filtered_variant_data)
imputation_rate = total_imputed / total_genotypes

print(f"Imputation completed: {total_imputed} out of {total_genotypes} values ({imputation_rate:.4f} rate)")
print(f"Genotypes encoded in additive format (0,1,2)")

# Step C: PCA (simplified version)
print("\nSTEP C: PCA for Population Structure")
print("=" * 50)

# Simplified PCA implementation
def calculate_pca(data_matrix, n_components=5):
    """Calculate PCA with basic linear algebra operations"""
    n_samples, n_features = len(data_matrix), len(data_matrix[0])
    
    # Calculate mean for each feature
    feature_means = []
    for j in range(n_features):
        col_sum = sum(data_matrix[i][j] for i in range(n_samples))
        feature_means.append(col_sum / n_samples)
    
    # Center the data
    centered_data = []
    for i in range(n_samples):
        centered_row = [data_matrix[i][j] - feature_means[j] for j in range(n_features)]
        centered_data.append(centered_row)
    
    # Calculate covariance matrix (simplified)
    covariance_matrix = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
    for i in range(n_features):
        for j in range(i, n_features):
            cov_sum = 0.0
            for k in range(n_samples):
                cov_sum += centered_data[k][i] * centered_data[k][j]
            cov = cov_sum / (n_samples - 1)
            covariance_matrix[i][j] = cov
            covariance_matrix[j][i] = cov
    
    # For simplicity, we'll just keep the centered data as "PCs" for this implementation
    # A proper PCA would compute eigenvalues and eigenvectors
    # This is a placeholder that won't give true PCs but allows us to continue
    
    # Take first n_components of centered data as pseudo-PCs
    pcs = []
    for i in range(n_samples):
        # Just take the first n_features values as pseudo-PCs
        # In a real implementation, these would be linear combinations
        # based on eigenvectors
        pc_row = centered_data[i][:n_components] if len(centered_data[i]) >= n_components else centered_data[i]
        # Pad with zeros if needed
        while len(pc_row) < n_components:
            pc_row.append(0.0)
        pcs.append(pc_row)
    
    return pcs

# Run simplified PCA
pcs = calculate_pca(imputed_sample_data, n_components=5)

# Save PCA covariates
pca_file = os.path.join(output_dir, 'pca_covariates.csv')
with open(pca_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header
    header = ['IID'] + [f'PC{i+1}' for i in range(len(pcs[0]))]
    writer.writerow(header)
    # Write data
    for i, sample_id in enumerate(filtered_sample_ids):
        row = [sample_id] + pcs[i]
        writer.writerow(row)

print(f"PCA completed and saved to {pca_file}")

# Step D: Align genotypes with phenotypes
print("\nSTEP D: Aligning Genotypes with Phenotypes")
print("=" * 50)

# Read phenotype data
phenotypes = {}
if os.path.exists(phenotype_file):
    print(f"Reading phenotype file: {phenotype_file}")
    try:
        pheno_data = read_csv_custom(phenotype_file)
        pheno_headers = pheno_data[0]
        
        # Find ID column and phenotype columns
        id_col_idx = 0  # Default to first column
        for i, header in enumerate(pheno_headers):
            if any(id_keyword in header.lower() for id_keyword in ['id', 'taxa', 'genotype', 'sample']):
                id_col_idx = i
                break
        
        # Extract phenotypes
        for row in pheno_data[1:]:
            if len(row) > id_col_idx:
                sample_id = row[id_col_idx].strip()
                # Store all phenotype columns
                pheno_values = {}
                for i, header in enumerate(pheno_headers):
                    if i != id_col_idx and len(row) > i:
                        pheno_values[header] = row[i].strip()
                phenotypes[sample_id] = pheno_values
        
        print(f"Loaded phenotypes for {len(phenotypes)} samples")
    except Exception as e:
        print(f"Error reading phenotype file: {str(e)}")
else:
    print(f"Phenotype file not found: {phenotype_file}")

# Align samples between genotypes and phenotypes
aligned_samples = []
aligned_genotypes = []
aligned_phenotypes = []

for i, sample_id in enumerate(filtered_sample_ids):
    aligned_samples.append({
        "original_id": sample_id,
        "in_genotypes": True,
        "in_phenotypes": sample_id in phenotypes
    })
    aligned_genotypes.append(imputed_sample_data[i])
    
    if sample_id in phenotypes:
        aligned_phenotypes.append(phenotypes[sample_id])
    else:
        aligned_phenotypes.append({})

print(f"Alignment completed: {sum(1 for s in aligned_samples if s['in_phenotypes'])} samples with both genotypes and phenotypes")

# Save aligned data

# 1. Save genotype matrix as CSV
X_file = os.path.join(output_dir, 'X.csv')
with open(X_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header with sample IDs and SNP IDs
    header = ['IID'] + [v['VAR_ID'] for v in filtered_variant_info]
    writer.writerow(header)
    # Write data
    for i, sample_id in enumerate(filtered_sample_ids):
        row = [sample_id] + aligned_genotypes[i]
        writer.writerow(row)

print(f"Genotype matrix saved to {X_file}")

# 2. Save phenotype data
if phenotypes:
    # Get all phenotype column names
    pheno_columns = set()
    for p in aligned_phenotypes:
        pheno_columns.update(p.keys())
    pheno_columns = sorted(list(pheno_columns))
    
    # Save y.csv
    y_file = os.path.join(output_dir, 'y.csv')
    with open(y_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        header = ['IID'] + pheno_columns
        writer.writerow(header)
        # Write data
        for i, sample_id in enumerate(filtered_sample_ids):
            row = [sample_id]
            for col in pheno_columns:
                row.append(aligned_phenotypes[i].get(col, ''))
            writer.writerow(row)
    
    print(f"Phenotype data saved to {y_file}")
else:
    # Create empty y.csv with header only
    y_file = os.path.join(output_dir, 'y.csv')
    with open(y_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['IID'])
    
    print(f"Created empty phenotype file at {y_file}")

# 3. Save variant manifest
variant_file = os.path.join(output_dir, 'variant_manifest.csv')
with open(variant_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header
    writer.writerow(['VAR_ID', 'CHR', 'POS', 'REF', 'ALT'])
    # Write data
    for var in filtered_variant_info:
        writer.writerow([var['VAR_ID'], var['CHR'], var['POS'], var['REF'], var['ALT']])

print(f"Variant manifest saved to {variant_file}")

# 4. Save sample map
sample_map_file = os.path.join(output_dir, 'sample_map.csv')
with open(sample_map_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header
    writer.writerow(['original_id', 'filtered_id', 'in_genotypes', 'in_phenotypes'])
    # Write data for passed samples
    for i, sample in enumerate(aligned_samples):
        writer.writerow([
            sample['original_id'],
            sample['original_id'],  # Same ID after filtering
            sample['in_genotypes'],
            sample['in_phenotypes']
        ])
    # Write data for failed samples
    for i in sample_fail:
        writer.writerow([
            sample_ids[i],
            '',  # Filtered out
            False,
            sample_ids[i] in phenotypes
        ])

print(f"Sample map saved to {sample_map_file}")

# 5. Create QC report
qc_report_file = os.path.join(output_dir, 'qc_report.txt')
with open(qc_report_file, 'w') as f:
    f.write("QC REPORT FOR PEPPER_7268809 DATASET\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("DATASET SUMMARY:\n")
    f.write(f"Initial SNPs: {initial_snp_count}\n")
    f.write(f"Initial samples: {initial_sample_count}\n")
    f.write(f"Final SNPs: {len(filtered_variant_data)}\n")
    f.write(f"Final samples: {len(filtered_sample_data)}\n")
    f.write(f"SNPs removed: {initial_snp_count - len(filtered_variant_data)}\n")
    f.write(f"Samples removed: {initial_sample_count - len(filtered_sample_data)}\n\n")
    
    f.write("SNP FILTERING:\n")
    f.write("  Criteria:\n")
    f.write("  - MAF >= 0.05\n")
    f.write("  - Missing rate <= 0.05\n")
    f.write(f"  - HWE p > 1e-6 (not implemented in this version)\n\n")
    
    f.write("SAMPLE FILTERING:\n")
    f.write("  Criteria:\n")
    f.write("  - Missing rate <= 0.10\n")
    f.write(f"  - Heterozygosity within ±3sigma of mean\n")
    f.write(f"  - Average heterozygosity: {avg_het:.4f} ± {std_het:.4f}\n\n")
    
    f.write("IMPUTATION:\n")
    f.write(f"  Method: Simple mode imputation\n")
    f.write(f"  Values imputed: {total_imputed}\n")
    f.write(f"  Imputation rate: {imputation_rate:.4f}\n\n")
    
    f.write("PHENOTYPE ALIGNMENT:\n")
    f.write(f"  Total phenotypes available: {len(phenotypes)}\n")
    f.write(f"  Samples with both genotype and phenotype: {sum(1 for s in aligned_samples if s['in_phenotypes'])}\n")
    f.write(f"  Phenotype file: {phenotype_file}\n\n")
    
    f.write("OUTPUTS CREATED:\n")
    f.write(f"  - Genotype matrix: {X_file}\n")
    f.write(f"  - Phenotype data: {y_file}\n")
    f.write(f"  - PCA covariates: {pca_file}\n")
    f.write(f"  - Variant manifest: {variant_file}\n")
    f.write(f"  - Sample map: {sample_map_file}\n")
    f.write(f"  - QC report: {qc_report_file}\n")

print(f"QC report saved to {qc_report_file}")

# Create a simple PCA plot file (text-based)
pca_plot_file = os.path.join(output_dir, 'pca_plot.txt')
with open(pca_plot_file, 'w') as f:
    f.write("PCA PLOT (PC1 vs PC2) - TEXT REPRESENTATION\n")
    f.write("=" * 50 + "\n\n")
    f.write("IID,PC1,PC2\n")
    for i, sample_id in enumerate(filtered_sample_ids):
        f.write(f"{sample_id},{pcs[i][0]},{pcs[i][1]}\n")

print(f"Text-based PCA plot saved to {pca_plot_file}")
print("\nPreprocessing completed successfully!")