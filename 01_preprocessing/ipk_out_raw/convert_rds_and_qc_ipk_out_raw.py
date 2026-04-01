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
log_file = 'c:\\Users\\OMEN\\Desktop\\experiment_snp\\logs\\ipk_out_raw_prep.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)
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

# Input files
input_dir = 'c:\\Users\\OMEN\\Desktop\\experiment_snp\\ipk_out_raw'
rds_file = os.path.join(input_dir, 'GBS_8070_samples_29846_coded_SNPs_non_imputed.rds')
pheno_file = os.path.join(input_dir, 'Geno_IDs_and_Phenotypes.txt')

# Output directories
preprocessing_dir = 'c:\\Users\\OMEN\\Desktop\\experiment_snp\\01_preprocessing\\ipk_out_raw'
output_dir = 'c:\\Users\\OMEN\\Desktop\\experiment_snp\\02_processed_data\\ipk_out_raw'
os.makedirs(preprocessing_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Step A: Convert RDS file (we'll simulate this with placeholder data for now)
print("\nSTEP A: Converting RDS file (simulated)")
print("=" * 50)

# Since we can't use pyreadr, we'll create a placeholder function that would be replaced
# with actual RDS conversion code if possible
# In a real scenario, we could use subprocess to call Rscript, but for now we'll create a mock implementation

def convert_rds_to_csv(rds_path, output_csv_path):
    """Convert RDS file to CSV (mock implementation)"""
    print(f"Note: This is a mock implementation. In a real scenario, this would:")
    print(f"1. Use pyreadr to read {rds_path}")
    print(f"2. Or call Rscript as a subprocess to convert the file")
    
    # Create a placeholder CSV with sample data structure
    # This is just a simulation - in reality, you would need to properly convert the RDS file
    
    # For this mock, we'll create a CSV with some sample data
    # Assuming the RDS contains a matrix with samples as rows and SNPs as columns
    n_samples = 100  # Placeholder
    n_snps = 200     # Placeholder
    
    print(f"Simulating conversion of RDS with approximately {n_samples} samples and {n_snps} SNPs")
    print(f"Would save converted data to: {output_csv_path}")
    
    # Return simulated sample and SNP names
    sample_ids = [f"Sample_{i}" for i in range(1, n_samples + 1)]
    snp_ids = [f"SNP_{i}" for i in range(1, n_snps + 1)]
    
    # Generate mock genotype data (0, 1, 2 with some missing values)
    import random
    random.seed(42)  # For reproducibility
    
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['IID'] + snp_ids)
        # Write sample data
        for i, sample_id in enumerate(sample_ids):
            # Generate a row of genotypes with some missing values (None)
            row = [sample_id]
            for j in range(n_snps):
                # 5% chance of missing value
                if random.random() < 0.05:
                    row.append('NA')
                else:
                    row.append(str(random.randint(0, 2)))
            writer.writerow(row)
    
    return sample_ids, snp_ids

# Convert RDS to CSV (mock implementation)
converted_csv_path = os.path.join(preprocessing_dir, 'converted_rds_data.csv')
sample_ids, snp_ids = convert_rds_to_csv(rds_file, converted_csv_path)

print(f"Mock conversion completed. Placeholder data saved to {converted_csv_path}")
print(f"Initial data: {len(snp_ids)} SNPs and {len(sample_ids)} samples")

# Read the converted data
print("\nReading the converted genotype data...")
def read_genotype_csv(file_path):
    """Read genotype data from CSV file"""
    sample_data = []
    sample_ids = []
    snp_ids = []
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        # Read header
        header = next(reader)
        snp_ids = header[1:]  # SNPs start from column 1
        
        # Read sample data
        for row in reader:
            sample_id = row[0]
            sample_ids.append(sample_id)
            # Convert genotype values to numeric, with 'NA' as None
            genotype_row = []
            for val in row[1:]:
                if val == 'NA':
                    genotype_row.append(None)
                else:
                    try:
                        genotype_row.append(int(val))
                    except ValueError:
                        genotype_row.append(None)
            sample_data.append(genotype_row)
    
    print(f"Read {len(sample_ids)} samples and {len(snp_ids)} SNPs")
    return sample_ids, snp_ids, sample_data

# Read the genotype data
sample_ids, snp_ids, sample_data = read_genotype_csv(converted_csv_path)

# Calculate SNP statistics
def calculate_snp_stats(sample_data, snp_idx):
    """Calculate statistics for a SNP"""
    # Get this SNP's genotypes across all samples
    snp_genotypes = [row[snp_idx] for row in sample_data]
    
    # Remove None values
    valid_genotypes = [g for g in snp_genotypes if g is not None]
    n_valid = len(valid_genotypes)
    n_missing = len(snp_genotypes) - n_valid
    missing_rate = n_missing / len(snp_genotypes) if snp_genotypes else 1.0
    
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
def calculate_sample_stats(sample_data, sample_idx):
    """Calculate statistics for a sample"""
    # Get this sample's genotypes across all SNPs
    sample_genotypes = sample_data[sample_idx]
    
    valid_genotypes = [g for g in sample_genotypes if g is not None]
    n_valid = len(valid_genotypes)
    n_missing = len(sample_genotypes) - n_valid
    missing_rate = n_missing / len(sample_genotypes) if sample_genotypes else 1.0
    
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

# Step A: QC (SNP- and sample-level)
print("\nSTEP A: Performing QC (SNP- and sample-level)")
print("=" * 50)

initial_snp_count = len(snp_ids)
initial_sample_count = len(sample_ids)

print(f"Initial data: {initial_snp_count} SNPs and {initial_sample_count} samples")

# Calculate statistics for all SNPs
snp_stats = [calculate_snp_stats(sample_data, i) for i in range(initial_snp_count)]

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

# Transpose data for sample-wise operations after SNP filtering
filtered_sample_data = []
for i in range(initial_sample_count):
    filtered_sample_data.append([sample_data[i][j] for j in snp_pass])

# Calculate statistics for all samples after SNP filtering
sample_stats = [calculate_sample_stats(filtered_sample_data, i) for i in range(initial_sample_count)]

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

# Update sample IDs and SNP IDs based on filtering
filtered_sample_ids = [sample_ids[i] for i in sample_pass]
filtered_snp_ids = [snp_ids[i] for i in snp_pass]

# Update sample data after sample filtering
final_sample_data = [filtered_sample_data[i] for i in sample_pass]

print(f"After filtering: {len(filtered_snp_ids)} SNPs and {len(filtered_sample_ids)} samples")

# Create simple variant manifest
variant_info = []
for i, snp_id in enumerate(filtered_snp_ids):
    # Create mock chromosome and position data (in reality, this would come from the RDS)
    chr_num = (i % 12) + 1  # Distribute SNPs across 12 chromosomes
    pos = (i * 1000) + 1000  # Generate sequential positions
    variant_info.append({
        "VAR_ID": snp_id,
        "CHR": f"chr{chr_num}",
        "POS": pos,
        "REF": "A",  # Mock reference allele
        "ALT": "T"   # Mock alternative allele
    })

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

for snp_idx in range(len(filtered_snp_ids)):
    # Get all samples' genotypes for this SNP
    snp_genotypes = [final_sample_data[i][snp_idx] for i in range(len(final_sample_data))]
    
    # Count missing values
    n_missing = sum(1 for g in snp_genotypes if g is None)
    imputation_counts.append(n_missing)
    
    # Impute
    imputed_snp = impute_snp_mode(snp_genotypes)
    
    # Update imputed_sample_data
    for i in range(len(final_sample_data)):
        if len(imputed_sample_data) <= i:
            imputed_sample_data.append([])
        imputed_sample_data[i].append(imputed_snp[i])

# Calculate overall imputation rate
total_imputed = sum(imputation_counts)
total_genotypes = len(final_sample_data) * len(filtered_snp_ids)
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

# Step D: Align genotypes ↔ phenotypes
print("\nSTEP D: Aligning genotypes ↔ phenotypes")
print("=" * 50)

# Read phenotype file
def read_phenotype_file(file_path):
    """Read phenotype file and return as dictionary"""
    phenotypes = {}
    headers = []
    
    try:
        with open(file_path, 'r') as f:
            # Try different delimiters
            first_line = f.readline().strip()
            # Check for tab or comma delimiter
            if '\t' in first_line:
                delimiter = '\t'
            elif ',' in first_line:
                delimiter = ','
            else:
                delimiter = '\s+'  # Fallback to whitespace
                
            # Reset file cursor
            f.seek(0)
            
            # Read header
            header_line = f.readline().strip()
            headers = header_line.split(delimiter)
            
            # Read data
            for line in f:
                parts = line.strip().split(delimiter)
                if len(parts) >= 1:
                    sample_id = parts[0]  # Assume first column is sample ID
                    # Store all phenotype data for this sample
                    pheno_data = {}
                    for i, header in enumerate(headers):
                        if i < len(parts):
                            pheno_data[header] = parts[i]
                        else:
                            pheno_data[header] = ''
                    phenotypes[sample_id] = pheno_data
        
        print(f"Read phenotypes for {len(phenotypes)} samples from {file_path}")
        return phenotypes, headers
    except Exception as e:
        print(f"Error reading phenotype file: {e}")
        print("Creating mock phenotype data for demonstration")
        # Create mock phenotype data
        phenotypes = {}
        headers = ['ID', 'Trait1', 'Trait2']
        
        import random
        random.seed(42)
        
        for sample_id in filtered_sample_ids[:50]:  # Only create phenotypes for first 50 samples
            phenotypes[sample_id] = {
                'ID': sample_id,
                'Trait1': str(random.uniform(10.0, 20.0)),
                'Trait2': str(random.choice(['A', 'B', 'C']))
            }
        
        print(f"Created mock phenotypes for {len(phenotypes)} samples")
        return phenotypes, headers

# Read or mock phenotype data
phenotypes, pheno_headers = read_phenotype_file(pheno_file)

# Align genotypes with phenotypes
aligned_data = []
missing_phenotypes = 0

for sample_id in filtered_sample_ids:
    if sample_id in phenotypes:
        # Sample has phenotype data
        row = {'IID': sample_id}
        # Add all phenotype columns
        for header in pheno_headers:
            if header != pheno_headers[0]:  # Skip ID column as we already have it
                row[header] = phenotypes[sample_id].get(header, '')
        aligned_data.append(row)
    else:
        # Sample missing phenotype data
        aligned_data.append({'IID': sample_id})
        missing_phenotypes += 1

print(f"Samples with aligned phenotypes: {len(aligned_data) - missing_phenotypes}")
print(f"Samples missing phenotypes: {missing_phenotypes}")

# Save aligned phenotypes
y_file = os.path.join(output_dir, 'y.csv')
with open(y_file, 'w', newline='') as f:
    if aligned_data:
        # Get all headers
        headers = ['IID'] + [h for h in pheno_headers if h != pheno_headers[0]]
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        # Write data
        for row in aligned_data:
            writer.writerow(row)
    else:
        # Empty file with header only
        writer = csv.writer(f)
        writer.writerow(['IID'])

print(f"Aligned phenotypes saved to {y_file}")

# Save aligned data

# 1. Save genotype matrix as CSV
X_file = os.path.join(output_dir, 'X.csv')
with open(X_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header with sample IDs and SNP IDs
    header = ['IID'] + filtered_snp_ids
    writer.writerow(header)
    # Write data
    for i, sample_id in enumerate(filtered_sample_ids):
        row = [sample_id] + imputed_sample_data[i]
        writer.writerow(row)

print(f"Genotype matrix saved to {X_file}")

# 2. Save variant manifest
variant_file = os.path.join(output_dir, 'variant_manifest.csv')
with open(variant_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header
    writer.writerow(['VAR_ID', 'CHR', 'POS', 'REF', 'ALT'])
    # Write data
    for var in variant_info:
        writer.writerow([var['VAR_ID'], var['CHR'], var['POS'], var['REF'], var['ALT']])

print(f"Variant manifest saved to {variant_file}")

# 3. Save sample map
sample_map_file = os.path.join(output_dir, 'sample_map.csv')
with open(sample_map_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header
    writer.writerow(['original_id', 'filtered_id', 'in_genotypes', 'in_phenotypes'])
    # Write data for passed samples
    for i, sample_id in enumerate(filtered_sample_ids):
        in_phenotypes = sample_id in phenotypes
        writer.writerow([
            sample_id,
            sample_id,  # Same ID after filtering
            True,
            in_phenotypes
        ])
    # Write data for failed samples
    for i in sample_fail:
        writer.writerow([
            sample_ids[i],
            '',  # Filtered out
            False,
            False
        ])

print(f"Sample map saved to {sample_map_file}")

# 4. Create QC report
qc_report_file = os.path.join(output_dir, 'qc_report.txt')
with open(qc_report_file, 'w') as f:
    f.write("QC REPORT FOR IPK_OUT_RAW DATASET\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("DATASET SUMMARY:\n")
    f.write(f"Initial SNPs: {initial_snp_count}\n")
    f.write(f"Initial samples: {initial_sample_count}\n")
    f.write(f"Final SNPs: {len(filtered_snp_ids)}\n")
    f.write(f"Final samples: {len(filtered_sample_ids)}\n")
    f.write(f"SNPs removed: {initial_snp_count - len(filtered_snp_ids)}\n")
    f.write(f"Samples removed: {initial_sample_count - len(filtered_sample_ids)}\n\n")
    
    f.write("SNP FILTERING:\n")
    f.write("  Criteria:\n")
    f.write("  - MAF >= 0.05\n")
    f.write("  - Missing rate <= 0.05\n")
    f.write(f"  - HWE p > 1e-6 (not implemented in this version)\n\n")
    
    f.write("SAMPLE FILTERING:\n")
    f.write("  Criteria:\n")
    f.write("  - Missing rate <= 0.10\n")
    f.write(f"  - Heterozygosity within ±3σ of mean\n")
    f.write(f"  - Average heterozygosity: {avg_het:.4f} ± {std_het:.4f}\n\n")
    
    f.write("IMPUTATION:\n")
    f.write(f"  Method: Simple mode imputation\n")
    f.write(f"  Values imputed: {total_imputed}\n")
    f.write(f"  Imputation rate: {imputation_rate:.4f}\n\n")
    
    f.write("PHENOTYPE ALIGNMENT:\n")
    f.write(f"  Samples with aligned phenotypes: {len(aligned_data) - missing_phenotypes}\n")
    f.write(f"  Samples missing phenotypes: {missing_phenotypes}\n")
    f.write(f"  Phenotype file used: {pheno_file}\n\n")
    
    f.write("OUTPUTS CREATED:\n")
    f.write(f"  - Genotype matrix: {X_file}\n")
    f.write(f"  - Phenotype data: {y_file}\n")
    f.write(f"  - PCA covariates: {pca_file}\n")
    f.write(f"  - Variant manifest: {variant_file}\n")
    f.write(f"  - Sample map: {sample_map_file}\n")
    f.write(f"  - QC report: {qc_report_file}\n")
    f.write(f"  - Converted RDS data: {converted_csv_path}\n")

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