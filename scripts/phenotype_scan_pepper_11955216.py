import os
import logging
import time
from datetime import datetime
import csv
import json
import sys

# Setup logging with UTF-8 encoding
log_file = r'c:\Users\OMEN\Desktop\experiment_snp\logs\pepper_11955216_prep.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Clear existing handlers
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Configure logging with UTF-8 encoding
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create file handler with UTF-8 encoding
file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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

print(f"Starting phenotype scan for pepper_11955216 dataset at {datetime.now()}")

# Input files - corrected path
input_dir = r'c:\Users\OMEN\Desktop\experiment_snp\pepper_11955216'
training_data_file = os.path.join(input_dir, 'trainingData.csv')
training_yield_file = os.path.join(input_dir, 'trainingYield.csv')
table_s1_file = os.path.join(input_dir, 'Table_S1.csv')

# Output directories
output_dir = 'c:\\Users\\OMEN\\Desktop\\experiment_snp\\02_processed_data\\pepper_11955216'
os.makedirs(output_dir, exist_ok=True)

# Common ID column names to look for
common_id_columns = ['IID', 'Taxa', 'Genotype', 'ID', 'Sample', 'Accession', 'Accession_ID', 'Variety', 'Line', 'Name']

# Function to read CSV file with automatic delimiter detection
def read_csv_with_delim(file_path):
    """Read a CSV file, automatically detecting the delimiter with UTF-8 encoding"""
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found")
        return None, None
    
    print(f"Reading file: {file_path}")
    
    try:
        # Read first few lines to detect delimiter with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        # Try different delimiters
        possible_delimiters = ['\t', ',', ';', '|']
        delimiter_scores = {}
        
        for delim in possible_delimiters:
            parts = first_line.split(delim)
            if len(parts) > 1:  # At least two columns
                delimiter_scores[delim] = len(parts)
        
        # Choose delimiter with most columns
        if delimiter_scores:
            delimiter = max(delimiter_scores.items(), key=lambda x: x[1])[0]
        else:
            # Fallback to comma
            delimiter = ','
        
        print(f"Detected delimiter: '{delimiter}'")
        
        # Read the file with the detected delimiter and UTF-8 encoding
        headers = []
        data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter)
            # Read header
            headers = next(reader)
            # Read data
            for row in reader:
                # Ensure row has at least as many elements as headers
                while len(row) < len(headers):
                    row.append('')
                data.append(row)
        
        print(f"Successfully read {len(data)} rows and {len(headers)} columns")
        return headers, data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

# Function to detect ID columns in a dataset
def detect_id_columns(headers):
    """Detect potential ID columns based on header names"""
    if not headers:
        return []
    
    id_columns = []
    for header in headers:
        # Check if header matches common ID column names
        header_lower = header.lower()
        for id_col in common_id_columns:
            if id_col.lower() in header_lower or header_lower in id_col.lower():
                id_columns.append(header)
                break
    
    return id_columns

# Function to detect potential target columns
def detect_target_columns(headers, data, min_numeric_rows=5):
    """Detect potential target columns that contain numeric data"""
    if not headers or not data:
        return []
    
    target_columns = []
    
    for col_idx, header in enumerate(headers):
        # Skip potential ID columns
        header_lower = header.lower()
        is_id_col = False
        for id_col in common_id_columns:
            if id_col.lower() in header_lower or header_lower in id_col.lower():
                is_id_col = True
                break
        if is_id_col:
            continue
        
        # Check if column has numeric data
        numeric_count = 0
        for row in data:
            if col_idx < len(row):
                val = row[col_idx].strip()
                if val and val.replace('.', '', 1).replace('-', '', 1).isdigit():
                    numeric_count += 1
        
        # Consider as target if enough numeric values
        if numeric_count >= min_numeric_rows:
            target_columns.append({
                'name': header,
                'numeric_count': numeric_count,
                'total_rows': len(data)
            })
    
    # Sort by percentage of numeric values
    target_columns.sort(key=lambda x: x['numeric_count'] / x['total_rows'], reverse=True)
    return target_columns

# Read all available files
datasets = {}

# Read trainingData.csv
headers, data = read_csv_with_delim(training_data_file)
if headers and data:
    datasets['trainingData'] = {
        'headers': headers,
        'data': data,
        'file': training_data_file
    }

# Read trainingYield.csv
headers, data = read_csv_with_delim(training_yield_file)
if headers and data:
    datasets['trainingYield'] = {
        'headers': headers,
        'data': data,
        'file': training_yield_file
    }

# Read Table_S1.csv
headers, data = read_csv_with_delim(table_s1_file)
if headers and data:
    datasets['Table_S1'] = {
        'headers': headers,
        'data': data,
        'file': table_s1_file
    }

# Analyze each dataset
print("\nAnalyzing datasets for ID and target columns...")
print("=" * 70)

dataset_analysis = {}

for name, dataset in datasets.items():
    print(f"\nDataset: {name}")
    print(f"File: {dataset['file']}")
    
    # Detect ID columns
    id_columns = detect_id_columns(dataset['headers'])
    print(f"Detected ID columns: {', '.join(id_columns) if id_columns else 'None'}")
    
    # Detect target columns
    target_columns = detect_target_columns(dataset['headers'], dataset['data'])
    print(f"Detected target columns:")
    for target in target_columns:
        percent = (target['numeric_count'] / target['total_rows']) * 100
        print(f"  - {target['name']}: {target['numeric_count']}/{target['total_rows']} ({percent:.1f}% numeric)")
    
    # Store analysis
    dataset_analysis[name] = {
        'id_columns': id_columns,
        'target_columns': target_columns,
        'total_samples': len(dataset['data']),
        'total_columns': len(dataset['headers'])
    }

# Look for potential join keys between datasets
print("\nLooking for potential join keys between datasets...")
print("=" * 70)

join_keys = {}
dataset_names = list(datasets.keys())

for i in range(len(dataset_names)):
    for j in range(i+1, len(dataset_names)):
        dataset1_name = dataset_names[i]
        dataset2_name = dataset_names[j]
        
        # Get ID columns for both datasets
        dataset1_ids = dataset_analysis[dataset1_name]['id_columns']
        dataset2_ids = dataset_analysis[dataset2_name]['id_columns']
        
        # Check all combinations of ID columns
        potential_joins = []
        
        for id_col1 in dataset1_ids:
            # Get all values from dataset1 for this ID column
            id_col1_idx = datasets[dataset1_name]['headers'].index(id_col1)
            dataset1_values = set()
            
            for row in datasets[dataset1_name]['data']:
                if id_col1_idx < len(row):
                    val = row[id_col1_idx].strip().lower()
                    if val:  # Only non-empty values
                        dataset1_values.add(val)
            
            for id_col2 in dataset2_ids:
                # Get all values from dataset2 for this ID column
                id_col2_idx = datasets[dataset2_name]['headers'].index(id_col2)
                dataset2_values = set()
                
                for row in datasets[dataset2_name]['data']:
                    if id_col2_idx < len(row):
                        val = row[id_col2_idx].strip().lower()
                        if val:  # Only non-empty values
                            dataset2_values.add(val)
                
                # Find intersection
                common_values = dataset1_values.intersection(dataset2_values)
                
                if common_values:
                    # Calculate join quality
                    dataset1_coverage = len(common_values) / len(dataset1_values) if dataset1_values else 0
                    dataset2_coverage = len(common_values) / len(dataset2_values) if dataset2_values else 0
                    
                    potential_joins.append({
                        'dataset1': dataset1_name,
                        'column1': id_col1,
                        'dataset2': dataset2_name,
                        'column2': id_col2,
                        'common_values': len(common_values),
                        'dataset1_coverage': dataset1_coverage,
                        'dataset2_coverage': dataset2_coverage
                    })
        
        # Sort by join quality (sum of coverages)
        potential_joins.sort(key=lambda x: x['dataset1_coverage'] + x['dataset2_coverage'], reverse=True)
        
        if potential_joins:
            print(f"\nPotential join between {dataset1_name} and {dataset2_name}:")
            for join in potential_joins[:3]:  # Show top 3 joins
                print(f"  {join['dataset1']}.{join['column1']} <-> {join['dataset2']}.{join['column2']}")
                print(f"    - {join['common_values']} common values")
                print(f"    - {join['dataset1_coverage']:.1%} coverage in {join['dataset1']}")
                print(f"    - {join['dataset2_coverage']:.1%} coverage in {join['dataset2']}")
            
            join_keys[f"{dataset1_name}_{dataset2_name}"] = potential_joins

# Create y.csv file with the best available data
def create_y_csv(output_path):
    """Create a y.csv file with sample IDs and target variables"""
    # Find the dataset with the most target variables
    best_dataset = None
    best_targets = []
    best_id_column = None
    
    for name, analysis in dataset_analysis.items():
        if analysis['target_columns'] and analysis['id_columns']:
            if len(analysis['target_columns']) > len(best_targets):
                best_dataset = name
                best_targets = analysis['target_columns']
                best_id_column = analysis['id_columns'][0]  # Use first ID column
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        if best_dataset and best_id_column and best_targets:
            print(f"\nCreating y.csv using {best_dataset} with {len(best_targets)} target columns")
            
            # Get the dataset
            dataset = datasets[best_dataset]
            id_col_idx = dataset['headers'].index(best_id_column)
            
            # Get target column indices
            target_col_indices = []
            target_names = []
            for target in best_targets:
                try:
                    idx = dataset['headers'].index(target['name'])
                    target_col_indices.append(idx)
                    target_names.append(target['name'])
                except ValueError:
                    continue
            
            # Write header
            headers = ['IID'] + target_names
            writer.writerow(headers)
            
            # Write data
            written_count = 0
            for row in dataset['data']:
                if id_col_idx < len(row):
                    sample_id = row[id_col_idx].strip()
                    if sample_id:  # Only include rows with non-empty ID
                        # Extract target values
                        target_values = []
                        for col_idx in target_col_indices:
                            if col_idx < len(row):
                                val = row[col_idx].strip()
                                # Try to convert to numeric if possible
                                try:
                                    val = float(val)
                                except ValueError:
                                    pass
                                target_values.append(val)
                            else:
                                target_values.append('')
                        
                        # Write row
                        writer.writerow([sample_id] + target_values)
                        written_count += 1
            
            print(f"Wrote {written_count} samples to {output_path}")
        else:
            print(f"\nNo suitable target data found. Creating empty y.csv with header only")
            # Create empty file with header only
            writer.writerow(['IID'])

# Create QC report
def create_qc_report(output_path):
    """Create a QC report for the phenotype scan"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("QC REPORT FOR PEPPER_11955216 DATASET\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("DATASETS ANALYZED:\n")
        for name, dataset in datasets.items():
            f.write(f"  - {name}: {dataset['file']}\n")
            f.write(f"    Rows: {len(dataset['data'])}, Columns: {len(dataset['headers'])}\n")
            
            # Write ID columns
            id_columns = dataset_analysis[name]['id_columns']
            if id_columns:
                f.write(f"    ID Columns: {', '.join(id_columns)}\n")
            else:
                f.write(f"    ID Columns: None detected\n")
            
            # Write target columns
            target_columns = dataset_analysis[name]['target_columns']
            if target_columns:
                f.write(f"    Potential Target Columns:\n")
                for target in target_columns[:5]:  # Show top 5 targets
                    percent = (target['numeric_count'] / target['total_rows']) * 100
                    f.write(f"      - {target['name']}: {target['numeric_count']}/{target['total_rows']} ({percent:.1f}% numeric)\n")
                if len(target_columns) > 5:
                    f.write(f"      - ... and {len(target_columns) - 5} more\n")
            else:
                f.write(f"    Potential Target Columns: None detected\n")
            f.write("\n")
        
        # Write join key information
        f.write("POTENTIAL JOIN KEYS:\n")
        if join_keys:
            for pair, joins in join_keys.items():
                f.write(f"  {pair}:\n")
                for join in joins[:2]:  # Show top 2 joins
                    f.write(f"    - {join['column1']} <-> {join['column2']}: {join['common_values']} common values\n")
        else:
            f.write("  No potential join keys detected between datasets\n")
        f.write("\n")
        
        # Write output information
        f.write("OUTPUTS CREATED:\n")
        f.write(f"  - Phenotype data: {os.path.join(output_dir, 'y.csv')}\n")
        f.write(f"  - Documentation: {os.path.join(output_dir, 'README.md')}\n")
        f.write(f"  - QC report: {output_path}\n")

# Create README.md
def create_readme(output_path):
    """Create a README.md file with dataset information"""
    with open(output_path, 'w') as f:
        f.write("# PEPPER_11955216 Dataset Summary\n\n")
        
        f.write("## Dataset Overview\n")
        f.write("This directory contains processed phenotype data from the pepper_11955216 dataset.\n\n")
        
        f.write("## Source Files\n")
        for name, dataset in datasets.items():
            f.write(f"- **{name}**: {dataset['file']}\n")
        
        if not datasets:
            f.write("*Note: No phenotype files were found or could be read.*\n")
        
        f.write("\n## Available Phenotypes\n")
        
        # Find all unique target columns across datasets
        all_targets = {}
        for name, analysis in dataset_analysis.items():
            for target in analysis['target_columns']:
                if target['name'] not in all_targets:
                    all_targets[target['name']] = {
                        'source': name,
                        'numeric_count': target['numeric_count'],
                        'total_rows': target['total_rows']
                    }
        
        if all_targets:
            f.write("The following phenotype targets were detected:\n\n")
            f.write("| Target | Source Dataset | Numeric Values | Total Values |\n")
            f.write("|--------|---------------|----------------|--------------|\n")
            
            for target_name, info in all_targets.items():
                percent = (info['numeric_count'] / info['total_rows']) * 100
                f.write(f"| {target_name} | {info['source']} | {info['numeric_count']} ({percent:.1f}%) | {info['total_rows']} |\n")
        else:
            f.write("No clear phenotype targets were detected in the available files.\n")
        
        f.write("\n## Sample Identification\n")
        
        # Find all unique ID columns across datasets
        all_id_columns = {}
        for name, analysis in dataset_analysis.items():
            for id_col in analysis['id_columns']:
                if id_col not in all_id_columns:
                    all_id_columns[id_col] = name
        
        if all_id_columns:
            f.write("The following ID columns were detected that could be used for sample matching:\n\n")
            for id_col, source in all_id_columns.items():
                f.write(f"- **{id_col}** (from {source})\n")
            
            f.write("\n## Join Keys\n")
            if join_keys:
                f.write("Potential join keys between datasets:\n\n")
                for pair, joins in join_keys.items():
                    dataset1, dataset2 = pair.split('_')
                    f.write(f"### {dataset1} <-> {dataset2}\n")
                    for join in joins[:2]:  # Show top 2 joins
                        f.write(f"- **{join['column1']}** (in {dataset1}) and **{join['column2']}** (in {dataset2}): ")
                        f.write(f"{join['common_values']} common values, ")
                        f.write(f"{join['dataset1_coverage']:.1%} coverage in {dataset1}, ")
                        f.write(f"{join['dataset2_coverage']:.1%} coverage in {dataset2}\n")
            else:
                f.write("No strong join keys were identified between the datasets.\n")
        else:
            f.write("No clear ID columns were detected for sample identification.\n")
        
        f.write("\n## Integration Notes\n")
        f.write("### To integrate with genotype data:\n")
        f.write("1. Identify a common sample ID format between the phenotype data and genotype data\n")
        f.write("2. If necessary, create a mapping file to standardize sample IDs\n")
        f.write("3. Join the standardized phenotype data with genotype data using the common IDs\n")
        
        if not datasets:
            f.write("\n### N/A\n")
            f.write("This dataset folder contains primarily phenotype information. ")
            f.write("No direct genotype-phenotype alignment is possible without additional genotype data.\n")

# Execute the file creation
print("\nCreating output files...")
print("=" * 50)

# Create y.csv
y_file = os.path.join(output_dir, 'y.csv')
create_y_csv(y_file)

# Create QC report
qc_report_file = os.path.join(output_dir, 'qc_report.txt')
create_qc_report(qc_report_file)

# Create README.md
readme_file = os.path.join(output_dir, 'README.md')
create_readme(readme_file)

print("\nSummary of created files:")
print(f"- Phenotype data: {y_file}")
print(f"- QC report: {qc_report_file}")
print(f"- Documentation: {readme_file}")

print("\nPhenotype scan completed successfully!")