import os
import pandas as pd
from pathlib import Path

# Define the main processed data directory
processed_data_dir = Path("c:\\Users\\OMEN\\Desktop\\experiment_snp\\02_processed_data")

# Define the datasets to check
datasets = ["ipk_out_raw", "pepper_10611831", "pepper_11955216", "pepper_7268809"]

# Define the standard files to check for each dataset
standard_files = [
    "X.csv",
    "y.csv",
    "pca_covariates.csv",
    "variant_manifest.csv",
    "sample_map.csv",
    "qc_report.txt"
]

# Initialize results dictionary
results = {}

# Check each dataset
for dataset in datasets:
    dataset_dir = processed_data_dir / dataset
    dataset_results = {}
    
    # Check if dataset directory exists
    if not dataset_dir.exists():
        results[dataset] = "Directory does not exist - not preprocessed"
        continue
    
    # Check each standard file
    for file_name in standard_files:
        file_path = dataset_dir / file_name
        
        # Check if file exists
        if not file_path.exists():
            dataset_results[file_name] = "Missing"
        else:
            # Check if file has content (non-zero size)
            if file_path.stat().st_size == 0:
                dataset_results[file_name] = "Empty"
            else:
                # For CSV files, try to read and report number of rows/columns
                if file_name.endswith(".csv"):
                    try:
                        df = pd.read_csv(file_path)
                        dataset_results[file_name] = f"Contains data ({df.shape[0]} rows, {df.shape[1]} columns)"
                    except Exception as e:
                        dataset_results[file_name] = f"Has content but error reading: {str(e)}"
                # For text files, just report that they have content
                elif file_name.endswith(".txt"):
                    try:
                        with open(file_path, 'r') as f:
                            lines = len(f.readlines())
                        dataset_results[file_name] = f"Contains data ({lines} lines)"
                    except Exception as e:
                        dataset_results[file_name] = f"Has content but error reading: {str(e)}"
                else:
                    dataset_results[file_name] = "Has content"
    
    results[dataset] = dataset_results

# Print summary table
print("\n=== Preprocessing Status Summary ===\n")
print(f"{'Dataset':<20} {'File':<25} {'Status':<50}")
print("-" * 100)

for dataset, files in results.items():
    if isinstance(files, str):  # If dataset directory doesn't exist
        print(f"{dataset:<20} {'N/A':<25} {files:<50}")
    else:
        # Print dataset name only once, then list files
        print(f"{dataset:<20} {'-':<25} {'-':<50}")
        for file_name, status in files.items():
            print(f"{'':<20} {file_name:<25} {status:<50}")
    print("-" * 100)

# Identify datasets that haven't been fully preprocessed
print("\n=== Preprocessing Completeness Analysis ===\n")
for dataset, files in results.items():
    if isinstance(files, str):  # If dataset directory doesn't exist
        print(f"❌ {dataset}: Not preprocessed at all (directory missing)")
    else:
        # Check if all standard files are present and non-empty
        missing_files = [f for f, status in files.items() if status == "Missing"]
        empty_files = [f for f, status in files.items() if status == "Empty"]
        
        if not missing_files and not empty_files:
            print(f"✅ {dataset}: Fully preprocessed (all files present and contain data)")
        elif missing_files:
            print(f"❌ {dataset}: Partially preprocessed but missing {len(missing_files)} file(s): {', '.join(missing_files)}")
        elif empty_files:
            print(f"⚠️ {dataset}: Preprocessed but {len(empty_files)} file(s) are empty: {', '.join(empty_files)}")
        # Special case for y.csv which might legitimately be empty for some datasets
        elif "y.csv" in files and "Empty" in files["y.csv"]:
            print(f"⚠️ {dataset}: All files present, but y.csv is empty")