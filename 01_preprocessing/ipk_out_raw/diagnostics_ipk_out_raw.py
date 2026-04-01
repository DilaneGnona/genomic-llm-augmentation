import os
import json
import logging
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/ipk_out_raw_prep.log"),
        logging.StreamHandler()
    ]
)

# Create necessary directories
os.makedirs("03_modeling_results/ipk_out_raw/models", exist_ok=True)
os.makedirs("03_modeling_results/ipk_out_raw/metrics", exist_ok=True)
os.makedirs("03_modeling_results/ipk_out_raw/plots", exist_ok=True)
os.makedirs("03_modeling_results/ipk_out_raw/logs", exist_ok=True)

# Configure file logging for diagnostics
logger = logging.getLogger()
diagnostic_logger = logging.FileHandler("03_modeling_results/ipk_out_raw/logs/diagnostics.log")
diagnostic_logger.setLevel(logging.INFO)
diagnostic_logger.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(diagnostic_logger)


def verify_sample_alignment():
    """Verify that sample IDs match across X, y, and PCA covariates"""
    logging.info("Verifying sample alignment across datasets...")
    
    try:
        # Read first column of X.csv (assuming first column is Sample_ID)
        X_sample_ids = pd.read_csv("02_processed_data/ipk_out_raw/X.csv", engine='python', usecols=[0])
        # Read pca_covariates.csv to get Sample_IDs
        pca_df = pd.read_csv("02_processed_data/ipk_out_raw/pca_covariates.csv", engine='python')
        
        # Extract sample IDs
        X_samples = set(X_sample_ids.iloc[:, 0])
        pca_samples = set(pca_df['Sample_ID'])
        
        # For y, since it only has Sample_ID, we'll just count it
        y_sample_ids = pd.read_csv("02_processed_data/ipk_out_raw/y.csv", engine='python')
        y_samples = set(y_sample_ids.iloc[:, 0])
        
        # Find intersection
        common_samples = X_samples & y_samples & pca_samples
        
        # Log sample counts
        logging.info(f"Sample counts: X={len(X_samples)}, y={len(y_samples)}, PCA={len(pca_samples)}")
        logging.info(f"Common samples: {len(common_samples)}")
        
        # For this dataset, we'll consider alignment valid if all samples are in the intersection
        if len(common_samples) == len(X_samples) and len(common_samples) == len(y_samples) and len(common_samples) == len(pca_samples):
            logging.info(f"All {len(common_samples)} samples are properly aligned across datasets")
            return True, len(common_samples)
        else:
            logging.warning(f"Some samples may not be aligned across datasets")
            # Calculate overlaps
            logging.warning(f"Samples in X but not in all: {len(X_samples - common_samples)}")
            logging.warning(f"Samples in y but not in all: {len(y_samples - common_samples)}")
            logging.warning(f"Samples in PCA but not in all: {len(pca_samples - common_samples)}")
            return len(common_samples) > 0, len(common_samples)
    except Exception as e:
        logging.error(f"Error during sample alignment verification: {str(e)}")
        return False, 0


def compute_target_statistics():
    """Compute descriptive statistics for the available target variables"""
    logging.info("Computing target variable statistics...")
    
    try:
        # Load original phenotype data
        phenotype_file = "ipk_out_raw/Geno_IDs_and_Phenotypes.txt"
        if not os.path.exists(phenotype_file):
            logging.error(f"Phenotype file {phenotype_file} not found")
            return None
        
        # Read phenotype data
        pheno_df = pd.read_csv(phenotype_file, sep='\t')
        
        # Possible target columns
        target_cols = ['YR_LS', 'YR_precision', 'Yield_BV']
        stats_dict = {}
        
        for col in target_cols:
            if col in pheno_df.columns:
                values = pheno_df[col].dropna().values
                if len(values) > 0:
                    stats_dict[col] = {
                        "count": len(values),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "median": float(np.median(values)),
                        "skew": float(stats.skew(values)),
                        "na_count": int(pheno_df[col].isna().sum())
                    }
                    logging.info(f"Statistics for {col}: count={stats_dict[col]['count']}, mean={stats_dict[col]['mean']:.4f}, "
                              f"std={stats_dict[col]['std']:.4f}, NA={stats_dict[col]['na_count']}")
        
        # Save statistics
        with open("03_modeling_results/ipk_out_raw/metrics/target_stats.json", "w") as f:
            json.dump(stats_dict, f, indent=2)
        
        return stats_dict
    except Exception as e:
        logging.error(f"Error computing target statistics: {str(e)}")
        return None


def compute_baseline_performance():
    """Train and evaluate a simple baseline model using the most complete target variable"""
    logging.info("Computing baseline model performance...")
    
    try:
        # Load original phenotype data
        pheno_df = pd.read_csv("ipk_out_raw/Geno_IDs_and_Phenotypes.txt", sep='\t')
        
        # Find the target column with most non-NA values
        target_cols = ['YR_LS', 'YR_precision', 'Yield_BV']
        best_target = None
        max_non_na = 0
        
        for col in target_cols:
            if col in pheno_df.columns:
                non_na_count = pheno_df[col].count()
                if non_na_count > max_non_na:
                    max_non_na = non_na_count
                    best_target = col
        
        if best_target is None:
            logging.error("No valid target variables found")
            return None
        
        logging.info(f"Using {best_target} as target variable with {max_non_na} non-NA values")
        
        # Get target values
        y_values = pheno_df[best_target].dropna().values
        
        # Split for baseline evaluation (using simple split since we don't need features)
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        # Simple split (80/20) for baseline
        y_train, y_test = train_test_split(y_values, test_size=0.2, random_state=42)
        
        # Baseline: predict mean of training data
        train_mean = np.mean(y_train)
        y_pred = np.full_like(y_test, train_mean)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        baseline_metrics = {
            "target_variable": best_target,
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "train_mean": float(train_mean),
            "sample_count": len(y_values)
        }
        
        logging.info(f"Baseline performance (predict-mean) for {best_target}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        return baseline_metrics
    except Exception as e:
        logging.error(f"Error computing baseline performance: {str(e)}")
        return None


def generate_diagnostics_report(sample_alignment, target_stats, baseline_metrics):
    """Generate a comprehensive diagnostics report"""
    logging.info("Generating diagnostics report...")
    
    try:
        report = {
            "timestamp": datetime.now().isoformat(),
            "sample_alignment": {
                "status": "aligned" if sample_alignment[0] else "misaligned",
                "common_samples_count": sample_alignment[1]
            },
            "target_statistics": target_stats,
            "baseline_performance": baseline_metrics,
            "data_info": {
                "processed_data_path": "02_processed_data/ipk_out_raw/",
                "notes": "y.csv only contains Sample_IDs; actual target variables in Geno_IDs_and_Phenotypes.txt"
            }
        }
        
        with open("03_modeling_results/ipk_out_raw/metrics/diagnostics_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        logging.info("Diagnostics report generated successfully")
        return True
    except Exception as e:
        logging.error(f"Error generating diagnostics report: {str(e)}")
        return False


def main():
    """Main diagnostics function"""
    logging.info("Starting diagnostics for ipk_out_raw dataset...")
    
    # Run all diagnostics
    sample_alignment = verify_sample_alignment()
    if not sample_alignment[0]:
        logging.warning("Proceeding despite sample alignment issues")
    
    target_stats = compute_target_statistics()
    baseline_metrics = compute_baseline_performance()
    
    # Generate final report
    generate_diagnostics_report(sample_alignment, target_stats, baseline_metrics)
    
    logging.info("Diagnostics completed")


if __name__ == "__main__":
    main()