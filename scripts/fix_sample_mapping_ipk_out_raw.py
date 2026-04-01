import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    base_dir = "c:\\Users\\OMEN\\Desktop\\experiment_snp"
    data_dir = os.path.join(base_dir, "02_processed_data", "ipk_out_raw")
    
    # Load the first few rows of X.csv to understand Sample_ID format
    logger.info("Loading a sample of X.csv to understand Sample_ID format...")
    try:
        # Read just the first 10 rows and only the index column
        X_sample = pd.read_csv(os.path.join(data_dir, "X.csv"), usecols=[0], nrows=10)
        logger.info(f"Sample X indices: {list(X_sample.iloc[:, 0])}")
    except Exception as e:
        logger.error(f"Error reading X.csv sample: {str(e)}")
        return
    
    # Load the clean phenotype file
    logger.info("Loading clean phenotype file...")
    y_clean = pd.read_csv(os.path.join(data_dir, "y_ipk_out_raw_clean.csv"))
    logger.info(f"Loaded y_ipk_out_raw_clean.csv with {len(y_clean)} samples")
    logger.info(f"Sample GBS_BIOSAMPLE_IDs: {list(y_clean['GBS_BIOSAMPLE_ID'].head())}")
    
    # Load sample_map.csv to see if it contains any mapping information
    sample_map_path = os.path.join(data_dir, "sample_map.csv")
    if os.path.exists(sample_map_path):
        logger.info("Loading sample_map.csv...")
        sample_map = pd.read_csv(sample_map_path)
        logger.info(f"Loaded sample_map.csv with {len(sample_map)} entries")
        logger.info(f"Sample map columns: {sample_map.columns.tolist()}")
        logger.info(f"Sample map data:")
        logger.info(sample_map.head().to_string())
    
    # Since we can't read the full X.csv directly, let's create a new approach:
    # 1. Create a new y.csv file with Sample_ID as index (same as GBS_BIOSAMPLE_ID for now)
    # 2. This will help the diagnostic script run and properly handle sample alignment
    
    logger.info("Creating new y.csv with Sample_ID as index...")
    y_new = y_clean.copy()
    y_new = y_new.rename(columns={"GBS_BIOSAMPLE_ID": "Sample_ID"})
    y_new = y_new.set_index("Sample_ID")
    
    # Save the new y.csv file
    new_y_path = os.path.join(data_dir, "y.csv")
    y_new.to_csv(new_y_path)
    logger.info(f"Saved new y.csv with {len(y_new)} samples")
    
    # Print statistics about the phenotype data
    logger.info(f"YR_LS statistics:")
    logger.info(f"  Count: {len(y_new['YR_LS'])}")
    logger.info(f"  Non-NA count: {y_new['YR_LS'].notna().sum()}")
    logger.info(f"  Mean: {y_new['YR_LS'].mean():.4f}")
    logger.info(f"  Std: {y_new['YR_LS'].std():.4f}")
    logger.info(f"  Min/Max: {y_new['YR_LS'].min():.4f}/{y_new['YR_LS'].max():.4f}")
    
if __name__ == "__main__":
    main()