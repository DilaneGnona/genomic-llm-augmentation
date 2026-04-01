import pandas as pd
import os
import sys
import logging

# Set up logging to write to a file
logging.basicConfig(filename='x_loading_test.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def test_x_loading():
    logging.info("Starting X.csv loading test")
    
    x_path = '02_processed_data/pepper/X.csv'
    logging.info(f"X.csv path: {x_path}")
    logging.info(f"File exists: {os.path.exists(x_path)}")
    logging.info(f"File size: {os.path.getsize(x_path)} bytes")
    
    try:
        logging.info("Attempting to load X.csv with pd.read_csv(x_path, dtype={'ID_12': 'object'}, on_bad_lines='skip', engine='python')")
        X_df = pd.read_csv(x_path, dtype={'ID_12': 'object'}, on_bad_lines='skip', engine='python')
        logging.info(f"X.csv loaded successfully!")
        logging.info(f"Number of rows: {X_df.shape[0]}")
        logging.info(f"Number of columns: {X_df.shape[1]}")
        logging.info(f"Column names: {list(X_df.columns[:5])}...")
        
        # Test some basic operations
        logging.info(f"Sample ID column: {_detect_id_column(X_df)}")
        logging.info(f"Data types of first 5 columns: {X_df.dtypes[:5]}")
        
        # Check for NaN values
        logging.info(f"NaN values in ID_12: {X_df['ID_12'].isna().sum()}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error loading X.csv: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

# Copy the _detect_id_column function from the pipeline
def _detect_id_column(df):
    """Detect the ID column in a DataFrame"""
    common_id_columns = ['Sample_ID', 'sample_id', 'ID', 'id', 'Individual_ID', 'individual_id']
    for col in df.columns:
        if col in common_id_columns:
            return col
    # If no common ID column found, check the first column
    if len(df.columns) > 0:
        return df.columns[0]
    return None

if __name__ == "__main__":
    success = test_x_loading()
    sys.exit(0 if success else 1)