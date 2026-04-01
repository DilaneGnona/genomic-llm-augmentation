import pandas as pd

print("Testing small load...")

try:
    # Try reading just the first few rows
    df = pd.read_csv('02_processed_data/pepper/X_aligned.csv', engine='python', nrows=5)
    print("Successfully read first 5 rows")
    print(f"Columns: {len(df.columns)}")
    print(f"First column names: {list(df.columns[:5])}")
    print("\nFirst few rows:")
    print(df.head())
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
