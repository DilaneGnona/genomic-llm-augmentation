import pandas as pd
import os

x_path = '02_processed_data/pepper/X.csv'

print(f"Reading first 10 lines of {x_path}")
print(f"File size: {os.path.getsize(x_path)} bytes")

try:
    df = pd.read_csv(x_path, dtype={'ID_12': 'object'}, on_bad_lines='skip', engine='python', nrows=10)
    print(f"\nSuccessfully loaded first 10 lines!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData:")
    print(df)
    
    # Check ID_12 column
    print(f"\nID_12 column:")
    print(df['ID_12'])
    print(f"ID_12 dtype: {df['ID_12'].dtype}")
    print(f"Unique values: {df['ID_12'].nunique()}")
    
    # Check for any issues with the data
    print(f"\nChecking for issues...")
    print(f"NaN values: {df.isna().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Test loading without on_bad_lines
    print(f"\nTesting without on_bad_lines...")
    df2 = pd.read_csv(x_path, dtype={'ID_12': 'object'}, engine='python', nrows=10)
    print(f"Successfully loaded without on_bad_lines!")
    print(f"Shape: {df2.shape}")
    
    # Test loading with dtype=str for all columns
    print(f"\nTesting with dtype=str for all columns...")
    df3 = pd.read_csv(x_path, dtype=str, engine='python', nrows=10)
    print(f"Successfully loaded with dtype=str!")
    print(f"Shape: {df3.shape}")
    print(f"ID_12 dtype: {df3['ID_12'].dtype}")
    
except Exception as e:
    print(f"\nError loading X.csv: {e}")
    import traceback
    traceback.print_exc()