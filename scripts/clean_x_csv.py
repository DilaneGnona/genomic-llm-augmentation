"""
Clean X.csv by removing metadata rows (POS, REF, ALT)
Uses chunking to handle large files efficiently
"""
import pandas as pd
import os

BASE = r'c:\Users\OMEN\Desktop\experiment_snp'
input_path = os.path.join(BASE, '02_processed_data', 'pepper', 'X.csv')
output_path = os.path.join(BASE, '02_processed_data', 'pepper', 'X_cleaned.csv')

print("Starting to clean X.csv...")
print(f"Input: {input_path}")
print(f"Output: {output_path}")
print()

# Process in chunks to handle large file
chunk_size = 10000
chunks_cleaned = []
total_rows = 0
cleaned_rows = 0

for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size, low_memory=False)):
    # Filter rows where Sample_ID starts with SAMEA
    mask = chunk['Sample_ID'].astype(str).str.startswith('SAMEA', na=False)
    chunk_clean = chunk[mask].copy()
    
    chunks_cleaned.append(chunk_clean)
    total_rows += len(chunk)
    cleaned_rows += len(chunk_clean)
    
    if i % 10 == 0:
        print(f"Processed {total_rows} rows, kept {cleaned_rows} samples...")

print(f"\nTotal rows processed: {total_rows}")
print(f"Samples kept: {cleaned_rows}")

# Combine all cleaned chunks
print("\nCombining chunks...")
df_clean = pd.concat(chunks_cleaned, ignore_index=True)

# Convert feature columns to numeric
print("Converting feature columns to numeric...")
feature_cols = [c for c in df_clean.columns if c != 'Sample_ID']
for col in feature_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Save cleaned file
print(f"\nSaving to: {output_path}")
df_clean.to_csv(output_path, index=False)

print(f"\n✓ Done! Cleaned file saved.")
print(f"Final shape: {df_clean.shape}")
print(f"Sample IDs: {df_clean['Sample_ID'].head(5).tolist()}")
