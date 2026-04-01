import pandas as pd
import numpy as np

# Lire seulement les headers d'abord
df = pd.read_csv('02_processed_data/pepper/X_aligned.csv', nrows=5)
print("Colonnes:", len(df.columns))
print("Noms (début):", list(df.columns)[:5])
print("Noms (fin):", list(df.columns)[-5:])

snp_cols = [c for c in df.columns if c != 'Sample_ID']
print("\nNombre de SNPs:", len(snp_cols))
print("Exemple valeurs SNP:")
print(df[snp_cols[0]].values[:5])
