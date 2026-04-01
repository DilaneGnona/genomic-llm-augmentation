import os
import csv
import json
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE = "02_processed_data/ipk_out_raw"
X_PATH = os.path.join(BASE, "X.csv")
PCA_PATH = os.path.join(BASE, "pca_covariates.csv")
Y_PATH = os.path.join(BASE, "y.csv")
MAP_PATH = os.path.join(BASE, "sample_id_map.csv")

# 1) Ensure X.csv and pca_covariates.csv use 'Sample_ID' as first column
for path in [X_PATH, PCA_PATH]:
    df = pd.read_csv(path)
    first_col = df.columns[0]
    if first_col != 'Sample_ID':
        logging.info(f"Renaming first column '{first_col}' to 'Sample_ID' in {path}")
        df = df.rename(columns={first_col: 'Sample_ID'})
        df.to_csv(path, index=False)
    else:
        logging.info(f"'{path}' already has 'Sample_ID' column")

# 2) Build sample_id_map.csv by pairing first N GBS_BIOSAMPLE_IDs to Sample_IDs in X
X_df = pd.read_csv(X_PATH)
Y_df = pd.read_csv(Y_PATH)

sample_ids = X_df['Sample_ID'].tolist()
gbs_ids = Y_df['GBS_BIOSAMPLE_ID'].tolist() if 'GBS_BIOSAMPLE_ID' in Y_df.columns else []

if not gbs_ids:
    logging.error("y.csv does not have 'GBS_BIOSAMPLE_ID' column; cannot build mapping")
    raise SystemExit(1)

N = min(len(sample_ids), len(gbs_ids))
map_df = pd.DataFrame({
    'GBS_BIOSAMPLE_ID': gbs_ids[:N],
    'Sample_ID': sample_ids[:N]
})
map_df.to_csv(MAP_PATH, index=False)
logging.info(f"Wrote mapping with {N} pairs to {MAP_PATH}")

# 3) Create processed y.csv with Sample_ID and YR_LS via join
if 'YR_LS' not in Y_df.columns:
    logging.error("y.csv missing 'YR_LS' target column; cannot process")
    raise SystemExit(1)

Y_proc = map_df.merge(Y_df[['GBS_BIOSAMPLE_ID', 'YR_LS']], on='GBS_BIOSAMPLE_ID', how='left')
Y_proc = Y_proc[['Sample_ID', 'YR_LS']]
Y_proc.to_csv(Y_PATH, index=False)
logging.info(f"Wrote processed y.csv with {Y_proc.shape[0]} rows to {Y_PATH}")

# 4) Save alignment report
report = {
    'total_X_samples': len(sample_ids),
    'total_Y_rows': len(Y_df),
    'mapped_pairs': N,
    'aligned_samples_in_y': int(Y_proc['YR_LS'].notna().sum())
}
logging.info(f"Alignment report: {json.dumps(report)}")
with open(os.path.join(BASE, 'mapping_report.json'), 'w') as f:
    json.dump(report, f, indent=2)