import os
import time
import pandas as pd

base = os.path.abspath(os.path.join('02_processed_data', 'pepper'))
csv_path = os.path.join(base, 'X.csv')
parquet_path = os.path.join(base, 'X.parquet')

csv_size = os.path.getsize(csv_path) if os.path.exists(csv_path) else 0
parquet_size = os.path.getsize(parquet_path) if os.path.exists(parquet_path) else 0

csv_time = None
parquet_time = None
csv_mem = None
parq_mem = None

try:
    t0 = time.time()
    df_csv = pd.read_csv(csv_path, nrows=10000)
    csv_time = time.time() - t0
    csv_mem = int(df_csv.memory_usage(deep=False).sum())
except Exception:
    pass

try:
    t1 = time.time()
    df_parq = pd.read_parquet(parquet_path)
    if len(df_parq) > 10000:
        df_parq = df_parq.head(10000)
    parquet_time = time.time() - t1
    parq_mem = int(df_parq.memory_usage(deep=False).sum())
except Exception:
    pass

out = os.path.join(base, 'parquet_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('--- Summary ---\n')
    f.write(f'X.csv size: {csv_size/1024/1024:.2f} MB\n')
    f.write(f'X.parquet size: {parquet_size/1024/1024:.2f} MB\n')
    if csv_time is not None and csv_mem is not None:
        f.write(f'CSV load (10k rows): {csv_time:.2f}s, memory={csv_mem/1024/1024:.2f} MB\n')
    if parquet_time is not None and parq_mem is not None:
        f.write(f'Parquet load (~1 row group): {parquet_time:.2f}s, memory={parq_mem/1024/1024:.2f} MB\n')
    f.write('Parquet will be preferred by augmented pipeline for X features.\n')

