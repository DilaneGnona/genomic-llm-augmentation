import os
import time
import pandas as pd

base = os.path.join('02_processed_data', 'pepper')
csv_path = os.path.join(base, 'X.csv')
parquet_path = os.path.join(base, 'X.parquet')

csv_size = os.path.getsize(csv_path) if os.path.exists(csv_path) else 0
parquet_size = os.path.getsize(parquet_path) if os.path.exists(parquet_path) else 0

sample_rows = 10000

t0 = time.time()
df_csv = pd.read_csv(csv_path, nrows=sample_rows)
csv_time = time.time() - t0
csv_mem = df_csv.memory_usage(deep=False).sum()
csv_line = f'CSV load (10k rows): {csv_time:.2f}s, memory={csv_mem/1024/1024:.2f} MB'

t1 = time.time()
try:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    table = pf.read_row_groups([0]) if pf.num_row_groups > 0 else pf.read()
    df_parq = table.to_pandas()
    if len(df_parq) > sample_rows:
        df_parq = df_parq.head(sample_rows)
except Exception:
    df_parq = pd.read_parquet(parquet_path)
    if len(df_parq) > sample_rows:
        df_parq = df_parq.head(sample_rows)
parquet_time = time.time() - t1
parq_mem = df_parq.memory_usage(deep=False).sum()
parq_line = f'Parquet load (~1 row group): {parquet_time:.2f}s, memory={parq_mem/1024/1024:.2f} MB'

out = os.path.join(base, 'parquet_benchmark.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('--- Benchmark ---\n')
    f.write(f'X.csv size: {csv_size/1024/1024:.2f} MB\n')
    f.write(f'X.parquet size: {parquet_size/1024/1024:.2f} MB\n')
    f.write(csv_line + '\n')
    f.write(parq_line + '\n')
