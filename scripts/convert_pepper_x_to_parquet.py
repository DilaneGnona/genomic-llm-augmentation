import os
import time
import argparse
import pandas as pd
import numpy as np

def ensure_engines():
    try:
        import pyarrow  # noqa: F401
        return 'pyarrow'
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return 'fastparquet'
        except Exception:
            return None

def write_parquet_incremental_csv(csv_path, parquet_path, chunksize=5000):
    engine = ensure_engines()
    if engine is None:
        raise RuntimeError('No Parquet engine available (pyarrow or fastparquet required)')

    pa = None
    pq = None
    fastparquet = None
    if engine == 'pyarrow':
        import pyarrow as pa
        import pyarrow.parquet as pq
    elif engine == 'fastparquet':
        import fastparquet

    first = True
    total_rows = 0
    columns = None
    writer = None

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if columns is None:
            columns = list(chunk.columns)
        # Preserve Sample_ID as string; cast feature columns to float32
        if 'Sample_ID' in chunk.columns:
            feat_cols = [c for c in chunk.columns if c != 'Sample_ID']
            chunk[feat_cols] = chunk[feat_cols].astype(np.float32)
        else:
            chunk = chunk.astype(np.float32)

        if engine == 'pyarrow':
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if first:
                writer = pq.ParquetWriter(parquet_path, table.schema, compression='snappy')
                first = False
            writer.write_table(table)
        else:
            fastparquet.write(
                parquet_path,
                chunk,
                compression='snappy',
                write_index=False,
                append=not first
            )
            first = False
        total_rows += len(chunk)

    if engine == 'pyarrow' and writer is not None:
        writer.close()

    return total_rows, columns

def verify_integrity(csv_path, parquet_path):
    # Count rows in CSV cheaply via iterator
    csv_rows = 0
    for _ in pd.read_csv(csv_path, chunksize=100000):
        csv_rows += _.shape[0]
    # Read Parquet metadata for row count
    try:
        import pyarrow.parquet as pq
        meta = pq.ParquetFile(parquet_path)
        parquet_rows = meta.metadata.num_rows
        parquet_cols = meta.schema_arrow.names
    except Exception:
        df = pd.read_parquet(parquet_path)
        parquet_rows = df.shape[0]
        parquet_cols = df.columns.tolist()
    # Read CSV header for column count
    header = pd.read_csv(csv_path, nrows=0)
    csv_cols = header.shape[1]
    return {
        'csv_rows': csv_rows,
        'parquet_rows': parquet_rows,
        'csv_cols': csv_cols,
        'parquet_cols': len(parquet_cols)
    }

def benchmark_load(csv_path, parquet_path, sample_rows=10000):
    # CSV load time (nrows)
    t0 = time.time()
    df_csv = pd.read_csv(csv_path, nrows=sample_rows)
    csv_time = time.time() - t0
    # Parquet load time (try minimal read)
    t1 = time.time()
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(parquet_path)
        # Read first row group or at least sample_rows rows
        table = pf.read_row_groups([0]) if pf.num_row_groups > 0 else pf.read()
        df_parq = table.to_pandas()
        if len(df_parq) > sample_rows:
            df_parq = df_parq.head(sample_rows)
    except Exception:
        df_parq = pd.read_parquet(parquet_path)
        if len(df_parq) > sample_rows:
            df_parq = df_parq.head(sample_rows)
    parquet_time = time.time() - t1
    # Memory usage estimation
    csv_mem = df_csv.memory_usage(deep=False).sum()
    parq_mem = df_parq.memory_usage(deep=False).sum()
    return {
        'csv_time_s': csv_time,
        'parquet_time_s': parquet_time,
        'csv_memory_bytes': int(csv_mem),
        'parquet_memory_bytes': int(parq_mem)
    }

def main():
    parser = argparse.ArgumentParser(description='Convert pepper X.csv to X.parquet with chunked writing')
    parser.add_argument('--dataset', type=str, default='pepper')
    args = parser.parse_args()

    base = os.path.join('02_processed_data', args.dataset)
    csv_path = os.path.join(base, 'X.csv')
    parquet_path = os.path.join(base, 'X.parquet')

    if os.path.exists(parquet_path):
        print('X.parquet already exists. Skipping conversion.')
    else:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        start = time.time()
        total_rows, columns = write_parquet_incremental_csv(csv_path, parquet_path, chunksize=10000)
        elapsed = time.time() - start
        print(f'Converted to Parquet: rows={total_rows}, columns={len(columns)}, time={elapsed:.2f}s')
        info = verify_integrity(csv_path, parquet_path)
        same_rows = info['csv_rows'] == info['parquet_rows']
        same_cols = info['csv_cols'] == info['parquet_cols']
        print(f'Integrity check: rows_match={same_rows}, cols_match={same_cols}')

    # Summary metrics
    csv_size = os.path.getsize(csv_path) if os.path.exists(csv_path) else 0
    parquet_size = os.path.getsize(parquet_path) if os.path.exists(parquet_path) else 0
    bench = benchmark_load(csv_path, parquet_path, sample_rows=10000) if os.path.exists(parquet_path) else None
    print('--- Summary ---')
    print(f'X.csv size: {csv_size/1024/1024:.2f} MB')
    print(f'X.parquet size: {parquet_size/1024/1024:.2f} MB')
    if bench:
        print(f'CSV load (10k rows): {bench["csv_time_s"]:.2f}s, memory={bench["csv_memory_bytes"]/1024/1024:.2f} MB')
        print(f'Parquet load (~1 row group): {bench["parquet_time_s"]:.2f}s, memory={bench["parquet_memory_bytes"]/1024/1024:.2f} MB')
    print('Parquet will be preferred by augmented pipeline for X features.')

    # Persist summary for external retrieval
    summary_path = os.path.join(os.path.abspath(base), 'parquet_summary.txt')
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('--- Summary ---\n')
            f.write(f'X.csv size: {csv_size/1024/1024:.2f} MB\n')
            f.write(f'X.parquet size: {parquet_size/1024/1024:.2f} MB\n')
            if bench:
                f.write(f'CSV load (10k rows): {bench["csv_time_s"]:.2f}s, memory={bench["csv_memory_bytes"]/1024/1024:.2f} MB\n')
                f.write(f'Parquet load (~1 row group): {bench["parquet_time_s"]:.2f}s, memory={bench["parquet_memory_bytes"]/1024/1024:.2f} MB\n')
            f.write('Parquet will be preferred by augmented pipeline for X features.\n')
            f.flush()
    except Exception:
        pass

if __name__ == '__main__':
    main()
