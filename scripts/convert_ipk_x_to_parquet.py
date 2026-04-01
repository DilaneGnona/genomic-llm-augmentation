import os
import time
import argparse
import pandas as pd

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

def write_parquet_incremental_csv(csv_path, parquet_path, chunksize=50000):
    engine = ensure_engines()
    if engine is None:
        raise RuntimeError('No Parquet engine available (pyarrow or fastparquet required)')

    pa = None
    pq = None
    fastparquet_mod = None
    if engine == 'pyarrow':
        import pyarrow as pa
        import pyarrow.parquet as pq
    else:
        import fastparquet as fastparquet_mod

    first = True
    total_rows = 0
    columns = None
    writer = None

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if columns is None:
            columns = list(chunk.columns)
        # Do not change dtypes; preserve Sample_ID and SNP values exactly
        if engine == 'pyarrow':
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if first:
                writer = pq.ParquetWriter(parquet_path, table.schema, compression='snappy')
                first = False
            writer.write_table(table)
        else:
            fastparquet_mod.write(
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

def verify_full_equality(csv_path, parquet_path):
    df_csv = pd.read_csv(csv_path)
    df_parq = pd.read_parquet(parquet_path)
    same_shape = df_csv.shape == df_parq.shape
    same_columns = list(df_csv.columns) == list(df_parq.columns)
    values_equal = True
    if same_shape and same_columns:
        for c in df_csv.columns:
            s1, s2 = df_csv[c], df_parq[c]
            if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
                try:
                    values_equal &= (s1.values == s2.values).all()
                except Exception:
                    values_equal = False
                    break
            else:
                values_equal &= (s1.astype(str).values == s2.astype(str).values).all()
                if not values_equal:
                    break
    else:
        values_equal = False
    return {
        'rows_csv': df_csv.shape[0],
        'cols_csv': df_csv.shape[1],
        'rows_parquet': df_parq.shape[0],
        'cols_parquet': df_parq.shape[1],
        'same_shape': same_shape,
        'same_columns': same_columns,
        'values_equal': values_equal,
    }

def main():
    parser = argparse.ArgumentParser(description='Convert ipk_out_raw X.csv to X.parquet (snappy) without altering values')
    parser.add_argument('--chunksize', type=int, default=50000)
    args = parser.parse_args()

    base = os.path.join('02_processed_data', 'ipk_out_raw')
    csv_path = os.path.join(base, 'X.csv')
    parquet_path = os.path.join(base, 'X.parquet')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    start = time.time()
    total_rows, columns = write_parquet_incremental_csv(csv_path, parquet_path, chunksize=args.chunksize)
    elapsed = time.time() - start
    print(f'Conversion terminée: rows={total_rows}, columns={len(columns)}, time={elapsed:.2f}s')

    info = verify_full_equality(csv_path, parquet_path)
    print(f'Vérification: rows_csv={info["rows_csv"]}, rows_parquet={info["rows_parquet"]}, cols_csv={info["cols_csv"]}, cols_parquet={info["cols_parquet"]}')
    print(f'Colonnes identiques: {info["same_columns"]}, Forme identique: {info["same_shape"]}, Valeurs identiques: {info["values_equal"]}')

    if not (info['same_columns'] and info['same_shape'] and info['values_equal']):
        raise RuntimeError('La vérification a échoué: les données Parquet diffèrent du CSV')

if __name__ == '__main__':
    main()

