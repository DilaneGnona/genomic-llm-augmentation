import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = r"c:\Users\OMEN\Desktop\experiment_snp"
PROCESSED_DIR = os.path.join(BASE_DIR, "02_processed_data")

REQUIRED_FILES = ["X.csv", "y.csv", "pca_covariates.csv"]

def check_files_exist(dataset):
    ds_dir = os.path.join(PROCESSED_DIR, dataset)
    missing = []
    for fname in REQUIRED_FILES:
        if not os.path.exists(os.path.join(ds_dir, fname)):
            missing.append(fname)
    return ds_dir, missing

def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        return f"ERROR: {e}"

def ensure_sample_id_string(df, name):
    if "Sample_ID" not in df.columns:
        return f"{name}: colonne Sample_ID absente"
    try:
        df["Sample_ID"] = df["Sample_ID"].astype(str)
        return None
    except Exception as e:
        return f"{name}: échec conversion Sample_ID en str ({e})"

def check_numeric_features(df, exclude_cols):
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith("Unnamed")]
    if not feature_cols:
        return {"ok": False, "message": "Aucune colonne de features détectée"}
    sample = df[feature_cols].head(100)
    non_numeric = []
    for c in sample.columns:
        if not pd.api.types.is_numeric_dtype(sample[c]):
            converted = pd.to_numeric(sample[c], errors="coerce")
            if converted.isna().all():
                non_numeric.append(c)
    ratio_non_numeric = len(non_numeric) / len(sample.columns)
    return {
        "ok": ratio_non_numeric < 0.05,
        "message": f"Features non-numériques échantillon: {len(non_numeric)}/{len(sample.columns)}",
        "non_numeric_examples": non_numeric[:10]
    }

def check_alignment(x_df, y_df, pca_df):
    x_ids = set(x_df["Sample_ID"])
    y_ids = set(y_df["Sample_ID"])
    pca_ids = set(pca_df["Sample_ID"])
    common = x_ids & y_ids & pca_ids
    only_x = x_ids - common
    only_y = y_ids - common
    only_p = pca_ids - common
    return {
        "common_count": len(common),
        "only_in_X": list(sorted(only_x))[:10],
        "only_in_y": list(sorted(only_y))[:10],
        "only_in_pca": list(sorted(only_p))[:10]
    }

def check_target_columns(dataset, y_df):
    if dataset == "ipk_out_raw":
        num_cols = y_df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if "ID" not in c.upper() and "SAMPLE" not in c.upper()]
        return {
            "ok": len(num_cols) > 0,
            "message": f"Colonnes cibles numériques détectées: {num_cols[:5]}"
        }
    elif dataset == "pepper":
        ok = "Yield_BV" in y_df.columns
        return {"ok": ok, "message": "Présence de Yield_BV dans y.csv" if ok else "Yield_BV absent de y.csv"}
    else:
        return {"ok": True, "message": "Dataset non reconnu, vérification cibles ignorée"}

def verify_dataset(dataset):
    print(f"\n=== Vérification dataset: {dataset} ===")
    ds_dir, missing = check_files_exist(dataset)
    if missing:
        print(f"Fichiers manquants: {missing}")
        return {"dataset": dataset, "ok": False, "errors": [f"Manque: {', '.join(missing)}"]}
    x_df = load_csv(os.path.join(ds_dir, "X.csv"))
    y_df = load_csv(os.path.join(ds_dir, "y.csv"))
    pca_df = load_csv(os.path.join(ds_dir, "pca_covariates.csv"))
    errors = []
    for name, df in [("X.csv", x_df), ("y.csv", y_df), ("pca_covariates.csv", pca_df)]:
        if isinstance(df, str):
            errors.append(f"{name}: {df}")
    if errors:
        for e in errors:
            print(e)
        return {"dataset": dataset, "ok": False, "errors": errors}
    for name, df in [("X", x_df), ("y", y_df), ("pca", pca_df)]:
        err = ensure_sample_id_string(df, name)
        if err:
            errors.append(err)
    x_num = check_numeric_features(x_df, exclude_cols=["Sample_ID"])
    p_num = check_numeric_features(pca_df, exclude_cols=["Sample_ID"])
    align = check_alignment(x_df, y_df, pca_df)
    tgt = check_target_columns(dataset, y_df)
    dup_x = x_df["Sample_ID"].duplicated().sum()
    dup_y = y_df["Sample_ID"].duplicated().sum()
    dup_p = pca_df["Sample_ID"].duplicated().sum()
    summary = {
        "dataset": dataset,
        "files_ok": True,
        "sample_id_errors": errors,
        "x_rows": len(x_df),
        "y_rows": len(y_df),
        "pca_rows": len(pca_df),
        "x_features_count": max(len(x_df.columns) - 1, 0),
        "pca_features_count": max(len(pca_df.columns) - 1, 0),
        "x_numeric_check": x_num,
        "pca_numeric_check": p_num,
        "alignment": align,
        "target_check": tgt,
        "duplicates": {"X": int(dup_x), "y": int(dup_y), "pca": int(dup_p)},
        "ok": True
    }
    if errors:
        summary["ok"] = False
    if not x_num["ok"] or not p_num["ok"]:
        summary["ok"] = False
    if align["common_count"] == 0:
        summary["ok"] = False
    if not tgt["ok"]:
        summary["ok"] = False
    print(f"- Lignes: X={summary['x_rows']} y={summary['y_rows']} pca={summary['pca_rows']}")
    print(f"- IDs communs: {summary['alignment']['common_count']}")
    if summary['duplicates']['X'] or summary['duplicates']['y'] or summary['duplicates']['pca']:
        print(f"- Duplicats IDs: X={summary['duplicates']['X']} y={summary['duplicates']['y']} pca={summary['duplicates']['pca']}")
    print(f"- X numérique: {summary['x_numeric_check']['message']}")
    print(f"- PCA numérique: {summary['pca_numeric_check']['message']}")
    print(f"- Cible: {summary['target_check']['message']}")
    if summary["sample_id_errors"]:
        print(f"- Problèmes Sample_ID: {summary['sample_id_errors']}")
    print(f"=> OK: {summary['ok']}")
    out_dir = os.path.join(BASE_DIR, "03_modeling_results", dataset, "logs")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "data_format_check.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(str(summary))
    print(f"Rapport sauvegardé: {report_path}")
    return summary

def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["pepper", "ipk_out_raw"]
    results = []
    for ds in datasets:
        results.append(verify_dataset(ds))
    ok_all = all(r.get("ok") for r in results)
    sys.exit(0 if ok_all else 1)

if __name__ == "__main__":
    main()
