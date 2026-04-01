import os
import argparse
import json
import pandas as pd
import numpy as np
from src.config import ProjectConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

def _read_csv_candidates(base, names):
    for name in names:
        path = os.path.join(base, name)
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    raise SystemExit(f"Failed to read any of: {', '.join(names)}")

def load_processed(cfg, dataset):
    base = os.path.join(cfg.paths.get("PROCESSED_DIR", "02_processed_data"), dataset)
    X = _read_csv_candidates(base, ["X.csv", "X_cleaned.csv", "X_mini.csv"])
    y = pd.read_csv(os.path.join(base, "y.csv"))
    pca = pd.read_csv(os.path.join(base, "pca_covariates.csv"))
    X["Sample_ID"] = X["Sample_ID"].astype(str)
    y["Sample_ID"] = y["Sample_ID"].astype(str)
    pca["Sample_ID"] = pca["Sample_ID"].astype(str)
    ids = list(set(X["Sample_ID"]).intersection(set(y["Sample_ID"])).intersection(set(pca["Sample_ID"])))
    X = X[X["Sample_ID"].isin(ids)].sort_values("Sample_ID").reset_index(drop=True)
    y = y[y["Sample_ID"].isin(ids)].sort_values("Sample_ID").reset_index(drop=True)
    pca = pca[pca["Sample_ID"].isin(ids)].sort_values("Sample_ID").reset_index(drop=True)
    target = cfg.get_target_column(dataset)
    if target is None or target not in y.columns:
        cols = y.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in cols if "ID" not in c.upper() and "SAMPLE" not in c.upper()]
        if not cols:
            raise SystemExit("No numeric target in y.csv")
        target = cols[0]
    y_target = pd.to_numeric(y[target], errors="coerce")
    mask = y_target.notna()
    X = X[mask].reset_index(drop=True)
    pca = pca[mask].reset_index(drop=True)
    y_target = y_target[mask].reset_index(drop=True)
    X_feat = X.drop(columns=["Sample_ID"])
    pca_feat = pca.drop(columns=["Sample_ID"])
    X_all = pd.concat([X_feat, pca_feat], axis=1)
    X_all = X_all.replace([np.inf, -np.inf], np.nan)
    X_all = X_all.fillna(X_all.median())
    return X_all.values.astype(float), y_target.values.astype(float), X["Sample_ID"].tolist(), target

def build_model(name):
    needs_scaling = False
    if name == "ridge":
        model = Ridge()
        needs_scaling = True
    elif name == "xgboost":
        if not XGB_AVAILABLE:
            raise SystemExit("xgboost is not installed")
        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    elif name == "lightgbm":
        if not LGBM_AVAILABLE:
            raise SystemExit("lightgbm is not installed")
        model = LGBMRegressor(n_estimators=500, num_leaves=31, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    else:
        model = Ridge()
        needs_scaling = True
    return model, needs_scaling

def run(dataset, model_name):
    cfg = ProjectConfig("config.yaml")
    X, y, ids, target = load_processed(cfg, dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model, needs_scaling = build_model(model_name)
    
    if needs_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "dataset": dataset,
        "model": model_name,
        "target": target,
        "samples": int(len(y)),
        "features": int(X.shape[1]),
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }
    outdir = os.path.join("03_modeling_results", dataset, "metrics")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "run_experiment_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=False)
    parser.add_argument("--model", required=False, choices=["ridge", "xgboost", "lightgbm"], default="ridge")
    args = parser.parse_args()
    cfg = ProjectConfig("config.yaml")
    ds = args.dataset or cfg.get_dataset()
    run(ds, args.model)

if __name__ == "__main__":
    main()
