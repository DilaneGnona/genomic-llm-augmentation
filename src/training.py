import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import json
from src.config import ProjectConfig
from src.models import build_model

def load_data(cfg, dataset):
    processed_dir = os.path.join(cfg.paths.get("PROCESSED_DIR", "02_processed_data"), dataset)
    X = pd.read_csv(os.path.join(processed_dir, "X.csv"))
    y = pd.read_csv(os.path.join(processed_dir, "y.csv"))
    pca = pd.read_csv(os.path.join(processed_dir, "pca_covariates.csv"))
    
    # Align
    X['Sample_ID'] = X['Sample_ID'].astype(str)
    y['Sample_ID'] = y['Sample_ID'].astype(str)
    pca['Sample_ID'] = pca['Sample_ID'].astype(str)
    
    # Drop duplicates in case they exist
    X = X.drop_duplicates('Sample_ID')
    y = y.drop_duplicates('Sample_ID')
    pca = pca.drop_duplicates('Sample_ID')
    
    ids = list(set(X['Sample_ID']).intersection(set(y['Sample_ID'])).intersection(set(pca['Sample_ID'])))
    X = X[X['Sample_ID'].isin(ids)].sort_values('Sample_ID').set_index('Sample_ID')
    y = y[y['Sample_ID'].isin(ids)].sort_values('Sample_ID').set_index('Sample_ID')
    pca = pca[pca['Sample_ID'].isin(ids)].sort_values('Sample_ID').set_index('Sample_ID')
    
    # Get target from YAML config
    import yaml
    with open("config.yaml", "r") as f:
        conf_data = yaml.safe_load(f)
    targets = conf_data.get("TARGET_COLUMNS", {})
    target = targets.get(dataset)
    
    if target not in y.columns:
        # Fallback
        num_cols = y.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if "ID" not in c.upper() and "SAMPLE" not in c.upper()]
        target = num_cols[0] if num_cols else y.columns[-1]

    y_target = pd.to_numeric(y[target], errors='coerce')
    mask = y_target.notna()
    
    X_feat = X[mask].copy()
    pca_feat = pca[mask].copy()
    y_final = y_target[mask].values.astype(float)
    
    X_combined = pd.concat([X_feat, pca_feat], axis=1).fillna(0).values.astype(float)
    return X_combined, y_final

def train_mlp(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy().flatten()
    return preds

def run_baseline(dataset, model_name="ridge"):
    cfg = ProjectConfig("config.yaml")
    X, y = load_data(cfg, dataset)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling always for baseline consistency if needed
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    if model_name == "mlp":
        model = build_model("mlp", input_dim=X.shape[1])
        y_pred = train_mlp(model, X_train, y_train, X_test, y_test)
    else:
        model = build_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} could not be built (check if library is installed)")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    metrics = {
        "dataset": dataset,
        "model": model_name,
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred))
    }
    
    out_dir = os.path.join("03_modeling_results", "baselines", dataset)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics
