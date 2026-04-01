print("SCRIPT STARTING")
import os
import sys
print("IMPORTS DONE")
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json

# Add project root to path
sys.path.append(os.getcwd())
from src.models import build_model

class AugmentedTrainer:
    def __init__(self, dataset_path, target_col="YR_LS"):
        self.df = pd.read_csv(dataset_path)
        self.target_col = target_col
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def prepare_data(self):
        X = self.df.drop(columns=['Sample_ID', self.target_col, 'Source'], errors='ignore')
        y = self.df[self.target_col].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test

    def train_pytorch_model(self, model_name, X_train, y_train, X_test, y_test, epochs=1, batch_size=128):
        input_dim = X_train.shape[1]
        print(f"DEBUG: Building {model_name} with input_dim={input_dim}", flush=True)
        model = build_model(model_name, input_dim=input_dim).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        print(f"DEBUG: Starting training loop for {model_name}", flush=True)
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"DEBUG:  Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}", flush=True)
        
        print(f"DEBUG: Evaluating {model_name}", flush=True)
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(self.device)
            preds = model(X_test_t).cpu().numpy().flatten()
            
        return preds

    def train_sklearn_model(self, model_name, X_train, y_train, X_test, y_test):
        if model_name == "ridge":
            from sklearn.linear_model import Ridge
            model = Ridge()
        elif model_name == "xgboost":
            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8)
        elif model_name == "lightgbm":
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(n_estimators=500, learning_rate=0.05, verbosity=-1)
        elif model_name == "mlp_sklearn":
            from sklearn.neural_network import MLPRegressor
            model = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=500, early_stopping=True)
        else:
            return None
            
        model.fit(X_train, y_train)
        return model.predict(X_test)

    def run_benchmark(self, models=["ridge", "xgboost"]):
        print(f"DEBUG: Starting benchmark on {len(self.df)} samples", flush=True)
        X_train, X_test, y_train, y_test = self.prepare_data()
        print(f"DEBUG: Data split done: {X_train.shape[0]} train, {X_test.shape[0]} test", flush=True)
        results = {}
        
        pytorch_models = ["mlp", "cnn", "lstm", "transformer", "hybrid"]
        
        for m in models:
            print(f"DEBUG: Training {m}...", flush=True)
            try:
                if m in pytorch_models:
                    # Train for 50 epochs for DL models
                    y_pred = self.train_pytorch_model(m, X_train, y_train, X_test, y_test, epochs=50)
                else:
                    y_pred = self.train_sklearn_model(m, X_train, y_train, X_test, y_test)
                
                if y_pred is None: continue
                
                metrics = {
                    "r2": float(r2_score(y_test, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "mae": float(mean_absolute_error(y_test, y_pred))
                }
                results[m] = metrics
                print(f"DEBUG:  {m} Result: R2={metrics['r2']:.4f}", flush=True)
            except Exception as e:
                print(f"DEBUG:  Error training {m}: {e}", flush=True)
                
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ipk_out_raw")
    parser.add_argument("--target", default="YR_LS")
    args = parser.parse_args()
    
    data_path = f"02_processed_data/{args.dataset}_augmented/augmented_combined.csv"
    if not os.path.exists(data_path):
        print(f"Error: Augmented data not found at {data_path}. Run consolidation script first.")
        sys.exit(1)
        
    trainer = AugmentedTrainer(data_path, target_col=args.target)
    results = trainer.run_benchmark(models=["ridge", "xgboost", "lightgbm", "mlp", "cnn", "lstm", "transformer", "hybrid"])
    
    out_path = f"03_modeling_results/augmented_{args.dataset}_summary.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark completed. Results saved to {out_path}")
