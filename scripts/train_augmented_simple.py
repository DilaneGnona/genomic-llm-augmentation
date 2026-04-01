import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
import json

# Manual Model Definitions to avoid import issues
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.layers(x)

def run_train():
    print("Loading data...")
    df = pd.read_csv("02_processed_data/ipk_out_raw_augmented/augmented_combined.csv")
    X = df.drop(columns=['Sample_ID', 'YR_LS', 'Source'], errors='ignore').values
    y = df['YR_LS'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    results = {}
    
    # Ridge
    print("Training Ridge...")
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    results["ridge"] = {"r2": float(r2_score(y_test, y_pred))}
    print(f"  Ridge R2: {results['ridge']['r2']:.4f}")
    
    # MLP
    print("Training MLP...")
    device = torch.device("cpu")
    model = SimpleMLP(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train).view(-1, 1)
    
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        y_pred_mlp = model(torch.FloatTensor(X_test)).numpy().flatten()
    results["mlp"] = {"r2": float(r2_score(y_test, y_pred_mlp))}
    print(f"  MLP R2: {results['mlp']['r2']:.4f}")
    
    with open("03_modeling_results/augmented_ipk_simple_summary.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_train()
