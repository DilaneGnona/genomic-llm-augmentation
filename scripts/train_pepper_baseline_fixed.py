import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import json

def train_baseline_fixed():
    print("--- TRAINING PEPPER BASELINE (FIXED FEATURE SET) ---")
    
    # 1. Get SNP list from context file (The 'correct format' features)
    context_file = "04_augmentation/pepper/context_learning/contexts/pepper_context_stats.csv"
    ctx_df = pd.read_csv(context_file)
    # Exclude Sample_ID and target
    exclude = ['Sample_ID', 'Yield_BV', 'YR_LS', 'Yield']
    feature_cols = [c for c in ctx_df.columns if c not in exclude]
    print(f"  Using {len(feature_cols)} features from context learning set.")

    # 2. Load Real Data
    print("  Loading real data...")
    # Peak to check available columns in X.csv
    x_sample = pd.read_csv("02_processed_data/pepper/X.csv", nrows=1)
    available_features = [c for c in feature_cols if c in x_sample.columns]
    print(f"  {len(available_features)} / {len(feature_cols)} features found in X.csv")
    
    X = pd.read_csv("02_processed_data/pepper/X.csv", usecols=['Sample_ID'] + available_features)
    y = pd.read_csv("02_processed_data/pepper/y.csv")
    
    df = X.merge(y, on="Sample_ID")
    target_col = "Yield_BV"
    
    X_feat = df[available_features].values
    y_feat = df[target_col].values
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X_feat, y_feat, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    results = {}
    
    # XGBoost
    print("  Training XGBoost...")
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    results["xgboost"] = float(r2_score(y_test, y_pred_xgb))
    
    # Ridge
    print("  Training Ridge...")
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    y_pred_r = ridge.predict(X_test)
    results["ridge"] = float(r2_score(y_test, y_pred_r))
    
    # MLP
    print("  Training MLP...")
    mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=500, early_stopping=True)
    mlp.fit(X_train, y_train)
    y_pred_m = mlp.predict(X_test)
    results["mlp_sklearn"] = float(r2_score(y_test, y_pred_m))
    
    print("\nBASELINE RESULTS (on 121 SNPs):")
    for m, score in results.items():
        print(f"  {m}: R2 = {score:.4f}")
        
    out_dir = "03_modeling_results/baselines/pepper"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "baseline_fixed_features.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    train_baseline_fixed()
