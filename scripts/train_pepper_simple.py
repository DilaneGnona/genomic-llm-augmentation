import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json
import os

print("DEBUG: Loading pepper augmented data")
data_path = "02_processed_data/pepper_augmented/augmented_combined.csv"
df = pd.read_csv(data_path)
print(f"DEBUG: Data loaded, shape {df.shape}")

X = df.drop(columns=['Sample_ID', 'Yield_BV', 'Source'], errors='ignore')
y = df['Yield_BV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("DEBUG: Training XGBoost")
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"DEBUG: XGBoost R2 for Pepper Augmented: {score:.4f}")

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

results = {"xgboost": score}

print("DEBUG: Training Ridge")
ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred_r = ridge.predict(X_test)
results["ridge"] = r2_score(y_test, y_pred_r)
print(f"DEBUG: Ridge R2: {results['ridge']:.4f}")

print("DEBUG: Training MLP")
mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=500, early_stopping=True)
mlp.fit(X_train, y_train)
y_pred_m = mlp.predict(X_test)
results["mlp_sklearn"] = r2_score(y_test, y_pred_m)
print(f"DEBUG: MLP R2: {results['mlp_sklearn']:.4f}")

out_path = "03_modeling_results/augmented_pepper_summary.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
