import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json

print("DEBUG: Loading data")
df = pd.read_csv("02_processed_data/ipk_out_raw_augmented/augmented_combined.csv")
X = df.drop(columns=['Sample_ID', 'YR_LS', 'Source'], errors='ignore')
y = df['YR_LS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("DEBUG: Training XGBoost")
model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"DEBUG: XGBoost R2: {score}")
