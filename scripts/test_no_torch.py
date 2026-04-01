import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json

print("DEBUG: Loading data")
df = pd.read_csv("02_processed_data/ipk_out_raw_augmented/augmented_combined.csv")
X = df.drop(columns=['Sample_ID', 'YR_LS', 'Source'], errors='ignore')
y = df['YR_LS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("DEBUG: Training Ridge")
model = Ridge()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"DEBUG: Ridge R2: {score}")

with open("03_modeling_results/test_no_torch.json", "w") as f:
    json.dump({"ridge": score}, f)
