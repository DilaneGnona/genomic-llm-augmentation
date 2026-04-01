import pandas as pd
import os
import sys

print("DEBUG: Script started")
data_path = "02_processed_data/ipk_out_raw_augmented/augmented_combined.csv"
if not os.path.exists(data_path):
    print(f"DEBUG: File not found {data_path}")
    sys.exit(1)

df = pd.read_csv(data_path)
print(f"DEBUG: Data loaded, shape {df.shape}")

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Sample_ID', 'YR_LS', 'Source'], errors='ignore')
y = df['YR_LS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("DEBUG: Training Ridge")
model = Ridge()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"DEBUG: Ridge score: {score}")
