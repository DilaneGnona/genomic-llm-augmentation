"""
================================================================================
ANALYSE DE LA QUALITÉ DES DONNÉES PEPPER
================================================================================
Objectifs:
1. Vérifier la corrélation SNP-Yield dans les données réelles
2. Analyser la variance des SNPs
3. Identifier les SNPs les plus corrélés avec le yield
4. Tester l'entraînement sans augmentation
================================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import json

BASE_DIR = Path("c:/Users/OMEN/Desktop/experiment_snp")

print("="*80)
print("ANALYSE DE LA QUALITÉ DES DONNÉES PEPPER")
print("="*80)

# 1. CHARGEMENT DES DONNÉES RÉELLES
print("\n" + "="*80)
print("1. CHARGEMENT DES DONNÉES RÉELLES PEPPER")
print("="*80)

real_data_dir = BASE_DIR / "02_processed_data/pepper"
X_real = pd.read_csv(real_data_dir / "X_aligned.csv")
y_real = pd.read_csv(real_data_dir / "y_aligned.csv")
df_real = X_real.merge(y_real, on='Sample_ID', how='inner')

feature_cols = [c for c in df_real.columns if c.startswith('SNP_')]
X = df_real[feature_cols].values
y = df_real['YR_LS'].values

print(f"Données réelles: {len(X)} échantillons, {len(feature_cols)} SNPs")
print(f"Yield range: [{y.min():.2f}, {y.max():.2f}]")
print(f"Yield mean: {y.mean():.2f}, std: {y.std():.2f}")

# 2. ANALYSE DE CORRÉLATION SNP-YIELD
print("\n" + "="*80)
print("2. ANALYSE DE CORRÉLATION SNP-YIELD (CRITIQUE)")
print("="*80)

correlations = []
p_values = []

for col in feature_cols:
    corr, pval = pearsonr(df_real[col], df_real['YR_LS'])
    correlations.append(corr)
    p_values.append(pval)

corr_df = pd.DataFrame({
    'SNP': feature_cols,
    'correlation': correlations,
    'p_value': p_values,
    'abs_correlation': np.abs(correlations)
}).sort_values('abs_correlation', ascending=False)

print(f"\nStatistiques des corrélations:")
print(f"  Mean |corr|: {np.mean(np.abs(correlations)):.4f}")
print(f"  Max |corr|:  {np.max(np.abs(correlations)):.4f}")
print(f"  Min |corr|:  {np.min(np.abs(correlations)):.4f}")
print(f"  SNPs avec |corr| > 0.1: {np.sum(np.abs(correlations) > 0.1)}")
print(f"  SNPs avec |corr| > 0.2: {np.sum(np.abs(correlations) > 0.2)}")
print(f"  SNPs avec |corr| > 0.3: {np.sum(np.abs(correlations) > 0.3)}")
print(f"  SNPs avec |corr| > 0.5: {np.sum(np.abs(correlations) > 0.5)}")

print(f"\nTop 10 SNPs les plus corrélés:")
print(corr_df.head(10)[['SNP', 'correlation', 'abs_correlation']].to_string(index=False))

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribution des corrélations
axes[0, 0].hist(correlations, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].axvline(x=0, color='r', linestyle='--', label='corr = 0')
axes[0, 0].axvline(x=0.1, color='g', linestyle='--', label='|corr| = 0.1')
axes[0, 0].axvline(x=-0.1, color='g', linestyle='--')
axes[0, 0].set_xlabel('Correlation')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution des corrélations SNP-Yield (Pepper)')
axes[0, 0].legend()

# Manhattan plot
axes[0, 1].scatter(range(len(correlations)), np.abs(correlations), alpha=0.6, s=10)
axes[0, 1].axhline(y=0.1, color='r', linestyle='--', label='|corr| = 0.1')
axes[0, 1].axhline(y=0.2, color='orange', linestyle='--', label='|corr| = 0.2')
axes[0, 1].axhline(y=0.3, color='green', linestyle='--', label='|corr| = 0.3')
axes[0, 1].set_xlabel('SNP Index')
axes[0, 1].set_ylabel('|Correlation|')
axes[0, 1].set_title('Corrélation absolue par SNP (Pepper)')
axes[0, 1].legend()

# Distribution des valeurs SNP
snp_values = df_real[feature_cols].values.flatten()
axes[1, 0].hist(snp_values, bins=[-0.5, 0.5, 1.5, 2.5], edgecolor='black', alpha=0.7, color='lightgreen')
axes[1, 0].set_xlabel('SNP Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution des valeurs SNP (Pepper)')
axes[1, 0].set_xticks([0, 1, 2])

# Distribution du yield
axes[1, 1].hist(y, bins=30, edgecolor='black', alpha=0.7, color='coral')
axes[1, 1].set_xlabel('Yield (YR_LS)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution du Yield (Pepper)')

plt.tight_layout()
plt.savefig(BASE_DIR / '03_modeling_results/pepper_data_quality_analysis.png', dpi=150)
print(f"\nGraphique sauvegardé: {BASE_DIR / '03_modeling_results/pepper_data_quality_analysis.png'}")

# 3. ANALYSE DE VARIANCE
print("\n" + "="*80)
print("3. ANALYSE DE VARIANCE DES SNPs")
print("="*80)

snp_vars = df_real[feature_cols].var()
print(f"\nStatistiques de variance:")
print(f"  Mean variance: {snp_vars.mean():.4f}")
print(f"  Max variance:  {snp_vars.max():.4f}")
print(f"  Min variance:  {snp_vars.min():.4f}")
print(f"  SNPs avec variance > 0.1: {np.sum(snp_vars > 0.1)}")
print(f"  SNPs avec variance > 0.2: {np.sum(snp_vars > 0.2)}")
print(f"  SNPs avec variance = 0 (constants): {np.sum(snp_vars == 0)}")

# 4. TEST SANS AUGMENTATION
print("\n" + "="*80)
print("4. TEST D'ENTRAÎNEMENT SANS AUGMENTATION (529 échantillons)")
print("="*80)

# Simple LSTM pour test rapide
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :]).squeeze()

def create_sequences(X, y, seq_length=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length-1])
    return np.array(X_seq), np.array(y_seq)

# Préparation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Créer séquences
SEQ_LENGTH = 10
X_seq, y_seq = create_sequences(X_scaled, y, SEQ_LENGTH)

print(f"Séquences créées: {len(X_seq)}")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Convertir en tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

# Modèle
model = SimpleLSTM(X_train.shape[2])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Entraînement
print(f"\nEntraînement LSTM simple (100 epochs):")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_t).numpy()
            test_pred = model(X_test_t).numpy()
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
        print(f"  Epoch {epoch+1}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}")

# Résultat final
model.eval()
with torch.no_grad():
    final_pred = model(X_test_t).numpy()
    final_r2 = r2_score(y_test, final_pred)
    final_rmse = np.sqrt(mean_squared_error(y_test, final_pred))

print(f"\nRésultat final (sans augmentation):")
print(f"  R² = {final_r2:.4f}")
print(f"  RMSE = {final_rmse:.4f}")

# 5. RÉSUMÉ ET CONCLUSIONS
print("\n" + "="*80)
print("5. RÉSUMÉ ET CONCLUSIONS")
print("="*80)

print(f"""
RÉSULTATS DE L'ANALYSE:

1. CORRÉLATION SNP-YIELD:
   - Mean |corr|: {np.mean(np.abs(correlations)):.4f}
   - SNPs significatifs (|corr| > 0.1): {np.sum(np.abs(correlations) > 0.1)}
   - Interprétation: {'✅ CORRÉLATION PRÉSENTE' if np.sum(np.abs(correlations) > 0.1) > 10 else '❌ CORRÉLATION FAIBLE'}

2. VARIANCE DES SNPs:
   - SNPs constants (var=0): {np.sum(snp_vars == 0)}
   - SNPs avec bonne variance (>0.1): {np.sum(snp_vars > 0.1)}
   - Interprétation: {'✅ VARIANCE SUFFISANTE' if np.sum(snp_vars == 0) < 10 else '❌ TROP DE SNPs CONSTANTS'}

3. TEST SANS AUGMENTATION:
   - R² final: {final_r2:.4f}
   - Interprétation: {'✅ MODÈLE APPREND' if final_r2 > 0.5 else '❌ MODÈLE N APPREND PAS'}

CONCLUSION:
""")

if final_r2 > 0.5:
    print("""
✅ Les données Pepper réelles sont de BONNE QUALITÉ
✅ Le problème vient de la GÉNÉRATION des données synthétiques
✅ Il faut recréer les données context learning en préservant la corrélation
    """)
else:
    print("""
❌ Les données Pepper réelles ont un PROBLÈME
❌ Même sans augmentation, le modèle n'apprend pas
❌ Il faut vérifier le prétraitement des données
    """)

print("="*80)
