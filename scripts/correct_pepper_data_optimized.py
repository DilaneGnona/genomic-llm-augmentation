"""
================================================================================
CORRECTION DES DONNÉES PEPPER - VERSION OPTIMISÉE
================================================================================
Objectif: Générer des données synthétiques Pepper avec format correct (0,1,2)
Stratégie: Utiliser seulement les SNPs les plus corrélés pour accélérer
================================================================================
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import time

BASE_DIR = Path("c:/Users/OMEN/Desktop/experiment_snp")

print("="*80)
print("CORRECTION DES DONNÉES PEPPER - OPTIMISÉ")
print("="*80)

start_time = time.time()

# 1. CHARGEMENT DES DONNÉES
print("\n[1/6] Chargement des données réelles Pepper...")
real_data_dir = BASE_DIR / "02_processed_data/pepper"
X_real = pd.read_csv(real_data_dir / "X_aligned.csv")
y_real = pd.read_csv(real_data_dir / "y_aligned.csv")
df_real = X_real.merge(y_real, on='Sample_ID', how='inner')

feature_cols = [c for c in df_real.columns if c.startswith('SNP_')]
print(f"   ✓ {len(df_real)} échantillons, {len(feature_cols)} SNPs")

# 2. SÉLECTION DES SNPs LES PLUS IMPORTANTS
print("\n[2/6] Sélection des SNPs les plus corrélés (TOP 500)...")

correlations = []
for i, col in enumerate(feature_cols):
    if i % 500 == 0:
        print(f"   Progression: {i}/{len(feature_cols)} SNPs analysés...")
    corr, _ = pearsonr(df_real[col], df_real['YR_LS'])
    correlations.append(abs(corr))

correlations = np.array(correlations)
top_k = 500  # Sélectionner les 500 meilleurs SNPs
top_indices = np.argsort(correlations)[-top_k:]
selected_snp_cols = [feature_cols[i] for i in top_indices]

print(f"   ✓ Top {top_k} SNPs sélectionnés")
print(f"   ✓ Corrélations: min={correlations[top_indices].min():.3f}, max={correlations[top_indices].max():.3f}")

# 3. CALCULER LES PROBABILITÉS
print("\n[3/6] Calcul des probabilités conditionnelles...")

y = df_real['YR_LS'].values
yield_percentiles = np.percentile(y, [0, 20, 40, 60, 80, 100])
yield_classes = np.digitize(y, yield_percentiles[1:-1])

# Probabilités marginales
snp_probs = {}
for col in selected_snp_cols:
    values = df_real[col].values.astype(int)
    counts = np.bincount(values, minlength=3)
    probs = counts / counts.sum() if counts.sum() > 0 else np.array([0.33, 0.33, 0.34])
    snp_probs[col] = probs

# Probabilités conditionnelles
conditional_probs = {}
for idx, snp_col in enumerate(selected_snp_cols):
    if idx % 100 == 0:
        print(f"   Progression: {idx}/{len(selected_snp_cols)} SNPs...")
    
    snp_values = df_real[snp_col].values.astype(int)
    probs = {}
    for cls in range(5):
        mask = yield_classes == cls
        if mask.sum() > 3:
            snp_cls = snp_values[mask]
            counts = np.bincount(snp_cls, minlength=3)
            p = counts / counts.sum()
            probs[cls] = p
        else:
            probs[cls] = snp_probs[snp_col]
    conditional_probs[snp_col] = probs

print(f"   ✓ Modèles conditionnels calculés")

# 4. GÉNÉRATION DES DONNÉES
print("\n[4/6] Génération des données synthétiques...")

def generate_samples(n_samples, yield_range=None, seed=42):
    np.random.seed(seed)
    
    if yield_range:
        y_synth = np.random.uniform(yield_range[0], yield_range[1], n_samples)
    else:
        y_synth = np.random.choice(y, size=n_samples, replace=True)
    
    y_classes = np.digitize(y_synth, yield_percentiles[1:-1])
    y_classes = np.clip(y_classes, 0, 4)
    
    data = {'Sample_ID': [f'SYNTH_PEPPER_{i:05d}' for i in range(n_samples)],
            'YR_LS': y_synth}
    
    # Générer les SNPs sélectionnés
    for snp_col in selected_snp_cols:
        snp_values = []
        for i in range(n_samples):
            cls = y_classes[i]
            probs = conditional_probs[snp_col][cls]
            snp_val = np.random.choice([0, 1, 2], p=probs)
            snp_values.append(int(snp_val))
        data[snp_col] = snp_values
    
    return pd.DataFrame(data)

# Créer dossier de sortie
output_dir = BASE_DIR / "04_augmentation/pepper/context_learning_CORRECTED"
output_dir.mkdir(parents=True, exist_ok=True)

# Générer 4 contextes
contexts = {
    'context_A': (y.min(), y.max()),
    'context_B': (y.min(), np.percentile(y, 50)),
    'context_C': (np.percentile(y, 50), y.max()),
    'context_D': (np.percentile(y, 25), np.percentile(y, 75)),
}

generated_files = []
for ctx_name, y_range in contexts.items():
    print(f"\n   Génération {ctx_name}...")
    df_synth = generate_samples(500, y_range, seed=42)
    
    # Vérifier format
    snp_vals = df_synth[selected_snp_cols].values.flatten()
    unique_vals = np.unique(snp_vals)
    is_valid = set(unique_vals).issubset({0, 1, 2})
    
    print(f"     Yield: [{df_synth['YR_LS'].min():.2f}, {df_synth['YR_LS'].max():.2f}]")
    print(f"     SNPs: {sorted(unique_vals)} | Valid: {'✓' if is_valid else '✗'}")
    
    # Sauvegarder
    output_file = output_dir / f"synthetic_pepper_{ctx_name}_500samples_CORRECTED.csv"
    df_synth.to_csv(output_file, index=False)
    generated_files.append(output_file)
    print(f"     ✓ Sauvegardé: {output_file.name}")

# 5. VÉRIFICATION
print("\n[5/6] Vérification de la qualité...")

df_test = pd.read_csv(generated_files[0])

print("\n   Comparaison corrélations (10 premiers SNPs):")
print(f"   {'SNP':<15} {'Réel':>8} {'Synth':>8} {'Diff':>8}")
print("   " + "-"*45)

diffs = []
for col in selected_snp_cols[:10]:
    cr, _ = pearsonr(df_real[col], df_real['YR_LS'])
    cs, _ = pearsonr(df_test[col], df_test['YR_LS'])
    diff = abs(cr - cs)
    diffs.append(diff)
    print(f"   {col:<15} {cr:>8.3f} {cs:>8.3f} {diff:>8.3f}")

avg_diff = np.mean(diffs)
print(f"\n   Différence moyenne: {avg_diff:.3f}")

if avg_diff < 0.1:
    print("   ✅ CORRÉLATION BIEN PRÉSERVÉE!")
elif avg_diff < 0.2:
    print("   ⚠️  Corrélation partiellement préservée")
else:
    print("   ❌ Corrélation faible")

# 6. TEST RAPIDE
print("\n[6/6] Test d'apprentissage rapide...")

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

X_synth = df_test[selected_snp_cols].values
y_synth = df_test['YR_LS'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_synth)

class QuickNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.fc(x).squeeze()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_synth, test_size=0.2, random_state=42)

model = QuickNN(X_scaled.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    pred = model(X_test_t).numpy()
    r2 = r2_score(y_test, pred)

print(f"\n   R² sur données synthétiques: {r2:.4f}")

if r2 > 0.5:
    print("   ✅ EXCELLENT - Les données permettent l'apprentissage!")
elif r2 > 0.3:
    print("   ✅ BON - Apprentissage possible")
else:
    print("   ⚠️  FAIBLE - Besoin d'amélioration")

# RÉSUMÉ
elapsed = time.time() - start_time
print("\n" + "="*80)
print("RÉSUMÉ")
print("="*80)
print(f"\n⏱️  Temps d'exécution: {elapsed:.1f} secondes")
print(f"📊 SNPs utilisés: {len(selected_snp_cols)} / {len(feature_cols)}")
print(f"📁 Fichiers générés: {len(generated_files)}")
print(f"📍 Dossier: {output_dir}")
print(f"\n🎯 R² test rapide: {r2:.4f}")

print("\n" + "="*80)
print("TERMINÉ")
print("="*80)
