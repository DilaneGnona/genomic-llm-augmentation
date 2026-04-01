"""
================================================================================
GÉNÉRATION DE DONNÉES CONTEXT LEARNING AVEC CORRÉLATION PRÉSERVÉE
================================================================================
Méthode: Génération conditionnelle basée sur le yield cible
- Les SNPs sont générés en fonction du yield souhaité
- Préservation de la structure de corrélation réelle
- Utilisation de modèles de régression pour chaque SNP
================================================================================
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path("c:/Users/OMEN/Desktop/experiment_snp")

print("="*80)
print("GÉNÉRATION DE DONNÉES AVEC CORRÉLATION PRÉSERVÉE")
print("="*80)

# 1. CHARGEMENT DES DONNÉES RÉELLES
print("\n1. Chargement des données réelles Pepper...")
real_data_dir = BASE_DIR / "02_processed_data/pepper"
X_real = pd.read_csv(real_data_dir / "X_aligned.csv")
y_real = pd.read_csv(real_data_dir / "y_aligned.csv")
df_real = X_real.merge(y_real, on='Sample_ID', how='inner')

feature_cols = [c for c in df_real.columns if c.startswith('SNP_')]
X = df_real[feature_cols].values
y = df_real['YR_LS'].values

print(f"   Données réelles: {len(X)} échantillons, {len(feature_cols)} SNPs")

# 2. CALCUL DES CORRÉLATIONS ET SÉLECTION DES SNPs IMPORTANTS
print("\n2. Analyse des corrélations SNP-Yield...")
from scipy.stats import pearsonr

correlations = []
for col in feature_cols:
    corr, _ = pearsonr(df_real[col], df_real['YR_LS'])
    correlations.append(abs(corr))

correlations = np.array(correlations)

# Sélectionner les SNPs les plus corrélés (top 20%)
top_k = max(50, int(len(feature_cols) * 0.2))
top_snp_indices = np.argsort(correlations)[-top_k:]
top_snp_cols = [feature_cols[i] for i in top_snp_indices]

print(f"   SNPs sélectionnés (top {top_k}): corrélations de {correlations[top_snp_indices].min():.3f} à {correlations[top_snp_indices].max():.3f}")

# 3. CONSTRUCTION DES MODÈLES POUR CHAQUE SNP
print("\n3. Construction des modèles de prédiction SNP...")

# Discrétiser le yield en classes pour la classification
n_classes = 5
yield_percentiles = np.percentile(y, np.linspace(0, 100, n_classes + 1))
yield_classes = np.digitize(y, yield_percentiles[1:-1])

snp_models = {}
for snp_col in top_snp_cols:
    # Modèle simple: prédire le SNP en fonction de la classe de yield
    snp_values = df_real[snp_col].values.astype(int)
    
    # Calculer les probabilités conditionnelles P(SNP | yield_class)
    probs = {}
    for cls in range(n_classes):
        mask = yield_classes == cls
        if mask.sum() > 0:
            snp_cls = snp_values[mask]
            # Probabilités de chaque génotype
            p0 = (snp_cls == 0).sum() / len(snp_cls) if len(snp_cls) > 0 else 0.33
            p1 = (snp_cls == 1).sum() / len(snp_cls) if len(snp_cls) > 0 else 0.33
            p2 = (snp_cls == 2).sum() / len(snp_cls) if len(snp_cls) > 0 else 0.33
            
            # Normaliser
            total = p0 + p1 + p2
            probs[cls] = [p0/total, p1/total, p2/total]
        else:
            probs[cls] = [0.33, 0.33, 0.34]
    
    snp_models[snp_col] = probs

print(f"   Modèles construits pour {len(snp_models)} SNPs")

# 4. GÉNÉRATION DES DONNÉES SYNTHÉTIQUES
print("\n4. Génération des données synthétiques...")

def generate_synthetic_samples(n_samples, target_yield_range=None):
    """
    Génère des échantillons synthétiques avec corrélation préservée
    """
    synthetic_data = []
    
    # Générer des yields dans la plage cible ou aléatoire
    if target_yield_range:
        y_synth = np.random.uniform(target_yield_range[0], target_yield_range[1], n_samples)
    else:
        y_synth = np.random.choice(y, size=n_samples, replace=True)
    
    # Discrétiser les yields synthétiques
    y_synth_classes = np.digitize(y_synth, yield_percentiles[1:-1])
    y_synth_classes = np.clip(y_synth_classes, 0, n_classes - 1)
    
    for i in range(n_samples):
        sample = {'Sample_ID': f'SYNTH_CORR_{i:05d}', 'YR_LS': y_synth[i]}
        
        # Générer chaque SNP en fonction de la classe de yield
        cls = y_synth_classes[i]
        
        for snp_col in feature_cols:
            if snp_col in snp_models:
                # Utiliser le modèle conditionnel
                probs = snp_models[snp_col][cls]
                snp_val = np.random.choice([0, 1, 2], p=probs)
            else:
                # Pour les autres SNPs, utiliser la distribution marginale
                snp_vals = df_real[snp_col].values
                p0 = (snp_vals == 0).mean()
                p1 = (snp_vals == 1).mean()
                p2 = 1 - p0 - p1
                snp_val = np.random.choice([0, 1, 2], p=[p0, p1, p2])
            
            sample[snp_col] = snp_val
        
        synthetic_data.append(sample)
    
    return pd.DataFrame(synthetic_data)

# Générer plusieurs contextes avec différentes plages de yield
contexts = {
    'context_A': (y.min(), y.max()),  # Toute la plage
    'context_B': (y.min(), y.mean()),  # Yield faible
    'context_C': (y.mean(), y.max()),  # Yield élevé
    'context_D': (y.min() + (y.max() - y.min()) * 0.25, y.min() + (y.max() - y.min()) * 0.75),  # Moyen
    'context_E': (y.min(), y.min() + (y.max() - y.min()) * 0.5),  # Bas-moyen
}

output_dir = BASE_DIR / "04_augmentation/pepper/context_learning_CORR_PRESERVED"
output_dir.mkdir(parents=True, exist_ok=True)

for ctx_name, yield_range in contexts.items():
    print(f"\n   Génération {ctx_name} (yield: {yield_range[0]:.2f} - {yield_range[1]:.2f})...")
    
    df_synth = generate_synthetic_samples(500, yield_range)
    
    # Vérification
    print(f"     Yield range: [{df_synth['YR_LS'].min():.2f}, {df_synth['YR_LS'].max():.2f}]")
    print(f"     Samples: {len(df_synth)}")
    
    # Sauvegarder
    output_file = output_dir / f"synthetic_corr_preserved_{ctx_name}_500samples.csv"
    df_synth.to_csv(output_file, index=False)
    print(f"     Sauvegardé: {output_file}")

# 5. VÉRIFICATION DE LA CORRÉLATION
print("\n5. Vérification de la corrélation préservée...")

df_test = pd.read_csv(output_dir / "synthetic_corr_preserved_context_A_500samples.csv")

corr_preserved = []
for col in top_snp_cols[:10]:  # Vérifier les 10 premiers SNPs corrélés
    corr_real, _ = pearsonr(df_real[col], df_real['YR_LS'])
    corr_synth, _ = pearsonr(df_test[col], df_test['YR_LS'])
    corr_preserved.append({
        'SNP': col,
        'corr_real': corr_real,
        'corr_synth': corr_synth,
        'diff': abs(corr_real - corr_synth)
    })

print("\n   Comparaison corrélations (réel vs synthétique):")
print(f"   {'SNP':<15} {'Réel':>8} {'Synth':>8} {'Diff':>8}")
print("   " + "-"*45)
for item in corr_preserved:
    print(f"   {item['SNP']:<15} {item['corr_real']:>8.3f} {item['corr_synth']:>8.3f} {item['diff']:>8.3f}")

avg_diff = np.mean([c['diff'] for c in corr_preserved])
print(f"\n   Différence moyenne: {avg_diff:.3f}")

if avg_diff < 0.1:
    print("   ✅ CORRÉLATION BIEN PRÉSERVÉE!")
else:
    print("   ⚠️  Corrélation partiellement préservée")

# 6. TEST RAPIDE
print("\n6. Test rapide d'entraînement...")

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Préparer données
X_synth = df_test[feature_cols].values
y_synth = df_test['YR_LS'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_synth)

# Simple modèle
class QuickTest(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.fc(x).squeeze()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_synth, test_size=0.2)

model = QuickTest(X_scaled.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

for epoch in range(50):
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

print(f"   R² sur données synthétiques: {r2:.4f}")
if r2 > 0.3:
    print("   ✅ Les données permettent l'apprentissage!")
else:
    print("   ⚠️  R² faible - besoin d'amélioration")

print("\n" + "="*80)
print("GÉNÉRATION TERMINÉE")
print("="*80)
print(f"\nDonnées sauvegardées dans: {output_dir}")
print("\nContextes générés:")
for ctx_name in contexts.keys():
    print(f"  - {ctx_name}: 500 échantillons")
