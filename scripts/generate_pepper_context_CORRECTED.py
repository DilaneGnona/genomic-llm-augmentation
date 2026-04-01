"""
Génération de données context learning pour PEPPER - VERSION CORRIGÉE
SNPs MUST être des entiers: 0, 1, 2
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

BASE_DIR = Path("c:/Users/OMEN/Desktop/experiment_snp")

print("="*80)
print("GÉNÉRATION DE DONNÉES PEPPER CORRIGÉES - SNPs 0,1,2")
print("="*80)

# 1. CHARGEMENT DES DONNÉES RÉELLES PEPPER
print("\n1. Chargement des données réelles Pepper...")
real_data_dir = BASE_DIR / "02_processed_data/pepper"
X_real = pd.read_csv(real_data_dir / "X_aligned.csv")
y_real = pd.read_csv(real_data_dir / "y_aligned.csv")
df_real = X_real.merge(y_real, on='Sample_ID', how='inner')

feature_cols = [c for c in df_real.columns if c.startswith('SNP_')]
print(f"   Données réelles: {len(df_real)} échantillons, {len(feature_cols)} SNPs")

# Vérifier le format
unique_vals = np.unique(df_real[feature_cols].values)
print(f"   Valeurs SNP uniques: {sorted(unique_vals)}")
print(f"   Format valide (0,1,2): {set(unique_vals).issubset({0, 1, 2})}")

# 2. CALCULER LES PROBABILITÉS DES SNPs
print("\n2. Calcul des probabilités de génotypes...")

snp_probs = {}
for col in feature_cols:
    values = df_real[col].values.astype(int)
    counts = np.bincount(values, minlength=3)
    probs = counts / counts.sum() if counts.sum() > 0 else np.array([0.33, 0.33, 0.34])
    snp_probs[col] = probs

print(f"   Probabilités calculées pour {len(snp_probs)} SNPs")

# 3. CALCULER CORRÉLATIONS SNP-YIELD
print("\n3. Analyse des corrélations...")
correlations = []
for col in feature_cols[:100]:  # Limiter pour la vitesse
    corr, _ = pearsonr(df_real[col], df_real['YR_LS'])
    correlations.append(abs(corr))

top_k = 50
top_indices = np.argsort(correlations)[-top_k:]
top_snp_cols = [feature_cols[i] for i in top_indices]

print(f"   Top {top_k} SNPs corrélés sélectionnés")
print(f"   Corrélations: min={min(correlations):.3f}, max={max(correlations):.3f}")

# 4. CONSTRUIRE MODÈLES CONDITIONNELS
print("\n4. Construction des modèles conditionnels P(SNP | Yield)...")

y = df_real['YR_LS'].values
yield_percentiles = np.percentile(y, [0, 20, 40, 60, 80, 100])
yield_classes = np.digitize(y, yield_percentiles[1:-1])

conditional_probs = {}
for snp_col in top_snp_cols:
    snp_values = df_real[snp_col].values.astype(int)
    probs = {}
    for cls in range(5):
        mask = yield_classes == cls
        if mask.sum() > 5:  # Minimum 5 échantillons
            snp_cls = snp_values[mask]
            counts = np.bincount(snp_cls, minlength=3)
            p = counts / counts.sum()
            probs[cls] = p
        else:
            probs[cls] = snp_probs[snp_col]  # Utiliser distribution marginale
    conditional_probs[snp_col] = probs

print(f"   Modèles conditionnels construits")

# 5. GÉNÉRATION DES DONNÉES SYNTHÉTIQUES
print("\n5. Génération des données synthétiques...")

def generate_context_data(n_samples, yield_range=None, seed=42):
    """Génère des échantillons avec corrélation préservée"""
    np.random.seed(seed)
    
    if yield_range:
        y_synth = np.random.uniform(yield_range[0], yield_range[1], n_samples)
    else:
        y_synth = np.random.choice(y, size=n_samples, replace=True)
    
    y_classes = np.digitize(y_synth, yield_percentiles[1:-1])
    y_classes = np.clip(y_classes, 0, 4)
    
    data = []
    for i in range(n_samples):
        sample = {'Sample_ID': f'SYNTH_PEPPER_{i:05d}', 'YR_LS': y_synth[i]}
        cls = y_classes[i]
        
        for snp_col in feature_cols:
            if snp_col in conditional_probs:
                probs = conditional_probs[snp_col][cls]
            else:
                probs = snp_probs[snp_col]
            
            # Générer SNP discret (0, 1, 2)
            snp_val = np.random.choice([0, 1, 2], p=probs)
            sample[snp_col] = int(snp_val)
        
        data.append(sample)
    
    return pd.DataFrame(data)

# Créer dossier de sortie
output_dir = BASE_DIR / "04_augmentation/pepper/context_learning_CORRECTED"
output_dir.mkdir(parents=True, exist_ok=True)

# Générer différents contextes
contexts = {
    'context_A': (y.min(), y.max()),  # Toute la plage
    'context_B': (y.min(), np.percentile(y, 50)),  # Yield faible
    'context_C': (np.percentile(y, 50), y.max()),  # Yield élevé
    'context_D': (np.percentile(y, 25), np.percentile(y, 75)),  # Moyen
}

for ctx_name, y_range in contexts.items():
    print(f"\n   Génération {ctx_name}...")
    
    df_synth = generate_context_data(500, y_range, seed=42)
    
    # Vérification
    snp_values = df_synth[feature_cols].values.flatten()
    unique_synth = np.unique(snp_values)
    is_valid = set(unique_synth).issubset({0, 1, 2})
    
    print(f"     Yield: [{df_synth['YR_LS'].min():.2f}, {df_synth['YR_LS'].max():.2f}]")
    print(f"     SNPs: {sorted(unique_synth)}")
    print(f"     Format valide: {'✓' if is_valid else '✗'}")
    
    # Sauvegarder
    output_file = output_dir / f"synthetic_pepper_{ctx_name}_500samples_CORRECTED.csv"
    df_synth.to_csv(output_file, index=False)
    print(f"     Sauvegardé: {output_file}")

# 6. VÉRIFICATION FINALE
print("\n" + "="*80)
print("6. VÉRIFICATION DE LA CORRÉLATION PRÉSERVÉE")
print("="*80)

df_test = pd.read_csv(output_dir / "synthetic_pepper_context_A_500samples_CORRECTED.csv")

print("\nComparaison corrélations (10 premiers SNPs):")
print(f"{'SNP':<15} {'Réel':>8} {'Synth':>8} {'Diff':>8}")
print("-" * 45)

diffs = []
for col in top_snp_cols[:10]:
    cr, _ = pearsonr(df_real[col], df_real['YR_LS'])
    cs, _ = pearsonr(df_test[col], df_test['YR_LS'])
    diff = abs(cr - cs)
    diffs.append(diff)
    print(f"{col:<15} {cr:>8.3f} {cs:>8.3f} {diff:>8.3f}")

avg_diff = np.mean(diffs)
print(f"\nDifférence moyenne: {avg_diff:.3f}")

if avg_diff < 0.1:
    print("✅ CORRÉLATION BIEN PRÉSERVÉE!")
elif avg_diff < 0.2:
    print("⚠️ Corrélation partiellement préservée")
else:
    print("❌ Corrélation faible")

# 7. TEST RAPIDE
print("\n" + "="*80)
print("7. TEST RAPIDE D'APPRENTISSAGE")
print("="*80)

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

# Modèle simple
class QuickNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
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
y_test_t = torch.FloatTensor(y_test)

# Entraînement rapide
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

# Évaluation
model.eval()
with torch.no_grad():
    pred = model(X_test_t).numpy()
    r2 = r2_score(y_test, pred)

print(f"\nR² sur données synthétiques: {r2:.4f}")

if r2 > 0.3:
    print("✅ Les données permettent l'apprentissage!")
else:
    print("⚠️ R² faible - besoin d'amélioration")

print("\n" + "="*80)
print("GÉNÉRATION TERMINÉE")
print("="*80)
print(f"\nDonnées sauvegardées dans: {output_dir}")
print("\nFichiers générés:")
for ctx in contexts.keys():
    print(f"  - synthetic_pepper_{ctx}_500samples_CORRECTED.csv")
