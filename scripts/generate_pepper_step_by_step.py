"""
Génération étape par étape avec sauvegarde
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

BASE_DIR = Path("c:/Users/OMEN/Desktop/experiment_snp")
CACHE_FILE = BASE_DIR / "04_augmentation/pepper/generation_cache.pkl"
OUTPUT_DIR = BASE_DIR / "04_augmentation/pepper/context_learning_CORRECTED"

print("="*80)
print("GÉNÉRATION ÉTAPE PAR ÉTAPE")
print("="*80)

# ÉTAPE 1: Chargement
print("\n[ÉTAPE 1/4] Chargement des données...")
real_data_dir = BASE_DIR / "02_processed_data/pepper"
X_real = pd.read_csv(real_data_dir / "X_aligned.csv")
y_real = pd.read_csv(real_data_dir / "y_aligned.csv")
df_real = pd.merge(X_real, y_real, on='Sample_ID', how='inner')

feature_cols = [c for c in df_real.columns if c.startswith('SNP_')]
print(f"   ✓ {len(df_real)} échantillons, {len(feature_cols)} SNPs")

# Prendre seulement 50 SNPs pour la vitesse
sample_cols = feature_cols[:50]
print(f"   ✓ Utilisés: {len(sample_cols)} SNPs")

# ÉTAPE 2: Calcul des corrélations
print("\n[ÉTAPE 2/4] Calcul des corrélations...")
correlations = []
for col in sample_cols:
    corr = np.corrcoef(df_real[col], df_real['YR_LS'])[0, 1]
    correlations.append(abs(corr))

top_30_indices = np.argsort(correlations)[-30:]
selected_cols = [sample_cols[i] for i in top_30_indices]
print(f"   ✓ Top 30 SNPs sélectionnés")

# ÉTAPE 3: Calcul des probabilités
print("\n[ÉTAPE 3/4] Calcul des probabilités conditionnelles...")

y = df_real['YR_LS'].values
y_min, y_max = y.min(), y.max()

# 3 classes de yield
yield_classes = np.digitize(y, [y_min + (y_max-y_min)*0.33, y_min + (y_max-y_min)*0.66])

conditional_probs = {}
for snp_col in selected_cols:
    snp_values = df_real[snp_col].values.astype(int)
    probs = {}
    for cls in range(3):
        mask = yield_classes == cls
        if mask.sum() > 0:
            counts = np.bincount(snp_values[mask], minlength=3)
            total = counts.sum()
            if total > 0:
                probs[cls] = counts / total
            else:
                probs[cls] = np.array([0.33, 0.33, 0.34])
        else:
            probs[cls] = np.array([0.33, 0.33, 0.34])
    conditional_probs[snp_col] = probs

print(f"   ✓ Probabilités calculées pour {len(conditional_probs)} SNPs")

# Sauvegarder le cache
with open(CACHE_FILE, 'wb') as f:
    pickle.dump({
        'selected_cols': selected_cols,
        'conditional_probs': conditional_probs,
        'y_min': y_min,
        'y_max': y_max
    }, f)
print(f"   ✓ Cache sauvegardé: {CACHE_FILE}")

# ÉTAPE 4: Génération
print("\n[ÉTAPE 4/4] Génération des données...")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Contexte A uniquement pour commencer
print("\n   Génération Context A...")
np.random.seed(42)
y_synth = np.random.uniform(y_min, y_max, 500)
y_synth_classes = np.digitize(y_synth, [y_min + (y_max-y_min)*0.33, y_min + (y_max-y_min)*0.66])

# Créer données
print("   Création du DataFrame...")
data = {
    'Sample_ID': [f'SYNTH_A_{i:04d}' for i in range(500)],
    'YR_LS': y_synth
}

print("   Génération des SNPs...")
for idx, snp_col in enumerate(selected_cols):
    if idx % 10 == 0:
        print(f"      Progression: {idx}/{len(selected_cols)} SNPs...")
    
    snp_values = []
    for i in range(500):
        cls = min(y_synth_classes[i], 2)
        probs = conditional_probs[snp_col][cls]
        # S'assurer que les probas somment à 1
        probs = probs / probs.sum()
        snp_val = np.random.choice([0, 1, 2], p=probs)
        snp_values.append(int(snp_val))
    data[snp_col] = snp_values

print("   Sauvegarde...")
df_synth = pd.DataFrame(data)

# Vérifier
unique_vals = np.unique(df_synth[selected_cols].values)
print(f"   Valeurs SNP: {sorted(unique_vals)}")

# Sauvegarder
output_file = OUTPUT_DIR / "synthetic_pepper_context_A_500samples_CORRECTED.csv"
df_synth.to_csv(output_file, index=False)
print(f"   ✓ Sauvegardé: {output_file}")
print(f"   ✓ Taille: {len(df_synth)} échantillons, {len(df_synth.columns)} colonnes")

print("\n" + "="*80)
print("TERMINÉ AVEC SUCCÈS!")
print("="*80)
print(f"\n📁 Fichier: {output_file}")
print(f"📊 Format SNP: {sorted(unique_vals)} (valide: {set(unique_vals).issubset({0,1,2})})")
