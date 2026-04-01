"""
CORRECTION ULTRA-RAPIDE DES DONNÉES PEPPER
Utilise seulement 50 SNPs pour maximiser la vitesse
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import time

start_time = time.time()
BASE_DIR = Path("c:/Users/OMEN/Desktop/experiment_snp")

print("="*80)
print("CORRECTION ULTRA-RAPIDE - 50 SNPs")
print("="*80)

# 1. CHARGEMENT RAPIDE
print("\n[1/3] Chargement des données...")
real_data_dir = BASE_DIR / "02_processed_data/pepper"

# Charger seulement les colonnes nécessaires
X_real = pd.read_csv(real_data_dir / "X_aligned.csv")
y_real = pd.read_csv(real_data_dir / "y_aligned.csv")

# Merge
df_real = pd.merge(X_real, y_real, on='Sample_ID', how='inner')

# Sélectionner seulement 100 SNPs aléatoires pour accélérer
all_snp_cols = [c for c in df_real.columns if c.startswith('SNP_')]
print(f"   Total SNPs: {len(all_snp_cols)}")

# Prendre seulement 100 SNPs (pas 3000!)
sample_snp_cols = all_snp_cols[:100]
print(f"   Utilisés: {len(sample_snp_cols)}")

# 2. CALCUL RAPIDE DES CORRÉLATIONS
print("\n[2/3] Calcul des corrélations...")

correlations = []
for col in sample_snp_cols:
    corr = np.corrcoef(df_real[col], df_real['YR_LS'])[0, 1]
    correlations.append(abs(corr))

# Sélectionner top 50
top_indices = np.argsort(correlations)[-50:]
selected_snp_cols = [sample_snp_cols[i] for i in top_indices]

print(f"   Top 50 SNPs sélectionnés")
print(f"   Meilleure corrélation: {max(correlations):.3f}")

# 3. CALCUL DES PROBABILITÉS CONDITIONNELLES
print("\n[3/3] Calcul des probabilités et génération...")

y = df_real['YR_LS'].values

# Discrétiser le yield en 3 classes simples
y_min, y_max = y.min(), y.max()
yield_classes = np.digitize(y, [y_min + (y_max-y_min)*0.33, y_min + (y_max-y_min)*0.66])

# Calculer probabilités conditionnelles
conditional_probs = {}
for snp_col in selected_snp_cols:
    snp_values = df_real[snp_col].values.astype(int)
    probs = {}
    for cls in range(3):
        mask = yield_classes == cls
        if mask.sum() > 0:
            counts = np.bincount(snp_values[mask], minlength=3)
            probs[cls] = counts / counts.sum()
        else:
            probs[cls] = np.array([0.33, 0.33, 0.34])
    conditional_probs[snp_col] = probs

# GÉNÉRATION
output_dir = BASE_DIR / "04_augmentation/pepper/context_learning_CORRECTED"
output_dir.mkdir(parents=True, exist_ok=True)

# Générer 3 contextes simples
contexts = {
    'context_A': (y_min, y_max),
    'context_B': (y_min, (y_min+y_max)/2),
    'context_C': ((y_min+y_max)/2, y_max),
}

for ctx_name, (y_low, y_high) in contexts.items():
    print(f"\n   Génération {ctx_name}...")
    
    # Générer yields
    np.random.seed(42)
    y_synth = np.random.uniform(y_low, y_high, 500)
    y_synth_classes = np.digitize(y_synth, [y_min + (y_max-y_min)*0.33, y_min + (y_max-y_min)*0.66])
    
    # Créer DataFrame
    data = {
        'Sample_ID': [f'SYNTH_{ctx_name}_{i:04d}' for i in range(500)],
        'YR_LS': y_synth
    }
    
    # Générer SNPs
    for snp_col in selected_snp_cols:
        snp_list = []
        for i in range(500):
            cls = min(y_synth_classes[i], 2)
            probs = conditional_probs[snp_col][cls]
            snp_val = np.random.choice([0, 1, 2], p=probs)
            snp_list.append(int(snp_val))
        data[snp_col] = snp_list
    
    df_synth = pd.DataFrame(data)
    
    # Vérifier
    unique_vals = np.unique(df_synth[selected_snp_cols].values)
    is_valid = set(unique_vals).issubset({0, 1, 2})
    
    print(f"     Yield: [{y_synth.min():.2f}, {y_synth.max():.2f}]")
    print(f"     SNPs: {sorted(unique_vals)} | Valid: {'✓' if is_valid else '✗'}")
    
    # Sauvegarder
    output_file = output_dir / f"synthetic_pepper_{ctx_name}_500samples_CORRECTED.csv"
    df_synth.to_csv(output_file, index=False)
    print(f"     ✓ Sauvegardé: {output_file.name}")

elapsed = time.time() - start_time
print("\n" + "="*80)
print(f"TERMINÉ en {elapsed:.1f} secondes")
print("="*80)
print(f"\n📁 Dossier: {output_dir}")
print(f"📊 {len(contexts)} fichiers générés")
