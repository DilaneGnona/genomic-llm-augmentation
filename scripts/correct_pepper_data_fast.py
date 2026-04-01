"""
CORRECTION RAPIDE DES DONNÉES PEPPER - 100 SNPs uniquement
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

BASE_DIR = Path("c:/Users/OMEN/Desktop/experiment_snp")

print("="*80)
print("CORRECTION RAPIDE - 100 SNPs")
print("="*80)

# 1. CHARGEMENT
print("\n[1/4] Chargement...")
real_data_dir = BASE_DIR / "02_processed_data/pepper"
X_real = pd.read_csv(real_data_dir / "X_aligned.csv")
y_real = pd.read_csv(real_data_dir / "y_aligned.csv")
df_real = X_real.merge(y_real, on='Sample_ID', how='inner')

feature_cols = [c for c in df_real.columns if c.startswith('SNP_')]
print(f"   ✓ {len(df_real)} échantillons, {len(feature_cols)} SNPs")

# 2. SÉLECTION RAPIDE (100 premiers SNPs)
print("\n[2/4] Sélection des 100 SNPs les plus corrélés...")

correlations = []
for col in feature_cols[:500]:  # Limiter à 500 pour la vitesse
    corr, _ = pearsonr(df_real[col], df_real['YR_LS'])
    correlations.append((col, abs(corr)))

correlations.sort(key=lambda x: x[1], reverse=True)
selected_snp_cols = [c[0] for c in correlations[:100]]

print(f"   ✓ Top 100 SNPs sélectionnés")

# 3. CALCULER PROBABILITÉS
print("\n[3/4] Calcul des probabilités...")

y = df_real['YR_LS'].values
yield_percentiles = np.percentile(y, [0, 20, 40, 60, 80, 100])

snp_probs = {}
conditional_probs = {}

for snp_col in selected_snp_cols:
    values = df_real[snp_col].values.astype(int)
    counts = np.bincount(values, minlength=3)
    probs = counts / counts.sum() if counts.sum() > 0 else np.array([0.33, 0.33, 0.34])
    snp_probs[snp_col] = probs
    
    # Probabilités conditionnelles simples
    yield_classes = np.digitize(y, yield_percentiles[1:-1])
    cond_probs = {}
    for cls in range(5):
        mask = yield_classes == cls
        if mask.sum() > 3:
            snp_cls = values[mask]
            counts_cls = np.bincount(snp_cls, minlength=3)
            cond_probs[cls] = counts_cls / counts_cls.sum()
        else:
            cond_probs[cls] = probs
    conditional_probs[snp_col] = cond_probs

print(f"   ✓ Probabilités calculées")

# 4. GÉNÉRATION
print("\n[4/4] Génération des données...")

output_dir = BASE_DIR / "04_augmentation/pepper/context_learning_CORRECTED"
output_dir.mkdir(parents=True, exist_ok=True)

contexts = {
    'context_A': (y.min(), y.max()),
    'context_B': (y.min(), np.percentile(y, 50)),
    'context_C': (np.percentile(y, 50), y.max()),
}

for ctx_name, y_range in contexts.items():
    print(f"\n   Génération {ctx_name}...")
    
    # Générer 500 échantillons
    np.random.seed(42)
    y_synth = np.random.uniform(y_range[0], y_range[1], 500)
    y_classes = np.digitize(y_synth, yield_percentiles[1:-1])
    y_classes = np.clip(y_classes, 0, 4)
    
    data = {'Sample_ID': [f'SYNTH_{i:05d}' for i in range(500)], 'YR_LS': y_synth}
    
    for snp_col in selected_snp_cols:
        snp_values = []
        for i in range(500):
            probs = conditional_probs[snp_col][y_classes[i]]
            snp_val = np.random.choice([0, 1, 2], p=probs)
            snp_values.append(int(snp_val))
        data[snp_col] = snp_values
    
    df_synth = pd.DataFrame(data)
    
    # Vérifier
    unique_vals = np.unique(df_synth[selected_snp_cols].values)
    print(f"     SNPs: {sorted(unique_vals)} | Valid: {set(unique_vals).issubset({0,1,2})}")
    
    # Sauvegarder
    output_file = output_dir / f"synthetic_pepper_{ctx_name}_500samples_CORRECTED.csv"
    df_synth.to_csv(output_file, index=False)
    print(f"     ✓ Sauvegardé")

print("\n" + "="*80)
print("TERMINÉ")
print("="*80)
print(f"\n📁 Dossier: {output_dir}")
