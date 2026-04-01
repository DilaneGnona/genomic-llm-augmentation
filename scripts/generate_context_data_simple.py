"""
Génération simple de données context learning avec corrélation préservée
Version simplifiée pour test rapide
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

BASE_DIR = Path("c:/Users/OMEN/Desktop/experiment_snp")

print("="*80)
print("GÉNÉRATION SIMPLE DE DONNÉES AVEC CORRÉLATION")
print("="*80)

# 1. Chargement des données
print("\n1. Chargement des données réelles Pepper...")
real_data_dir = BASE_DIR / "02_processed_data/pepper"
X_real = pd.read_csv(real_data_dir / "X_aligned.csv")
y_real = pd.read_csv(real_data_dir / "y_aligned.csv")
df_real = X_real.merge(y_real, on='Sample_ID', how='inner')

feature_cols = [c for c in df_real.columns if c.startswith('SNP_')]
print(f"   Données: {len(df_real)} échantillons, {len(feature_cols)} SNPs")

# 2. Sélection des SNPs les plus corrélés (top 100 pour accélérer)
print("\n2. Sélection des SNPs corrélés...")
correlations = []
for col in feature_cols[:200]:  # Limiter à 200 pour la vitesse
    corr, _ = pearsonr(df_real[col], df_real['YR_LS'])
    correlations.append(abs(corr))

top_indices = np.argsort(correlations)[-100:]
top_snp_cols = [feature_cols[i] for i in top_indices]
print(f"   Top 100 SNPs sélectionnés")

# 3. Calculer les probabilités conditionnelles
print("\n3. Calcul des probabilités conditionnelles...")

# Discrétiser le yield en 5 classes
y = df_real['YR_LS'].values
yield_percentiles = np.percentile(y, [0, 20, 40, 60, 80, 100])
yield_classes = np.digitize(y, yield_percentiles[1:-1])

snp_models = {}
for snp_col in top_snp_cols:
    snp_values = df_real[snp_col].values.astype(int)
    probs = {}
    for cls in range(5):
        mask = yield_classes == cls
        if mask.sum() > 0:
            snp_cls = snp_values[mask]
            p0 = (snp_cls == 0).sum() / len(snp_cls)
            p1 = (snp_cls == 1).sum() / len(snp_cls)
            p2 = 1 - p0 - p1
            probs[cls] = [max(0.01, p0), max(0.01, p1), max(0.01, p2)]
            # Normaliser
            total = sum(probs[cls])
            probs[cls] = [p/total for p in probs[cls]]
        else:
            probs[cls] = [0.33, 0.33, 0.34]
    snp_models[snp_col] = probs

print(f"   Modèles construits pour {len(snp_models)} SNPs")

# 4. Génération
print("\n4. Génération des données...")

output_dir = BASE_DIR / "04_augmentation/pepper/context_learning_v2"
output_dir.mkdir(parents=True, exist_ok=True)

# Générer 500 échantillons
n_samples = 500
y_synth = np.random.uniform(y.min(), y.max(), n_samples)
y_synth_classes = np.digitize(y_synth, yield_percentiles[1:-1])
y_synth_classes = np.clip(y_synth_classes, 0, 4)

synthetic_data = []
for i in range(n_samples):
    sample = {'Sample_ID': f'SYNTH_V2_{i:05d}', 'YR_LS': y_synth[i]}
    cls = y_synth_classes[i]
    
    for snp_col in feature_cols:
        if snp_col in snp_models:
            probs = snp_models[snp_col][cls]
            snp_val = np.random.choice([0, 1, 2], p=probs)
        else:
            snp_vals = df_real[snp_col].values
            p0 = (snp_vals == 0).mean()
            p1 = (snp_vals == 1).mean()
            p2 = 1 - p0 - p1
            snp_val = np.random.choice([0, 1, 2], p=[p0, p1, p2])
        sample[snp_col] = snp_val
    
    synthetic_data.append(sample)

df_synth = pd.DataFrame(synthetic_data)
output_file = output_dir / "synthetic_v2_500samples.csv"
df_synth.to_csv(output_file, index=False)

print(f"   Généré: {len(df_synth)} échantillons")
print(f"   Sauvegardé: {output_file}")

# 5. Vérification rapide
print("\n5. Vérification...")
corr_real_list = []
corr_synth_list = []

for col in top_snp_cols[:10]:
    cr, _ = pearsonr(df_real[col], df_real['YR_LS'])
    cs, _ = pearsonr(df_synth[col], df_synth['YR_LS'])
    corr_real_list.append(abs(cr))
    corr_synth_list.append(abs(cs))

print(f"   Corr moyenne (réel): {np.mean(corr_real_list):.3f}")
print(f"   Corr moyenne (synth): {np.mean(corr_synth_list):.3f}")

print("\n" + "="*80)
print("TERMINÉ")
print("="*80)
