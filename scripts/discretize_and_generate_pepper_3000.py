"""
Discrétisation et génération des 3000+ SNPs Pepper
Convertit les valeurs continues en 0,1,2 puis génère des données synthétiques
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time

BASE_DIR = Path("c:/Users/OMEN/Desktop/experiment_snp")

print("="*80)
print("DISCRÉTISATION ET GÉNÉRATION - 3000+ SNPs PEPPER")
print("="*80)

start_time = time.time()

# 1. CHARGEMENT
print("\n[1/6] Chargement des données Pepper...")
real_data_dir = BASE_DIR / "02_processed_data/pepper"
X_real = pd.read_csv(real_data_dir / "X_aligned.csv")
y_real = pd.read_csv(real_data_dir / "y_aligned.csv")
df_real = pd.merge(X_real, y_real, on='Sample_ID', how='inner')

# Toutes les colonnes sauf Sample_ID sont des SNPs
snp_cols = [c for c in df_real.columns if c != 'Sample_ID' and c != 'YR_LS']
print(f"   ✓ {len(df_real)} échantillons, {len(snp_cols)} SNPs")
print(f"   ✓ Exemple de noms: {snp_cols[:3]}")

# Vérifier le format actuel
sample_values = df_real[snp_cols[0]].values[:5]
print(f"   ✓ Valeurs exemple (avant): {sample_values}")

# 2. DISCRÉTISATION EN 3 CLASSES (0, 1, 2)
print("\n[2/6] Discrétisation des SNPs en 3 classes (0, 1, 2)...")

def discretize_snp(values):
    """Discrétise les valeurs continues en 3 classes"""
    # Utiliser les quantiles pour créer 3 classes équilibrées
    q33, q66 = np.percentile(values, [33.33, 66.67])
    result = np.zeros(len(values), dtype=int)
    result[values > q33] = 1
    result[values > q66] = 2
    return result

# Discrétiser tous les SNPs
snp_matrix = np.zeros((len(df_real), len(snp_cols)), dtype=int)
for i, col in enumerate(snp_cols):
    snp_matrix[:, i] = discretize_snp(df_real[col].values)
    if i % 500 == 0:
        print(f"      Progression: {i}/{len(snp_cols)} SNPs...")

print(f"   ✓ Discrétisation terminée")
print(f"   ✓ Valeurs uniques: {np.unique(snp_matrix)}")

# Créer DataFrame discrétisé
df_discrete = pd.DataFrame(snp_matrix, columns=snp_cols)
df_discrete['Sample_ID'] = df_real['Sample_ID'].values
df_discrete['YR_LS'] = df_real['YR_LS'].values

# 3. CALCUL DES PROBABILITÉS MARGINALES
print("\n[3/6] Calcul des probabilités marginales...")

probs_0 = (snp_matrix == 0).mean(axis=0)
probs_1 = (snp_matrix == 1).mean(axis=0)
probs_2 = 1 - probs_0 - probs_1

marginal_probs = np.stack([probs_0, probs_1, probs_2], axis=1)
print(f"   ✓ Probabilités marginales: shape {marginal_probs.shape}")

# 4. CALCUL DES PROBABILITÉS CONDITIONNELLES
print("\n[4/6] Calcul des probabilités conditionnelles (par classe de yield)...")

y = df_discrete['YR_LS'].values
y_min, y_max = y.min(), y.max()

# Discrétiser en 3 classes
def get_yield_class(y_values):
    q33, q66 = np.percentile(y_values, [33.33, 66.67])
    classes = np.zeros(len(y_values), dtype=int)
    classes[y_values > q33] = 1
    classes[y_values > q66] = 2
    return classes

yield_classes = get_yield_class(y)
print(f"   ✓ Distribution des classes: {np.bincount(yield_classes)}")

# Calculer les probabilités conditionnelles
conditional_probs = {}
for cls in range(3):
    mask = yield_classes == cls
    if mask.sum() > 0:
        snp_cls = snp_matrix[mask]
        p0 = (snp_cls == 0).mean(axis=0)
        p1 = (snp_cls == 1).mean(axis=0)
        p2 = 1 - p0 - p1
        # Normaliser
        total = p0 + p1 + p2
        total[total == 0] = 1  # Éviter division par zéro
        conditional_probs[cls] = np.stack([p0/total, p1/total, p2/total], axis=1)
    else:
        conditional_probs[cls] = marginal_probs

print(f"   ✓ Probabilités conditionnelles calculées")

# 5. GÉNÉRATION SYNTHÉTIQUE
print("\n[5/6] Génération de 500 échantillons synthétiques...")

output_dir = BASE_DIR / "04_augmentation/pepper/context_learning_CORRECTED"
output_dir.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
n_samples = 500

# Générer les yields
y_synth = np.random.uniform(y_min, y_max, n_samples)
y_synth_classes = get_yield_class(y_synth)

print(f"   ✓ Classes des échantillons synthétiques: {np.bincount(y_synth_classes)}")

# Créer DataFrame de base
df_synth = pd.DataFrame({
    'Sample_ID': [f'SYNTH_PEPPER_{i:05d}' for i in range(n_samples)],
    'YR_LS': y_synth
})

# Générer les SNPs en batches
batch_size = 200
n_batches = (len(snp_cols) + batch_size - 1) // batch_size

print(f"   Génération en {n_batches} batches de {batch_size} SNPs...")

for batch_idx in range(n_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(snp_cols))
    batch_cols = snp_cols[start_idx:end_idx]
    
    if batch_idx % 5 == 0:
        print(f"      Batch {batch_idx+1}/{n_batches}...")
    
    # Pour chaque SNP du batch
    for snp_idx, snp_col in enumerate(batch_cols):
        global_idx = start_idx + snp_idx
        
        # Probabilités pour ce SNP selon les classes de yield
        sample_probs = np.array([conditional_probs[cls][global_idx] for cls in y_synth_classes])
        
        # Générer les valeurs
        snp_values = np.array([np.random.choice([0, 1, 2], p=probs) for probs in sample_probs])
        df_synth[snp_col] = snp_values

print(f"   ✓ {len(snp_cols)} SNPs générés")

# 6. SAUVEGARDE
print("\n[6/6] Sauvegarde...")

output_file = output_dir / "synthetic_pepper_3000snps_DISCRETIZED.csv"
df_synth.to_csv(output_file, index=False)

elapsed = time.time() - start_time
print(f"   ✓ Sauvegardé: {output_file}")
print(f"   ✓ Taille: {len(df_synth)} échantillons × {len(df_synth.columns)} colonnes")

# Vérification
snp_cols_check = [c for c in df_synth.columns if c not in ['Sample_ID', 'YR_LS']]
unique_vals = np.unique(df_synth[snp_cols_check].values)
print(f"   ✓ Valeurs SNP: {sorted(unique_vals)}")
print(f"   ✓ Format valide: {set(unique_vals).issubset({0, 1, 2})}")

# Sauvegarder aussi les données réelles discrétisées pour référence
real_discrete_file = output_dir / "pepper_real_DISCRETIZED.csv"
df_discrete.to_csv(real_discrete_file, index=False)
print(f"   ✓ Données réelles discrétisées: {real_discrete_file}")

print("\n" + "="*80)
print(f"TERMINÉ EN {elapsed:.1f} SECONDES!")
print("="*80)
