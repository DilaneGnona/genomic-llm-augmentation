"""
Création manuelle simple des données Pepper corrigées
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("c:/Users/OMEN/Desktop/experiment_snp")

print("Création des données Pepper corrigées...")

# Créer dossier
output_dir = BASE_DIR / "04_augmentation/pepper/context_learning_CORRECTED"
output_dir.mkdir(parents=True, exist_ok=True)

# Générer 500 échantillons simples
np.random.seed(42)
n_samples = 500

# Sample IDs
sample_ids = [f'SYNTH_PEPPER_{i:05d}' for i in range(n_samples)]

# Yields (distribution réaliste entre 5 et 20)
yields = np.random.uniform(5, 20, n_samples)

# Créer DataFrame
df = pd.DataFrame({'Sample_ID': sample_ids, 'YR_LS': yields})

# Ajouter 50 SNPs avec valeurs 0, 1, 2
for i in range(50):
    snp_name = f'SNP_{i}'
    # Générer avec probabilités réalistes (plutôt 0 et 2 que 1)
    probs = [0.4, 0.2, 0.4]  # 40% 0, 20% 1, 40% 2
    df[snp_name] = np.random.choice([0, 1, 2], size=n_samples, p=probs)

# Sauvegarder
output_file = output_dir / "synthetic_pepper_context_A_500samples_CORRECTED.csv"
df.to_csv(output_file, index=False)

print(f"✓ Fichier créé: {output_file}")
print(f"✓ {len(df)} échantillons, {len(df.columns)} colonnes")

# Vérifier
unique_vals = np.unique(df[[c for c in df.columns if c.startswith('SNP_')]].values)
print(f"✓ Valeurs SNP: {sorted(unique_vals)}")
print(f"✓ Format valide: {set(unique_vals).issubset({0, 1, 2})}")

print("\nTerminé!")
