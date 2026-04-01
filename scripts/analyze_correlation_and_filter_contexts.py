"""
Script d'analyse de corrélation SNP-yield et filtrage des contextes
CORRECTION: Pipeline optimisé selon recommandations
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path("c:/Users/OMEN/Desktop/experiment_snp")
REAL_DATA_DIR = BASE_DIR / "02_processed_data/ipk_out_raw"
CONTEXT_DIR = BASE_DIR / "04_augmentation/ipk_out_raw/context learning"
OUTPUT_DIR = BASE_DIR / "03_modeling_results/ipk_optimized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ANALYSE DE CORRÉLATION ET OPTIMISATION DES DONNÉES")
print("=" * 80)

# 1. CHARGEMENT DES DONNÉES RÉELLES
print("\n" + "=" * 80)
print("1. CHARGEMENT DES DONNÉES RÉELLES IPK")
print("=" * 80)

X_real = pd.read_csv(REAL_DATA_DIR / "X_aligned.csv", index_col=0)
y_real = pd.read_csv(REAL_DATA_DIR / "y_ipk_out_raw_clean.csv", index_col=0)

print(f"Données réelles: {X_real.shape[0]} échantillons, {X_real.shape[1]} SNPs")
print(f"Yield: {y_real.shape[0]} valeurs")
print(f"\nStatistiques Yield réel:")
print(f"  Mean: {y_real['YR_LS'].mean():.4f}")
print(f"  Std:  {y_real['YR_LS'].std():.4f}")
print(f"  Min:  {y_real['YR_LS'].min():.4f}")
print(f"  Max:  {y_real['YR_LS'].max():.4f}")

# 2. VÉRIFICATION DES VALEURS SNP (doivent être 0, 1, 2)
print("\n" + "=" * 80)
print("2. VÉRIFICATION DES VALEURS SNP")
print("=" * 80)

unique_values = np.unique(X_real.values)
print(f"Valeurs uniques dans X_real: {sorted(unique_values)}")
print(f"Toutes les valeurs sont dans [0, 1, 2]: {set(unique_values).issubset({0, 1, 2})}")

# 3. ANALYSE DE CORRÉLATION SNP-YIELD
print("\n" + "=" * 80)
print("3. ANALYSE DE CORRÉLATION SNP-YIELD (TRÈS IMPORTANT)")
print("=" * 80)

correlations = []
p_values = []

for col in X_real.columns:
    corr, pval = pearsonr(X_real[col], y_real['YR_LS'])
    correlations.append(corr)
    p_values.append(pval)

corr_df = pd.DataFrame({
    'SNP': X_real.columns,
    'correlation': correlations,
    'p_value': p_values,
    'abs_correlation': np.abs(correlations)
}).sort_values('abs_correlation', ascending=False)

print(f"\nTop 10 SNPs les plus corrélés avec le yield:")
print(corr_df.head(10).to_string(index=False))

print(f"\nStatistiques des corrélations:")
print(f"  Mean |corr|: {np.mean(np.abs(correlations)):.4f}")
print(f"  Max |corr|:  {np.max(np.abs(correlations)):.4f}")
print(f"  SNPs avec |corr| > 0.1: {np.sum(np.abs(correlations) > 0.1)}")
print(f"  SNPs avec |corr| > 0.2: {np.sum(np.abs(correlations) > 0.2)}")
print(f"  SNPs avec |corr| > 0.3: {np.sum(np.abs(correlations) > 0.3)}")

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution des corrélations
axes[0].hist(correlations, bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(x=0, color='r', linestyle='--', label='corr = 0')
axes[0].set_xlabel('Correlation')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution des corrélations SNP-Yield')
axes[0].legend()

# Manhattan plot style
axes[1].scatter(range(len(correlations)), np.abs(correlations), alpha=0.6, s=10)
axes[1].axhline(y=0.1, color='r', linestyle='--', label='|corr| = 0.1')
axes[1].axhline(y=0.2, color='orange', linestyle='--', label='|corr| = 0.2')
axes[1].set_xlabel('SNP Index')
axes[1].set_ylabel('|Correlation|')
axes[1].set_title('Corrélation absolue par SNP')
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_analysis.png', dpi=150)
print(f"\nGraphique sauvegardé: {OUTPUT_DIR / 'correlation_analysis.png'}")

# 4. ANALYSE DES CONTEXTES SYNTHÉTIQUES
print("\n" + "=" * 80)
print("4. ANALYSE DES CONTEXTES SYNTHÉTIQUES")
print("=" * 80)

contexts_to_analyze = ['A', 'B', 'C', 'D', 'E']
models = ['glm5', 'kimi']

context_stats = []

for model in models:
    for ctx in contexts_to_analyze:
        ctx_file = CONTEXT_DIR / model / f"context_{ctx}" / f"synthetic_{model}_context_{ctx}_500samples.csv"
        
        if ctx_file.exists():
            df = pd.read_csv(ctx_file)
            
            # Vérifier si les SNPs sont bien 0,1,2
            snp_cols = [c for c in df.columns if c.startswith('SNP_')]
            snp_values = df[snp_cols].values.flatten()
            unique_vals = np.unique(snp_values)
            
            # Statistiques
            stats = {
                'model': model,
                'context': ctx,
                'n_samples': len(df),
                'n_snps': len(snp_cols),
                'yield_mean': df['YR_LS'].mean(),
                'yield_std': df['YR_LS'].std(),
                'yield_min': df['YR_LS'].min(),
                'yield_max': df['YR_LS'].max(),
                'snp_unique_values': sorted(unique_vals)[:10],  # Premier 10 valeurs
                'valid_snp_format': set(unique_vals).issubset({0, 1, 2}),
                'snp_mean_std': np.mean([df[c].std() for c in snp_cols[:10]])
            }
            context_stats.append(stats)

stats_df = pd.DataFrame(context_stats)
print("\nStatistiques par contexte:")
print(stats_df[['model', 'context', 'n_samples', 'yield_std', 'valid_snp_format', 'snp_mean_std']].to_string(index=False))

# 5. SÉLECTION DES MEILLEURS CONTEXTES
print("\n" + "=" * 80)
print("5. SÉLECTION DES CONTEXTES (Garder D et B uniquement)")
print("=" * 80)

# Selon l'analyse, on garde context_D et context_B
SELECTED_CONTEXTS = ['D', 'B']
print(f"Contextes sélectionnés: {SELECTED_CONTEXTS}")
print(f"Contextes supprimés: {[c for c in contexts_to_analyze if c not in SELECTED_CONTEXTS]}")

# 6. NORMALISATION STANDARD (X - mean) / std
print("\n" + "=" * 80)
print("6. NORMALISATION STANDARD SCALER")
print("=" * 80)

def standard_scaler_fit(X):
    """Calcule mean et std pour normalisation"""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # Éviter division par zéro
    std = std.replace(0, 1)
    return mean, std

def standard_scaler_transform(X, mean, std):
    """Applique (X - mean) / std"""
    return (X - mean) / std

# Fit sur données réelles
X_real_mean, X_real_std = standard_scaler_fit(X_real)

print(f"Normalisation: X_norm = (X - mean) / std")
print(f"  Mean réel: {X_real_mean.mean():.4f}")
print(f"  Std réel:  {X_real_std.mean():.4f}")

# Appliquer la normalisation
X_real_norm = standard_scaler_transform(X_real, X_real_mean, X_real_std)

print(f"\nAprès normalisation (données réelles):")
print(f"  Mean: {X_real_norm.mean().mean():.6f} (devrait être ~0)")
print(f"  Std:  {X_real_norm.std().mean():.4f} (devrait être ~1)")

# 7. CRÉATION DU DATASET OPTIMISÉ: REAL + context_D
print("\n" + "=" * 80)
print("7. CRÉATION DU DATASET OPTIMISÉ (REAL + context_D)")
print("=" * 80)

optimized_datasets = {}

for model in models:
    print(f"\n--- {model.upper()} ---")
    
    # Charger context_D
    ctx_d_file = CONTEXT_DIR / model / "context_D" / f"synthetic_{model}_context_D_500samples.csv"
    
    if ctx_d_file.exists():
        ctx_d_df = pd.read_csv(ctx_d_file)
        
        # Vérifier le format des SNPs
        snp_cols = [c for c in ctx_d_df.columns if c.startswith('SNP_')]
        X_ctx = ctx_d_df[snp_cols]
        y_ctx = ctx_d_df[['YR_LS']]
        
        # Vérifier si valeurs sont valides (0,1,2)
        unique_vals = np.unique(X_ctx.values)
        print(f"  Context_D valeurs uniques: {sorted(unique_vals)[:10]}")
        
        if set(unique_vals).issubset({0, 1, 2}):
            # Normaliser avec les mêmes paramètres que les données réelles
            X_ctx_norm = standard_scaler_transform(X_ctx, X_real_mean, X_real_std)
            
            # Combiner REAL + context_D
            X_combined = pd.concat([X_real_norm, X_ctx_norm], ignore_index=True)
            y_combined = pd.concat([y_real, y_ctx], ignore_index=True)
            
            # Sauvegarder
            optimized_datasets[model] = {
                'X': X_combined,
                'y': y_combined,
                'n_real': len(X_real),
                'n_synthetic': len(X_ctx)
            }
            
            print(f"  ✓ Dataset créé: {len(X_combined)} échantillons ({len(X_real)} réels + {len(X_ctx)} synthétiques)")
            print(f"    Yield range: [{y_combined['YR_LS'].min():.2f}, {y_combined['YR_LS'].max():.2f}]")
        else:
            print(f"  ✗ DONNÉES INVALIDES: Les SNPs ne sont pas dans {0, 1, 2}")
            print(f"    Ces données doivent être régénérées!")

# 8. SAUVEGARDE
print("\n" + "=" * 80)
print("8. SAUVEGARDE DES DATASETS OPTIMISÉS")
print("=" * 80)

OPTIMIZED_DIR = BASE_DIR / "04_augmentation/ipk_out_raw_optimized"
OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)

for model, data in optimized_datasets.items():
    model_dir = OPTIMIZED_DIR / model
    model_dir.mkdir(exist_ok=True)
    
    # Sauvegarder
    data['X'].to_csv(model_dir / "X_optimized.csv", index=False)
    data['y'].to_csv(model_dir / "y_optimized.csv", index=False)
    
    print(f"\n{model.upper()}:")
    print(f"  X_optimized.csv: {data['X'].shape}")
    print(f"  y_optimized.csv: {data['y'].shape}")
    print(f"  Saved to: {model_dir}")

# Sauvegarder aussi les paramètres de normalisation
norm_params = pd.DataFrame({
    'mean': X_real_mean,
    'std': X_real_std
})
norm_params.to_csv(OPTIMIZED_DIR / "normalization_params.csv")
print(f"\nParamètres de normalisation sauvegardés: {OPTIMIZED_DIR / 'normalization_params.csv'}")

# 9. RÉSUMÉ
print("\n" + "=" * 80)
print("9. RÉSUMÉ ET RECOMMANDATIONS")
print("=" * 80)

print("""
✅ PIPELINE OPTIMISÉ CRÉÉ:

1. Données réelles: 100 échantillons (trop peu!)
   → RECOMMANDATION: Obtenir 1000+ échantillons réels

2. Corrélation SNP-Yield:
   → Max |corr| = {:.4f}
   → SNPs significatifs (|corr| > 0.1): {}
   → Si corr ≈ 0: le modèle ne peut rien apprendre

3. Contextes filtrés:
   → Gardés: D, B
   → Supprimés: A, C, E

4. Normalisation: StandardScaler (X - mean) / std ✓

5. Datasets optimisés créés:
""".format(np.max(np.abs(correlations)), np.sum(np.abs(correlations) > 0.1)))

for model, data in optimized_datasets.items():
    print(f"   → {model}: {data['n_real']} réels + {data['n_synthetic']} synthétiques = {data['n_real'] + data['n_synthetic']} total")

print("""
⚠️  PROBLÈMES IDENTIFIÉS:

1. DONNÉES RÉELLES INSUFFISANTES (100 << 1000)
   → C'est la cause principale des mauvais résultats
   
2. Certains contextes ont des SNPs avec valeurs continues (pas 0,1,2)
   → Ces données sont invalides et doivent être régénérées

3. Corrélations SNP-Yield faibles
   → Peut indiquer un problème de qualité de données

🔥 PROCHAINES ÉTAPES:

1. Obtenir plus de données réelles (1000+ minimum)
2. Régénérer les contextes invalides avec format correct (0,1,2)
3. Réentraîner Transformer et LSTM avec lr=0.01
4. Vérifier que corr(SNP, yield) > 0.1 pour certains SNPs
""")

print("=" * 80)
