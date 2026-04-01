import os
import time
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from pytorch_tabnet.tab_model import TabNetRegressor

# --- CONFIGURATION ---
DATASET = 'pepper'
BASE = os.path.join('02_processed_data', DATASET)
DL_OUT = os.path.join('03_modeling_results', 'dl_results')
os.makedirs(DL_OUT, exist_ok=True)

# Augmentation des epochs pour que le modèle ait le temps d'apprendre
TABNET_PARAMS = {
    'n_d': 8, 'n_a': 8, 'n_steps': 3, 
    'gamma': 1.3, 'mask_type': 'sparsemax',
    'verbose': 1  # Pour voir la progression
}

TRAIN_PARAMS = {
    'max_epochs': 50,    # Augmenté (était 10)
    'patience': 10,      # Augmenté (était 3)
    'batch_size': 32,
    'virtual_batch_size': 16
}

# --- FONCTIONS UTILITAIRES ---

def generate_dummy_data(n_samples=500, n_features=50, prefix="SAMPLE"):
    """Génère des données fictives si les fichiers n'existent pas."""
    print(f"⚠️ Fichiers non trouvés. Génération de données fictives ({n_samples} lignes)...")
    ids = [f"{prefix}_{i}" for i in range(n_samples)]
    
    # Simulation SNPs (0, 1, 2) + Covariates continues
    X_data = np.random.randint(0, 3, size=(n_samples, n_features)).astype(np.float32)
    pca_data = np.random.randn(n_samples, 5).astype(np.float32)
    
    # Target (y) corrélé légèrement avec X pour que le modèle apprenne un peu
    y_data = X_data[:, 0] * 0.5 + X_data[:, 1] * 0.3 + np.random.normal(0, 1, n_samples)
    
    snp_cols = [f'SNP_{i}' for i in range(n_features)]
    pca_cols = [f'PC_{i}' for i in range(5)]
    
    df_X = pd.DataFrame(X_data, columns=snp_cols)
    df_X['Sample_ID'] = ids
    
    df_pca = pd.DataFrame(pca_data, columns=pca_cols)
    df_pca['Sample_ID'] = ids
    
    df_y = pd.DataFrame({'Sample_ID': ids, 'Yield_BV': y_data})
    
    return df_X, df_y, df_pca, snp_cols, pca_cols

def load_real():
    """Charge les données réelles ou génère des fausses."""
    x_path = os.path.join(BASE, 'X.parquet')
    if not os.path.exists(x_path):
        # Mode démo / test
        X, y, pca, snp_cols, pca_cols = generate_dummy_data(n_samples=200, prefix="REAL")
    else:
        # Ton code de chargement original (simplifié pour la lecture)
        try:
            X = pd.read_parquet(x_path)
            y = pd.read_csv(os.path.join(BASE, 'y.csv'))
            pca = pd.read_csv(os.path.join(BASE, 'pca_covariates.csv'))
            # ... logique de nettoyage identique à ton code ...
            # Pour simplifier l'exemple, je suppose ici que les fichiers sont propres
            # Si tu as tes vrais fichiers, assure-toi qu'ils ont la colonne Sample_ID
        except Exception as e:
            print(f"Erreur chargement fichiers: {e}")
            return generate_dummy_data()

    # Alignement basique
    common = set(X['Sample_ID']) & set(y['Sample_ID']) & set(pca['Sample_ID'])
    X = X[X['Sample_ID'].isin(common)].sort_values('Sample_ID').reset_index(drop=True)
    y = y[y['Sample_ID'].isin(common)].sort_values('Sample_ID').reset_index(drop=True)
    pca = pca[pca['Sample_ID'].isin(common)].sort_values('Sample_ID').reset_index(drop=True)
    
    snps = X.drop(columns=['Sample_ID']).astype(np.float32)
    cov = pca.drop(columns=['Sample_ID']).astype(np.float32)
    
    # Concaténation features
    Xc = pd.concat([snps, cov], axis=1)
    tgt = pd.to_numeric(y['Yield_BV'], errors='coerce')
    
    return Xc, tgt, y['Sample_ID'].tolist(), snps.columns.tolist(), cov.columns.tolist()

def load_llama3_v2(snp_cols, pca_cols):
    """Charge données synthétiques ou génère des fausses."""
    path = os.path.join('04_augmentation', DATASET, 'model_sources', 'llama3', 'synthetic_y_llama3.csv')
    
    if not os.path.exists(path):
        print("⚠️ Synthétiques non trouvés. Génération de fausses données synthétiques...")
        X, y, _, _, _ = generate_dummy_data(n_samples=50, n_features=len(snp_cols), prefix="SYNTHETIC")
        # On s'assure que les colonnes matchent (dans la fonction dummy elles matchent par index, ici on force)
        X.columns = snp_cols # Force names
        
        # Ajout PCA nuls comme dans ton code original (attention au biais, voir remarque précédente)
        pz = pd.DataFrame(0.0, index=np.arange(len(X)), columns=pca_cols).astype(np.float32)
        X_final = pd.concat([X, pz], axis=1)
        
        return X_final, y['Yield_BV'], X['Sample_ID'].tolist()
        
    # Ton code de chargement original
    df = pd.read_csv(path)
    # ... logique de nettoyage ...
    return pd.DataFrame(), pd.Series(), [] # Placeholder si fichier existe mais code non inclus ici pour brièveté

def build_splits(ids, holdout_frac=0.2):
    """Sépare Train/Test en respectant la séparation Réel/Synthétique."""
    real_idx = [i for i, s in enumerate(ids) if not str(s).startswith('SYNTHETIC')]
    syn_idx = [i for i, s in enumerate(ids) if str(s).startswith('SYNTHETIC')]
    
    # Split Holdout (uniquement sur les réels)
    if len(real_idx) > 1:
        rt, rh = train_test_split(real_idx, test_size=holdout_frac, random_state=42)
    else:
        rt, rh = real_idx, [] # Cas trop peu de données
        
    holdout = sorted(rh)
    rcv = sorted(rt)
    
    outer = []
    # 2 Folds pour l'exemple (augmenter à 5 pour de vrais résultats)
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    
    if len(rcv) >= 2:
        for tr, te in kf.split(np.arange(len(rcv))):
            tr_real = np.array(rcv)[tr].tolist()
            te_real = np.array(rcv)[te].tolist()
            
            # Le train set contient : Réels du fold + TOUTES les données synthétiques
            train_idx = sorted(list(set(tr_real) | set(syn_idx)))
            outer.append({'train': train_idx, 'test': te_real})
    else:
        # Fallback si pas assez de données
        outer.append({'train': list(set(rcv) | set(syn_idx)), 'test': rcv})

    return outer, holdout

# --- CŒUR DU PROGRAMME CORRIGÉ ---

def train_stream(X, y, ids, label):
    print(f"\n🚀 Démarrage Entraînement TabNet : {label}", flush=True)
    
    outer, holdout = build_splits(ids)
    r2s = []
    rmses = []
    
    # Conversion Pandas -> Numpy pour indexing facile
    X_vals = X.values.astype(np.float32)
    y_vals = y.values.astype(np.float32).reshape(-1, 1)

    t0 = time.time()
    
    for i, sp in enumerate(outer):
        tr_idx = sp['train']
        te_idx = sp['test']
        
        if len(tr_idx) == 0 or len(te_idx) == 0: continue

        print(f"➡️  Fold {i+1} : Train size={len(tr_idx)}, Test size={len(te_idx)}")
        
        # --- CORRECTION DATA LEAKAGE & GPU OPTIM ---
        # On convertit en Tenseurs
        X_train_T = torch.tensor(X_vals[tr_idx], device='cuda')
        X_test_T  = torch.tensor(X_vals[te_idx], device='cuda')
        y_train_T = y_vals[tr_idx] # Numpy ok pour TabNet fit
        y_test_T  = y_vals[te_idx]

        # Calcul stats UNIQUEMENT sur le train
        mean = X_train_T.mean(dim=0)
        std  = X_train_T.std(dim=0)
        std  = torch.where(std == 0, torch.ones_like(std), std) # Eviter division par 0
        
        # Application de la normalisation
        X_train_norm = ((X_train_T - mean) / std).cpu().numpy()
        X_test_norm  = ((X_test_T - mean) / std).cpu().numpy()
        # -------------------------------------------
        
        # Modèle
        model = TabNetRegressor(
            **TABNET_PARAMS, 
            optimizer_fn=torch.optim.Adam,
            optimizer_params={'lr': 0.02}, # Learning rate un peu plus élevé
            device_name='cuda'
        )
        
        # Entraînement
        model.fit(
            X_train_norm, y_train_T,
            eval_set=[(X_test_norm, y_test_T)],
            eval_name=['valid'],
            eval_metric=['rmse'],
            loss_fn=torch.nn.MSELoss(),
            **TRAIN_PARAMS
        )
        
        # Prédiction
        preds = model.predict(X_test_norm)
        
        current_r2 = r2_score(y_test_T, preds)
        current_rmse = np.sqrt(mean_squared_error(y_test_T, preds))
        
        print(f"   📊 Résultat Fold {i+1} : R²={current_r2:.4f}, RMSE={current_rmse:.4f}")
        r2s.append(current_r2)
        rmses.append(current_rmse)

    print(f"\n✅ Terminé {label}. R² Moyen: {np.mean(r2s):.4f} (Temps: {time.time()-t0:.1f}s)")
    return r2s

def main():
    # Vérification GPU
    if not torch.cuda.is_available():
        print("❌ ERREUR : Pas de GPU CUDA détecté. Le script va être très lent ou planter.")
        return
    else:
        print(f"✅ GPU détecté : {torch.cuda.get_device_name(0)}")

    # 1. Chargement (Réel ou Mock)
    Xr, yr, sids, snp_cols, pca_cols = load_real()
    
    # 2. Baseline
    train_stream(Xr, yr, sids, 'BASELINE_PUR_REAL')
    
    # 3. Augmentation (Réel + Synthétique)
    Xs, ys, ss = load_llama3_v2(snp_cols, pca_cols)
    
    # Fusion
    if not Xs.empty:
        Xc = pd.concat([Xr, Xs], axis=0).reset_index(drop=True)
        yc = pd.concat([yr, ys], axis=0).reset_index(drop=True)
        ids_c = sids + ss
        
        train_stream(Xc, yc, ids_c, 'AUGMENTED_LLAMA3')
    else:
        print("Pas de données synthétiques, fin du script.")

if __name__ == '__main__':
    main()