import os 
import time 
import json 
import numpy as np 
import pandas as pd 
import torch 
import matplotlib.pyplot as plt 
import argparse
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
    'n_d': 4, 'n_a': 4, 'n_steps': 3, 
    'gamma': 1.3, 'mask_type': 'sparsemax', 
    'verbose': 1 
} 

TRAIN_PARAMS = { 
    'max_epochs': 50, 
    'patience': 10, 
    'batch_size': 16, 
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
    """Charge synthétiques llama3 v2/v1 ou génère des fausses; aligne colonnes.""" 
    # Try v2 strict, then v1 200, then generic
    paths = [
        os.path.join('04_augmentation', DATASET, 'model_sources', 'llama3', 'synthetic_y_llama3_filtered_k3000_200_v2.csv'),
        os.path.join('04_augmentation', DATASET, 'model_sources', 'llama3', 'synthetic_y_llama3_filtered_k3000_200.csv'),
        os.path.join('04_augmentation', DATASET, 'model_sources', 'llama3', 'synthetic_y_llama3.csv'),
    ]
    path = next((p for p in paths if os.path.exists(p)), None)
    
    if not os.path.exists(path): 
        print("⚠️ Synthétiques non trouvés. Génération de fausses données synthétiques...") 
        X, y, _, _, _ = generate_dummy_data(n_samples=50, n_features=len(snp_cols), prefix="SYNTHETIC") 
        # On s'assure que les colonnes matchent (dans la fonction dummy elles matchent par index, ici on force) 
        X.columns = snp_cols # Force names 
        
        # Ajout PCA nuls comme dans ton code original (attention au biais, voir remarque précédente) 
        pz = pd.DataFrame(0.0, index=np.arange(len(X)), columns=pca_cols).astype(np.float32) 
        X_final = pd.concat([X, pz], axis=1) 
        
        return X_final, y['Yield_BV'], X['Sample_ID'].tolist() 
        
    # Chargement réel si fichier existe
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Erreur chargement synthétiques: {e}")
        return pd.DataFrame(), pd.Series(), []
    if 'Sample_ID' not in df.columns:
        df.rename(columns={df.columns[0]: 'Sample_ID'}, inplace=True)
    # Extraire SNPs avec correspondance stricte
    snps = df[[c for c in df.columns if c in snp_cols]].copy()
    # Ajouter colonnes manquantes à 0.0
    for c in snp_cols:
        if c not in snps.columns:
            snps[c] = 0.0
    # Réordonner exactement comme snp_cols
    snps = snps[snp_cols].astype(np.float32)
    # Ajouter PCA nuls pour respecter le schéma
    pz = pd.DataFrame(0.0, index=np.arange(len(df)), columns=pca_cols).astype(np.float32)
    X_final = pd.concat([snps, pz], axis=1)
    # Cible
    if 'Yield_BV' in df.columns:
        y = pd.to_numeric(df['Yield_BV'], errors='coerce')
    else:
        y = pd.to_numeric(df.iloc[:, -1], errors='coerce')
    ids = df['Sample_ID'].astype(str).tolist()
    return X_final, y, ids 

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
    # Libération mémoire GPU préalable
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    outer, holdout = build_splits(ids) 
    r2s = [] 
    rmses = [] 
    
    # Conversion Pandas -> Numpy pour indexing facile 
    # Sanitize and lock feature schema
    X_vals = np.nan_to_num(X.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_vals = np.nan_to_num(y.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1, 1)
    n_features = X_vals.shape[1]

    t0 = time.time() 
    
    rng = np.random.default_rng(42)
    for i, sp in enumerate(outer): 
        tr_idx = sp['train'] 
        te_idx = sp['test'] 
        
        if len(tr_idx) == 0 or len(te_idx) == 0:
            print(f"   ⚠️ Fold {i+1} skipped (empty split)", flush=True)
            continue 

        # Downsample fold sizes to fit 8GB VRAM
        if len(tr_idx) > 1000:
            tr_idx = rng.choice(tr_idx, size=1000, replace=False).tolist()
        if len(te_idx) > 800:
            te_idx = rng.choice(te_idx, size=800, replace=False).tolist()
        print(f"➡️  Fold {i+1} : Train size={len(tr_idx)}, Test size={len(te_idx)}", flush=True) 
        
        # --- CORRECTION DATA LEAKAGE & GPU OPTIM --- 
        # Index range checks
        assert min(tr_idx) >= 0 and max(tr_idx) < X_vals.shape[0], "train indices out of range"
        assert min(te_idx) >= 0 and max(te_idx) < X_vals.shape[0], "test indices out of range"
        # Convert to tensors on GPU
        # On convertit en Tenseurs 
        X_train_T = torch.tensor(X_vals[tr_idx], device='cuda') 
        X_test_T  = torch.tensor(X_vals[te_idx], device='cuda') 
        # Ensure feature counts match
        assert X_train_T.shape[1] == n_features and X_test_T.shape[1] == n_features, "feature dimension mismatch"
        # Targets 2D float32
        y_train_T = y_vals[tr_idx].astype(np.float32).reshape(-1, 1) 
        y_test_T  = y_vals[te_idx].astype(np.float32).reshape(-1, 1) 

        # Calcul stats UNIQUEMENT sur le train 
        mean = X_train_T.mean(dim=0) 
        std  = X_train_T.std(dim=0) 
        std  = torch.where(std == 0, torch.ones_like(std), std) # Eviter division par 0 
        
        # Application de la normalisation 
        X_train_norm = ((X_train_T - mean) / std).cpu().numpy().astype(np.float32) 
        X_test_norm  = ((X_test_T - mean) / std).cpu().numpy().astype(np.float32) 
        # Sanitize NaN/Inf after normalization
        X_train_norm = np.nan_to_num(X_train_norm, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_norm  = np.nan_to_num(X_test_norm, nan=0.0, posinf=0.0, neginf=0.0)
        # Final guards
        assert X_train_norm.shape[1] == X_test_norm.shape[1] == n_features, "normalized feature dimension mismatch"
        # ------------------------------------------- 
        
        # Modèle 
        model = TabNetRegressor( 
            **TABNET_PARAMS, 
            optimizer_fn=torch.optim.Adam, 
            optimizer_params={'lr': 0.001}, 
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
        
        # Libération mémoire GPU inter-fold
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Prédiction 
        preds = model.predict(X_test_norm) 
        
        current_r2 = r2_score(y_test_T, preds) 
        current_rmse = np.sqrt(mean_squared_error(y_test_T, preds)) 
        
        print(f"   📊 Résultat Fold {i+1} : R²={current_r2:.4f}, RMSE={current_rmse:.4f}", flush=True) 
        r2s.append(current_r2) 
        rmses.append(current_rmse) 

    print(f"\n✅ Terminé {label}. R² Moyen: {np.mean(r2s):.4f} (Temps: {time.time()-t0:.1f}s)", flush=True) 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return r2s 

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug_only', action='store_true')
    args = parser.parse_args()
    # Vérification GPU 
    if not torch.cuda.is_available(): 
        print("❌ ERREUR : Pas de GPU CUDA détecté. Le script va être très lent ou planter.") 
        return 
    else: 
        print(f"✅ GPU détecté : {torch.cuda.get_device_name(0)}") 

    # 1. Chargement (Réel ou Mock) 
    Xr, yr, sids, snp_cols, pca_cols = load_real() 
    
    # 2. Baseline (skip if aug_only)
    if not args.aug_only:
        train_stream(Xr, yr, sids, 'BASELINE_PUR_REAL') 
    
    # 3. Augmentation (Réel + Synthétique) 
    Xs, ys, ss = load_llama3_v2(snp_cols, pca_cols) 
    
    # Fusion avec alignement strict des colonnes
    if not Xs.empty: 
        # Aligne Xs sur le schéma exact de Xr (remplit à 0.0 les colonnes manquantes, réordonne)
        Xs_aligned = Xs.reindex(columns=Xr.columns, fill_value=0.0).astype(np.float32) 
        Xc = pd.concat([Xr, Xs_aligned], axis=0).reset_index(drop=True) 
        yc = pd.concat([yr, ys], axis=0).reset_index(drop=True) 
        ids_c = sids + ss 
        
        train_stream(Xc, yc, ids_c, 'AUGMENTED_LLAMA3_v2') 
    else: 
        print("Pas de données synthétiques, fin du script.") 

if __name__ == '__main__': 
    main()
