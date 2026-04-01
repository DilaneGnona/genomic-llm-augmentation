import pandas as pd 
import numpy as np 
import os 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error, r2_score 
import xgboost as xgb 

# --- CONFIGURATION --- 
BASE_DIR = "04_augmentation/pepper/context_learning/" 
# Priorité au fichier réduit s'il existe pour la vitesse
REAL_DATA_X_MINI = "02_processed_data/pepper/X_mini.csv"
REAL_DATA_X_FULL = "02_processed_data/pepper/X.csv"

REAL_DATA_Y = "02_processed_data/pepper/y.csv"

SYNTH_DATA_PATH = os.path.join(BASE_DIR, "synthetic_data_final.csv") 

def load_and_align_data(): 
    print("🔄 Chargement et alignement des données...") 
    
    # 1. Charger les Synthétiques
    if not os.path.exists(SYNTH_DATA_PATH): 
        raise FileNotFoundError("Pas de données synthétiques trouvées !") 
    df_synth = pd.read_csv(SYNTH_DATA_PATH) 
    
    # On récupère la liste exacte des colonnes utilisées par l'IA 
    cols_to_use = [c for c in df_synth.columns if c not in ['is_synthetic', 'Sample_ID', 'dist_to_median']] 
    print(f"   -> Le synthétique a {len(cols_to_use)} colonnes utiles (SNPs + Yield).") 

    # 2. Reconstruire les Vraies Données
    print("   -> Reconstruction des vraies données...")
    
    if os.path.exists(REAL_DATA_X_MINI):
        print(f"   🚀 Utilisation du fichier optimisé : {REAL_DATA_X_MINI}")
        X = pd.read_csv(REAL_DATA_X_MINI, header=0, low_memory=False)
    else:
        print(f"   🐢 Utilisation du fichier complet (LENT) : {REAL_DATA_X_FULL}")
        # On charge uniquement ces colonnes
        cols_to_load = ['Sample_ID'] + [c for c in cols_to_use if c != 'Yield_BV']
        try:
            X = pd.read_csv(REAL_DATA_X_FULL, header=0, usecols=lambda c: c in cols_to_load, low_memory=False)
        except ValueError:
            X = pd.read_csv(REAL_DATA_X_FULL, header=0, low_memory=False)

    # Nettoyage metadata si nécessaire (si on vient du full X, les 3 premières lignes sont metadata)
    # Si on vient du mini, normalement c'est déjà propre si create_mini_X.py a bien fait le job
    # Vérifions si la colonne Sample_ID contient "Sample_ID" ou des métadonnées
    # Dans X.csv brut, ligne 0=Header, 1=POS, 2=REF, 3=ALT. Donc data commence ligne 3 (index 3).
    # Si create_mini_X a fait read_csv sans skiprows, il a gardé les metadata.
    # On va assumer que le mini est brut aussi pour l'instant.
    
    # Petit check : si la première valeur de la 1ère colonne est un entier ou string ressemblant à un ID, c'est bon.
    # Si c'est "POS" ou "REF", c'est de la métadata.
    # X.csv structure : Header, then line 0 is metadata POS... wait.
    # X.csv:
    # Sample_ID, SNP1, SNP2...
    # 0, 1234, A... (POS)
    # 1, REF, G... (REF)
    # 2, ALT, T... (ALT)
    # 3, S_001, 0... (Data)
    
    # Donc il faut toujours skipper les 3 premières lignes de données si elles sont présentes.
    # On regarde si la colonne Sample_ID a des valeurs bizarres au début.
    
    # Simplification : On applique toujours le skip des 3 premières lignes si on détecte que ce n'est pas un ID.
    if len(X) > 3:
         # Check simple : est-ce que les premières valeurs ressemblent à des entiers (POS) ?
         # Ou juste on applique la règle aveuglément comme avant si on est sûr de la source.
         # Pour être safe, on garde la logique "X.iloc[3:]" car create_mini_X a fait un read_csv simple sur le fichier brut.
         X = X.iloc[3:].copy()
    
    X['Sample_ID'] = X['Sample_ID'].astype(str)
    print(f"   -> X chargé : {X.shape}")
    
    y = pd.read_csv(REAL_DATA_Y)
    y['Sample_ID'] = y['Sample_ID'].astype(str)
    print(f"   -> y chargé : {y.shape}")
    
    # Merge
    print("   -> Fusion X+y...")
    df_real = pd.merge(X, y[['Sample_ID', 'Yield_BV']], on='Sample_ID', how='inner')
    print(f"   -> Fusion terminée : {df_real.shape}")
    
    # 3. Alignement Chirurgical 
    # On ne garde dans le Réel QUE les colonnes présentes dans le Synthétique (plus Yield)
    cols_needed = cols_to_use # Contient déjà Yield_BV si présent dans synth (oui car df_synth a Yield_BV)
    
    try: 
        df_real_aligned = df_real[cols_needed].copy() 
    except KeyError as e: 
        print(f"❌ ERREUR D'ALIGNEMENT : Les colonnes ne correspondent pas !") 
        # print(f"   Synthétique demande : {cols_needed[:5]}...") 
        missing = [c for c in cols_needed if c not in df_real.columns]
        print(f"   Manquantes dans réel : {missing}")
        raise e 
    
    # Conversion en numeric pour être sûr
    for c in df_real_aligned.columns:
        df_real_aligned[c] = pd.to_numeric(df_real_aligned[c], errors='coerce')
    df_real_aligned = df_real_aligned.dropna()

    # 4. Préparation finale 
    # Ajout du flag pour savoir qui est qui 
    df_real_aligned['is_synthetic'] = 0 
    df_synth['is_synthetic'] = 1 
    
    # On enlève Sample_ID pour l'entraînement (déjà fait car cols_to_use l'excluait, mais on vérifie)
    if 'Sample_ID' in df_real_aligned.columns: 
        df_real_aligned = df_real_aligned.drop(columns=['Sample_ID']) 
        df_synth = df_synth.drop(columns=['Sample_ID']) 
        
    print(f"   -> Alignement réussi. Features utilisées : {len(df_real_aligned.columns) - 2} SNPs.") 
    return df_real_aligned, df_synth 

def train_and_evaluate(name, model, X_train, y_train, X_test, y_test): 
    # Entraînement 
    model.fit(X_train, y_train) 
    
    # Prédiction 
    preds = model.predict(X_test) 
    
    # Scores 
    rmse = np.sqrt(mean_squared_error(y_test, preds)) 
    r2 = r2_score(y_test, preds) 
    
    return {"Model": name, "RMSE": rmse, "R2": r2} 

def run_benchmark(): 
    # 1. Prépare les données 
    try:
        df_real, df_synth = load_and_align_data() 
    except Exception as e:
        print(f"Arrêt critique : {e}")
        return

    # 2. Séparation Train/Test sur le RÉEL UNIQUEMENT 
    # IMPORTANT : On ne teste JAMAIS sur du synthétique. Le juge de paix, c'est la réalité. 
    X = df_real.drop(columns=['Yield_BV', 'is_synthetic']) 
    y = df_real['Yield_BV'] 
    
    X_train_real, X_test, y_train_real, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    
    # 3. Création du Dataset Augmenté (Train Réel + Train Synth) 
    X_synth = df_synth.drop(columns=['Yield_BV', 'is_synthetic']) 
    y_synth = df_synth['Yield_BV'] 
    
    # S'assurer que les colonnes sont dans le même ordre
    X_synth = X_synth[X_train_real.columns]

    X_train_aug = pd.concat([X_train_real, X_synth]) 
    y_train_aug = pd.concat([y_train_real, y_synth]) 
    
    print(f"\n🥊 DÉBUT DU TOURNOI") 
    print(f"   - Train Réel : {len(X_train_real)} échantillons") 
    print(f"   - Renforts Synthétiques : {len(X_synth)} échantillons") 
    print(f"   - Train Total : {len(X_train_aug)} échantillons") 
    print(f"   - Test Set (Intouchable) : {len(X_test)} échantillons") 
    
    results = [] 
    
    # --- COMBAT 1 : Random Forest --- 
    rf = RandomForestRegressor(n_estimators=100, random_state=42) 
    
    # Round A : Baseline 
    res_base = train_and_evaluate("RF Baseline", rf, X_train_real, y_train_real, X_test, y_test) 
    results.append(res_base) 
    
    # Round B : Augmented 
    res_aug = train_and_evaluate("RF Augmented", rf, X_train_aug, y_train_aug, X_test, y_test) 
    results.append(res_aug) 
    
    # --- COMBAT 2 : XGBoost --- 
    xg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42) 
    
    results.append(train_and_evaluate("XGB Baseline", xg, X_train_real, y_train_real, X_test, y_test)) 
    results.append(train_and_evaluate("XGB Augmented", xg, X_train_aug, y_train_aug, X_test, y_test)) 
    
    # --- RÉSULTATS --- 
    print("\n🏆 RÉSULTATS FINAUX 🏆") 
    res_df = pd.DataFrame(results) 
    print(res_df.to_string(index=False)) 
    
    # Analyse rapide 
    base_rmse = res_df.iloc[0]['RMSE'] 
    aug_rmse = res_df.iloc[1]['RMSE'] 
    diff = base_rmse - aug_rmse 
    
    print("\n🧐 ANALYSE :") 
    if diff > 0: 
        print(f"✅ SUCCÈS ! L'ajout de synthétiques a réduit l'erreur de {diff:.4f} points.") 
    else: 
        print(f"⚠️ Pas d'amélioration immédiate (Diff: {diff:.4f}).") 
        print("   C'est normal avec seulement 64 échantillons. Il faut lancer la génération complète !") 

if __name__ == "__main__": 
    run_benchmark()
