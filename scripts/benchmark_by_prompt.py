import os 
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score 
import glob 

# --- CONFIGURATION --- 
BASE_PATH = "04_augmentation/pepper/context_learning/" 
# Attention : les fichiers bruts sont dans outputs/llama3/seed_X/
# INPUT_DIR pointe vers la racine des outputs, il faudra chercher récursivement
INPUT_DIR = os.path.join(BASE_PATH, "outputs") 

# Chemins vers les données réelles (Optimisé)
REAL_DATA_X_MINI = "02_processed_data/pepper/X_mini.csv"
REAL_DATA_X_FULL = "02_processed_data/pepper/X.csv"
REAL_DATA_Y = "02_processed_data/pepper/y.csv"

# Liste des types de prompts (d'après vos noms de fichiers) 
PROMPT_TYPES = { 
    "A_statistical": "prompt_A", 
    "B_genetic_structure": "prompt_B", 
    "C_prediction_utility": "prompt_C", 
    "D_baseline": "prompt_D", 
    "E_flexible": "prompt_E" 
} 

def get_real_data(synth_cols): 
    # Charge et aligne les vraies données 
    print("   -> Chargement des données réelles...")
    
    # 1. Chargement Y
    y = pd.read_csv(REAL_DATA_Y)
    y['Sample_ID'] = y['Sample_ID'].astype(str)
    
    # 2. Chargement X (Priorité au Mini)
    if os.path.exists(REAL_DATA_X_MINI):
        X = pd.read_csv(REAL_DATA_X_MINI, header=0, low_memory=False)
    else:
        # Fallback (plus lent)
        cols_to_load = ['Sample_ID'] + [c for c in synth_cols if c not in ['is_synthetic', 'Sample_ID', 'Yield_BV']]
        try:
             X = pd.read_csv(REAL_DATA_X_FULL, header=0, usecols=lambda c: c in cols_to_load, low_memory=False)
        except:
             X = pd.read_csv(REAL_DATA_X_FULL, header=0, low_memory=False)
        X = X.iloc[3:].copy() # Skip metadata si full
    
    X['Sample_ID'] = X['Sample_ID'].astype(str)
    
    # 3. Fusion
    df_real = pd.merge(X, y[['Sample_ID', 'Yield_BV']], on='Sample_ID', how='inner')
    
    # 4. Alignement
    cols_to_use = [c for c in synth_cols if c not in ['is_synthetic', 'Sample_ID']]
    # Assurons-nous que Yield_BV est inclus s'il n'est pas dans cols_to_use (il ne devrait pas l'être car c'est la target)
    # Mais on veut retourner X + y
    
    # On vérifie que toutes les cols sont là
    missing = [c for c in cols_to_use if c not in df_real.columns]
    if missing:
        print(f"   ⚠️ Colonnes manquantes dans réel : {missing}")
        # On ne peut pas inventer les données manquantes pour l'entraînement réel
        # Si c'est juste dist_to_median, on l'ignore
        cols_to_use = [c for c in cols_to_use if c in df_real.columns]
        
    final_df = df_real[cols_to_use + ['Yield_BV']].copy()
    
    # Dédoublonnage des colonnes (sécurité)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    # Conversion numeric
    # On utilise apply pour être plus robuste
    cols_numeric = [c for c in final_df.columns if c != 'Sample_ID']
    for c in cols_numeric:
        # Check if duplicated, if so, keep first
        if isinstance(final_df[c], pd.DataFrame):
            final_df[c] = final_df[c].iloc[:, 0]
            
        final_df[c] = pd.to_numeric(final_df[c], errors='coerce')
        
    final_df = final_df.dropna()
    
    return final_df

def train_and_score(name, df_synth, df_real): 
    # Préparation 
    # On prend l'intersection des colonnes (au cas où le synthétique a des extras)
    common_cols = [c for c in df_real.columns if c != 'Yield_BV']
    
    # Train/Test sur le RÉEL 
    X = df_real[common_cols] 
    y = df_real['Yield_BV'] 
    
    # Split
    X_train_real, X_test, y_train_real, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    
    # Ajout du Synthétique au Train 
    # Il faut s'assurer que df_synth a bien ces colonnes
    try:
        X_synth = df_synth[common_cols]
        y_synth = df_synth['Yield_BV']
    except KeyError as e:
        print(f"   ❌ Erreur colonnes synthétiques : {e}")
        return 999, -999, 0
    
    X_train_aug = pd.concat([X_train_real, X_synth]) 
    y_train_aug = pd.concat([y_train_real, y_synth]) 
    
    # Entraînement RF 
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) 
    rf.fit(X_train_aug, y_train_aug) 
    
    # Evaluation 
    preds = rf.predict(X_test) 
    rmse = np.sqrt(mean_squared_error(y_test, preds)) 
    r2 = r2_score(y_test, preds) 
    
    return rmse, r2, len(df_synth) 

def run_comparison(): 
    print("📊 DÉMARRAGE DU BENCHMARK PAR PROMPT...") 
    
    # 1. Charger tous les fichiers récursivement
    all_files = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))
                
    print(f"    -> {len(all_files)} fichiers trouvés.") 
    
    results = [] 
    
    # 2. Pour chaque type de prompt 
    for nice_name, pattern in PROMPT_TYPES.items(): 
        print(f"\n🔎 Analyse du Prompt : {nice_name}...") 
        
        # Trouver les fichiers correspondants 
        prompt_files = [f for f in all_files if pattern in os.path.basename(f)] 
        
        if not prompt_files: 
            print("    ⚠️ Aucune donnée trouvée.") 
            continue 
            
        # Fusionner et Nettoyer 
        dfs = [] 
        for f in prompt_files: 
            try: 
                df = pd.read_csv(f) 
                # Nettoyage rapide 
                if 'Yield_BV' in df.columns: 
                    # Conversion numeric pour tout sauf Sample_ID
                    cols_numeric = [c for c in df.columns if c != 'Sample_ID']
                    for c in cols_numeric: 
                        df[c] = pd.to_numeric(df[c], errors='coerce') 
                    
                    df = df.dropna(subset=['Yield_BV']) 
                    
                    # On ne garde que les colonnes intéressantes (SNPs)
                    # Filtre basique : valeurs 0, 1, 2 pour les SNPs
                    # On suppose que tout ce qui n'est pas Yield/SampleID est un SNP
                    snp_cols = [c for c in df.columns if c not in ['Yield_BV', 'Sample_ID', 'is_synthetic', 'dist_to_median']]
                    
                    # Nettoyage NaN
                    df = df.dropna()
                    
                    # Round SNPs
                    for c in snp_cols:
                        df[c] = df[c].round()
                    
                    if len(df) > 0: dfs.append(df) 
            except: pass 
            
        if not dfs: 
            print("    ⚠️ Données vides après nettoyage.") 
            continue 
            
        df_prompt_synth = pd.concat(dfs, ignore_index=True) 
        print(f"    ✅ {len(df_prompt_synth)} échantillons valides générés.") 
        
        # Lancer le Benchmark 
        try: 
            df_real = get_real_data(df_prompt_synth.columns) 
            rmse, r2, n_samples = train_and_score(nice_name, df_prompt_synth, df_real) 
            
            results.append({ 
                "Context (Prompt)": nice_name, 
                "N_Samples": n_samples, 
                "RMSE (Error)": round(rmse, 6), 
                "R2 Score": round(r2, 4) 
            }) 
            print(f"    -> Score : RMSE={rmse:.5f}") 
        except Exception as e: 
            print(f"    ❌ Erreur Benchmark : {e}") 
            import traceback
            traceback.print_exc()

    # 3. BASELINE (Référence sans synthétique) 
    # On la calcule pour de vrai pour être sûr (Dataset Real Only)
    try:
        # On charge un dataset réel standard (basé sur le dernier df_real valide ou rechargé)
        # Pour faire simple, on utilise le Mini directement
        print("\n🔎 Calcul Baseline...")
        df_base = get_real_data(['Sample_ID', 'Yield_BV'] + [c for c in pd.read_csv(REAL_DATA_X_MINI, nrows=0).columns if c != 'Sample_ID'])
        
        # Train/Test Real Only
        X = df_base.drop(columns=['Yield_BV'])
        y = df_base['Yield_BV']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        base_rmse = np.sqrt(mean_squared_error(y_test, preds))
        base_r2 = r2_score(y_test, preds)
        
        results.append({ 
            "Context (Prompt)": "BASELINE (Real Only)", 
            "N_Samples": 0, 
            "RMSE (Error)": round(base_rmse, 6), 
            "R2 Score": round(base_r2, 4) 
        })
    except Exception as e:
        print(f"Erreur Baseline: {e}")

    # 4. Affichage du Tableau Final 
    if results:
        results_df = pd.DataFrame(results).sort_values("RMSE (Error)", ascending=True) 
        
        print("\n\n🏆 --- RÉSULTAT FINAL POUR LE SUPERVISEUR --- 🏆") 
        print(results_df.to_string(index=False)) 
        print("\nCopiez ce tableau dans votre présentation !") 
        
        # Sauvegarde CSV
        results_df.to_csv(os.path.join(BASE_PATH, "benchmark_by_prompt_results.csv"), index=False)
    else:
        print("\n❌ Aucun résultat à afficher.")

if __name__ == "__main__": 
    run_comparison()
