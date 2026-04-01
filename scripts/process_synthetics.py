import os 
import pandas as pd 
import glob 

# --- CONFIGURATION --- 
BASE_PATH = "04_augmentation/pepper/context_learning/" 
INPUT_DIR = os.path.join(BASE_PATH, "outputs") 
OUTPUT_FILE = os.path.join(BASE_PATH, "synthetic_data_final.csv") 

def process_synthetics(): 
    print("🕵️‍♂️  Démarrage de l'inspection des données...") 
    
    all_files = [] 
    # On cherche tous les fichiers CSV dans les sous-dossiers (seed_*) 
    for root, dirs, files in os.walk(INPUT_DIR): 
        for file in files: 
            if file.endswith(".csv"): 
                all_files.append(os.path.join(root, file)) 
    
    print(f"    -> {len(all_files)} fichiers bruts trouvés.") 
    
    valid_dfs = [] 
    total_rows_raw = 0 
    
    for f in all_files: 
        try: 
            # 1. Chargement 
            df = pd.read_csv(f) 
            total_rows_raw += len(df) 
            
            # 2. Vérification de structure 
            if 'Yield_BV' not in df.columns: 
                print(f"    ⚠️ Ignoré (Pas de Yield): {os.path.basename(f)}") 
                continue 
                
            # 3. Séparation SNPs / Yield 
            # On suppose que la dernière colonne est Yield, le reste des SNPs 
            snp_cols = [c for c in df.columns if c != 'Yield_BV' and c != 'Sample_ID'] 
            
            # 4. Nettoyage des valeurs SNP (Doit être 0, 1, 2) 
            # On force en numérique, les erreurs deviennent NaN 
            for col in snp_cols: 
                df[col] = pd.to_numeric(df[col], errors='coerce') 
            
            # On ne garde que les lignes complètes (pas de NaN) 
            df = df.dropna() 
            
            # On filtre pour être sûr d'avoir 0, 1 ou 2 (arrondi à l'entier le plus proche) 
            # Cela corrige si l'IA a mis "1.0" ou "0.99" 
            for col in snp_cols: 
                df[col] = df[col].round().astype(int) 
                # On vire les valeurs hors limites (ex: 3, -1) 
                df = df[df[col].isin([0, 1, 2])] 
                
            # 5. Nettoyage Yield 
            df['Yield_BV'] = pd.to_numeric(df['Yield_BV'], errors='coerce') 
            df = df.dropna(subset=['Yield_BV']) 
            
            if len(df) > 0: 
                valid_dfs.append(df) 
            else: 
                print(f"    ⚠️ Ignoré (Vide après nettoyage): {os.path.basename(f)}") 

        except Exception as e: 
            print(f"    ❌ Erreur lecture {os.path.basename(f)}: {e}") 

    # --- FUSION --- 
    if valid_dfs: 
        final_df = pd.concat(valid_dfs, ignore_index=True) 
        
        # Petit bonus : On ajoute une colonne pour dire "C'est de la data synthétique" 
        final_df['is_synthetic'] = 1 
        
        print("\n📊 BILAN DU NETTOYAGE :") 
        print(f"   - Lignes brutes trouvées : {total_rows_raw}") 
        print(f"   - Lignes valides conservées : {len(final_df)}") 
        print(f"   - Nombre de colonnes (SNPs) : {len(final_df.columns) - 2}") # -2 pour Yield et is_synth 
        
        # Sauvegarde 
        final_df.to_csv(OUTPUT_FILE, index=False) 
        print(f"\n✅ Fichier final généré : {OUTPUT_FILE}") 
        print("   (Prêt pour l'entraînement !)") 
        
    else: 
        print("\n❌ Aucun dataframe valide n'a pu être reconstitué.") 

if __name__ == "__main__": 
    process_synthetics()
