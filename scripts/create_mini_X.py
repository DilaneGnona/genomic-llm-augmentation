import pandas as pd
import os

# CONFIG
FULL_X_PATH = "02_processed_data/pepper/X.csv"
MINI_X_PATH = "02_processed_data/pepper/X_mini.csv"
SYNTH_DATA_PATH = "04_augmentation/pepper/context_learning/synthetic_data_final.csv"

def create_mini_dataset():
    print("🚀 Création d'un dataset X réduit (Optimisé)...")
    
    # 1. Lire les colonnes nécessaires depuis le synthétique
    if not os.path.exists(SYNTH_DATA_PATH):
        print("❌ Pas de données synthétiques ! Lancez process_synthetics.py d'abord.")
        return

    df_synth = pd.read_csv(SYNTH_DATA_PATH)
    cols_needed = [c for c in df_synth.columns if c not in ['is_synthetic', 'Sample_ID', 'Yield_BV', 'dist_to_median']]
    print(f"   -> Colonnes ciblées : {len(cols_needed)} SNPs")
    
    # 2. Lire X.csv par morceaux (chunksize) pour ne pas exploser la RAM
    # On veut juste extraire ces colonnes pour TOUTES les lignes (ou un subset)
    # Pour un benchmark rapide, prenons TOUTES les lignes mais SEULEMENT les colonnes utiles.
    
    # On lit juste le header d'abord
    header_df = pd.read_csv(FULL_X_PATH, nrows=0)
    all_cols = header_df.columns.tolist()
    
    # On vérifie que les colonnes existent
    cols_to_load = ['Sample_ID'] + [c for c in cols_needed if c in all_cols]
    
    print(f"   -> Extraction de {len(cols_to_load)} colonnes depuis {FULL_X_PATH} (4.5GB)...")
    
    # Lecture optimisée
    df_mini = pd.read_csv(FULL_X_PATH, usecols=cols_to_load, low_memory=False)
    
    # Sauvegarde
    df_mini.to_csv(MINI_X_PATH, index=False)
    print(f"✅ Dataset réduit généré : {MINI_X_PATH}")
    print(f"   -> Dimensions : {df_mini.shape}")

if __name__ == "__main__":
    create_mini_dataset()