import os
import shutil

def clean_augmentation_and_modeling():
    print("--- DEEP CLEANING: 04_augmentation & 03_modeling_results ---")
    
    # Target directories
    AUG_DIR = "04_augmentation"
    MODEL_DIR = "03_modeling_results"
    
    # 1. Cleaning 04_augmentation
    if os.path.exists(AUG_DIR):
        print(f"Cleaning {AUG_DIR}...")
        # Keep only the base directories, but clear all files inside
        for dataset in ["pepper", "ipk_out_raw"]:
            ds_path = os.path.join(AUG_DIR, dataset)
            if os.path.exists(ds_path):
                # Completely wipe the dataset folder and recreate structure
                shutil.rmtree(ds_path)
                print(f"  Resetting {dataset} structure...")
                
                # Recreate standard structure
                os.makedirs(os.path.join(ds_path, "context_learning", "contexts"), exist_ok=True)
                os.makedirs(os.path.join(ds_path, "context_learning", "prompts"), exist_ok=True)
                os.makedirs(os.path.join(ds_path, "context_learning", "logs"), exist_ok=True)
                os.makedirs(os.path.join(ds_path, "model_sources"), exist_ok=True)
                
                # Pre-create model folders for known models
                for model in ["kimi", "glm5", "phi3", "deepseek"]:
                    for ctx in ["context_A", "context_B", "context_C", "context_D", "context_E"]:
                        os.makedirs(os.path.join(ds_path, "model_sources", model, ctx), exist_ok=True)

    # 2. Cleaning 03_modeling_results
    if os.path.exists(MODEL_DIR):
        print(f"Cleaning {MODEL_DIR}...")
        for f in os.listdir(MODEL_DIR):
            path = os.path.join(MODEL_DIR, f)
            # Keep the baselines we just ran if they are good, otherwise wipe all
            # Since the user wants to "efface les entrainements... avec les mauvais donnes", 
            # and our baselines were on 0-1-2, we keep 'baselines' but wipe others.
            if f != "baselines":
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                    else:
                        shutil.rmtree(path)
                    print(f"  Removed old modeling: {f}")
                except Exception as e:
                    print(f"  Error removing {f}: {e}")

    # 3. Clean root residues from previous augmentation tests
    root_residues = ["test_write.csv", "synthetic_snps.csv", "synthetic_y.csv", "run_output.txt"]
    for f in root_residues:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed root residue: {f}")

    print("\n--- RESTRUCTURING COMPLETE ---")
    print("New Hierarchy: 04_augmentation/{dataset}/model_sources/{model}/{context_type}/")

if __name__ == "__main__":
    clean_augmentation_and_modeling()
