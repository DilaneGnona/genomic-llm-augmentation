import os
import shutil
import glob

def clean_workspace():
    print("--- STARTING WORKSPACE CLEANUP ---")
    
    # 1. Root level files to delete
    root_files_to_delete = [
        "cl_output.txt", "prep_output.txt", "fix_output.txt", "benchmark_output.txt",
        "pipeline_output.log", "glm46_log.txt", "rds_check.csv", "rds_sample.txt",
        "test_int.csv", "simple_test.csv", "complete_tree.txt", "debug_types.txt",
        "test_simple.txt", "simple_test.py", "check_data_format.py", "check_packages.py",
        "check_y_data.py", "create_small_cleaned_sample.py", "create_synthetic_cleaned_data.py",
        "debug_data.py", "diagnose_snp_data.py", "dump_prompt.py", "fast_x_load_test.py",
        "find_sigma_values.py", "fix_all_snp_columns.py", "fix_data.py", "fix_data_v2.py",
        "fix_generate_script.py", "fix_pepper_data.py", "fix_snp_data_memory_efficient.py",
        "fix_snp_format.py", "monitor_pipeline.py", "simple_x_load.py", "test_data_loading.py",
        "test_model_registry.py", "test_ollama_large.py", "preprocessing_verification_report.md",
        "test_simple.txt", "rds_sample.txt", "rds_check.csv", "simple_test.csv", "test_int.csv",
        "debug_types.txt", "test_simple.txt", "test_env.py", "test_pyreadr.py", "test_simple.txt",
        "run_output.txt", "run_output_v2.txt", "run_output_v3.txt", "run_output_v4.txt",
        "run_output_v5.txt", "run_output_v6.txt", "run_output_v7.txt", "run_output_v8.txt",
        "run_output_debug.txt", "test_int.csv", "simple_test.csv"
    ]
    
    for f in root_files_to_delete:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"Removed root file: {f}")
            except Exception as e:
                print(f"Error removing {f}: {e}")

    # 2. Scripts to clean (keep only essential ones)
    essential_scripts = [
        "unified_modeling_pipeline.py", "model_registry.py", "prepare_contexts.py",
        "generate_cl_automatic.py", "verify_cl_data.py", "clean_pepper_outputs.py",
        "verify_preprocessed_format.py", "clean_workspace.py"
    ]
    
    script_patterns_to_delete = [
        "scripts/fix_pepper_*.py", "scripts/verify_cl_data_v*.py",
        "scripts/test_*.py", "scripts/simple_test.py", "scripts/train_dl_*.py",
        "scripts/train_mlp_*.py", "scripts/generate_pepper_3000_*.py",
        "scripts/ml_pipeline_*.py", "scripts/train_ae_*.py", "scripts/train_cnn_*.py"
    ]
    
    for pattern in script_patterns_to_delete:
        for f in glob.glob(pattern):
            if os.path.basename(f) not in essential_scripts:
                try:
                    os.remove(f)
                    print(f"Removed redundant script: {f}")
                except Exception as e:
                    print(f"Error removing script {f}: {e}")

    # 3. Clean 02_processed_data/pepper (keep only X.csv, y.csv, and manifests)
    pepper_dir = "02_processed_data/pepper"
    if os.path.exists(pepper_dir):
        keep_in_pepper = ["X.csv", "y.csv", "variant_manifest.csv", "sample_map.csv", "pca_covariates.csv", "qc_report.txt"]
        for f in os.listdir(pepper_dir):
            if f not in keep_in_pepper:
                path = os.path.join(pepper_dir, f)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                        print(f"Removed pepper data: {f}")
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                        print(f"Removed pepper subdir: {f}")
                except Exception as e:
                    print(f"Error removing {path}: {e}")

    # 4. Clean 04_augmentation/pepper
    aug_pepper_dir = "04_augmentation/pepper"
    if os.path.exists(aug_pepper_dir):
        keep_in_aug = ["context_learning", "model_sources"]
        for f in os.listdir(aug_pepper_dir):
            if f not in keep_in_aug:
                path = os.path.join(aug_pepper_dir, f)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                        print(f"Removed aug file: {f}")
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                        print(f"Removed aug subdir: {f}")
                except Exception as e:
                    print(f"Error removing {path}: {e}")

    # 5. Clean 03_modeling_results (KEEP NOTHING OLD)
    modeling_dir = "03_modeling_results"
    if os.path.exists(modeling_dir):
        try:
            # Use shell command for forced removal on Windows if shutil fails
            if os.name == 'nt':
                import subprocess
                subprocess.run(['rmdir', '/s', '/q', modeling_dir.replace('/', '\\')], shell=True)
            else:
                shutil.rmtree(modeling_dir)
            os.makedirs(modeling_dir)
            print(f"Cleaned modeling result: {modeling_dir}")
        except Exception as e:
            print(f"Error cleaning modeling {modeling_dir}: {e}")

    # 6. Final directory restructuring (ensure all exist)
    dirs_to_ensure = [
        "01_raw_data", "02_processed_data", "03_modeling_results/baselines",
        "04_augmentation/pepper/model_sources", "logs", "scripts", "src", "tests", "docs"
    ]
    for d in dirs_to_ensure:
        os.makedirs(d, exist_ok=True)

    print("\n--- CLEANUP COMPLETE ---")

if __name__ == "__main__":
    clean_workspace()
