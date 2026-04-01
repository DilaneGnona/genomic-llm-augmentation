import sys
import os
import json

# Add project root to path
sys.path.append(os.getcwd())

from src.training import run_baseline

def main():
    datasets = ["ipk_out_raw", "pepper"]
    models = ["ridge", "xgboost", "mlp"]
    
    overall_results = {}
    
    for ds in datasets:
        print(f"\n>>> TRAINING BASELINE FOR {ds} <<<")
        ds_results = {}
        for m in models:
            try:
                print(f"Running {m}...")
                res = run_baseline(ds, m)
                ds_results[m] = res
                print(f"  Result: R2={res['r2']:.4f}")
            except Exception as e:
                import traceback
                print(f"  Error training {m} on {ds}: {e}")
                traceback.print_exc()
        overall_results[ds] = ds_results

    # Final summary
    summary_path = os.path.join("03_modeling_results", "baselines", "baseline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(overall_results, f, indent=2)
    
    print(f"\n--- ALL BASELINES COMPLETED ---")
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
