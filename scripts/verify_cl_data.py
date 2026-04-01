import os
import pandas as pd
import numpy as np
import json

BASE_DIR = r"c:\Users\OMEN\Desktop\experiment_snp"
AUGMENT_DIR = os.path.join(BASE_DIR, "04_augmentation", "pepper", "model_sources")
MODELS = ["kimi", "glm5", "phi3"]
CONTEXTS = ["context_A", "context_B", "context_C", "context_D", "context_E"]

def verify_cl_data():
    results = {}
    for model in MODELS:
        model_dir = os.path.join(AUGMENT_DIR, model)
        if not os.path.exists(model_dir):
            results[model] = "Directory missing"
            continue
        
        model_results = {}
        for ctx in CONTEXTS:
            ctx_dir = os.path.join(model_dir, ctx)
            if not os.path.exists(ctx_dir):
                model_results[ctx] = "Context directory missing"
                continue
            
            # Find the main 500 samples file
            files = [f for f in os.listdir(ctx_dir) if "500samples.csv" in f]
            if not files:
                model_results[ctx] = "Data file missing"
                continue
            
            filepath = os.path.join(ctx_dir, files[0])
            try:
                df = pd.read_csv(filepath)
                
                # Check basic stats
                n_samples = len(df)
                n_cols = len(df.columns)
                sample_id_ok = "Sample_ID" in df.columns
                yield_ok = "Yield_BV" in df.columns
                
                # Check SNP types (are they 0,1,2 or float?)
                snp_cols = [c for c in df.columns if c not in ["Sample_ID", "Yield_BV"]]
                is_integer = True
                if snp_cols:
                    # Check first 10 rows/cols for speed
                    sample_data = df[snp_cols].head(10)
                    is_integer = all(pd.api.types.is_integer_dtype(sample_data[c]) for c in sample_data.columns)
                    
                model_results[ctx] = {
                    "file": files[0],
                    "samples": n_samples,
                    "features": len(snp_cols),
                    "sample_id_present": sample_id_ok,
                    "yield_present": yield_ok,
                    "is_snp_integer": is_integer,
                    "yield_range": [float(df["Yield_BV"].min()), float(df["Yield_BV"].max())] if yield_ok else None
                }
            except Exception as e:
                model_results[ctx] = f"Error reading file: {str(e)}"
        
        results[model] = model_results
    
    return results

if __name__ == "__main__":
    report = verify_cl_data()
    print(json.dumps(report, indent=2))
    with open(os.path.join(BASE_DIR, "03_modeling_results", "cl_verification_report.json"), "w") as f:
        json.dump(report, f, indent=2)
