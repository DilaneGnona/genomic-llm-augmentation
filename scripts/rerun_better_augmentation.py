import subprocess
import time

def run_gen(dataset, model, samples):
    cmd = [
        "python", "scripts/generate_cl_automatic.py",
        "--dataset", dataset,
        "--model", model,
        "--samples", str(samples)
    ]
    print(f"\n>>> RE-GENERATING WITH MORE SNPS: {model} on {dataset} ({samples} samples) <<<")
    try:
        subprocess.run(cmd, check=True)
        print(f">>> SUCCESS: {model} <<<")
    except Exception as e:
        print(f">>> ERROR: {model} failed - {e} <<<")

def main():
    dataset = "ipk_out_raw"
    samples = 20
    models = ["deepseek", "gpt", "llama3", "phi3"]
    
    for model in models:
        run_gen(dataset, model, samples)
        time.sleep(2)

if __name__ == "__main__":
    main()
