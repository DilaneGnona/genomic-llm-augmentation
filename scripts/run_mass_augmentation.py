import subprocess
import time

def run_gen(dataset, model, samples):
    cmd = [
        "python", "scripts/generate_cl_automatic.py",
        "--dataset", dataset,
        "--model", model,
        "--samples", str(samples)
    ]
    print(f"\n>>> STARTING GENERATION: {model} on {dataset} ({samples} samples) <<<")
    try:
        subprocess.run(cmd, check=True)
        print(f">>> SUCCESS: {model} on {dataset} <<<")
    except Exception as e:
        print(f">>> ERROR: {model} failed - {e} <<<")

def main():
    dataset = "pepper"
    samples = 200
    
    # Models to use for the full run, prioritizing DeepSeek and Kimi as requested
    all_models = ["deepseek", "kimi", "gpt", "llama3", "phi3"]
    
    for model in all_models:
        run_gen(dataset, model, samples)
        # Larger delay between models for stability
        print("Waiting 10 seconds before next model...")
        time.sleep(10)

if __name__ == "__main__":
    main()
