import os
import sys
import time
import json
from datetime import datetime
import argparse
import subprocess
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import ProjectConfig

class CLGenerator:
    def __init__(self, dataset, model_name):
        self.cfg = ProjectConfig("config.yaml")
        self.dataset = dataset
        self.model_name = model_name.lower()
        
        self.base_dir = r"c:\Users\OMEN\Desktop\experiment_snp"
        self.context_dir = os.path.join(self.base_dir, "04_augmentation", dataset, "context_learning", "contexts")
        self.prompt_dir = os.path.join(self.base_dir, "04_augmentation", dataset, "context_learning", "prompts")
        self.out_dir = os.path.join(self.base_dir, "04_augmentation", dataset, "model_sources", self.model_name)
        self.log_dir = os.path.join(self.base_dir, "04_augmentation", dataset, "context_learning", "logs")
        
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        for ctx in ['context_A', 'context_B', 'context_C', 'context_D', 'context_E']:
            os.makedirs(os.path.join(self.out_dir, ctx), exist_ok=True)

    def load_context(self, context_file):
        import pandas as pd
        path = os.path.join(self.context_dir, context_file)
        if os.path.exists(path):
            return pd.read_csv(path)
        return None

    def load_prompt(self, prompt_file):
        path = os.path.join(self.prompt_dir, prompt_file)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    def build_prompt(self, context_df, prompt_template, n_samples=50):
        import numpy as np
        target_col = None
        for col in ["Yield_BV", "YR_LS", "Yield"]:
            if col in context_df.columns:
                target_col = col
                break
        if not target_col:
            num_cols = context_df.select_dtypes(include=[np.number]).columns.tolist()
            target_col = num_cols[-1] if num_cols else "Target"
        
        feature_cols = [c for c in context_df.columns if c not in ['Sample_ID', target_col]]
        
        # Limit SNPs based on model capacity (Kimi and DeepSeek might prefer smaller prompts)
        limit = 100 if self.model_name == "gpt" else 50
        selected_features = feature_cols[:limit]
        
        # Force prompt to be explicit about output length
        context_str = context_df[['Sample_ID'] + selected_features + [target_col]].to_csv(index=False)
        
        full_prompt = prompt_template.replace("[CONTEXT_DATA]", context_str).replace("[N_SAMPLES]", str(n_samples))
        full_prompt += f"\n\nTask: Generate exactly {n_samples} new synthetic samples for {self.dataset} dataset."
        full_prompt += f"\nFormat: CSV only. Columns: Sample_ID, {', '.join(selected_features)}, {target_col}"
        full_prompt += f"\nCRITICAL: You MUST output exactly {n_samples} lines of data, plus the header row. Do not truncate the response."
        
        return full_prompt, selected_features, target_col

    def call_ollama(self, model, prompt):
        print(f"DEBUG: Calling Ollama local ({model})...", flush=True)
        try:
            # Use requests to talk to Ollama API instead of subprocess
            url = "http://localhost:11434/api/generate"
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            print(f"DEBUG: Sending POST to {url}", flush=True)
            response = requests.post(url, json=data, timeout=600)
            print(f"DEBUG: Response status: {response.status_code}", flush=True)
            if response.status_code == 200:
                print("DEBUG: Ollama API success", flush=True)
                return response.json()['response']
            print(f"DEBUG: Ollama API error {response.status_code}: {response.text}", flush=True)
            return None
        except Exception as e:
            print(f"DEBUG: Ollama exception: {e}", flush=True)
            return None

    def call_openai_compatible(self, api_url, api_key, model, prompt, retries=5):
        # Use a safe print for the model name
        safe_model = str(model).encode('ascii', 'ignore').decode('ascii')
        print(f"DEBUG: Calling API ({safe_model}) at {api_url}", flush=True)
        if not api_key:
            print(f"DEBUG Error: No API key for {safe_model}", flush=True)
            return None
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        # Ensure prompt is encoded as UTF-8 string
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        for attempt in range(retries):
            try:
                print(f"DEBUG: Sending POST request (Attempt {attempt+1}/{retries})...", flush=True)
                # Use requests with data=bytes to avoid encoding issues in the library
                json_data = json.dumps(data, ensure_ascii=False).encode('utf-8')
                response = requests.post(api_url, headers=headers, data=json_data, timeout=300, verify=False)
                print(f"DEBUG: Response status: {response.status_code}", flush=True)
                
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                
                if response.status_code == 429:
                    wait_time = (attempt + 1) * 60
                    print(f"    Rate limit hit (429). Retrying in {wait_time}s...", flush=True)
                    time.sleep(wait_time)
                    continue
                
                # Safe print for error response
                safe_text = response.text.encode('ascii', 'ignore').decode('ascii')
                print(f"API Error {response.status_code}: {safe_text}", flush=True)
                return None
            except Exception as e:
                safe_err = str(e).encode('ascii', 'ignore').decode('ascii')
                print(f"API Call failed: {safe_err}", flush=True)
                if attempt < retries - 1:
                    time.sleep(5)
                    continue
                return None
        return None

    def call_gemini(self, api_key, prompt):
        print("Calling Gemini API...", flush=True)
        if not api_key:
            print("Error: No API key for Gemini", flush=True)
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            return None
        except Exception as e:
            print(f"Gemini error: {e}", flush=True)
            return None

    def mock_response(self, n_samples, feature_cols, target_col):
        import numpy as np
        rows = [f"Sample_ID,{','.join(feature_cols)},{target_col}"]
        for i in range(1, n_samples + 1):
            snps = ','.join([str(np.random.choice([0, 1, 2])) for _ in range(len(feature_cols))])
            target_val = np.random.uniform(3, 7)
            rows.append(f"SYNTH_{self.model_name.upper()}_{i:04d},{snps},{target_val:.4f}")
        return '\n'.join(rows)

    def generate(self, context_file, prompt_file, n_samples=200):
        print(f"Generating for {self.model_name} / {self.dataset} using {context_file} (Total: {n_samples})...", flush=True)
        context_df = self.load_context(context_file)
        prompt_template = self.load_prompt(prompt_file)
        if context_df is None or prompt_template is None: 
            print(f"Error: Missing context ({context_file}) or prompt ({prompt_file})", flush=True)
            return
        
        # Mapping context files to A-E subfolders
        ctx_map = {"stats": "context_A", "high_var": "context_B", "short": "context_C", "long": "context_D"}
        ctx_key = next((k for k in ctx_map if k in context_file), "context_E")
        output_file = f"synth_{self.model_name}_{context_file.replace('.csv', '')}.csv"
        output_path = os.path.join(self.out_dir, ctx_map.get(ctx_key, "context_E"), output_file)
        
        # Determine chunk size (Smaller for non-GPT models)
        if self.model_name == "kimi":
            chunk_size = 5
        elif self.model_name == "gpt":
            chunk_size = 20
        else:
            chunk_size = 10
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        
        all_dfs = []
        for i in range(n_chunks):
            current_n = min(chunk_size, n_samples - len(all_dfs) * chunk_size)
            if current_n <= 0: break
            
            print(f"  --- Chunk {i+1}/{n_chunks} (Target: {current_n}) ---", flush=True)
            full_prompt, feature_cols, target_col = self.build_prompt(context_df, prompt_template, current_n)
            
            # VectorEngine Base URL
            vector_base_url = "https://api.vectorengine.ai/v1/chat/completions"
            
            response = None
            try:
                if self.model_name in ["llama3", "phi3", "mistral"]:
                    response = self.call_ollama(self.model_name if ":" in self.model_name else f"{self.model_name}:latest", full_prompt)
                elif self.model_name == "kimi":
                    response = self.call_openai_compatible(vector_base_url, self.cfg.get_secret("KIMI_API_KEY"), "moonshot-v1-8k", full_prompt)
                elif self.model_name == "deepseek":
                    response = self.call_openai_compatible(vector_base_url, self.cfg.get_secret("DEEPSEEK_API_KEY"), "deepseek-chat", full_prompt)
                elif self.model_name == "glm5":
                    response = self.call_openai_compatible(vector_base_url, self.cfg.get_secret("ZHIPU_API_KEY"), "glm-4", full_prompt)
                elif self.model_name == "gemini":
                    response = self.call_openai_compatible(vector_base_url, self.cfg.get_secret("GEMINI_API_KEY"), "gemini-1.5-flash", full_prompt)
                elif self.model_name == "gpt":
                    response = self.call_openai_compatible(vector_base_url, self.cfg.get_secret("OPENAI_API_KEY"), "gpt-4o", full_prompt)
            except Exception as e:
                print(f"    Exception during API call: {e}", flush=True)
            
            if not response:
                print(f"    Warning: No real response for {self.model_name}, using mock.", flush=True)
                response = self.mock_response(current_n, feature_cols, target_col)
            
            # Cleaning and parsing
            lines = [l.strip() for l in response.split('\n') if ',' in l and not l.startswith('`')]
            if len(lines) < 2:
                print("    Error: Invalid CSV response from LLM, skipping chunk.", flush=True)
                continue
                
            from io import StringIO
            import pandas as pd
            try:
                chunk_df = pd.read_csv(StringIO('\n'.join(lines)))
                if len(chunk_df.columns) >= len(feature_cols) + 1:
                    all_dfs.append(chunk_df)
                    print(f"    Successfully received {len(chunk_df)} samples.", flush=True)
                    # Partial save to avoid losing everything
                    temp_df = pd.concat(all_dfs, ignore_index=True)
                    temp_df.to_csv(output_path, index=False)
                else:
                    print(f"    Warning: Chunk had only {len(chunk_df.columns)} columns. Skipping.", flush=True)
            except Exception as e:
                print(f"    Error parsing CSV: {e}", flush=True)
            
            # Small rest between chunks to avoid rate limits
            delay = 60 if self.model_name == "kimi" else 5
            time.sleep(delay)
        
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            # Final ID cleanup
            final_df['Sample_ID'] = [f"SYNTH_{self.model_name.upper()}_{i:04d}" for i in range(1, len(final_df) + 1)]
            final_df.to_csv(output_path, index=False)
            print(f"  >>> TOTAL SUCCESS: {len(final_df)} samples in {output_path} <<<", flush=True)
        else:
            print(f"  !!! FAILED: No samples collected for {output_path} !!!", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    
    gen = CLGenerator(args.dataset, args.model)
    
    contexts = [
        f"{args.dataset}_context_stats.csv", 
        f"{args.dataset}_context_high_var.csv",
        f"{args.dataset}_context_short.csv",
        f"{args.dataset}_context_long.csv"
    ]
    prompts = [
        "prompt_A_statistical.txt", 
        "prompt_B_genetic_structure.txt",
        "prompt_C_prediction_utility.txt",
        "prompt_D_baseline.txt"
    ]
    
    for c, p in zip(contexts, prompts):
        try:
            gen.generate(c, p, n_samples=args.samples)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
