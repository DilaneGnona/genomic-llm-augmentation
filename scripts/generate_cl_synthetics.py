import os 
import pandas as pd 
import json 
from datetime import datetime 
import subprocess 
import io 
import time 
import re 

# --- CONFIGURATION STRICTE --- 
LLMS = ["llama3"]  # Focus sur Llama 3 

# On garde un nombre de seeds conséquent pour avoir de la masse, mais on peut réduire si on veut tester vite
# Pour un vrai test comparatif, gardons la même rigueur que pour Llama3
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 
         51, 52, 53, 54, 55, 56, 57, 58, 59, 60] # 19 seeds (environ 19000 échantillons théoriques)

N_SAMPLES = 200 
BASE_PATH = "04_augmentation/pepper/context_learning/" 

PROMPT_CONTEXT_MAP = { 
    "prompt_A_statistical.txt":      "pepper_context_stats.csv", 
    "prompt_B_genetic_structure.txt":"pepper_context_high_var.csv", 
    "prompt_C_prediction_utility.txt":"pepper_context_short.csv", 
    "prompt_D_baseline.txt":         "pepper_context_short.csv", 
    "prompt_E_flexible.txt":         "pepper_context_long.csv" 
} 

# --- FONCTIONS --- 

def load_context_reduced(context_name): 
    """Charge le contexte et ne garde STRICTEMENT que 15 SNPs max""" 
    path = os.path.join(BASE_PATH, "contexts", context_name) 
    if not os.path.exists(path): 
        raise FileNotFoundError(f"Context file not found: {path}") 
    
    df = pd.read_csv(path) 
    
    # FORCE BRUTE : On garde la 1ère colonne (ID), les colonnes 1 à 16 (15 SNPs), et la dernière (Yield) 
    # Peu importe combien il y en a au départ, on coupe tout ce qui dépasse. 
    if len(df.columns) > 17: 
        cols_to_keep = [df.columns[0]] + list(df.columns[1:16]) + [df.columns[-1]] 
        df = df[cols_to_keep] 
    
    return df 

def load_prompt(prompt_name): 
    path = os.path.join(BASE_PATH, "prompts", prompt_name) 
    if not os.path.exists(path): 
        raise FileNotFoundError(f"Prompt file not found: {path}") 
    with open(path, "r", encoding="utf-8") as f: 
        prompt = f.read().replace("[N_SAMPLES]", str(N_SAMPLES)) 
    return prompt 

def generate_with_ollama_subprocess(model, prompt, seed): 
    try: 
        cmd = ['ollama', 'run', model] 
        # Timeout réduit à 5 min car avec 15 SNPs ça doit aller vite 
        p = subprocess.run( 
            cmd, 
            input=prompt, 
            capture_output=True, 
            encoding='utf-8', 
            timeout=300 
        ) 
        if p.returncode == 0: 
            return p.stdout 
        else: 
            raise RuntimeError(f"Ollama CLI Error: {p.stderr}") 
    except Exception as e: 
        raise RuntimeError(f"Ollama Call Failed: {str(e)}") 

def parse_llm_csv_aggressive(text): 
    """Parsing nettoyeur agressif""" 
    # 1. Nettoyage caractères 
    text = text.replace('"', '').replace("'", "") 
    
    lines = text.split('\n') 
    csv_lines = [] 
    
    for line in lines: 
        clean_line = line.strip() 
        # Filtres anti-bavardage 
        if clean_line == '' or '...' in clean_line: continue 
        if clean_line.startswith('Here') or clean_line.startswith('REFERENCE') or clean_line.startswith('Alright'): continue 
        if clean_line.startswith('1.') or clean_line.startswith('2.'): continue # Listes numérotées 
        
        # Doit contenir au moins 2 virgules pour être une ligne de données 
        if clean_line.count(',') < 2: continue 
        
        csv_lines.append(clean_line) 

    csv_text = '\n'.join(csv_lines).replace('```csv', '').replace('```', '') 
    
    if not csv_text.strip(): 
        raise ValueError("Aucune donnée CSV valide trouvée après filtrage.") 

    try: 
        df = pd.read_csv(io.StringIO(csv_text)) 
    except: 
        # Mode permissif 
        df = pd.read_csv(io.StringIO(csv_text), engine='python', on_bad_lines='skip') 

    return df 

# --- MAIN --- 

if __name__ == "__main__": 
    manifest = [] 
    generation_logs = [] 
    run_id = 1 
    
    os.makedirs(os.path.join(BASE_PATH, "logs"), exist_ok=True) 
    os.makedirs(os.path.join(BASE_PATH, "outputs"), exist_ok=True) 

    print(f"🚀 Démarrage OPTIMISÉ (15 SNPs max).") 

    for llm in LLMS: 
        for seed in SEEDS: 
            os.environ["PYTHONHASHSEED"] = str(seed) 

            for prompt_name, context_name in PROMPT_CONTEXT_MAP.items(): 
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {llm} | Seed {seed} | {prompt_name}") 
                
                start_time = datetime.now() 
                status = "failed" 
                error_msg = None 
                n_generated = 0 
                
                try: 
                    # Chargement réduit 
                    context_df = load_context_reduced(context_name) 
                    prompt_text = load_prompt(prompt_name) 
                    
                    # Prompt court et direct 
                    prompt_with_context = f""" 
Act as a Data Generator. Output ONLY raw CSV. 
NO TALKING. NO MARKDOWN. 

REFERENCE HEADER: 
{','.join(context_df.columns)} 

REFERENCE SAMPLES: 
{context_df.head(3).to_csv(index=False, header=False)} 

TASK: 
Generate exactly {N_SAMPLES} new rows following this pattern. 
Last column is Yield_BV. 
""" 
                    print(f"  -> Envoi (Contexte léger: {len(context_df.columns)} cols)...") 
                    
                    raw_response = generate_with_ollama_subprocess(llm, prompt_with_context, seed) 
                    synth_df = parse_llm_csv_aggressive(raw_response) 
                    
                    # Réparation automatique du header si manquant 
                    if 'Yield_BV' not in synth_df.columns: 
                        if len(synth_df.columns) == len(context_df.columns): 
                            print("  ⚠️ Header manquant -> Réparé.") 
                            synth_df.columns = context_df.columns 
                        else: 
                             # On tente de renommer quand même si le compte est proche 
                             print("  ⚠️ Header reconstruit de force.") 
                             synth_df = synth_df.iloc[:, :len(context_df.columns)] # On coupe si trop long 
                             synth_df.columns = context_df.columns 

                    # Sauvegarde 
                    llm_clean = llm.split(':')[0] 
                    output_dir = os.path.join(BASE_PATH, "outputs", llm_clean, f"seed_{seed}") 
                    os.makedirs(output_dir, exist_ok=True) 
                    output_filename = f"synth_{prompt_name[:-4]}_{context_name[:-4]}.csv" 
                    synth_df.to_csv(os.path.join(output_dir, output_filename), index=False) 
                    
                    n_generated = len(synth_df) 
                    status = "success" 
                    print(f"  -> ✅ Succès : {n_generated} lignes propres.") 

                except Exception as e: 
                    error_msg = str(e) 
                    print(f"  -> ❌ Erreur : {e}") 

                duration = (datetime.now() - start_time).total_seconds() 
                manifest.append({ 
                    "run_id": run_id, "llm": llm, "seed": seed, "prompt": prompt_name, 
                    "n_samples": n_generated, "status": status, "error": error_msg 
                }) 
                run_id += 1 
                pd.DataFrame(manifest).to_csv(os.path.join(BASE_PATH, "logs", "runs_manifest.csv"), index=False) 

    print("\n=== 🎉 TERMINE ===")
