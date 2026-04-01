import os
import sys
import json
import glob
import subprocess

DATASET='pepper'
PROCESSED=os.path.join('02_processed_data',DATASET)
AUG=os.path.join('04_augmentation',DATASET,'model_sources')
OUTDIR=os.path.join('03_modeling_results',f'{DATASET}_augmented_v2')
LLMS=['llama3','deepseek','glm46','qwen','minimax']

def ensure_dirs():
    os.makedirs(os.path.join(OUTDIR,'logs'),exist_ok=True)
    os.makedirs(os.path.join(OUTDIR,'metrics'),exist_ok=True)

def run_llm(llm):
    print(f"Lancement v2 RF pour {llm} (200 synth, k=3000, Parquet loader)...")
    synth=os.path.join(AUG,llm,f'synthetic_y_{llm}_filtered_k3000_200.csv')
    if not os.path.exists(synth):
        print(f"[ERREUR] {synth} introuvable; skip {llm}")
        return None
    llm_log_dir=os.path.join(OUTDIR,'logs',llm)
    os.makedirs(llm_log_dir,exist_ok=True)
    verbose=os.path.join(llm_log_dir,'verbose_logs.txt')
    cmd=[
        sys.executable,
        os.path.join('scripts','unified_modeling_pipeline_augmented.py'),
        '--dataset',DATASET,
        '--use_synthetic','True',
        '--selected_k','3000',
        '--sigma_resid_factor','0.1',
        '--cross_validation_outer','2',
        '--cross_validation_inner','2',
        '--holdout_size','0.2',
        '--overwrite_previous',
        '--models','randomforest',
        '--augment_file',synth,
        '--rf_n_estimators','75',
        '--rf_max_depth','15',
        '--rf_max_features','sqrt',
        '--rf_max_samples','0.8'
    ]
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
    with open(verbose,'w',encoding='utf-8') as vf:
        for line in proc.stdout:
            sys.stdout.write(line)
            vf.write(line)
    code=proc.wait()
    # find last randomforest metrics
    metrics_dir=os.path.join('03_modeling_results',f'{DATASET}_augmented','metrics')
    candidates=sorted(glob.glob(os.path.join(metrics_dir,'randomforest_metrics_*.json')),key=os.path.getmtime)
    if not candidates:
        print(f"[ERREUR] métriques v1 introuvables pour {llm}")
        return None
    last=candidates[-1]
    with open(last,'r',encoding='utf-8') as f:
        m=json.load(f)
    # write v2 metrics file
    out_m=os.path.join(OUTDIR,'metrics',f'{llm}_randomforest_200synth_metrics.json')
    with open(out_m,'w',encoding='utf-8') as f:
        json.dump(m,f,indent=2)
    print(f"[OK] {llm} métriques v2 → {out_m}")
    return {
        'LLM': llm,
        'Nb_Synthétiques': 200,
        'Holdout_R2': m.get('holdout_r2'),
        'Holdout_RMSE': m.get('holdout_rmse'),
        'CV_R2': m.get('cv_r2_mean'),
        'Nb_Echantillons_Totaux': m.get('samples_count')
    }

def run_baseline():
    print('Lancement baseline v2 (sans synthétiques)...')
    cmd=[
        sys.executable,
        os.path.join('scripts','unified_modeling_pipeline_augmented.py'),
        '--dataset',DATASET,
        '--use_synthetic','False',
        '--selected_k','3000',
        '--cross_validation_outer','2',
        '--cross_validation_inner','2',
        '--holdout_size','0.2',
        '--overwrite_previous',
        '--models','randomforest',
        '--rf_n_estimators','75','--rf_max_depth','15','--rf_max_features','sqrt','--rf_max_samples','0.8'
    ]
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
    for line in proc.stdout:
        sys.stdout.write(line)
    code=proc.wait()
    # read latest metrics
    metrics_dir=os.path.join('03_modeling_results',f'{DATASET}_augmented','metrics')
    candidates=sorted(glob.glob(os.path.join(metrics_dir,'randomforest_metrics_*.json')),key=os.path.getmtime)
    last=candidates[-1]
    with open(last,'r',encoding='utf-8') as f:
        m=json.load(f)
    out_m=os.path.join('03_modeling_results','baseline','randomforest_baseline_v2_metrics.json')
    os.makedirs(os.path.dirname(out_m),exist_ok=True)
    with open(out_m,'w',encoding='utf-8') as f:
        json.dump(m,f,indent=2)
    print(f"[OK] baseline métriques → {out_m}")
    return {
        'LLM':'baseline','Nb_Synthétiques':0,
        'Holdout_R2': m.get('holdout_r2'),
        'Holdout_RMSE': m.get('holdout_rmse'),
        'CV_R2': m.get('cv_r2_mean'),
        'Nb_Echantillons_Totaux': m.get('samples_count')
    }

def write_v2_csv(rows):
    import pandas as pd
    comp=os.path.join('03_modeling_results','comparative_analysis','cross_llm_k3000','randomforest')
    os.makedirs(comp,exist_ok=True)
    out=os.path.join(comp,'comparative_results_v2.csv')
    pd.DataFrame(rows)[['LLM','Nb_Synthétiques','Holdout_R2','Holdout_RMSE','CV_R2','Nb_Echantillons_Totaux']].to_csv(out,index=False)
    print(f"[OK] comparatif v2 → {out}")

def main():
    ensure_dirs()
    rows=[]
    for llm in LLMS:
        r=run_llm(llm)
        if r: rows.append(r)
    rows.append(run_baseline())
    write_v2_csv(rows)

if __name__=='__main__':
    main()

