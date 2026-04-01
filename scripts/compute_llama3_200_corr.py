import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR=os.path.join('03_modeling_results','quality_analysis')
os.makedirs(OUT_DIR,exist_ok=True)

def main():
    y_real=pd.read_csv(os.path.join('02_processed_data','pepper','y.csv'))
    if 'Sample_ID' not in y_real.columns:
        y_real.rename(columns={y_real.columns[0]:'Sample_ID'},inplace=True)
    real=np.float32(pd.to_numeric(y_real['Yield_BV'],errors='coerce').dropna().values)
    y_synth=pd.read_csv(os.path.join('04_augmentation','pepper','model_sources','llama3','synthetic_y_llama3_filtered_k3000_200.csv'))
    synth=np.float32(pd.to_numeric(y_synth['Yield_BV'] if 'Yield_BV' in y_synth.columns else y_synth.iloc[:,-1],errors='coerce').dropna().values)
    n=min(len(real),len(synth))
    corr=float(np.corrcoef(real[:n],synth[:n])[0,1])
    # Heatmap
    val=np.array([[corr]],dtype=np.float32)
    plt.figure(figsize=(3,3))
    plt.imshow(val,cmap='viridis',vmin=-1,vmax=1)
    plt.colorbar()
    plt.xticks([0],["Synthétiques llama3 (200)"])
    plt.yticks([0],["Données Réelles (Yield_BV)"])
    plt.title('Corrélation Yield_BV (k=3000) : Réelles vs Synthétiques llama3 (200)')
    plt.text(0,0,f"{corr:.3f}",ha='center',va='center',color='white')
    plt.tight_layout()
    corr_png=os.path.join(OUT_DIR,'llama3_200synth_correlation.png')
    plt.savefig(corr_png,dpi=300)
    plt.close()
    # Report
    md=os.path.join(OUT_DIR,'llama3_200synth_quality_short.md')
    with open(md,'w',encoding='utf-8') as f:
        f.write('# Qualité synthétiques llama3 (200, k=3000)\n\n')
        f.write('1. PCA: voir la figure `llama3_200synth_pca.png`.\n')
        f.write(f'2. Corrélation Yield_BV (réelles vs synthétiques llama3 200): {corr:.3f}\n')
        f.write('3. Conclusion: voir PCA et corrélation pour la cohérence globale.\n')
    print('corr:',corr)

if __name__=='__main__':
    main()

