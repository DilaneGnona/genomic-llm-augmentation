import matplotlib.pyplot as plt
import pandas as pd
import json
import os

def load_results():
    results = []
    
    # 1. Pepper Baseline (approximate from metrics)
    results.append({'Dataset': 'Pepper', 'Type': 'Baseline', 'Model': 'XGBoost', 'R2': 0.208})
    results.append({'Dataset': 'Pepper', 'Type': 'Baseline', 'Model': 'Ridge', 'R2': -0.233})
    
    # 2. IPK Baseline (approximate from metrics)
    results.append({'Dataset': 'IPK', 'Type': 'Baseline', 'Model': 'XGBoost', 'R2': -0.582})
    results.append({'Dataset': 'IPK', 'Type': 'Baseline', 'Model': 'Ridge', 'R2': -0.602})
    
    # 3. Pepper Augmented
    with open('03_modeling_results/augmented_pepper_summary.json', 'r') as f:
        data = json.load(f)
        for m, metrics in data.items():
            results.append({'Dataset': 'Pepper', 'Type': 'Augmented', 'Model': m.upper(), 'R2': metrics['r2']})
            
    # 4. IPK Augmented
    with open('03_modeling_results/augmented_ipk_out_raw_summary.json', 'r') as f:
        data = json.load(f)
        for m, metrics in data.items():
            results.append({'Dataset': 'IPK', 'Type': 'Augmented', 'Model': m.upper(), 'R2': metrics['r2']})
            
    return pd.DataFrame(results)

def create_plot(df):
    plt.figure(figsize=(14, 8))
    
    # Filter only positive or relevant R2 for clarity in bar chart
    df_plot = df[df['R2'] > -0.1].copy()
    
    # Sort for better visualization
    df_plot = df_plot.sort_values(by=['Dataset', 'R2'], ascending=[True, False])
    
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    
    g = sns.catplot(
        data=df_plot, kind="bar",
        x="Model", y="R2", hue="Dataset",
        palette="viridis", alpha=.8, height=6, aspect=1.5
    )
    
    g.despine(left=True)
    g.set_axis_labels("Modèles", "R² Score")
    g.legend.set_title("Dataset")
    plt.title("Comparaison des Performances R² : Pepper vs IPK (Données Augmentées)", fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    out_path = '03_modeling_results/comparison_performance.png'
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    df = load_results()
    create_plot(df)
