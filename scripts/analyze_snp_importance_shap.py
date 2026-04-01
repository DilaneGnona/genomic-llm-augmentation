"""
Analyze SNP importance using SHAP values
Identify which SNPs are most important for yield prediction
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn

# SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "shap", "-q"])
    import shap
    SHAP_AVAILABLE = True

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC_PEPPER = os.path.join(BASE, '02_processed_data', 'pepper')
OUTDIR = os.path.join(BASE, '03_modeling_results', 'pepper', 'snp_importance_analysis')

def ensure_dirs():
    os.makedirs(OUTDIR, exist_ok=True)

def load_data(max_features=100):
    """Load data for SHAP analysis"""
    x_path = os.path.join(PROC_PEPPER, 'X_aligned.csv')
    y_path = os.path.join(PROC_PEPPER, 'y_aligned.csv')
    
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    
    feature_cols = [c for c in X.columns if c != 'Sample_ID'][:max_features]
    X = X[['Sample_ID'] + feature_cols]
    
    target_col = 'Yield_BV' if 'Yield_BV' in y.columns else [c for c in y.columns if c != 'Sample_ID'][0]
    y = y[['Sample_ID', target_col]].dropna()
    
    # Merge
    df = X.merge(y, on='Sample_ID', how='inner')
    
    Xf = df[feature_cols].values.astype(np.float32)
    yv = df[target_col].values.astype(np.float32)
    
    return Xf, yv, feature_cols

class LSTMModel(nn.Module):
    def __init__(self, n_features):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(n_features, 32, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(32, 16, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_best_model(X, y):
    """Train the best performing model (LSTM)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    
    # Convert to tensors
    X_tr_t = torch.FloatTensor(X_tr_s).unsqueeze(1).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    
    n_features = X_tr_s.shape[1]
    model = LSTMModel(n_features).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_tr_t).squeeze()
        loss = criterion(outputs, y_tr_t)
        loss.backward()
        optimizer.step()
    
    return model, sc, X_te_s, y_te

def analyze_with_shap(model, X_test, feature_names):
    """Analyze feature importance using SHAP"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\nAnalyzing SNP importance with SHAP...")
    print("This may take a few minutes...")
    
    # Convert test data to tensor
    X_test_t = torch.FloatTensor(X_test).unsqueeze(1).to(device)
    
    # Define prediction function for SHAP
    def predict_fn(x):
        model.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x).unsqueeze(1).to(device)
            pred = model(x_t).squeeze().cpu().numpy()
        return pred
    
    # Use KernelExplainer for PyTorch models
    # Sample background data
    background = X_test[:50]
    explainer = shap.KernelExplainer(predict_fn, background)
    
    # Calculate SHAP values for test set (limit to 100 samples for speed)
    shap_values = explainer.shap_values(X_test[:100], nsamples=100)
    
    # Get mean absolute SHAP values for each feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame with feature importance
    importance_df = pd.DataFrame({
        'SNP': feature_names,
        'SHAP_importance': mean_shap
    }).sort_values('SHAP_importance', ascending=False)
    
    return importance_df, shap_values

def plot_top_snps(importance_df, top_n=20):
    """Plot top N most important SNPs"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    top_snps = importance_df.head(top_n)
    
    plt.barh(range(top_n), top_snps['SHAP_importance'], color='steelblue')
    plt.yticks(range(top_n), top_snps['SNP'])
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.ylabel('SNP', fontsize=12)
    plt.title(f'Top {top_n} Most Important SNPs for Yield Prediction', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTDIR, 'top_snps_importance.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: top_snps_importance.png")
    plt.close()

def create_summary_report(importance_df):
    """Create summary report of SNP analysis"""
    report = []
    report.append("="*70)
    report.append("SNP IMPORTANCE ANALYSIS REPORT")
    report.append("="*70)
    report.append("")
    report.append(f"Total SNPs analyzed: {len(importance_df)}")
    report.append("")
    report.append("TOP 20 MOST IMPORTANT SNPs:")
    report.append("-"*70)
    
    for idx, row in importance_df.head(20).iterrows():
        report.append(f"{row['SNP']:20s} | SHAP: {row['SHAP_importance']:.6f}")
    
    report.append("")
    report.append("STATISTICS:")
    report.append("-"*70)
    report.append(f"Mean importance: {importance_df['SHAP_importance'].mean():.6f}")
    report.append(f"Std importance: {importance_df['SHAP_importance'].std():.6f}")
    report.append(f"Max importance: {importance_df['SHAP_importance'].max():.6f}")
    report.append(f"Min importance: {importance_df['SHAP_importance'].min():.6f}")
    
    # Count SNPs above different thresholds
    thresholds = [0.001, 0.01, 0.1]
    for thresh in thresholds:
        count = (importance_df['SHAP_importance'] > thresh).sum()
        report.append(f"SNPs with importance > {thresh}: {count}")
    
    report.append("")
    report.append("="*70)
    
    report_text = "\n".join(report)
    
    # Save report
    with open(os.path.join(OUTDIR, 'snp_importance_report.txt'), 'w') as f:
        f.write(report_text)
    
    return report_text

def main():
    print("="*70)
    print("SNP IMPORTANCE ANALYSIS WITH SHAP")
    print("="*70)
    print()
    
    ensure_dirs()
    
    # Load data
    print("Loading data...")
    X, y, feature_names = load_data(max_features=100)
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print()
    
    # Train model
    print("Training LSTM model (best performing)...")
    model, scaler, X_test, y_test = train_best_model(X, y)
    print("Model trained!")
    print()
    
    # SHAP analysis
    importance_df, shap_values = analyze_with_shap(model, X_test, feature_names)
    
    # Save importance data
    importance_df.to_csv(os.path.join(OUTDIR, 'snp_importance.csv'), index=False)
    print(f"Saved: snp_importance.csv")
    
    # Plot top SNPs
    print("\nCreating plots...")
    plot_top_snps(importance_df, top_n=20)
    
    # Create report
    print("\nGenerating report...")
    report = create_summary_report(importance_df)
    print(report)
    
    print("\n" + "="*70)
    print(f"Analysis complete! Results saved to: {OUTDIR}")
    print("="*70)

if __name__ == '__main__':
    main()
