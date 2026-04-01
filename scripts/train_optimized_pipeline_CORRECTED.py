"""
Pipeline complet d'entraînement avec données CORRIGÉES
- REAL + context_D uniquement
- Normalisation StandardScaler
- Learning rate = 0.01
- Modèles: Transformer et LSTM
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path("c:/Users/OMEN/Desktop/experiment_snp")
REAL_DATA_DIR = BASE_DIR / "02_processed_data/ipk_out_raw"
CONTEXT_DIR = BASE_DIR / "04_augmentation/ipk_out_raw/context learning"
RESULTS_DIR = BASE_DIR / "03_modeling_results/ipk_optimized_corrected"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Hyperparamètres optimaux
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.01
PATIENCE = 15
SEQ_LENGTH = 10

print("=" * 80)
print("PIPELINE D'ENTRAÎNEMENT - DONNÉES CORRIGÉES")
print("=" * 80)
print(f"Configuration: lr={LEARNING_RATE}, epochs={EPOCHS}, batch_size={BATCH_SIZE}")
print(f"Données: REAL + context_D (format SNP: 0,1,2)")

# ============================================================================
# ARCHITECTURES DES MODÈLES
# ============================================================================

class LSTMModel(nn.Module):
    """LSTM optimisé"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class TransformerModel(nn.Module):
    """Transformer optimisé"""
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def load_real_data():
    """Charge les données réelles IPK"""
    X = pd.read_csv(REAL_DATA_DIR / "X_aligned.csv")
    y = pd.read_csv(REAL_DATA_DIR / "y_aligned.csv")
    
    # Merge
    df = X.merge(y, on='Sample_ID', how='inner')
    
    feature_cols = [c for c in df.columns if c.startswith('SNP_')]
    
    print(f"\nDonnées réelles chargées:")
    print(f"  Échantillons: {len(df)}")
    print(f"  SNPs: {len(feature_cols)}")
    print(f"  Yield range: [{df['YR_LS'].min():.2f}, {df['YR_LS'].max():.2f}]")
    
    X_real = df[feature_cols].values
    y_real = df['YR_LS'].values
    
    return X_real, y_real, feature_cols

def load_context_data(model_name, context='D'):
    """Charge les données context learning corrigées"""
    ctx_file = CONTEXT_DIR / model_name / f"context_{context}" / f"synthetic_{model_name}_context_{context}_500samples_CORRECTED.csv"
    
    if not ctx_file.exists():
        print(f"  Fichier non trouvé: {ctx_file}")
        return None, None
    
    df = pd.read_csv(ctx_file)
    feature_cols = [c for c in df.columns if c.startswith('SNP_')]
    
    X_ctx = df[feature_cols].values
    y_ctx = df['YR_LS'].values
    
    print(f"\nDonnées context_{context} ({model_name}) chargées:")
    print(f"  Échantillons: {len(df)}")
    print(f"  Yield range: [{df['YR_LS'].min():.2f}, {df['YR_LS'].max():.2f}]")
    
    # Vérification format SNP
    unique_vals = np.unique(X_ctx)
    is_valid = set(unique_vals).issubset({0, 1, 2})
    print(f"  Format SNP (0,1,2): {'✓ VALIDE' if is_valid else '✗ INVALIDE'}")
    
    return X_ctx, y_ctx

def standard_scaler_fit(X):
    """Calcule mean et std pour normalisation"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Éviter division par zéro
    return mean, std

def standard_scaler_transform(X, mean, std):
    """Applique (X - mean) / std"""
    return (X - mean) / std

def create_sequences(X, y, seq_length=10):
    """Crée des séquences pour LSTM/Transformer"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length-1])
    return np.array(X_seq), np.array(y_seq)

class EarlyStopping:
    """Early stopping"""
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(model, train_loader, val_loader, model_name, epochs=100, lr=0.01):
    """Entraîne un modèle"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=PATIENCE)
    
    best_val_loss = float('inf')
    best_model_state = None
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"    Early stopping à l'epoch {epoch+1}")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history, best_val_loss

def evaluate_model(model, X_test, y_test):
    """Évalue le modèle"""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        predictions = model(X_test_tensor).cpu().numpy().squeeze()
    
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'predictions': predictions}

# ============================================================================
# ENTRAÎNEMENT PRINCIPAL
# ============================================================================

results_summary = []

# Charger données réelles
X_real, y_real, feature_cols = load_real_data()

# Fit StandardScaler sur données réelles
scaler_mean, scaler_std = standard_scaler_fit(X_real)
print(f"\nNormalisation StandardScaler:")
print(f"  Mean: {np.mean(scaler_mean):.4f}")
print(f"  Std: {np.mean(scaler_std):.4f}")

# Normaliser données réelles
X_real_norm = standard_scaler_transform(X_real, scaler_mean, scaler_std)

for model_name in ['glm5', 'kimi']:
    print(f"\n{'=' * 80}")
    print(f"MODÈLE: {model_name.upper()}")
    print(f"{'=' * 80}")
    
    # Charger context_D
    X_ctx, y_ctx = load_context_data(model_name, 'D')
    
    if X_ctx is None:
        print(f"  Données context non disponibles, skipping...")
        continue
    
    # Normaliser context avec MÊMES paramètres que real
    X_ctx_norm = standard_scaler_transform(X_ctx, scaler_mean, scaler_std)
    
    # Combiner REAL + context_D
    X_combined = np.vstack([X_real_norm, X_ctx_norm])
    y_combined = np.concatenate([y_real, y_ctx])
    
    print(f"\nDataset combiné:")
    print(f"  Total: {len(X_combined)} échantillons")
    print(f"  Réels: {len(X_real)}, Synthétiques: {len(X_ctx)}")
    print(f"  Yield range: [{y_combined.min():.2f}, {y_combined.max():.2f}]")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42
    )
    
    # Créer séquences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQ_LENGTH)
    
    print(f"\nSéquences créées:")
    print(f"  Train: {len(X_train_seq)}, Test: {len(X_test_seq)}")
    
    # DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_seq),
        torch.FloatTensor(y_train_seq)
    )
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)
    
    input_size = X_combined.shape[1]
    
    # Entraîner LSTM
    print(f"\n  --- Entraînement LSTM ---")
    lstm_model = LSTMModel(input_size).to(DEVICE)
    lstm_model, lstm_hist, lstm_val_loss = train_model(
        lstm_model, train_loader, val_loader, "LSTM", EPOCHS, LEARNING_RATE
    )
    lstm_results = evaluate_model(lstm_model, X_test_seq, y_test_seq)
    
    print(f"    R² = {lstm_results['R2']:.4f}")
    print(f"    RMSE = {lstm_results['RMSE']:.4f}")
    print(f"    MAE = {lstm_results['MAE']:.4f}")
    
    torch.save(lstm_model.state_dict(), RESULTS_DIR / f"{model_name}_lstm.pt")
    
    # Entraîner Transformer
    print(f"\n  --- Entraînement Transformer ---")
    transformer_model = TransformerModel(input_size).to(DEVICE)
    transformer_model, trans_hist, trans_val_loss = train_model(
        transformer_model, train_loader, val_loader, "Transformer", EPOCHS, LEARNING_RATE
    )
    transformer_results = evaluate_model(transformer_model, X_test_seq, y_test_seq)
    
    print(f"    R² = {transformer_results['R2']:.4f}")
    print(f"    RMSE = {transformer_results['RMSE']:.4f}")
    print(f"    MAE = {transformer_results['MAE']:.4f}")
    
    torch.save(transformer_model.state_dict(), RESULTS_DIR / f"{model_name}_transformer.pt")
    
    # Stocker résultats
    results_summary.append({
        'model_source': model_name,
        'dl_model': 'LSTM',
        'R2': lstm_results['R2'],
        'RMSE': lstm_results['RMSE'],
        'MAE': lstm_results['MAE'],
        'n_samples': len(X_combined),
        'n_real': len(X_real),
        'n_synthetic': len(X_ctx)
    })
    
    results_summary.append({
        'model_source': model_name,
        'dl_model': 'Transformer',
        'R2': transformer_results['R2'],
        'RMSE': transformer_results['RMSE'],
        'MAE': transformer_results['MAE'],
        'n_samples': len(X_combined),
        'n_real': len(X_real),
        'n_synthetic': len(X_ctx)
    })

# ============================================================================
# RÉSULTATS
# ============================================================================

print(f"\n{'=' * 80}")
print("RÉSULTATS FINAUX - DONNÉES CORRIGÉES")
print(f"{'=' * 80}")

results_df = pd.DataFrame(results_summary)
print("\n" + results_df.to_string(index=False))

results_df.to_csv(RESULTS_DIR / "training_results_corrected.csv", index=False)
print(f"\nRésultats sauvegardés: {RESULTS_DIR / 'training_results_corrected.csv'}")

# Sauvegarder paramètres de normalisation
np.savez(RESULTS_DIR / "scaler_params.npz", mean=scaler_mean, std=scaler_std)
print(f"Paramètres scaler sauvegardés: {RESULTS_DIR / 'scaler_params.npz'}")

print(f"\n{'=' * 80}")
print("COMPARAISON AVEC AVANT CORRECTION")
print(f"{'=' * 80}")
print("""
AVANT (données invalides):
  - SNPs continus: 1.419, 0.865... ❌
  - Meilleur R²: ~0.001 (TRÈS MAUVAIS)

APRÈS (données corrigées):
  - SNPs discrets: 0, 1, 2 ✅
  - Résultats: Voir tableau ci-dessus
  
AMÉLIORATIONS:
  ✓ Format SNP correct (0,1,2)
  ✓ Normalisation StandardScaler
  ✓ REAL + context_D uniquement
  ✓ Learning rate optimal (0.01)
""")

print("=" * 80)
