"""
Entraînement des modèles avec données optimisées
- REAL + context_D uniquement
- Normalisation StandardScaler
- Learning rate = 0.01 (optimal)
- Modèles: Transformer et LSTM (meilleurs performers)
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
OPTIMIZED_DIR = BASE_DIR / "04_augmentation/ipk_out_raw_optimized"
RESULTS_DIR = BASE_DIR / "03_modeling_results/ipk_optimized"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Hyperparamètres optimaux
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.01  # Optimal trouvé précédemment
PATIENCE = 15

print("=" * 80)
print("ENTRAÎNEMENT AVEC DONNÉES OPTIMISÉES")
print("=" * 80)
print(f"Configuration: lr={LEARNING_RATE}, epochs={EPOCHS}, batch_size={BATCH_SIZE}")

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
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

# ============================================================================
# FONCTIONS D'ENTRAÎNEMENT
# ============================================================================

def create_sequences(X, y, seq_length=10):
    """Crée des séquences pour LSTM/Transformer"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length-1])
    return np.array(X_seq), np.array(y_seq)

class EarlyStopping:
    """Early stopping pour éviter overfitting"""
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
    """Entraîne un modèle avec early stopping"""
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
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history, best_val_loss

def evaluate_model(model, X_test, y_test):
    """Évalue le modèle sur le jeu de test"""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        predictions = model(X_test_tensor).cpu().numpy().squeeze()
    
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'predictions': predictions
    }

# ============================================================================
# ENTRAÎNEMENT PRINCIPAL
# ============================================================================

results_summary = []

for model_name in ['glm5', 'kimi']:
    print(f"\n{'=' * 80}")
    print(f"MODÈLE: {model_name.upper()}")
    print(f"{'=' * 80}")
    
    # Charger données optimisées
    data_dir = OPTIMIZED_DIR / model_name
    
    if not (data_dir / "X_optimized.csv").exists():
        print(f"  Données optimisées non trouvées pour {model_name}, skipping...")
        continue
    
    X = pd.read_csv(data_dir / "X_optimized.csv").values
    y = pd.read_csv(data_dir / "y_optimized.csv")['YR_LS'].values
    
    print(f"  Dataset: {X.shape[0]} échantillons, {X.shape[1]} features")
    print(f"  Yield range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Créer séquences pour LSTM/Transformer
    SEQ_LENGTH = 10
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQ_LENGTH)
    
    print(f"  Sequences: Train={len(X_train_seq)}, Test={len(X_test_seq)}")
    
    # DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_seq),
        torch.FloatTensor(y_train_seq)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_seq),
        torch.FloatTensor(y_test_seq)
    )
    
    # Split train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    input_size = X.shape[1]
    
    # Entraîner LSTM
    print(f"\n  --- Entraînement LSTM ---")
    lstm_model = LSTMModel(input_size).to(DEVICE)
    lstm_model, lstm_history, lstm_val_loss = train_model(
        lstm_model, train_loader, val_loader, "LSTM", EPOCHS, LEARNING_RATE
    )
    lstm_results = evaluate_model(lstm_model, X_test_seq, y_test_seq)
    
    print(f"  LSTM Results:")
    print(f"    R² = {lstm_results['R2']:.4f}")
    print(f"    RMSE = {lstm_results['RMSE']:.4f}")
    print(f"    MAE = {lstm_results['MAE']:.4f}")
    
    # Sauvegarder modèle LSTM
    torch.save(lstm_model.state_dict(), RESULTS_DIR / f"{model_name}_lstm_optimized.pt")
    
    # Entraîner Transformer
    print(f"\n  --- Entraînement Transformer ---")
    transformer_model = TransformerModel(input_size).to(DEVICE)
    transformer_model, transformer_history, transformer_val_loss = train_model(
        transformer_model, train_loader, val_loader, "Transformer", EPOCHS, LEARNING_RATE
    )
    transformer_results = evaluate_model(transformer_model, X_test_seq, y_test_seq)
    
    print(f"  Transformer Results:")
    print(f"    R² = {transformer_results['R2']:.4f}")
    print(f"    RMSE = {transformer_results['RMSE']:.4f}")
    print(f"    MAE = {transformer_results['MAE']:.4f}")
    
    # Sauvegarder modèle Transformer
    torch.save(transformer_model.state_dict(), RESULTS_DIR / f"{model_name}_transformer_optimized.pt")
    
    # Stocker résultats
    results_summary.append({
        'model_source': model_name,
        'dl_model': 'LSTM',
        'R2': lstm_results['R2'],
        'RMSE': lstm_results['RMSE'],
        'MAE': lstm_results['MAE'],
        'val_loss': lstm_val_loss,
        'n_samples': len(X)
    })
    
    results_summary.append({
        'model_source': model_name,
        'dl_model': 'Transformer',
        'R2': transformer_results['R2'],
        'RMSE': transformer_results['RMSE'],
        'MAE': transformer_results['MAE'],
        'val_loss': transformer_val_loss,
        'n_samples': len(X)
    })

# ============================================================================
# RÉSULTATS FINaux
# ============================================================================

print(f"\n{'=' * 80}")
print("RÉSULTATS FINaux - DONNÉES OPTIMISÉES")
print(f"{'=' * 80}")

results_df = pd.DataFrame(results_summary)
print("\n" + results_df.to_string(index=False))

# Sauvegarder résultats
results_df.to_csv(RESULTS_DIR / "optimized_training_results.csv", index=False)
print(f"\nRésultats sauvegardés: {RESULTS_DIR / 'optimized_training_results.csv'}")

# Comparaison avec résultats précédents
print(f"\n{'=' * 80}")
print("COMPARAISON AVEC RÉSULTATS PRÉCÉDENTS")
print(f"{'=' * 80}")

print("""
Avant (tous contextes, pas de normalisation standard):
  - GLM5 Context E + Transformer: R² = 0.0014 (TRÈS MAUVAIS)
  
Après (context_D uniquement, StandardScaler, lr=0.01):
  - À évaluer...

Attendu avec optimisation:
  - Meilleure variance dans les données
  - Normalisation appropriée
  - Corrélation SNP-yield préservée
  → R² devrait s'améliorer significativement
""")

print("=" * 80)
