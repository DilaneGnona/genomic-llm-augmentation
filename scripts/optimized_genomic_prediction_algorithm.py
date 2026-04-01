"""
================================================================================
ALGORITHME OPTIMISÉ POUR LA PRÉDICTION GÉNOMIQUE
================================================================================
Architecture: Deep Ensemble with Attention Mechanisms
Optimisations: 
- Multi-Head Self-Attention for SNP interactions
- Residual Connections for gradient flow
- Layer Normalization for training stability
- Advanced Regularization (Dropout + Weight Decay)
- Learning Rate Scheduling with Warmup
- Early Stopping with Model Checkpointing
- K-Fold Cross-Validation
================================================================================
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
import warnings
from typing import Tuple, List, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ================================================================================
# CONFIGURATION OPTIMALE
# ================================================================================

@dataclass
class Config:
    """Configuration optimale basée sur les résultats expérimentaux"""
    # Data
    BASE_DIR: Path = Path("c:/Users/OMEN/Desktop/experiment_snp")
    SEQ_LENGTH: int = 10
    TEST_SIZE: float = 0.2
    VAL_SIZE: float = 0.1
    
    # Model Architecture
    D_MODEL: int = 128  # Dimension du modèle (augmentée pour plus de capacité)
    NHEAD: int = 8      # Nombre de têtes d'attention
    NUM_LAYERS: int = 4 # Profondeur du réseau
    DROPOUT: float = 0.3
    DIM_FEEDFORWARD: int = 512
    
    # Training
    BATCH_SIZE: int = 16  # Plus petit pour stabilité
    EPOCHS: int = 200
    LEARNING_RATE: float = 0.001  # Plus conservateur
    WEIGHT_DECAY: float = 1e-4   # Régularisation L2
    WARMUP_EPOCHS: int = 10
    PATIENCE: int = 25
    
    # Cross-Validation
    N_FOLDS: int = 5
    
    # Device
    DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

print(f"Device: {config.DEVICE}")
print(f"Configuration: d_model={config.D_MODEL}, nhead={config.NHEAD}, layers={config.NUM_LAYERS}")

# ================================================================================
# ARCHITECTURE OPTIMISÉE: TRANSFORMER AVANCÉ AVEC RÉSIDUAL CONNECTIONS
# ================================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Encodage positionnel sinusoïdal pour capturer la position des SNPs"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SNPAttentionBlock(nn.Module):
    """Bloc d'attention spécialisé pour les données SNP"""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),  # Meilleur que ReLU pour les transformers
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, attn_weights

class OptimizedGenomicTransformer(nn.Module):
    """
    Architecture Transformer optimisée pour la prédiction génomique
    """
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 4, dropout: float = 0.3):
        super().__init__()
        
        self.d_model = d_model
        
        # Projection d'entrée
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Encodage positionnel
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        
        # Blocs d'attention empilés
        self.attention_blocks = nn.ModuleList([
            SNPAttentionBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        
        # Global Average Pooling avec attention apprise
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # Tête de prédiction avec connexions résiduelles
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        # Initialisation des poids
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation Xavier pour stabilité"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        
        # Projection
        x = self.input_projection(x)
        
        # Encodage positionnel
        x = self.pos_encoder(x)
        
        # Blocs d'attention
        attention_weights = []
        for block in self.attention_blocks:
            x, attn_w = block(x)
            attention_weights.append(attn_w)
        
        # Pooling attention-based
        attn_scores = self.attention_pool(x)  # (batch, seq_len, 1)
        x = torch.sum(x * attn_scores, dim=1)  # (batch, d_model)
        
        # Prédiction
        output = self.predictor(x)
        
        return output.squeeze(), attention_weights

# ================================================================================
# ARCHITECTURE ALTERNATIVE: LSTM AVANCÉ AVEC ATTENTION
# ================================================================================

class AttentionLSTM(nn.Module):
    """LSTM avec mécanisme d'attention pour la prédiction génomique"""
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Projection d'entrée
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # LSTM bidirectionnel
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Prédicteur
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # Projection
        x = self.input_proj(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)
        
        # Prédiction
        output = self.predictor(context)
        
        return output.squeeze(), attn_weights

# ================================================================================
# ENSEMBLE DE MODÈLES POUR ROBUSTESSE
# ================================================================================

class GenomicEnsemble(nn.Module):
    """Ensemble de modèles pour prédiction robuste"""
    def __init__(self, input_size: int, models: List[nn.Module] = None):
        super().__init__()
        
        if models is None:
            self.models = nn.ModuleList([
                OptimizedGenomicTransformer(input_size),
                AttentionLSTM(input_size),
            ])
        else:
            self.models = nn.ModuleList(models)
        
        # Poids appris pour l'ensemble
        self.ensemble_weights = nn.Parameter(torch.ones(len(self.models)))
    
    def forward(self, x):
        predictions = []
        for model in self.models:
            pred, _ = model(x)
            predictions.append(pred)
        
        # Combinaison pondérée
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_pred = sum(w * p for w, p in zip(weights, predictions))
        
        return ensemble_pred, predictions, weights

# ================================================================================
# FONCTIONS D'ENTRAÎNEMENT AVANCÉES
# ================================================================================

class WarmupCosineScheduler:
    """Scheduler avec warmup et décroissance cosine"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup linéaire
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Entraînement d'une epoch avec monitoring"""
    model.train()
    total_loss = 0
    predictions, targets = [], []
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend(outputs.cpu().detach().numpy())
        targets.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    r2 = r2_score(targets, predictions)
    
    return avg_loss, r2

def validate(model, val_loader, criterion, device):
    """Validation avec métriques complètes"""
    model.eval()
    total_loss = 0
    predictions, targets = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs, _ = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    r2 = r2_score(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    
    return avg_loss, r2, rmse, mae, predictions, targets

# ================================================================================
# PIPELINE COMPLET AVEC CROSS-VALIDATION
# ================================================================================

def create_sequences(X, y, seq_length=10):
    """Création de séquences temporelles"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length-1])
    return np.array(X_seq), np.array(y_seq)

def train_with_cross_validation(X, y, config, model_type='transformer'):
    """
    Entraînement avec K-Fold Cross-Validation
    """
    kfold = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    best_models = []
    
    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION: {config.N_FOLDS} FOLDS")
    print(f"{'='*80}")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n--- Fold {fold+1}/{config.N_FOLDS} ---")
        
        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Créer séquences
        X_train_seq, y_train_seq = create_sequences(X_train_fold, y_train_fold, config.SEQ_LENGTH)
        X_val_seq, y_val_seq = create_sequences(X_val_fold, y_val_fold, config.SEQ_LENGTH)
        
        # DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_seq),
            torch.FloatTensor(y_train_seq)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_seq),
            torch.FloatTensor(y_val_seq)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
        
        # Modèle
        input_size = X.shape[1]
        if model_type == 'transformer':
            model = OptimizedGenomicTransformer(
                input_size, config.D_MODEL, config.NHEAD, 
                config.NUM_LAYERS, config.DROPOUT
            ).to(config.DEVICE)
        else:
            model = AttentionLSTM(
                input_size, config.D_MODEL, config.NUM_LAYERS, config.DROPOUT
            ).to(config.DEVICE)
        
        # Optimiseur avec weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Scheduler
        scheduler = WarmupCosineScheduler(
            optimizer, config.WARMUP_EPOCHS, config.EPOCHS, config.LEARNING_RATE
        )
        
        criterion = nn.MSELoss()
        
        # Entraînement
        best_val_r2 = -np.inf
        patience_counter = 0
        history = {'train_loss': [], 'train_r2': [], 'val_loss': [], 'val_r2': []}
        
        for epoch in range(config.EPOCHS):
            # Training
            train_loss, train_r2 = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
            
            # Validation
            val_loss, val_r2, val_rmse, val_mae, _, _ = validate(model, val_loader, criterion, config.DEVICE)
            
            # Scheduler step
            current_lr = scheduler.step(epoch)
            
            # History
            history['train_loss'].append(train_loss)
            history['train_r2'].append(train_r2)
            history['val_loss'].append(val_loss)
            history['val_r2'].append(val_r2)
            
            # Early stopping
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= config.PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}: Train R²={train_r2:.4f}, Val R²={val_r2:.4f}, LR={current_lr:.6f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Final evaluation
        _, _, rmse, mae, preds, targets = validate(model, val_loader, criterion, config.DEVICE)
        
        fold_results.append({
            'fold': fold + 1,
            'r2': best_val_r2,
            'rmse': rmse,
            'mae': mae,
            'history': history
        })
        
        best_models.append(model)
        
        print(f"    Best Val R²: {best_val_r2:.4f}")
    
    # Moyennes des folds
    avg_r2 = np.mean([r['r2'] for r in fold_results])
    std_r2 = np.std([r['r2'] for r in fold_results])
    
    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"Average R²: {avg_r2:.4f} ± {std_r2:.4f}")
    fold_values = [f"{r['r2']:.4f}" for r in fold_results]
    print(f"Individual folds: {fold_values}")
    
    return fold_results, best_models

# ================================================================================
# FONCTION PRINCIPALE
# ================================================================================

def main():
    print("="*80)
    print("ALGORITHME OPTIMISÉ POUR LA PRÉDICTION GÉNOMIQUE - DATASET PEPPER")
    print("="*80)
    print(f"\nArchitecture: Transformer avancé avec attention multi-têtes")
    print(f"Optimisations: Residual connections, LayerNorm, Warmup+Cosine LR")
    print(f"Validation: {config.N_FOLDS}-Fold Cross-Validation")
    
    # Chargement des données PEPPER
    print("\n" + "="*80)
    print("CHARGEMENT DES DONNÉES PEPPER")
    print("="*80)
    
    # Données Pepper
    real_data_dir = config.BASE_DIR / "02_processed_data/pepper"
    context_dir = config.BASE_DIR / "04_augmentation/pepper/context_learning"
    
    # Charger données réelles Pepper
    X_real = pd.read_csv(real_data_dir / "X_aligned.csv")
    y_real = pd.read_csv(real_data_dir / "y_aligned.csv")
    df_real = X_real.merge(y_real, on='Sample_ID', how='inner')
    
    feature_cols = [c for c in df_real.columns if c.startswith('SNP_')]
    X = df_real[feature_cols].values
    y = df_real['YR_LS'].values
    
    print(f"Données réelles Pepper: {len(X)} échantillons, {len(feature_cols)} SNPs")
    
    # Option: Ajouter données synthétiques (meilleur contexte GLM5)
    use_synthetic = True
    if use_synthetic:
        # Utiliser le meilleur contexte identifié précédemment (Context A de GLM5)
        ctx_file = context_dir / "glm5/synthetic_glm5_context_A_500samples.csv"
        if ctx_file.exists():
            df_synth = pd.read_csv(ctx_file)
            # Sélectionner uniquement les colonnes SNP qui existent dans les données réelles
            available_cols = [c for c in feature_cols if c in df_synth.columns]
            X_synth = df_synth[available_cols].values
            y_synth = df_synth['YR_LS'].values
            
            # Combiner
            X_real_subset = df_real[available_cols].values
            X = np.vstack([X_real_subset, X_synth])
            y = np.concatenate([y, y_synth])
            print(f"Avec données synthétiques GLM5 Context A: {len(X)} échantillons")
            print(f"  - Réels: {len(X_real_subset)}")
            print(f"  - Synthétiques: {len(X_synth)}")
        else:
            print(f"  Fichier synthétique non trouvé: {ctx_file}")
            print(f"  Utilisation des données réelles uniquement")
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entraînement avec cross-validation
    print("\n" + "="*80)
    print("ENTRAÎNEMENT TRANSFORMER")
    print("="*80)
    transformer_results, transformer_models = train_with_cross_validation(
        X_scaled, y, config, model_type='transformer'
    )
    
    print("\n" + "="*80)
    print("ENTRAÎNEMENT LSTM+ATTENTION")
    print("="*80)
    lstm_results, lstm_models = train_with_cross_validation(
        X_scaled, y, config, model_type='lstm'
    )
    
    # Sauvegarde
    results_dir = config.BASE_DIR / "03_modeling_results/optimized_algorithm"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'transformer': {
            'avg_r2': np.mean([r['r2'] for r in transformer_results]),
            'std_r2': np.std([r['r2'] for r in transformer_results]),
            'folds': transformer_results
        },
        'lstm': {
            'avg_r2': np.mean([r['r2'] for r in lstm_results]),
            'std_r2': np.std([r['r2'] for r in lstm_results]),
            'folds': lstm_results
        }
    }
    
    with open(results_dir / "cv_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("RÉSULTATS FINAUX")
    print(f"{'='*80}")
    print(f"\nTransformer: {results['transformer']['avg_r2']:.4f} ± {results['transformer']['std_r2']:.4f}")
    print(f"LSTM+Attention: {results['lstm']['avg_r2']:.4f} ± {results['lstm']['std_r2']:.4f}")
    print(f"\nRésultats sauvegardés dans: {results_dir}")

if __name__ == '__main__':
    main()
