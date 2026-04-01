import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.layers(x)

class CNN1D(nn.Module):
    def __init__(self, input_dim):
        super(CNN1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        # x: [batch, input_dim] -> [batch, 1, input_dim]
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class LSTMModel(nn.Module):
    def __init__(self, input_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        # x: [batch, input_dim] -> [batch, input_dim, 1]
        x = x.unsqueeze(2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TransformerModel(nn.Module):
    def __init__(self, input_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(1, 32)
        # Use batch_first=True for older PyTorch compatibility
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(32 * input_dim, 1)
    def forward(self, x):
        # x: [batch, input_dim]
        x = x.unsqueeze(2) # [batch, input_dim, 1]
        x = self.embedding(x) # [batch, input_dim, 32]
        x = self.transformer(x) # [batch, input_dim, 32]
        x = x.reshape(x.size(0), -1) # [batch, input_dim * 32]
        return self.fc(x)

class HybridModel(nn.Module):
    """CNN + LSTM Hybrid"""
    def __init__(self, input_dim):
        super(HybridModel, self).__init__()
        self.conv = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        x = x.unsqueeze(1) # [B, 1, L]
        x = torch.relu(self.conv(x))
        x = x.transpose(1, 2) # [B, L, 32]
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def build_model(name="ridge", input_dim=None):
    name = name.lower()
    if name == "ridge":
        return Ridge()
    elif name == "xgboost":
        if not XGB_AVAILABLE: return None
        return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    elif name == "lightgbm":
        if not LGBM_AVAILABLE: return None
        return LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
    elif name == "mlp":
        return SimpleMLP(input_dim) if input_dim else None
    elif name == "cnn":
        return CNN1D(input_dim) if input_dim else None
    elif name == "lstm":
        return LSTMModel(input_dim) if input_dim else None
    elif name == "transformer":
        return TransformerModel(input_dim) if input_dim else None
    elif name == "hybrid":
        return HybridModel(input_dim) if input_dim else None
    return None
