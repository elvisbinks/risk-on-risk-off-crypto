# Enable future annotations for type hints
from __future__ import annotations

# Dataclass for clean configuration
from dataclasses import dataclass

# Type hints
from typing import Iterable, Optional, Tuple

# NumPy for numerical operations
import numpy as np

# Pandas for time-series data
import pandas as pd

# PyTorch: deep learning framework for building neural networks
import torch
import torch.nn as nn  # Neural network modules (layers, activations)
import torch.optim as optim  # Optimization algorithms (Adam, SGD, etc.)

# KMeans for clustering the learned embeddings
from sklearn.cluster import KMeans

# StandardScaler for feature normalization
from sklearn.preprocessing import StandardScaler


@dataclass
class AutoencoderConfig:
    # Dimension of bottleneck layer (compressed representation)
    # Lower values force model to learn most important features
    encoding_dim: int = 4

    # Hidden layer sizes between input and bottleneck
    # Creates deeper network for learning non-linear patterns
    hidden_dims: Tuple[int, ...] = (8,)

    # Number of clusters for KMeans on learned embeddings
    n_clusters: int = 2

    # Number of training epochs (full passes through data)
    epochs: int = 100

    # Mini-batch size for stochastic gradient descent
    # Smaller batches = more updates but noisier gradients
    batch_size: int = 32

    # Learning rate for Adam optimizer
    # Controls step size in parameter updates
    learning_rate: float = 1e-3

    # Random seed for reproducibility
    random_state: int = 42

    # Optional feature subset
    features: Optional[Tuple[str, ...]] = None


class SimpleAutoencoder(nn.Module):
    """Neural network autoencoder for non-linear dimensionality reduction.

    Architecture: Input -> Hidden Layers -> Bottleneck -> Hidden Layers -> Output
    The bottleneck forces the network to learn a compressed representation.
    """

    def __init__(self, input_dim: int, encoding_dim: int, hidden_dims: Tuple[int, ...] = (8,)):
        super().__init__()

        # Build encoder: compresses input to low-dimensional embedding
        layers = []
        prev_dim = input_dim

        # Add hidden layers with ReLU activation
        # ReLU(x) = max(0, x) introduces non-linearity
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))  # Fully connected layer
            layers.append(nn.ReLU())  # Non-linear activation
            prev_dim = h

        # Final encoder layer to bottleneck dimension (no activation)
        layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*layers)

        # Build decoder: reconstructs input from embedding
        # Mirror architecture of encoder
        dec_layers = []
        prev_dim = encoding_dim

        # Add hidden layers in reverse order
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev_dim, h))
            dec_layers.append(nn.ReLU())
            prev_dim = h

        # Final decoder layer to original input dimension
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        """Forward pass: encode then decode."""
        # Encode: compress to low-dimensional representation
        z = self.encoder(x)

        # Decode: reconstruct original input
        x_recon = self.decoder(z)

        # Return both reconstruction and embedding
        return x_recon, z

    def encode(self, x):
        """Get embedding without reconstruction."""
        return self.encoder(x)


def prepare_features(
    df: pd.DataFrame, feature_cols: Optional[Iterable[str]] = None
) -> Tuple[np.ndarray, pd.Index, StandardScaler, Tuple[str, ...]]:
    if feature_cols is None:
        feature_cols = tuple(c for c in df.columns if c not in ("Date",))
    X_df = df.loc[:, feature_cols].dropna()
    idx = X_df.index
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)
    return X, idx, scaler, tuple(feature_cols)


def train_autoencoder(
    X: np.ndarray, cfg: AutoencoderConfig
) -> Tuple[SimpleAutoencoder, np.ndarray]:
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    input_dim = X.shape[1]
    model = SimpleAutoencoder(input_dim, cfg.encoding_dim, cfg.hidden_dims)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    X_tensor = torch.FloatTensor(X)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for (batch_x,) in loader:
            optimizer.zero_grad()
            x_recon, _ = model(batch_x)
            loss = criterion(x_recon, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{cfg.epochs}, Loss: {epoch_loss/len(X):.6f}")

    model.eval()
    with torch.no_grad():
        _, embeddings = model(X_tensor)
        embeddings = embeddings.numpy()

    return model, embeddings


def fit_autoencoder(
    df: pd.DataFrame, cfg: Optional[AutoencoderConfig] = None
) -> Tuple[SimpleAutoencoder, KMeans, pd.Index, StandardScaler, Tuple[str, ...]]:
    cfg = cfg or AutoencoderConfig()
    X, idx, scaler, feats = prepare_features(df, cfg.features)
    model, embeddings = train_autoencoder(X, cfg)

    # Cluster embeddings
    kmeans = KMeans(n_clusters=cfg.n_clusters, random_state=cfg.random_state, n_init=10)
    kmeans.fit(embeddings)

    return model, kmeans, idx, scaler, feats


def predict_autoencoder(
    ae_model: SimpleAutoencoder,
    kmeans: KMeans,
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: Iterable[str],
) -> Tuple[pd.Series, np.ndarray]:
    X_df = df.loc[:, feature_cols].dropna()
    idx = X_df.index
    X = scaler.transform(X_df.values)
    X_tensor = torch.FloatTensor(X)

    ae_model.eval()
    with torch.no_grad():
        embeddings = ae_model.encode(X_tensor).numpy()

    z = kmeans.predict(embeddings)
    return pd.Series(z, index=idx, name="regime"), embeddings
