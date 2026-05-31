"""
cnn_encoder.py
--------------
Trains a 1D-CNN on raw telemetry sequences to produce a 32-dimensional
embedding (fingerprint) per lap. The embedding is the bottleneck layer —
we train with cross-entropy on driver ID, then throw away the classifier
head and keep only the encoder.

Architecture:
    Input: (batch, channels, seq_len)  — 5 channels, 750 timesteps
        → Conv1d(5, 32, kernel=7) + BatchNorm + ReLU + MaxPool
        → Conv1d(32, 64, kernel=5) + BatchNorm + ReLU + MaxPool
        → Conv1d(64, 128, kernel=3) + BatchNorm + ReLU + MaxPool
        → GlobalAveragePool  →  (batch, 128)
        → Linear(128, 32)    →  embedding  (batch, 32)
        → Linear(32, n_classes)  →  logits for training

Usage:
    python src/models/cnn_encoder.py
"""

import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

TELEMETRY_CHANNELS = ["Throttle", "Brake", "Speed", "nGear", "RPM"]
SEQ_LEN = 750  # pad/truncate all laps to this length


def load_raw_telemetry(processed_dir: str, race_tag: str, drivers: list) -> pd.DataFrame:
    dfs = []
    for driver in drivers:
        path = os.path.join(processed_dir, race_tag, f"{driver}.parquet")
        df = pd.read_parquet(path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def build_sequences(df: pd.DataFrame, seq_len: int) -> tuple:
    """
    For each (Driver, LapNumber) group, extract the telemetry channels,
    normalize each channel to [0, 1], and pad/truncate to seq_len.

    Returns:
        sequences: np.array of shape (n_laps, n_channels, seq_len)
        labels:    list of driver strings, one per lap
        lap_meta:  list of (driver, lap_number) tuples
    """
    sequences = []
    labels    = []
    lap_meta  = []

    # Per-channel min/max for normalization (computed across all drivers)
    channel_stats = {}
    for ch in TELEMETRY_CHANNELS:
        if ch in df.columns:
            ch_vals = df[ch].astype(float)
            channel_stats[ch] = (ch_vals.min(), ch_vals.max())

    for (driver, lap_num), lap_df in df.groupby(["Driver", "LapNumber"]):
        if len(lap_df) < 50:
            continue

        channels = []
        for ch in TELEMETRY_CHANNELS:
            if ch not in lap_df.columns:
                channels.append(np.zeros(seq_len))
                continue

            signal = lap_df[ch].fillna(0).values.astype(np.float32)

            # Normalize to [0, 1]
            ch_min, ch_max = channel_stats[ch]
            if ch_max > ch_min:
                signal = (signal - ch_min) / (ch_max - ch_min)

            # Pad or truncate to seq_len
            if len(signal) >= seq_len:
                signal = signal[:seq_len]
            else:
                signal = np.pad(signal, (0, seq_len - len(signal)), mode='edge')

            channels.append(signal)

        seq = np.stack(channels, axis=0)  # (n_channels, seq_len)
        sequences.append(seq)
        labels.append(driver)
        lap_meta.append((driver, lap_num))

    sequences = np.array(sequences, dtype=np.float32)
    print(f"[sequences] Shape: {sequences.shape}  |  Labels: {len(labels)}")
    return sequences, labels, lap_meta


class TelemetryDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels,    dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class DriverEncoder(nn.Module):
    """
    1D-CNN that maps a raw telemetry sequence to a 32-dim embedding.
    The encoder (everything before the final classifier head) is what
    we extract for UMAP visualization.
    """
    def __init__(self, n_channels: int, seq_len: int, embedding_dim: int, n_classes: int):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),          # seq_len → 375

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),          # 375 → 187

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),              # → (batch, 128, 1)
        )

        self.bottleneck = nn.Sequential(
            nn.Flatten(),                         # → (batch, 128)
            nn.Linear(128, embedding_dim),        # → (batch, 32)
            nn.ReLU(),
        )

        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):
        features   = self.encoder(x)
        embedding  = self.bottleneck(features)
        logits     = self.classifier(embedding)
        return logits

    def get_embedding(self, x):
        """Extract 32-dim embedding without the classifier head."""
        with torch.no_grad():
            features  = self.encoder(x)
            embedding = self.bottleneck(features)
        return embedding


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * len(y_batch)
        total_correct += (logits.argmax(1) == y_batch).sum().item()
        total         += len(y_batch)
    return total_loss / total, total_correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            total_loss    += loss.item() * len(y_batch)
            total_correct += (logits.argmax(1) == y_batch).sum().item()
            total         += len(y_batch)
    return total_loss / total, total_correct / total


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    config = load_config()

    race_tag     = f"{config['session']['year']}_{config['session']['race'].lower()}"
    processed_dir = config["data"]["processed_dir"]
    drivers       = config["drivers"]
    embedding_dim = config["model"]["embedding_dim"]
    seed          = config["model"]["random_seed"]
    models_dir    = os.path.join("outputs", "models")
    features_dir  = config["data"]["features_dir"]

    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"[device] Using: {device}")

    # Load and prepare data
    print("\n[data] Loading telemetry...")
    df = load_raw_telemetry(processed_dir, race_tag, drivers)
    sequences, labels_str, lap_meta = build_sequences(df, SEQ_LEN)

    le = LabelEncoder()
    labels = le.fit_transform(labels_str)
    n_classes  = len(le.classes_)
    n_channels = sequences.shape[1]

    print(f"[data] Classes: {list(le.classes_)}  |  n_channels: {n_channels}  |  seq_len: {SEQ_LEN}")

    # Train/val split — stratified by driver
    idx = np.arange(len(sequences))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, stratify=labels, random_state=seed
    )

    train_ds = TelemetryDataset(sequences[train_idx], labels[train_idx])
    val_ds   = TelemetryDataset(sequences[val_idx],   labels[val_idx])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

    print(f"[split] Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # Model
    model = DriverEncoder(
        n_channels=n_channels,
        seq_len=SEQ_LEN,
        embedding_dim=embedding_dim,
        n_classes=n_classes,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("\n[train] Starting training...")
    best_val_acc = 0.0
    best_epoch   = 0
    patience_counter = 0
    EPOCHS = 60
    EARLY_STOP_PATIENCE = 15

    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss":   val_loss,   "val_acc":   val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            patience_counter = 0
            # Save best model
            os.makedirs(models_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(models_dir, f"{race_tag}_cnn_encoder.pt"))
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | "
                  f"train loss {train_loss:.4f}  acc {train_acc*100:.1f}% | "
                  f"val loss {val_loss:.4f}  acc {val_acc*100:.1f}%")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n[early stop] No improvement for {EARLY_STOP_PATIENCE} epochs. Stopping.")
            break

    print(f"\n[best] Val accuracy: {best_val_acc*100:.1f}%  at epoch {best_epoch}")

    # Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(features_dir, f"{race_tag}_cnn_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"[saved] Training history → {history_path}")

    # Extract embeddings for ALL laps using best model
    print("\n[embed] Extracting embeddings from best model...")
    model.load_state_dict(
        torch.load(os.path.join(models_dir, f"{race_tag}_cnn_encoder.pt"),
                   map_location=device)
    )
    model.eval()

    all_embeddings = []
    full_ds    = TelemetryDataset(sequences, labels)
    full_loader = DataLoader(full_ds, batch_size=32, shuffle=False)

    with torch.no_grad():
        for X_batch, _ in full_loader:
            emb = model.get_embedding(X_batch.to(device))
            all_embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    print(f"[embed] Embeddings shape: {embeddings.shape}")

    # Save embeddings with metadata
    embed_df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embedding_dim)])
    embed_df["Driver"]    = labels_str
    embed_df["LapNumber"] = [m[1] for m in lap_meta]
    embed_path = os.path.join(features_dir, f"{race_tag}_embeddings.csv")
    embed_df.to_csv(embed_path, index=False)
    print(f"[saved] Embeddings → {embed_path}")

    # Save label encoder
    import joblib
    joblib.dump(le, os.path.join(models_dir, f"{race_tag}_cnn_label_encoder.pkl"))
    print(f"[saved] LabelEncoder → {models_dir}")


if __name__ == "__main__":
    main()