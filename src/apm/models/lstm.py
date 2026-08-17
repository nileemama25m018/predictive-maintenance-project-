from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SequenceDataset:
    X: np.ndarray
    y: np.ndarray


def make_sequences(
    df,
    feature_cols: list[str],
    target_col: str = "rul_capped",
    window_size: int = 30,
) -> SequenceDataset:
    """Build engine-local sliding windows for LSTM/GRU training."""
    X_parts: list[np.ndarray] = []
    y_parts: list[float] = []
    for _, engine in df.sort_values(["engine_id", "cycle"]).groupby("engine_id"):
        values = engine[feature_cols].to_numpy(dtype=np.float32)
        targets = engine[target_col].to_numpy(dtype=np.float32)
        if len(engine) < window_size:
            continue
        for end in range(window_size, len(engine) + 1):
            X_parts.append(values[end - window_size:end])
            y_parts.append(float(targets[end - 1]))
    if not X_parts:
        return SequenceDataset(np.empty((0, window_size, len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.float32))
    return SequenceDataset(np.stack(X_parts), np.array(y_parts, dtype=np.float32))


def train_lstm_regressor(
    train_ds: SequenceDataset,
    valid_ds: SequenceDataset,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden_size: int = 96,
    num_layers: int = 2,
    dropout: float = 0.2,
):
    """Train a compact PyTorch LSTM. Imported lazily so classical pipeline works without torch."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class LSTMRegressor(nn.Module):
        def __init__(self, input_size: int):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(train_ds.X.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_ds.X), torch.tensor(train_ds.y)),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_x = torch.tensor(valid_ds.X).to(device)
    valid_y = torch.tensor(valid_ds.y).to(device)

    best_state = None
    best_mae = float("inf")
    patience = 4
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(valid_x)
            mae = torch.mean(torch.abs(pred - valid_y)).item()
        history.append({"epoch": epoch, "valid_mae": mae})
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), history

