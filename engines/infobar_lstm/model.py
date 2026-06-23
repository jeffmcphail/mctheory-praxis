"""
engines/infobar_lstm/model.py -- the small info-bar LSTM (DEFINED, NOT FIT).

HARD PAUSE (protocol section 1): no model is fit in Cycle 58. This module
defines the architecture and the per-fold training routine so the apparatus is
complete and reviewable, but `train_one_fold` refuses to run unless an explicit
authorization flag is passed -- an executable expression of the pause, not just
a comment. run_audit.py (Run A) NEVER calls it.

torch is imported lazily so the rest of the pipeline (and the leakage audit)
runs without a torch install.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelSpec:
    input_size: int
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.3
    n_classes: int = 3          # triple-barrier {-1,0,+1} -> classes {0,1,2}
    seq_len: int = 32


def build_model(spec: ModelSpec):
    """Construct (do not train) the LSTM classifier. Imports torch lazily."""
    import torch.nn as nn

    class InfoBarLSTM(nn.Module):
        def __init__(self, s: ModelSpec):
            super().__init__()
            self.lstm = nn.LSTM(
                s.input_size, s.hidden_size, s.num_layers,
                batch_first=True,
                dropout=s.dropout if s.num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(s.hidden_size, s.hidden_size),
                nn.ReLU(),
                nn.Dropout(s.dropout),
                nn.Linear(s.hidden_size, s.n_classes),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    return InfoBarLSTM(spec)


def param_count(spec: ModelSpec) -> int:
    """Parameter count of the spec (no torch import; closed-form for an LSTM).

    Per-layer LSTM params = 4 * (in + hidden + 1) * hidden (+ recurrent bias).
    Head = hidden*hidden + hidden + hidden*n_classes + n_classes.
    """
    p = 0
    in_sz = spec.input_size
    for _ in range(spec.num_layers):
        p += 4 * (in_sz + spec.hidden_size) * spec.hidden_size  # weights
        p += 4 * spec.hidden_size * 2                           # 2 bias vectors
        in_sz = spec.hidden_size
    h = spec.hidden_size
    p += h * h + h + h * spec.n_classes + spec.n_classes
    return p


def train_one_fold(X_train, y_train, X_test, y_test, spec: ModelSpec,
                   *, authorized_run_b: bool = False, epochs: int = 60,
                   lr: float = 1e-3):
    """Train + evaluate one walk-forward fold. GATED.

    Refuses to run unless `authorized_run_b=True`, which Jeff sets ONLY after
    reviewing CYCLE58_VALIDATION_PROTOCOL.md (Run B authorization). This is the
    executable hard-pause: the deployment-grade run (Run C) is further gated on
    the data-maturity precondition (protocol section 7).
    """
    if not authorized_run_b:
        raise RuntimeError(
            "HARD PAUSE: model fitting is gated. Pass authorized_run_b=True "
            "only after Jeff reviews CYCLE58_VALIDATION_PROTOCOL.md (Run B). "
            "Cycle 58 ships Run A (plumbing/leakage audit) only -- no fit."
        )
    # --- Implementation below runs only post-authorization (Run B / Run C). ---
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    model = build_model(spec)
    # Map labels {-1,0,1} -> {0,1,2}.
    yt = torch.LongTensor(np.asarray(y_train, dtype=int) + 1)
    Xt = torch.FloatTensor(X_train)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test))
        pred = logits.argmax(1).numpy() - 1
    return {"pred": pred, "model": model}
