"""Model definitions: the PyTorch MLP and the scikit-learn / XGBoost baselines.

Kept separate from training code so the Streamlit app and the explainers can
import a model without pulling in any training logic.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from src import config

logger = logging.getLogger(__name__)

BASELINE_NAMES: tuple[str, ...] = ("logreg", "random_forest", "xgboost")
BASELINE_LABELS: dict[str, str] = {
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "mlp": "MLP (calibrated)",
}


# --------------------------------------------------------------------------- #
# PyTorch MLP
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Feed-forward classifier: n_in -> 256 -> 128 -> n_classes.

    Each hidden block is Linear -> BatchNorm -> ReLU -> Dropout(0.3).
    BatchNorm stabilises training on tiny datasets; Dropout regularises and is
    reused at prediction time for Monte-Carlo-Dropout uncertainty (Stage 3).
    The output is raw logits; softmax (with temperature) is applied outside.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_dims: Sequence[int] = config.MLP_HIDDEN_DIMS,
        dropout: float = config.MLP_DROPOUT,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return logits of shape ``(batch, n_out)``."""
        return self.net(x)


def build_mlp(n_in: int, n_out: int) -> MLP:
    """Create an MLP with the hyper-parameters from ``config``."""
    return MLP(n_in, n_out, config.MLP_HIDDEN_DIMS, config.MLP_DROPOUT)


def save_mlp(model: MLP, path: Path, temperature: float, extra: dict | None = None) -> None:
    """Save weights + architecture + calibration temperature in one file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    first = model.net[0]
    last = model.net[-1]
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_in": first.in_features,
            "n_out": last.out_features,
            "hidden_dims": list(config.MLP_HIDDEN_DIMS),
            "dropout": config.MLP_DROPOUT,
            "temperature": float(temperature),
            "extra": extra or {},
        },
        path,
    )


def load_mlp(path: Path = config.MODELS_DIR / "mlp.pt") -> tuple[MLP, float, dict]:
    """Load a saved MLP; returns ``(model_in_eval_mode, temperature, extra_info)``."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train first: python -m src.train_mlp")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MLP(ckpt["n_in"], ckpt["n_out"], ckpt["hidden_dims"], ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, float(ckpt["temperature"]), ckpt.get("extra", {})


@torch.no_grad()
def mlp_logits(model: MLP, X: np.ndarray) -> np.ndarray:
    """Deterministic logits (eval mode: dropout off, BatchNorm uses running stats)."""
    model.eval()
    return model(torch.as_tensor(np.asarray(X), dtype=torch.float32)).numpy()


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax of ``logits / temperature``."""
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def mlp_proba(model: MLP, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Class probabilities, optionally temperature-calibrated."""
    return softmax(mlp_logits(model, X), temperature)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
class BaselineClassifier:
    """Uniform wrapper so every baseline returns probabilities for ALL classes.

    XGBoost insists on labels 0..k-1 with every class present, and a CV fold may
    lack a rare class. We therefore re-index labels to a compact range for
    fitting and scatter the probabilities back into the full class space.
    """

    def __init__(self, name: str, n_classes: int, seed: int = config.SEED) -> None:
        if name not in BASELINE_NAMES:
            raise ValueError(f"Unknown baseline {name!r}; choose from {BASELINE_NAMES}")
        self.name = name
        self.n_classes = n_classes
        self.seed = seed
        self.classes_: np.ndarray | None = None
        self.est = self._make_estimator()

    def _make_estimator(self):
        # Imports are local so that a missing xgboost only breaks the xgboost baseline.
        if self.name == "logreg":
            from sklearn.linear_model import LogisticRegression

            # C=1 and L2 regularisation: a sensible default for 0/1 features.
            return LogisticRegression(max_iter=3000, C=1.0, random_state=self.seed)
        if self.name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(n_estimators=300, random_state=self.seed, n_jobs=-1)
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, tree_method="hist", random_state=self.seed,
            n_jobs=-1, verbosity=0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineClassifier":
        """Fit on ``X`` (n, d) and integer labels ``y``."""
        self.classes_ = np.unique(y)
        compact = np.searchsorted(self.classes_, y)
        self.est.fit(X, compact)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``(n, n_classes)`` probabilities (zero for classes unseen in training)."""
        assert self.classes_ is not None, "call fit() first"
        part = self.est.predict_proba(X)
        out = np.zeros((len(X), self.n_classes), dtype=np.float64)
        out[:, self.classes_] = part
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Most likely class index."""
        return self.predict_proba(X).argmax(1)
