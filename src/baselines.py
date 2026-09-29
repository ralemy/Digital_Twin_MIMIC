"""
Non-LLM baselines — Chapter 5, Section 5.5, step 3.

  - naive_forecast          : last-value-carried-forward, per patient, no training.
  - GBMBaseline              : one HistGradientBoostingRegressor per variable per
                                forecast hour, trained on observation-window summary
                                features. A standard, strong tabular baseline for
                                irregularly sampled clinical time series.
  - LSTMBaseline             : a small local PyTorch sequence-to-sequence LSTM,
                                trained from scratch on the training split.

All three run entirely on local CPU/GPU (the LSTM will use CUDA automatically
if available, i.e. the 1080 Ti) with no external services.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from common import get_logger
from harmonization_agent import summarize_observation

log = get_logger("baselines")

# torch is only needed for LSTMBaseline; importing it lazily means the naive
# and GBM baselines (and everything else in this file) work even in an
# environment where torch isn't installed / isn't wanted.
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Naive last-value-carried-forward
# ---------------------------------------------------------------------------

def naive_forecast(obs: np.ndarray, horizon_hours: int) -> dict:
    n_vars = obs.shape[1]
    forecast = np.zeros((horizon_hours, n_vars))
    halfwidths = np.zeros(n_vars)
    for v in range(n_vars):
        valid = obs[:, v][~np.isnan(obs[:, v])]
        last = valid[-1] if len(valid) else 0.0
        std = np.std(valid) if len(valid) > 1 else 1.0
        forecast[:, v] = last
        halfwidths[v] = max(std, 1e-3) * 1.5
    return {"forecast_array": forecast, "halfwidth_array": halfwidths}


# ---------------------------------------------------------------------------
# Gradient-boosted trees
# ---------------------------------------------------------------------------

def _summary_features(obs: np.ndarray, variables: list[str]) -> np.ndarray:
    summary = summarize_observation(obs, variables)
    feats = []
    for var in variables:
        s = summary[var]
        feats.extend([
            s["last"] if s["last"] is not None else 0.0,
            s["mean"] if s["mean"] is not None else 0.0,
            s["std"] if s["std"] is not None else 0.0,
            s["slope"] if s["slope"] is not None else 0.0,
            float(s["n_obs"]),
        ])
    return np.array(feats, dtype=float)


class GBMBaseline:
    """One regressor per (variable, forecast_hour) pair — simple, robust,
    trains in seconds to a few minutes on a lean-scope cohort on CPU."""

    def __init__(self, variables: list[str], horizon_hours: int):
        self.variables = variables
        self.horizon_hours = horizon_hours
        self.models: dict[tuple[str, int], HistGradientBoostingRegressor] = {}
        self.residual_std: dict[tuple[str, int], float] = {}

    def fit(self, train_tensors: dict[int, dict]) -> "GBMBaseline":
        X = np.stack([_summary_features(t["obs"], self.variables) for t in train_tensors.values()])
        for v_idx, var in enumerate(self.variables):
            for h in range(self.horizon_hours):
                y = np.array([t["horizon"][h, v_idx] for t in train_tensors.values()])
                mask = ~np.isnan(y)
                if mask.sum() < 10:
                    continue
                model = HistGradientBoostingRegressor(max_depth=4, max_iter=150, random_state=0)
                model.fit(X[mask], y[mask])
                self.models[(var, h)] = model
                preds = model.predict(X[mask])
                self.residual_std[(var, h)] = float(np.std(y[mask] - preds)) or 1.0
        log.info("GBM baseline trained: %d (variable, hour) models.", len(self.models))
        return self

    def predict(self, obs: np.ndarray) -> dict:
        x = _summary_features(obs, self.variables).reshape(1, -1)
        forecast = np.full((self.horizon_hours, len(self.variables)), np.nan)
        halfwidths = np.ones(len(self.variables))
        for v_idx, var in enumerate(self.variables):
            hw_vals = []
            for h in range(self.horizon_hours):
                model = self.models.get((var, h))
                if model is None:
                    continue
                forecast[h, v_idx] = model.predict(x)[0]
                hw_vals.append(self.residual_std[(var, h)])
            if hw_vals:
                halfwidths[v_idx] = np.mean(hw_vals) * 1.645  # ~90% interval assuming Gaussian residuals
        return {"forecast_array": forecast, "halfwidth_array": halfwidths}


# ---------------------------------------------------------------------------
# Small sequence-to-sequence LSTM
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _Seq2SeqLSTM(nn.Module):
        def __init__(self, n_vars: int, hidden: int = 64, horizon_hours: int = 24):
            super().__init__()
            self.encoder = nn.LSTM(input_size=n_vars, hidden_size=hidden, batch_first=True)
            self.decoder = nn.LSTM(input_size=n_vars, hidden_size=hidden, batch_first=True)
            self.head = nn.Linear(hidden, n_vars)
            self.horizon_hours = horizon_hours
            self.n_vars = n_vars

        def forward(self, obs: "torch.Tensor") -> "torch.Tensor":
            _, (h, c) = self.encoder(obs)
            # teacher-forcing-free autoregressive decode, seeded with the last observed step
            dec_input = obs[:, -1:, :]
            outputs = []
            for _ in range(self.horizon_hours):
                out, (h, c) = self.decoder(dec_input, (h, c))
                pred = self.head(out)
                outputs.append(pred)
                dec_input = pred
            return torch.cat(outputs, dim=1)  # (batch, horizon_hours, n_vars)


class LSTMBaseline:
    def __init__(self, variables: list[str], horizon_hours: int, device: str | None = None, hidden: int = 64):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "LSTMBaseline requires PyTorch. Install it with `pip install torch` "
                "(a CUDA build matching your GPU), or set baselines.run_lstm: false "
                "and drop 'lstm' from `conditions` in config.yaml to skip this baseline."
            )
        self.variables = variables
        self.horizon_hours = horizon_hours
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _Seq2SeqLSTM(len(variables), hidden=hidden, horizon_hours=horizon_hours).to(self.device)
        self.residual_std = np.ones(len(variables))
        self._norm_mean = None
        self._norm_std = None

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        filled = np.nan_to_num(arr, nan=0.0)
        return torch.tensor(filled, dtype=torch.float32)

    def fit(self, train_tensors: dict[int, dict], epochs: int = 100, lr: float = 1e-3) -> "LSTMBaseline":
        obs_stack = np.stack([t["obs"] for t in train_tensors.values()])
        hor_stack = np.stack([t["horizon"] for t in train_tensors.values()])

        self._norm_mean = np.nanmean(obs_stack, axis=(0, 1))
        self._norm_std = np.nanstd(obs_stack, axis=(0, 1))
        self._norm_std[self._norm_std == 0] = 1.0

        obs_norm = (np.nan_to_num(obs_stack, nan=np.nan) - self._norm_mean) / self._norm_std
        obs_norm = np.nan_to_num(obs_norm, nan=0.0)
        hor_norm = (hor_stack - self._norm_mean) / self._norm_std

        X = torch.tensor(obs_norm, dtype=torch.float32).to(self.device)
        Y = torch.tensor(np.nan_to_num(hor_norm, nan=0.0), dtype=torch.float32).to(self.device)
        Y_mask = torch.tensor(~np.isnan(hor_norm), dtype=torch.float32).to(self.device)

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for epoch in range(epochs):
            opt.zero_grad()
            pred = self.model(X)
            loss = ((pred - Y) ** 2 * Y_mask).sum() / Y_mask.sum().clamp(min=1)
            loss.backward()
            opt.step()
            if (epoch + 1) % 10 == 0:
                log.info("LSTM baseline epoch %d/%d, masked MSE loss=%.4f", epoch + 1, epochs, loss.item())

        with torch.no_grad():
            pred = self.model(X).cpu().numpy()
        residuals = (hor_norm - pred) * self._norm_std  # back to original units
        self.residual_std = np.nanstd(residuals, axis=(0, 1))
        self.residual_std[np.isnan(self.residual_std) | (self.residual_std == 0)] = 1.0

        self.model.eval()
        return self

    def predict(self, obs: np.ndarray) -> dict:
        obs_norm = (np.nan_to_num(obs, nan=np.nan) - self._norm_mean) / self._norm_std
        obs_norm = np.nan_to_num(obs_norm, nan=0.0)
        x = torch.tensor(obs_norm[None, :, :], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_norm = self.model(x).cpu().numpy()[0]
        pred = pred_norm * self._norm_std + self._norm_mean
        halfwidths = self.residual_std * 1.645
        return {"forecast_array": pred, "halfwidth_array": halfwidths}

    def predict_batch(self, obs_stack: np.ndarray, batch_size: int = 512) -> dict:
        """Same math as predict(), but for many patients in one (or a few)
        GPU forward passes instead of one Python call per patient. Enabled
        by performance.batch_predict_baselines in config.yaml — on a single
        consumer GPU (e.g. the 1080 Ti this project was first written
        against) the per-patient loop is already close to GPU-bound, so
        batching barely matters; on a 40GB card the per-call overhead
        dominates at cohort sizes in the hundreds-to-thousands, and this
        turns that into a handful of large matmuls. batch_size caps how many
        patients go through the encoder/decoder at once, purely to bound
        peak VRAM on very large cohorts — 512 is conservative for a 40GB
        card at this model's tiny hidden size; raise it if nvidia-smi shows
        headroom."""
        n = obs_stack.shape[0]
        forecasts = np.zeros((n, self.horizon_hours, len(self.variables)))
        obs_norm = (np.nan_to_num(obs_stack, nan=np.nan) - self._norm_mean) / self._norm_std
        obs_norm = np.nan_to_num(obs_norm, nan=0.0)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                x = torch.tensor(obs_norm[start:end], dtype=torch.float32).to(self.device)
                pred_norm = self.model(x).cpu().numpy()
                forecasts[start:end] = pred_norm * self._norm_std + self._norm_mean
        halfwidths = np.tile(self.residual_std * 1.645, (n, 1))
        return {"forecast_array": forecasts, "halfwidth_array": halfwidths}
