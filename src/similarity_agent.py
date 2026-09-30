"""
Patient-similarity agent — Chapter 5's architecture; the local counterpart to
the k-NN component motivated by the "Toward Clinical Digital Twins" pathway-
modeling precedent and, theoretically, by Ashby's law of requisite variety
(Theoretical Framework, Section 4.1): conditioning the forecast on a cohort of
similar prior patients gives the system more than one internal "mode of
response" instead of a single undifferentiated forecaster.

Fully local: a scikit-learn k-NN index built once over the training split's
observation-window summary features, queried at inference time for the k most
similar training patients to a given test patient. No network calls.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from common import get_logger
from harmonization_agent import summarize_observation

log = get_logger("similarity_agent")


def _feature_vector(obs: np.ndarray, variables: list[str]) -> np.ndarray:
    summary = summarize_observation(obs, variables)
    feats = []
    for var in variables:
        s = summary[var]
        feats.extend([
            s["last"] if s["last"] is not None else np.nan,
            s["mean"] if s["mean"] is not None else np.nan,
            s["std"] if s["std"] is not None else np.nan,
            s["slope"] if s["slope"] is not None else np.nan,
        ])
    return np.array(feats, dtype=float)


class SimilarityAgent:
    """Fit once on the training cohort; query per test patient."""

    def __init__(self, variables: list[str], k: int = 20):
        self.variables = variables
        self.k = k
        self.scaler = StandardScaler()
        self.nn = None
        self.train_stay_ids: list[int] = []
        self.train_horizons: dict[int, np.ndarray] = {}

    def fit(self, train_tensors: dict[int, dict]) -> SimilarityAgent:
        feats, ids = [], []
        for stay_id, t in train_tensors.items():
            fv = _feature_vector(t["obs"], self.variables)
            feats.append(fv)
            ids.append(stay_id)
            self.train_horizons[stay_id] = t["horizon"]
        feats = np.array(feats)
        col_means = np.nanmean(feats, axis=0)
        inds = np.where(np.isnan(feats))
        feats[inds] = np.take(col_means, inds[1])

        self.scaler.fit(feats)
        feats_scaled = self.scaler.transform(feats)
        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(ids)))
        self.nn.fit(feats_scaled)
        self.train_stay_ids = ids
        log.info("Similarity agent fit on %d training patients (k=%d).", len(ids), self.k)
        return self

    def query(self, obs: np.ndarray) -> dict:
        """Returns the k nearest training patients' stay_ids and their
        horizon-window outcomes, plus a simple cohort-mean trajectory that the
        forecasting agent can use as a conditioning signal."""
        fv = _feature_vector(obs, self.variables).reshape(1, -1)
        col_means = self.scaler.mean_
        inds = np.where(np.isnan(fv))
        if len(inds[0]) > 0:
            fv[inds] = np.take(col_means, inds[1]) # type: ignore
        fv_scaled = self.scaler.transform(fv)
        dist, idx = self.nn.kneighbors(fv_scaled) # type: ignore
        neighbor_ids = [self.train_stay_ids[i] for i in idx[0]]
        neighbor_horizons = np.stack([self.train_horizons[i] for i in neighbor_ids])  # (k, hor_h, n_var)
        cohort_mean_trajectory = np.nanmean(neighbor_horizons, axis=0)  # (hor_h, n_var)
        return {
            "neighbor_stay_ids": neighbor_ids,
            "distances": dist[0].tolist(),
            "cohort_mean_trajectory": cohort_mean_trajectory,
        }
