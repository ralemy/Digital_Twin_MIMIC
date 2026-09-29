"""
Evaluation metrics for MIMIC-IV digital twin forecasts.

Covers three families from the experiment plan:
  1. Point-forecast accuracy   (sMAPE, MAE, RMSE)
  2. Distributional fidelity   (marginal KS test, cross-variable correlation preservation)
  3. Calibration / plausibility (interval coverage, interval width, plausibility violation rate)

Conventions
-----------
- `y_true`, `y_pred`: shape (n_patients, n_timesteps, n_variables), float, np.nan for missing.
- `y_lower`, `y_upper`: same shape as y_pred; predicted (e.g. 90%) interval bounds.
- `variable_names`: list[str] of length n_variables, used for reporting and plausibility bounds.
- All functions ignore np.nan entries (treated as missing ground truth or unpredicted values)
  and return per-variable results plus an overall summary, since coverage varies a lot
  across MIMIC-IV variables and pooling silently would hide that.

Dependencies: numpy, scipy (for the KS test). Both are standard in a data-analysis environment.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. Point-forecast accuracy
# ---------------------------------------------------------------------------

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

        sMAPE = mean( 2 * |y_true - y_pred| / (|y_true| + |y_pred| + eps) )

    Reported as a fraction in [0, 2]; multiply by 100 for a percentage.
    Matches the metric DT-GPT reports, so use this for direct baseline comparison.
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    yt, yp = y_true[mask], y_pred[mask]
    return float(np.mean(2 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp) + eps)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def per_variable_point_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, variable_names: list[str]
) -> dict[str, dict[str, float]]:
    """Compute sMAPE / MAE / RMSE separately for each variable (axis=-1)."""
    n_vars = y_true.shape[-1]
    out = {}
    for i in range(n_vars):
        name = variable_names[i]
        yt, yp = y_true[..., i], y_pred[..., i]
        out[name] = {
            "smape": smape(yt, yp),
            "mae": mae(yt, yp),
            "rmse": rmse(yt, yp),
            "n_obs": int(np.sum(~(np.isnan(yt) | np.isnan(yp)))),
        }
    return out


# ---------------------------------------------------------------------------
# 2. Distributional fidelity
# ---------------------------------------------------------------------------

def marginal_ks_test(
    y_true: np.ndarray, y_pred: np.ndarray, variable_names: list[str]
) -> dict[str, dict[str, float]]:
    """
    Two-sample Kolmogorov-Smirnov test comparing the marginal distribution of
    predicted vs. actual values, per variable, pooled across patients/timesteps.

    A LOW ks_statistic (and high p-value) means the predicted distribution
    matches the true distribution well -- i.e. the twin isn't just accurate
    pointwise but also preserves realistic variable ranges/shapes.

    Note: KS p-values are sensitive to sample size (huge n makes tiny, clinically
    irrelevant differences "significant"), so report the KS statistic itself as
    the primary effect-size measure and treat the p-value as secondary context.
    """
    n_vars = y_true.shape[-1]
    out = {}
    for i in range(n_vars):
        name = variable_names[i]
        yt = y_true[..., i]
        yp = y_pred[..., i]
        yt = yt[~np.isnan(yt)]
        yp = yp[~np.isnan(yp)]
        if len(yt) == 0 or len(yp) == 0:
            out[name] = {"ks_statistic": np.nan, "p_value": np.nan}
            continue
        stat, p = ks_2samp(yt, yp)
        out[name] = {"ks_statistic": float(stat), "p_value": float(p)}
    return out


def cross_variable_correlation_preservation(
    y_true: np.ndarray, y_pred: np.ndarray, variable_names: list[str]
) -> dict[str, np.ndarray | float]:
    """
    Compares the true and predicted cross-variable Pearson correlation matrices.

    Flattens (patient, timestep) into one axis so correlations are computed
    across variables at matched observation points. Returns both matrices and
    a single summary number: the Frobenius norm of their difference (lower is
    better -- the twin preserves realistic co-movement between variables, e.g.
    heart rate and respiratory rate rising together).
    """
    n_vars = y_true.shape[-1]
    yt_flat = y_true.reshape(-1, n_vars)
    yp_flat = y_pred.reshape(-1, n_vars)

    # keep only rows with no missing values, so both matrices are computed
    # over exactly the same observation set
    valid = ~np.isnan(yt_flat).any(axis=1) & ~np.isnan(yp_flat).any(axis=1)
    yt_flat, yp_flat = yt_flat[valid], yp_flat[valid]

    if yt_flat.shape[0] < 2:
        return {"true_corr": None, "pred_corr": None, "frobenius_diff": np.nan}

    true_corr = np.corrcoef(yt_flat, rowvar=False)
    pred_corr = np.corrcoef(yp_flat, rowvar=False)
    frob_diff = float(np.linalg.norm(true_corr - pred_corr, ord="fro"))

    return {
        "true_corr": true_corr,
        "pred_corr": pred_corr,
        "frobenius_diff": frob_diff,
        "variable_names": variable_names,
    }


# ---------------------------------------------------------------------------
# 3. Calibration / plausibility
# ---------------------------------------------------------------------------

def interval_coverage(
    y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray
) -> float:
    """
    Empirical coverage of a predicted interval: fraction of true values that
    fall within [y_lower, y_upper]. For a well-calibrated 90% interval this
    should be close to 0.90 -- systematically higher means intervals are too
    wide (safe but uninformative); lower means overconfident.
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_lower) | np.isnan(y_upper))
    if mask.sum() == 0:
        return np.nan
    yt, lo, hi = y_true[mask], y_lower[mask], y_upper[mask]
    return float(np.mean((yt >= lo) & (yt <= hi)))


def mean_interval_width(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Average width of the predicted interval -- report alongside coverage,
    since a trivially wide interval gets perfect coverage but is useless."""
    mask = ~(np.isnan(y_lower) | np.isnan(y_upper))
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(y_upper[mask] - y_lower[mask]))


# Example physiologic plausibility bounds -- REPLACE with clinically reviewed
# ranges for your actual variable panel before using in the paper. These are
# illustrative starting points only, not a substitute for clinical sign-off.
DEFAULT_PLAUSIBLE_RANGES = {
    "heart_rate": (20, 250),          # bpm
    "resp_rate": (4, 60),             # breaths/min
    "spo2": (50, 100),                # %
    "map": (20, 180),                 # mmHg
    "lactate": (0.1, 30),             # mmol/L
    "creatinine": (0.1, 20),          # mg/dL
    "magnesium": (0.5, 6),            # mg/dL
    "wbc": (0.1, 100),                # x10^9/L
}


def plausibility_violation_rate(
    y_pred: np.ndarray,
    variable_names: list[str],
    plausible_ranges: dict[str, tuple[float, float]] = DEFAULT_PLAUSIBLE_RANGES,
) -> dict[str, float]:
    """
    For each variable, the fraction of predictions that fall outside a
    clinically plausible range. This is the metric that should most directly
    show the critic/validation agent's contribution -- compare this rate
    with vs. without the critic agent in the ablation.
    """
    out = {}
    for i, name in enumerate(variable_names):
        if name not in plausible_ranges:
            out[name] = np.nan
            continue
        lo, hi = plausible_ranges[name]
        preds = y_pred[..., i]
        preds = preds[~np.isnan(preds)]
        if len(preds) == 0:
            out[name] = np.nan
            continue
        out[name] = float(np.mean((preds < lo) | (preds > hi)))
    return out


# ---------------------------------------------------------------------------
# 4. Convenience: run everything and produce a summary report
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    point_metrics: dict = field(default_factory=dict)
    ks_results: dict = field(default_factory=dict)
    correlation_preservation: dict = field(default_factory=dict)
    coverage: float | None = None
    mean_width: float | None = None
    plausibility_violations: dict = field(default_factory=dict)

    def summary_table(self) -> str:
        """Render a compact plain-text summary suitable for a results table draft."""
        lines = ["variable\tsMAPE\tMAE\tKS_stat\tplausibility_violation_rate"]
        for name, pm in self.point_metrics.items():
            ks = self.ks_results.get(name, {}).get("ks_statistic", float("nan"))
            pv = self.plausibility_violations.get(name, float("nan"))
            lines.append(
                f"{name}\t{pm['smape']:.4f}\t{pm['mae']:.4f}\t{ks:.4f}\t{pv:.4f}"
            )
        if self.coverage is not None:
            lines.append(f"\nInterval coverage: {self.coverage:.3f}  "
                         f"(target ~0.90)  mean width: {self.mean_width:.3f}")
        if self.correlation_preservation.get("frobenius_diff") is not None:
            lines.append(
                f"Cross-variable correlation Frobenius diff: "
                f"{self.correlation_preservation['frobenius_diff']:.4f} (lower is better)"
            )
        return "\n".join(lines)


def evaluate_twin(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    variable_names: list[str],
    y_lower: np.ndarray | None = None,
    y_upper: np.ndarray | None = None,
    plausible_ranges: dict[str, tuple[float, float]] = DEFAULT_PLAUSIBLE_RANGES,
) -> EvaluationReport:
    """One-call entry point: computes every metric family and returns a report.

    Example
    -------
    >>> report = evaluate_twin(y_true, y_pred, variable_names,
    ...                         y_lower=y_lower, y_upper=y_upper)
    >>> print(report.summary_table())
    """
    report = EvaluationReport()
    report.point_metrics = per_variable_point_metrics(y_true, y_pred, variable_names)
    report.ks_results = marginal_ks_test(y_true, y_pred, variable_names)
    report.correlation_preservation = cross_variable_correlation_preservation(
        y_true, y_pred, variable_names
    )
    report.plausibility_violations = plausibility_violation_rate(
        y_pred, variable_names, plausible_ranges
    )
    if y_lower is not None and y_upper is not None:
        report.coverage = interval_coverage(y_true, y_lower, y_upper)
        report.mean_width = mean_interval_width(y_lower, y_upper)
    return report


if __name__ == "__main__":
    # Minimal smoke test with synthetic data, so you can confirm the pipeline
    # runs before plugging in real MIMIC-IV-derived arrays.
    rng = np.random.default_rng(0)
    n_patients, n_timesteps, n_vars = 50, 24, 3
    names = ["heart_rate", "resp_rate", "spo2"]

    y_true = rng.normal(loc=[85, 18, 96], scale=[10, 3, 2],
                         size=(n_patients, n_timesteps, n_vars))
    noise = rng.normal(scale=[5, 1.5, 1], size=y_true.shape)
    y_pred = y_true + noise
    y_lower = y_pred - 1.645 * np.array([5, 1.5, 1])
    y_upper = y_pred + 1.645 * np.array([5, 1.5, 1])

    report = evaluate_twin(y_true, y_pred, names, y_lower, y_upper)
    print(report.summary_table())
