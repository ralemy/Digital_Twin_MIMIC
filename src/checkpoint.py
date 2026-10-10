"""
Checkpoints for src/run_experiment.py, so a run stopped part-way (Slurm time
limit, preemption, node failure, Ctrl+C) resumes where it left off when it's
started again, instead of redoing hours of LLM calls.

Everything lives under <work_dir>/checkpoints/run_experiment/:

    models/gbm/        gbm.pkl — regressors fitted so far, saved after each variable
    models/lstm/       lstm.pt — weights + optimizer state, saved every N epochs
    conditions/<condition>/
        batch_00000.npz, batch_00001.npz, ...
                       predictions for one batch of test patients each,
                       saved as soon as the batch finishes
        summary_rows.json
                       the condition's rows of all_conditions_summary.csv,
                       written last — its presence marks the condition done

Each of those directories also holds a fingerprint.json: a hash of the config
values and train/test patient ids its contents depend on. Resuming with a
different config or cohort for that piece raises CheckpointMismatch rather
than silently mixing results from two setups; start over with
`run_experiment.py --full-refresh`. Prompt or code changes are NOT detected —
use --full-refresh after changing those too.

Every file is written to a temporary name and renamed into place, so a job
killed mid-write leaves either the old file or the new one, never half a file.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from common import LLM_CONDITIONS, cfg_for_llm_variant, forecasting_settings, get_logger, split_condition

log = get_logger("checkpoint")


class CheckpointMismatch(RuntimeError):
    pass


def checkpoint_root(cfg: dict) -> Path:
    """paths.checkpoint_dir if set (e.g. a tuned config keeps its own),
    else <work_dir>/checkpoints."""
    return Path(cfg["paths"].get("checkpoint_dir") or Path(cfg["paths"]["work_dir"]) / "checkpoints")


@contextmanager
def atomic_path(path: Path):
    """Yield a temporary path next to `path`; rename it onto `path` only if
    the block finishes without an exception."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# What every model and condition depends on beyond the config: how the data
# are prepared. v2: raw values outside each variable's valid range are dropped,
# and a variable with no observations is filled with the training median
# instead of 0 (naive forecast, LLM fallback, prompt). v3: LLM forecasts
# are sanitised (absurd values filled, hard ranges clipped) and saved before
# post-processing too. Checkpoints made before that must not be reused.
DATA_VERSION = "v3_sanitised_llm_output"


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _ids_hash(ids) -> str:
    return _hash(sorted(int(i) for i in ids))


def model_fingerprint(cfg: dict, model: str, train_ids) -> dict:
    """What a fitted baseline depends on: the panel, the cohort settings, the
    exact training patients and (for the LSTM) its training settings."""
    fp = {"model": model, "variables": cfg["variables"], "cohort": cfg["cohort"],
          "train_ids": _ids_hash(train_ids), "data": DATA_VERSION}
    if model == "lstm":
        fp["lstm"] = {k: cfg["baselines"].get(k) for k in ("lstm_hidden_size", "lstm_epochs")}
        fp["lstm"].update(_set_keys(cfg["baselines"], ("lstm_learning_rate", "lstm_seed", "lstm_input")))
    if model == "gbm":
        extra = _set_keys(cfg["baselines"], ("gbm_max_depth", "gbm_max_iter", "gbm_learning_rate"))
        if extra:
            fp["gbm"] = extra
    return fp


def _set_keys(section: dict, keys) -> dict:
    """The given keys that are present in a config section. Settings added
    after the first experiments are fingerprinted only when set, so existing
    checkpoints made without them stay valid."""
    return {k: section[k] for k in keys if k in section}


def condition_fingerprint(cfg: dict, condition: str, train_ids, test_ids,
                          postprocess: dict | None = None) -> dict:
    """What a condition's predictions depend on. Baseline conditions depend on
    their fitted model; LLM conditions on the (variant's) llm settings plus
    the similarity and critic agents' settings."""
    base, variant = split_condition(condition)
    fp = {"condition": condition, "variables": cfg["variables"], "cohort": cfg["cohort"],
          "train_ids": _ids_hash(train_ids), "test_ids": _ids_hash(test_ids), "data": DATA_VERSION}
    if base in ("lstm", "gbm"):
        fp.update({k: v for k, v in model_fingerprint(cfg, base, train_ids).items() if k == base})
    if base in ("recent_mean", "persistence_blend"):
        fp["recent_mean_hours"] = cfg["baselines"].get("recent_mean_hours", 6)
    if postprocess is not None:
        # Post-processing fitted per condition by calibrate.py, instead of
        # the config's forecasting_agent values.
        fp["postprocess"] = postprocess
    if base in LLM_CONDITIONS:
        fp["llm"] = {k: v for k, v in cfg_for_llm_variant(cfg, variant)["llm"].items() if k != "variants"}
        fp["similarity_agent"] = cfg["similarity_agent"]
        fp["critic_agent"] = cfg["critic_agent"]
        # The settings this condition's prompt group runs with (shared +
        # forecasting_agent.per_condition), so a per-condition change leaves
        # the other group's checkpoints valid. Without per_condition this is
        # the section as it is, as before.
        fa = forecasting_settings(cfg, base)
        if fa:
            fp["forecasting_agent"] = fa
        # A variable the LLM leaves out now gets the naive forecast for that
        # variable only; before, the whole forecast fell back. Predictions
        # made under the old rule must not be reused.
        fp["missing_variable"] = "naive_fill"
        # Forecasts and critic corrections are now decoded under a JSON
        # schema fixing each list's length (format=json before), which
        # changes the generations: don't reuse the old ones.
        fp["output_format"] = "json_schema_v1"
        # A critic on another model (llm.variants.<v>.critic_variant): its
        # model's settings shape the corrected forecasts too.
        cv = fp["llm"].get("critic_variant")
        if cv is not None:
            fp["critic_llm"] = {k: v for k, v in cfg_for_llm_variant(cfg, cv)["llm"].items() if k != "variants"}
    return fp


class CheckpointTakenOver(RuntimeError):
    pass


class RunCheckpoint:
    """The checkpoint directory of one run_experiment.py configuration.

    claim() marks this run as the directory's owner (owner.json); every save
    checks the mark first. If another run has since claimed the directory
    (e.g. a resubmitted job, or a --full-refresh started before the previous
    job had fully stopped), the old run raises CheckpointTakenOver instead of
    writing into the new run's checkpoints."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.run_id: str | None = None

    def claim(self) -> None:
        self.run_id = uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)
        owner = {"run_id": self.run_id, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                 "host": socket.gethostname(), "pid": os.getpid()}
        with atomic_path(self.root / "owner.json") as tmp:
            tmp.write_text(json.dumps(owner, indent=2))

    def check_owner(self) -> None:
        if self.run_id is None:          # not claimed (e.g. tests): no ownership checks
            return
        try:
            current = json.loads((self.root / "owner.json").read_text()).get("run_id")
        except (OSError, ValueError):
            current = None
        if current != self.run_id:
            raise CheckpointTakenOver(
                f"Another run has taken over the checkpoints in {self.root} "
                f"(owner.json changed) — stopping this run without saving.")

    def exists(self) -> bool:
        return self.root.is_dir() and any(self.root.iterdir())

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def _bind(self, rel: str, fingerprint: dict) -> Path:
        """The directory for one checkpointed piece, created with its
        fingerprint, or checked against the fingerprint already there."""
        d = self.root / rel
        fp_path = d / "fingerprint.json"
        digest = _hash(fingerprint)
        if fp_path.exists():
            saved = json.loads(fp_path.read_text())
            if saved.get("digest") != digest:
                raise CheckpointMismatch(
                    f"The checkpoint in {d} was made with a different config or cohort than "
                    f"this run's. Rerun with --full-refresh to start over, or restore the "
                    f"config it was made with."
                )
        else:
            d.mkdir(parents=True, exist_ok=True)
            with atomic_path(fp_path) as tmp:
                tmp.write_text(json.dumps({"digest": digest, "fingerprint": fingerprint},
                                          indent=2, default=str))
        return d

    def model_path(self, model: str, fingerprint: dict, filename: str) -> Path:
        self.check_owner()
        return self._bind(f"models/{model}", fingerprint) / filename

    def condition(self, condition: str, fingerprint: dict) -> ConditionCheckpoint:
        self.check_owner()
        return ConditionCheckpoint(self._bind(f"conditions/{condition}", fingerprint), self.check_owner)


class ConditionCheckpoint:
    """Per-batch predictions and the finished summary rows of one condition."""

    def __init__(self, directory: Path, check_owner=lambda: None):
        self.dir = directory
        self._check_owner = check_owner

    def _batch_path(self, index: int) -> Path:
        return self.dir / f"batch_{index:05d}.npz"

    def load_batch(self, index: int, stay_ids: list[int]) -> dict | None:
        """The saved predictions for batch `index`, or None if it wasn't saved
        or was saved for different patients (e.g. checkpoint_batch_size
        changed since), in which case it is recomputed."""
        path = self._batch_path(index)
        if not path.exists():
            return None
        with np.load(path) as z:
            if z["stay_ids"].tolist() != list(stay_ids):
                return None
            return {k: z[k] for k in ("y_pred", "y_lower", "y_upper")} | {
                "violations": int(z["violations"]), "fallbacks": int(z["fallbacks"]),
                "clipped": int(z["clipped"]) if "clipped" in z.files else 0,
                "y_raw": z["y_raw"] if "y_raw" in z.files else None,
                # Baseline batches saved before per-variable fill counts existed.
                "filled": z["filled"] if "filled" in z.files else np.zeros(z["y_pred"].shape[2], dtype=int)}

    def save_batch(self, index: int, stay_ids: list[int], batch: dict) -> None:
        self._check_owner()
        with atomic_path(self._batch_path(index)) as tmp:
            with open(tmp, "wb") as f:
                np.savez(f, stay_ids=np.array(stay_ids),
                         y_pred=batch["y_pred"], y_lower=batch["y_lower"], y_upper=batch["y_upper"],
                         violations=batch["violations"], fallbacks=batch["fallbacks"],
                         filled=batch["filled"], clipped=batch["clipped"],
                         **({"y_raw": batch["y_raw"]} if batch.get("y_raw") is not None else {}))

    def load_rows(self, n_batches: int) -> list[dict] | None:
        """The condition's summary rows if it already finished, else None.
        A finished condition whose batch files are no longer all present
        (e.g. a bad batch was deleted by hand to have it recomputed) counts
        as unfinished, so the missing batches are recomputed and the summary
        rebuilt from the full set."""
        path = self.dir / "summary_rows.json"
        if not path.exists():
            return None
        missing = [i for i in range(n_batches) if not self._batch_path(i).exists()]
        if missing:
            log.info("%s: summary exists but batch(es) %s are missing — recomputing them.",
                     self.dir.name, missing)
            return None
        return json.loads(path.read_text())

    def save_rows(self, rows: list[dict]) -> None:
        self._check_owner()
        with atomic_path(self.dir / "summary_rows.json") as tmp:
            tmp.write_text(json.dumps(rows, indent=2, default=_to_json))


def _to_json(o):
    """numpy scalars (np.float32 etc.) aren't JSON-serializable by default."""
    return o.item() if hasattr(o, "item") else str(o)
