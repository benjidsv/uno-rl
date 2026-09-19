"""Local MLflow tracking, with a local tabular metrics file that never depends on it.

docs/rl/pipeline-plan.md section 10. One experiment (`uno-v5-rl`), one run per
training seed/configuration, tags for `study_id`, arm, phase, seed, machine and
run kind. Only the coordinator logs; workers never write runs. Metrics are
batched once per update, with evaluation metrics at evaluation boundaries.

The MLflow **step** is cumulative learner decisions, and update index, games,
engine decisions, optimizer steps and cumulative active runtime are logged as
metrics beside it -- so a curve can be read against data, against updates or
against time without relying on the UI's step convention.

**The store is local and explicit.** `mlflow` is a project dependency, pinned by
`uv.lock`, and its version is recorded in every run's provenance. The tracking URI
is always set explicitly rather than left to a version-dependent default: SQLite
metadata at `<root>/mlflow.db` and artifacts under `<root>/mlflow-artifacts/`,
both inside the gitignored `logs/` tree. No tracking server, no model registry and
no serving deployment is required or used.

**Training never depends on it.** If the import fails, the store is unavailable, or
an upload raises mid-run, training continues and every metric still lands in the
local CSV; the run is marked as having pending uploads so an apparently successful
run is not mistaken for a complete artifact archive.

Undefined statistics -- an explained variance with zero return variance, for
instance -- are written as empty, never as an invented number.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

EXPERIMENT = "uno-v5-rl"
DEFAULT_ROOT = Path("logs/rl")  # gitignored; holds mlflow.db and mlflow-artifacts/


def _import_mlflow():
    """Import mlflow with its interactive hint silenced.

    The package prints an assistant hint to stdout on import, which would land in
    the middle of an experiment's own report. It is a display setting, not a
    behavior change."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow  # noqa: PLC0415

    return mlflow


def local_store(root: str | Path | None = None) -> tuple[str, str]:
    """`(tracking URI, artifact location)` for a machine-local store.

    Explicit by design (pipeline plan section 10): a version-dependent default
    would put the run somewhere else on the other machine, and cross-machine
    resume already creates a *new* local run rather than migrating a database.
    """
    base = Path(root or DEFAULT_ROOT).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{base / 'mlflow.db'}", (base / "mlflow-artifacts").as_uri()


def provenance(config_hash: str, extra: dict | None = None) -> dict:
    """Base commit, dirty patch marker, versions, device and schema hashes."""
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""

    import numpy
    import torch

    try:
        mlflow_version = _import_mlflow().__version__
    except Exception:
        mlflow_version = "unavailable"
    out = {
        "mlflow": mlflow_version,
        "base_commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "config_hash": config_hash,
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.node(),
        "cpu_count": os.cpu_count(),
    }
    out.update(extra or {})
    return out


@dataclass
class Tracker:
    """Local metrics file plus, when available, a local MLflow run."""

    directory: Path
    run_name: str
    tags: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    tracking_uri: str | None = None
    # Where mlflow.db and mlflow-artifacts/ live. One machine-local store holds every
    # run of the project, so `DEFAULT_ROOT` is the default rather than the run's own
    # directory: one database per run would defeat comparing runs in one experiment.
    store_root: Path | None = None
    # False keeps everything local: what a run does when the store is unavailable,
    # and what the outage tests exercise without touching a real database.
    use_mlflow: bool = True
    _run: object = None
    _mlflow: object = None
    pending_uploads: int = 0
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._csv = self.directory / "metrics.csv"
        self._fields: list[str] = []
        (self.directory / "params.json").write_text(json.dumps(self.params, indent=2, default=str))
        (self.directory / "tags.json").write_text(json.dumps(self.tags, indent=2, default=str))
        self._start_mlflow()

    def _start_mlflow(self) -> None:
        if not self.use_mlflow:
            self.unavailable_reason = "mlflow tracking disabled for this run; metrics are local only"
            return
        try:
            mlflow = _import_mlflow()
        except Exception as exc:
            self.unavailable_reason = f"mlflow is not importable ({exc.__class__.__name__}); metrics are local only"
            return
        try:
            uri, artifacts = local_store(self.store_root or DEFAULT_ROOT)
            mlflow.set_tracking_uri(self.tracking_uri or uri)  # explicit: never the version-dependent default
            client = mlflow.tracking.MlflowClient()
            found = client.get_experiment_by_name(EXPERIMENT)
            # The artifact root is fixed at creation, so it is set explicitly here
            # and never inherited from the process's working directory.
            experiment_id = found.experiment_id if found else client.create_experiment(EXPERIMENT, artifact_location=artifacts)
            self._run = mlflow.start_run(experiment_id=experiment_id, run_name=self.run_name, tags={k: str(v) for k, v in self.tags.items()})
            for i in range(0, len(self.params), 100):  # MLflow caps a batch at 100 params
                chunk = list(self.params.items())[i : i + 100]
                mlflow.log_params({k: str(v)[:500] for k, v in chunk})
            self._mlflow = mlflow
        except Exception as exc:  # a tracking outage must not lose a training run
            self.unavailable_reason = f"mlflow tracking unavailable: {exc}"
            self._run = None
            self._mlflow = None

    @property
    def available(self) -> bool:
        return self._mlflow is not None

    def log(
        self,
        metrics: dict,
        step: int,
        *,
        mlflow_metrics: dict | None = None,
    ) -> None:
        """Batch metrics, optionally with a namespaced MLflow presentation.

        The compact CSV retains the stable analysis names in ``metrics``. MLflow
        may instead receive slash-delimited names so its UI groups a large run
        into semantic folders. ``None`` values are unavailable in either view.
        """
        row = {"step": step, "wall_time": time.time(), **metrics}
        self._append(row)
        if self._mlflow is None:
            self.pending_uploads += 1
            return
        try:
            presented = metrics if mlflow_metrics is None else mlflow_metrics
            numeric = {k: float(v) for k, v in presented.items() if isinstance(v, (int, float)) and v is not None}
            self._mlflow.log_metrics(numeric, step=int(step))
        except Exception as exc:
            self.pending_uploads += 1
            self.unavailable_reason = f"metric upload failed: {exc}"

    def _append(self, row: dict) -> None:
        new = [k for k in row if k not in self._fields]
        if new and self._fields:
            # keep one stable column set: rewrite the header and the existing rows
            existing = list(csv.DictReader(self._csv.open())) if self._csv.exists() else []
            self._fields += new
            with self._csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._fields)
                w.writeheader()
                w.writerows(existing)
        elif new:
            self._fields = list(row)
            with self._csv.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._fields).writeheader()
        with self._csv.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fields).writerow({k: ("" if v is None else v) for k, v in row.items()})

    def artifact(self, path: str | Path) -> None:
        if self._mlflow is None:
            self.pending_uploads += 1
            return
        try:
            self._mlflow.log_artifact(str(path))
        except Exception as exc:
            self.pending_uploads += 1
            self.unavailable_reason = f"artifact upload failed: {exc}"

    def finish(self, status: str = "FINISHED") -> dict:
        summary = {
            "run_name": self.run_name,
            "mlflow": self.available,
            "pending_uploads": self.pending_uploads,
            "unavailable_reason": self.unavailable_reason,
            "metrics_file": str(self._csv),
            "status": status,
        }
        (self.directory / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        if self._mlflow is not None:
            try:
                self._mlflow.end_run(status=status)
            except Exception:
                pass
        return summary

    @property
    def run_id(self) -> str | None:
        return getattr(getattr(self._run, "info", None), "run_id", None)


__all__ = ["EXPERIMENT", "Tracker", "provenance"]
