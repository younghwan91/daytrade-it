"""MLflow integration for experiment tracking and model registry.

This module provides MLflowTracker for logging experiments, metrics,
parameters, and models during ML training.
"""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient


@dataclass
class MLflowTracker:
    """MLflow experiment tracking and model registry integration.

    Provides methods to track experiments, log parameters and metrics,
    and register models for deployment.

    Attributes:
        tracking_uri: MLflow tracking server URI
        artifact_location: Default artifact storage location
        _client: MLflow client instance
        _active_run_id: Currently active run ID
    """

    tracking_uri: str = "mlruns"
    artifact_location: str | None = None
    _client: MlflowClient = field(init=False, repr=False)
    _active_run_id: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize MLflow client and set tracking URI."""
        mlflow.set_tracking_uri(self.tracking_uri)
        self._client = MlflowClient(self.tracking_uri)

    def start_run(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> str:
        """Start a new MLflow run.

        Args:
            experiment_name: Name of the experiment
            run_name: Optional name for this specific run
            tags: Optional tags for the run
            nested: Whether this is a nested run

        Returns:
            Run ID of the started run
        """
        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=self.artifact_location,
            )
        else:
            experiment_id = experiment.experiment_id

        # Start the run
        run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags=tags,
            nested=nested,
        )
        self._active_run_id = run.info.run_id
        return self._active_run_id

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters for the current run.

        Args:
            params: Dictionary of parameter names and values
        """
        if not self._active_run_id:
            raise RuntimeError("No active run. Call start_run() first.")

        # MLflow requires string values, convert as needed
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics for the current run.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for time-series metrics
        """
        if not self._active_run_id:
            raise RuntimeError("No active run. Call start_run() first.")

        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_model(
        self,
        model: Any,
        model_name: str,
        registered_model_name: str | None = None,
        signature: Any | None = None,
        input_example: Any | None = None,
    ) -> str:
        """Log a model artifact and optionally register it.

        Args:
            model: The model object to log
            model_name: Artifact path name for the model
            registered_model_name: Name to register in model registry (optional)
            signature: Model signature for input/output schema
            input_example: Example input for documentation

        Returns:
            URI of the logged model
        """
        if not self._active_run_id:
            raise RuntimeError("No active run. Call start_run() first.")

        model_info = mlflow.xgboost.log_model(
            model,
            artifact_path=model_name,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
        )
        return str(model_info.model_uri)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log a local file or directory as an artifact.

        Args:
            local_path: Path to the file or directory
            artifact_path: Optional subdirectory within artifacts
        """
        if not self._active_run_id:
            raise RuntimeError("No active run. Call start_run() first.")

        mlflow.log_artifact(str(local_path), artifact_path)

    def log_dict(self, data: dict[str, Any], filename: str) -> None:
        """Log a dictionary as a JSON artifact.

        Args:
            data: Dictionary to log
            filename: Name of the artifact file
        """
        if not self._active_run_id:
            raise RuntimeError("No active run. Call start_run() first.")

        mlflow.log_dict(data, filename)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags on the current run.

        Args:
            tags: Dictionary of tag names and values
        """
        if not self._active_run_id:
            raise RuntimeError("No active run. Call start_run() first.")

        mlflow.set_tags(tags)

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run.

        Args:
            status: Final status ("FINISHED", "FAILED", "KILLED")
        """
        if self._active_run_id:
            mlflow.end_run(status=status)
            self._active_run_id = None

    def get_best_run(
        self,
        experiment_name: str,
        metric: str,
        ascending: bool = False,
    ) -> Run | None:
        """Get the best run from an experiment based on a metric.

        Args:
            experiment_name: Name of the experiment
            metric: Metric name to sort by
            ascending: If True, lower is better; if False, higher is better

        Returns:
            Best Run object, or None if no runs found
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None

        order = "ASC" if ascending else "DESC"
        runs = self._client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )

        return runs[0] if runs else None

    def get_run(self, run_id: str) -> Run:
        """Get a specific run by ID.

        Args:
            run_id: The run ID

        Returns:
            Run object
        """
        return self._client.get_run(run_id)

    def load_model(self, model_uri: str) -> Any:
        """Load a model from MLflow.

        Args:
            model_uri: URI of the model (e.g., "runs:/run_id/model")

        Returns:
            Loaded model object
        """
        return mlflow.xgboost.load_model(model_uri)

    @property
    def active_run_id(self) -> str | None:
        """Get the currently active run ID."""
        return self._active_run_id

    @contextmanager
    def run_context(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Generator[str, None, None]:
        """Context manager for MLflow runs.

        Automatically ends the run when exiting the context.

        Args:
            experiment_name: Name of the experiment
            run_name: Optional name for this specific run
            tags: Optional tags for the run

        Yields:
            Run ID of the started run
        """
        run_id = self.start_run(experiment_name, run_name, tags)
        try:
            yield run_id
        except Exception:
            self.end_run(status="FAILED")
            raise
        else:
            self.end_run(status="FINISHED")
