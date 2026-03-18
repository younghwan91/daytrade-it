"""ML Models API router for model serving and management."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from gpt_quant_v2.interfaces.api.dependencies import (
    CorrelationIdDep,
    CurrentUserDep,
)

router = APIRouter()


# ============================================================================
# Request/Response Schemas
# ============================================================================


class ModelInfoResponse(BaseModel):
    """Model information response schema."""

    name: str
    version: str
    stage: str
    created_at: datetime
    metrics: dict[str, float] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)


class ModelVersionResponse(BaseModel):
    """Model version information."""

    version: str
    stage: str
    created_at: datetime
    metrics: dict[str, float] = Field(default_factory=dict)


class PredictionRequest(BaseModel):
    """Single prediction request."""

    ticker: str = Field(min_length=1, max_length=10, description="Stock ticker symbol")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    tickers: list[str] = Field(
        min_length=1,
        max_length=100,
        description="List of stock ticker symbols",
    )


class PredictionResponse(BaseModel):
    """Prediction result response."""

    ticker: str
    direction: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    features_used: dict[str, Any] = Field(default_factory=dict)
    model_version: str
    predicted_at: datetime


class BatchPredictionResponse(BaseModel):
    """Batch prediction results."""

    predictions: list[PredictionResponse]
    total: int
    model_name: str
    model_version: str


class PromoteModelRequest(BaseModel):
    """Request to promote a model to a new stage."""

    stage: str = Field(
        description="Target stage (staging, production)",
        pattern="^(development|staging|production)$",
    )


class PromoteModelResponse(BaseModel):
    """Response after promoting a model."""

    name: str
    version: str
    old_stage: str
    new_stage: str
    message: str


# ============================================================================
# Helper functions
# ============================================================================


def _get_registry(request: Request) -> Any:
    """Get model registry from app state."""
    registry = getattr(request.app.state, "model_registry", None)
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="Model registry not available",
        )
    return registry


def _get_predictor(request: Request) -> Any:
    """Get model predictor from app state."""
    predictor = getattr(request.app.state, "model_predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model predictor not available",
        )
    return predictor


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "",
    response_model=list[ModelInfoResponse],
    summary="List available models",
    description="Retrieve a list of all registered models with their latest versions.",
)
async def list_models(
    request: Request,
    current_user: CurrentUserDep,  # noqa: ARG001
    name: str | None = Query(
        default=None,
        description="Filter by model name",
    ),
) -> list[ModelInfoResponse]:
    """List all available models in the registry.

    Args:
        request: FastAPI request object.
        current_user: Current authenticated user (required for auth).
        name: Optional model name filter.

    Returns:
        List of model information.
    """
    registry = _get_registry(request)
    models = registry.list_models(name)

    return [
        ModelInfoResponse(
            name=model.name,
            version=model.version,
            stage=model.stage,
            created_at=model.created_at,
            metrics=model.metrics,
            parameters=model.parameters,
        )
        for model in models
    ]


@router.get(
    "/{name}/versions",
    response_model=list[ModelVersionResponse],
    summary="List model versions",
    description="Retrieve all versions of a specific model.",
)
async def list_model_versions(
    name: str,
    request: Request,
    current_user: CurrentUserDep,  # noqa: ARG001
) -> list[ModelVersionResponse]:
    """List all versions of a specific model.

    Args:
        name: Model name.
        request: FastAPI request object.
        current_user: Current authenticated user (required for auth).

    Returns:
        List of model versions.
    """
    registry = _get_registry(request)
    models = registry.list_models(name)

    if not models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{name}' not found",
        )

    return [
        ModelVersionResponse(
            version=model.version,
            stage=model.stage,
            created_at=model.created_at,
            metrics=model.metrics,
        )
        for model in models
    ]


@router.post(
    "/{name}/predict",
    response_model=PredictionResponse,
    summary="Get prediction for ticker",
    description="Generate a prediction for a single stock ticker using the specified model.",
)
async def predict(
    name: str,
    prediction_request: PredictionRequest,
    request: Request,
    current_user: CurrentUserDep,  # noqa: ARG001
    correlation_id: CorrelationIdDep,  # noqa: ARG001
    version: str | None = Query(
        default=None,
        description="Model version (latest if not specified)",
    ),
) -> PredictionResponse:
    """Generate a prediction for a single ticker.

    Args:
        name: Model name.
        prediction_request: Prediction request with ticker.
        request: FastAPI request object.
        current_user: Current authenticated user (required for auth).
        correlation_id: Request correlation ID (for tracing).
        version: Optional model version.

    Returns:
        Prediction result.
    """
    predictor = _get_predictor(request)

    try:
        predictor.load_model(name, version)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    try:
        result = predictor.predict(prediction_request.ticker)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}",
        ) from e

    return PredictionResponse(
        ticker=result.ticker,
        direction=result.direction.value,
        confidence=result.confidence,
        probabilities=result.probabilities,
        features_used=result.features_used,
        model_version=result.model_version,
        predicted_at=result.predicted_at,
    )


@router.post(
    "/{name}/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch predictions",
    description="Generate predictions for multiple stock tickers.",
)
async def predict_batch(
    name: str,
    batch_request: BatchPredictionRequest,
    request: Request,
    current_user: CurrentUserDep,  # noqa: ARG001
    correlation_id: CorrelationIdDep,  # noqa: ARG001
    version: str | None = Query(
        default=None,
        description="Model version (latest if not specified)",
    ),
) -> BatchPredictionResponse:
    """Generate predictions for multiple tickers.

    Args:
        name: Model name.
        batch_request: Batch prediction request with ticker list.
        request: FastAPI request object.
        current_user: Current authenticated user (required for auth).
        correlation_id: Request correlation ID (for tracing).
        version: Optional model version.

    Returns:
        Batch prediction results.
    """
    predictor = _get_predictor(request)

    try:
        model_info = predictor.load_model(name, version)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    try:
        results = predictor.predict_batch(batch_request.tickers)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {e}",
        ) from e

    predictions = [
        PredictionResponse(
            ticker=result.ticker,
            direction=result.direction.value,
            confidence=result.confidence,
            probabilities=result.probabilities,
            features_used=result.features_used,
            model_version=result.model_version,
            predicted_at=result.predicted_at,
        )
        for result in results
    ]

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        model_name=name,
        model_version=model_info.version,
    )


@router.put(
    "/{name}/versions/{version}/promote",
    response_model=PromoteModelResponse,
    summary="Promote model stage",
    description="Promote a model version to a new stage (staging, production).",
)
async def promote_model(
    name: str,
    version: str,
    promote_request: PromoteModelRequest,
    request: Request,
    current_user: CurrentUserDep,  # noqa: ARG001
    correlation_id: CorrelationIdDep,  # noqa: ARG001
) -> PromoteModelResponse:
    """Promote a model version to a new stage.

    Args:
        name: Model name.
        version: Model version.
        promote_request: Promotion request with target stage.
        request: FastAPI request object.
        current_user: Current authenticated user (required for auth).
        correlation_id: Request correlation ID (for tracing).

    Returns:
        Promotion result.
    """
    registry = _get_registry(request)

    models = registry.list_models(name)
    model_info = next(
        (m for m in models if m.version == version),
        None,
    )

    if model_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{name}' version '{version}' not found",
        )

    old_stage = model_info.stage

    try:
        registry.promote_model(name, version, promote_request.stage)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return PromoteModelResponse(
        name=name,
        version=version,
        old_stage=old_stage,
        new_stage=promote_request.stage,
        message=f"Successfully promoted {name} v{version} from {old_stage} to {promote_request.stage}",
    )
