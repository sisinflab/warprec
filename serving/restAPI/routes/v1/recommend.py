"""Parameterized recommendation endpoints for the WarpRec REST API.

Instead of separate per-model/per-dataset routes, a single ``/recommend/{model_key}``
endpoint dispatches to the correct inference flow based on the model type declared
in the serving configuration.
"""

from typing import List, Dict, Optional

from fastapi import APIRouter, HTTPException, Request

from serving.common.models.base import BaseModelCustom

router = APIRouter(prefix="/api/warprec/v1", tags=["Recommendations"])


class RecommendRequest(BaseModelCustom):
    """Unified recommendation request body.

    Provide the fields relevant to the model type:
    - Sequential models require ``item_sequence``.
    - Collaborative models require ``user_index``.
    - Contextual models require ``user_index`` and ``context``.

    Attributes:
        top_k (int): Number of recommendations to return.
        item_sequence (Optional[List[int]]): Ordered external item IDs for sequential models.
        user_index (Optional[int]): User identifier for collaborative or contextual models.
        context (Optional[List[int]]): Context feature values for contextual models.
    """

    top_k: int = 10
    item_sequence: Optional[List[int]] = None
    user_index: Optional[int] = None
    context: Optional[List[int]] = None


class RecommendResponse(BaseModelCustom):
    """Unified recommendation response body.

    Attributes:
        model_key (str): The model-dataset identifier used for this request.
        model_type (str): The recommender category (sequential, collaborative, contextual).
        recommendations (List[int]): Ordered list of recommended items.
    """

    model_key: str
    model_type: str
    recommendations: List[int]


@router.post(
    "/recommend/{model_key}",
    status_code=200,
    response_model=RecommendResponse,
    description="Get recommendations from any loaded model-dataset pair.",
)
def recommend(
    model_key: str, data: RecommendRequest, request: Request
) -> RecommendResponse:
    """Dispatch a recommendation request to the appropriate model.

    The ``model_key`` path parameter selects which model-dataset pair to use
    (e.g., ``SASRec_movielens``). The request body fields are validated based
    on the model type.

    Args:
        model_key (str): Identifier in ``"{model}_{dataset}"`` format.
        data (RecommendRequest): Request body with parameters for the recommendation.
        request (Request): FastAPI request (used to access shared services).

    Returns:
        RecommendResponse: Recommendation results with model metadata.

    Raises:
        HTTPException: If the model key is not in the model manager.
    """
    inference_service = request.app.state.inference_service
    model_manager = request.app.state.model_manager

    try:
        model_type = model_manager.get_endpoint_type(model_key)
    except KeyError as exc:
        available = model_manager.get_available_endpoints()
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_key}' not found. Available: {', '.join(available) or '(none)'}",
        ) from exc

    recommendations = inference_service.recommend(
        model_key=model_key,
        top_k=data.top_k,
        item_sequence=data.item_sequence,
        user_index=data.user_index,
        context=data.context,
    )

    return RecommendResponse(
        model_key=model_key,
        model_type=model_type,
        recommendations=recommendations,
    )


@router.get(
    "/models",
    status_code=200,
    description="List all loaded model-dataset pairs and their types.",
)
def list_models(request: Request) -> Dict[str, str]:
    """Return available model keys and their recommender types.

    Clients can call this endpoint to discover which model keys are valid
    for the ``/recommend/{model_key}`` endpoint.

    Args:
        request (Request): FastAPI request (used to access shared services).

    Returns:
        Dict[str, str]: Dictionary mapping model keys to their types.
    """
    model_manager = request.app.state.model_manager
    return model_manager.get_available_endpoints()
