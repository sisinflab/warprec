from typing import Any, Dict, List

from fastapi import FastAPI
from ray import serve
from ray.serve.handle import DeploymentHandle

from warprec.serving.api import build_api
from warprec.serving.catalogue import read_item_names
from warprec.serving.servable import Query, ServableModel, ServingError, ServingPolicy
from warprec.utils.config.serving_configuration import EndpointConfig, ServerSettings


@serve.deployment
class ModelServer:
    """One served checkpoint, with concurrent requests scored together.

    Requests are checked one by one, so a bad request fails alone, and only the
    ones that pass are grouped into a single forward pass.

    Args:
        endpoint (Dict[str, Any]): The endpoint's configuration, as plain data
            so that it can be shipped to the replica.
    """

    def __init__(self, endpoint: Dict[str, Any]):
        config = EndpointConfig.model_validate(endpoint)
        names = read_item_names(config.item_metadata) if config.item_metadata else None
        self._model = ServableModel.from_checkpoint(
            config.checkpoint,
            device=config.device,
            policy=ServingPolicy(
                default_k=config.default_k,
                max_k=config.max_k,
                mask_seen=config.mask_seen,
                unknown_user=config.unknown_user,
            ),
            item_names=names,
        )
        # serve.batch wraps the method in an object carrying these setters,
        # which the decorator's type hints do not show.
        self._recommend_batch.set_max_batch_size(  # type: ignore[attr-defined]
            config.batching.max_batch_size
        )
        self._recommend_batch.set_batch_wait_timeout_s(  # type: ignore[attr-defined]
            config.batching.batch_wait_timeout_s
        )

    def describe(self) -> Dict[str, Any]:
        """What this endpoint serves.

        Returns:
            Dict[str, Any]: The model description.
        """
        return self._model.describe()

    async def recommend(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend for one request, batched with its concurrent neighbours.

        Args:
            request (Dict[str, Any]): The fields of a RecommendRequest.

        Returns:
            Dict[str, Any]: The items and whether popularity answered, or an error.
        """
        try:
            query = self._model.resolve(**request)
        except ServingError as error:
            return {"error": error.to_dict()}
        items = await self._recommend_batch(query)
        return {"items": items, "fallback": query.fallback}

    async def score(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Score the candidates of one request.

        Args:
            request (Dict[str, Any]): The fields of a ScoreRequest.

        Returns:
            Dict[str, Any]: The scores, or an error.
        """
        try:
            return {"scores": self._model.score(**request)}
        except ServingError as error:
            return {"error": error.to_dict()}

    @serve.batch(max_batch_size=64, batch_wait_timeout_s=0.005)
    async def _recommend_batch(
        self, queries: List[Query]
    ) -> List[List[Dict[str, Any]]]:
        """Answer the queries that arrived together with one forward pass.

        Args:
            queries (List[Query]): The queries of the batch.

        Returns:
            List[List[Dict[str, Any]]]: One answer per query, in order.
        """
        return self._model.recommend(queries)


# The gateway only forwards requests, so it reserves no CPU: on a machine whose
# CPUs all go to model replicas it would otherwise never be scheduled.
@serve.deployment(name="gateway", ray_actor_options={"num_cpus": 0})
@serve.ingress()
class Gateway:
    """The HTTP entry point, routing each request to its model deployment.

    Args:
        models (Dict[str, DeploymentHandle]): The model deployments by name.
        server (Dict[str, Any]): The server settings, as plain data.
    """

    def __init__(self, models: Dict[str, DeploymentHandle], server: Dict[str, Any]):
        self._models = models
        self._server = ServerSettings.model_validate(server)

    def __serve_build_asgi_app__(self) -> FastAPI:
        """Called by Ray Serve after __init__, once the handles exist.

        Returns:
            FastAPI: The application this replica serves.
        """
        return build_api(
            self._models,
            api_key=self._server.effective_api_key(),
            mcp=self._server.mcp,
        )
