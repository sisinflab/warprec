from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from ray import serve
from ray.serve.handle import DeploymentHandle

from warprec.serving.api import build_api
from warprec.serving.catalogue import read_catalogue
from warprec.serving.servable import (
    Presentation,
    Query,
    ServableModel,
    ServingError,
    ServingPolicy,
)
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
        catalogue = (
            read_catalogue(config.item_metadata) if config.item_metadata else None
        )
        self._model = ServableModel.from_checkpoint(
            config.checkpoint,
            device=config.device,
            policy=ServingPolicy(
                default_k=config.default_k,
                max_k=config.max_k,
                mask_seen=config.mask_seen,
                unknown_user=config.unknown_user,
            ),
            catalogue=catalogue,
            presentation=Presentation(
                description=config.description,
                item_noun=config.item_noun,
                context_descriptions=config.context_descriptions,
            ),
        )
        # serve.batch wraps each method in an object carrying these setters,
        # which the decorator's type hints do not show.
        for batched in (self._recommend_batch, self._score_batch):
            batched.set_max_batch_size(  # type: ignore[attr-defined]
                config.batching.max_batch_size
            )
            batched.set_batch_wait_timeout_s(  # type: ignore[attr-defined]
                config.batching.batch_wait_timeout_s
            )

    def describe(self) -> Dict[str, Any]:
        """What this endpoint serves.

        Returns:
            Dict[str, Any]: The model description.
        """
        return self._model.describe()

    def describe_context(self) -> Dict[str, Any]:
        """The context this endpoint accepts.

        Returns:
            Dict[str, Any]: The fields and an example, or an error.
        """
        return self._answer(self._model.describe_context)

    def search_items(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Search the items by name.

        Args:
            request (Dict[str, Any]): 'query' and optionally 'limit'.

        Returns:
            Dict[str, Any]: The matches and suggestions, or an error.
        """
        return self._answer(self._model.search_items, **request)

    def get_items(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Look items up by id or name.

        Args:
            request (Dict[str, Any]): 'items'.

        Returns:
            Dict[str, Any]: The items found and not found, or an error.
        """
        return self._answer(self._model.get_items, **request)

    def popular(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """The most popular items.

        Args:
            request (Dict[str, Any]): The fields of a PopularRequest.

        Returns:
            Dict[str, Any]: The items, or an error.
        """
        try:
            return {"items": self._model.popular_items(**request)}
        except ServingError as error:
            return {"error": error.to_dict()}

    @staticmethod
    def _answer(method: Any, **request: Any) -> Dict[str, Any]:
        """Call a method of the model, returning a refusal as a value.

        Args:
            method (Any): The method.
            **request (Any): Its arguments.

        Returns:
            Dict[str, Any]: What it returned, or the error it raised.
        """
        try:
            return method(**request)
        except ServingError as error:
            return {"error": error.to_dict()}

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
        """Score the candidates of one request, batched with its neighbours.

        Args:
            request (Dict[str, Any]): The fields of a ScoreRequest.

        Returns:
            Dict[str, Any]: The scores, or an error.
        """
        try:
            query = self._model.resolve_scoring(**request)
        except ServingError as error:
            return {"error": error.to_dict()}
        return {"scores": await self._score_batch(query)}

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

    @serve.batch(max_batch_size=64, batch_wait_timeout_s=0.005)
    async def _score_batch(self, queries: List[Query]) -> List[List[Dict[str, Any]]]:
        """Score the candidates of the queries that arrived together, in one pass.

        Args:
            queries (List[Query]): The scoring queries of the batch.

        Returns:
            List[List[Dict[str, Any]]]: One answer per query, in order.
        """
        return self._model.score_batch(queries)


# The gateway only forwards requests, so it reserves no CPU: on a machine whose
# CPUs all go to model replicas it would otherwise never be scheduled.
@serve.deployment(name="gateway", ray_actor_options={"num_cpus": 0})
@serve.ingress()
class Gateway:
    """The HTTP entry point, routing each request to its model deployment.

    Args:
        models (Dict[str, DeploymentHandle]): The model deployments by name.
        server (Dict[str, Any]): The server settings, as plain data.
        endpoints (Optional[List[Dict[str, Any]]]): Each endpoint's name,
            description and item noun, for the instructions MCP clients receive.
    """

    def __init__(
        self,
        models: Dict[str, DeploymentHandle],
        server: Dict[str, Any],
        endpoints: Optional[List[Dict[str, Any]]] = None,
    ):
        self._models = models
        self._server = ServerSettings.model_validate(server)
        self._endpoints = endpoints

    def __serve_build_asgi_app__(self) -> FastAPI:
        """Called by Ray Serve after __init__, once the handles exist.

        Returns:
            FastAPI: The application this replica serves.
        """
        return build_api(
            self._models,
            api_key=self._server.effective_api_key(),
            mcp=self._server.mcp,
            endpoints=self._endpoints,
        )
