from typing import Any, Dict

import ray
from ray import serve
from ray.serve import Application

from warprec.serving.deployments import Gateway, ModelServer
from warprec.utils.config.serving_configuration import ServingConfiguration

APP_NAME = "warprec"


def build_application(config: ServingConfiguration) -> Application:
    """The Ray Serve application for a serving configuration.

    Args:
        config (ServingConfiguration): The validated configuration.

    Returns:
        Application: A gateway bound to one model deployment per endpoint.
    """
    # serve.deployment turns both classes into Deployments, which mypy cannot
    # follow through the decorator, hence the ignores on options() and bind().
    models = {
        endpoint.name: ModelServer.options(  # type: ignore[attr-defined]
            name=endpoint.name, **endpoint.deployment_options()
        ).bind(endpoint.model_dump(mode="json"))
        for endpoint in config.endpoints
    }
    return Gateway.bind(  # type: ignore[attr-defined]
        models, config.server.model_dump(mode="json")
    )


def app_builder(args: Dict[str, Any]) -> Application:
    """The application builder Ray Serve config files point at.

    Args:
        args (Dict[str, Any]): The 'args' of the application, holding the
            serving configuration under 'config'.

    Returns:
        Application: The application.
    """
    return build_application(ServingConfiguration.model_validate(args["config"]))


def run(config: ServingConfiguration) -> None:
    """Start Ray Serve with the configured models and block until interrupted.

    Args:
        config (ServingConfiguration): The validated configuration.
    """
    ray.init(address=config.server.ray_address, ignore_reinit_error=True)
    serve.start(http_options={"host": config.server.host, "port": config.server.port})
    serve.run(
        build_application(config),
        name=APP_NAME,
        route_prefix=config.server.route_prefix,
        blocking=True,
    )
