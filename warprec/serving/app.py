import signal
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union

import ray
import yaml
from ray import serve
from ray.serve import Application

from warprec.serving.deployments import Gateway, ModelServer
from warprec.serving.servable import scores_on_device
from warprec.utils.config.serving_configuration import ServingConfiguration
from warprec.utils.logger import logger

APP_NAME = "warprec"


def build_application(config: ServingConfiguration) -> Application:
    """The Ray Serve application for a serving configuration.

    Args:
        config (ServingConfiguration): The validated configuration.

    Returns:
        Application: A gateway bound to one model deployment per endpoint.
    """
    # Replicas may run in another directory than this process - on a joined
    # cluster they run wherever it was started - so they get absolute paths.
    config = config.resolved()

    # A model that keeps no tensors scores on the CPU wherever it is placed, so
    # it is served there rather than holding a GPU it would never use.
    for endpoint in config.endpoints:
        if endpoint.device != "cpu" and not scores_on_device(endpoint.checkpoint):
            logger.attention(
                f"Endpoint '{endpoint.name}' serves a model that scores on the CPU "
                f"(it keeps no tensors), so device '{endpoint.device}' is ignored "
                "and no GPU is reserved for it."
            )
            endpoint.device = "cpu"

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


def export_serve_config(config: ServingConfiguration, path: Union[str, Path]) -> None:
    """Write the application as a Ray Serve config file, for serve deploy or KubeRay.

    Paths become absolute so the file works from any directory, and the API key
    is left out so that no secret ends up in a file meant to be shared. Unless
    the configuration sets server.host explicitly, the proxy listens on every
    interface: on a cluster it is reached from outside its machine or pod, where
    the local default of 127.0.0.1 would leave the application unreachable.

    Args:
        config (ServingConfiguration): The validated configuration.
        path (Union[str, Path]): Where to write the file.
    """
    if config.server.api_key:
        logger.attention(
            "The API key is not written to the exported file. Set WARPREC_API_KEY "
            "in the environment of the cluster that runs it."
        )
    portable = config.portable()
    if "host" not in config.server.model_fields_set:
        portable.server.host = "0.0.0.0"  # nosec B104 - a cluster proxy must be reachable
    document = {
        "proxy_location": "EveryNode",
        "http_options": {"host": portable.server.host, "port": portable.server.port},
        "applications": [
            {
                "name": APP_NAME,
                "route_prefix": config.server.route_prefix,
                "import_path": "warprec.serving.app:app_builder",
                "args": {"config": portable.model_dump(mode="json")},
            }
        ],
    }
    Path(path).write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    logger.positive(f"Ray Serve config written to {path}")


def run(config: ServingConfiguration) -> None:
    """Start Ray Serve with the configured models and serve until told to stop.

    Ctrl-C at a terminal and SIGTERM from a process manager or a container both
    shut the application down cleanly, rather than leaving a traceback or
    killing Ray mid-way. A signal during startup - a replica that cannot be
    scheduled waits forever - interrupts it at once.

    Args:
        config (ServingConfiguration): The validated configuration.
    """
    started = threading.Event()
    stop = threading.Event()

    def request_stop(signum: int, _frame: Any) -> None:
        logger.msg(f"Received {signal.Signals(signum).name}, stopping.")
        stop.set()
        if not started.is_set():
            # Still deploying: break out of Ray's wait instead of finishing it.
            raise KeyboardInterrupt

    def listen() -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, request_stop)

    listen()
    try:
        ray.init(address=config.server.ray_address, ignore_reinit_error=True)
        # Ray installs a SIGTERM handler of its own that aborts the process,
        # so the clean one is put back once Ray is up.
        listen()
        serve.start(
            http_options={"host": config.server.host, "port": config.server.port}
        )
        serve.run(
            build_application(config),
            name=APP_NAME,
            route_prefix=config.server.route_prefix,
        )
        started.set()
        names = ", ".join(endpoint.name for endpoint in config.endpoints)
        logger.positive(
            f"Serving {names} at http://{config.server.host}:{config.server.port}"
            f"{config.server.route_prefix.rstrip('/')}/v1/models"
        )
        # Waiting in short steps keeps the main thread responsive to signals.
        while not stop.wait(timeout=1.0):
            pass
    except KeyboardInterrupt:
        logger.msg("Interrupted before every model was ready.")
    finally:
        stop_serving(config.server.ray_address)
        logger.positive("Serving stopped.")


def stop_serving(ray_address: Optional[str]) -> None:
    """Take the application down, and only it when the cluster is shared.

    Started locally, Ray and Serve belong to this process and are shut down
    whole. On a cluster joined with ray_address, Serve may be running other
    applications, so only this one is deleted before disconnecting.

    Args:
        ray_address (Optional[str]): The cluster address the server joined, if any.
    """
    if not ray.is_initialized():
        return
    if ray_address is None:
        serve.shutdown()
    else:
        serve.delete(APP_NAME)
    ray.shutdown()
