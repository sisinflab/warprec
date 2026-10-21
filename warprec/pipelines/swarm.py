import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, Any, Optional, List, Tuple

import ray
from ray import tune

from warprec.common import (
    ModelState,
    ModelStatus,
    log_evaluation,
    reconcile_model_state,
)
from warprec.data.writer import WriterFactory
from warprec.data.timing import DataPreparationTimes
from warprec.pipelines.common import (
    PipelineContext,
    PreparedDatasets,
    bootstrap_pipeline,
    callback_needs_model,
    data_preparation_seconds,
    measure_emissions,
    model_timing_row,
    prepare_datasets,
    raise_failed_loads,
    report_statistical_significance,
    write_model_outputs,
)
from warprec.pipelines.train import complete_model, resolve_model_outcome
from warprec.recommenders.trainer import Trainer, TrainingOutcome
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config import (
    TrainConfiguration,
)
from warprec.utils.pause import PauseController
from warprec.utils.helpers import load_custom_modules, model_param_from_dict
from warprec.utils.logger import logger

if TYPE_CHECKING:
    from warprec.recommenders.trainer.emissions import EmissionsSettings

# The hooks Ray Tune calls on a callback while a sweep runs
TUNE_HOOKS = (
    "setup",
    "on_step_begin",
    "on_step_end",
    "on_trial_start",
    "on_trial_restore",
    "on_trial_save",
    "on_trial_result",
    "on_trial_complete",
    "on_trial_recover",
    "on_trial_error",
    "on_checkpoint",
    "on_experiment_end",
)


@dataclass
class SwarmModelResult:
    """What the task running one model of a swarm hands back to the driver.

    Attributes:
        model_name (str): The name of the model.
        outcome (TrainingOutcome): The outcome of its search, without the model.
        built (bool): Whether its best model was built and evaluated.
        model (Optional[Recommender]): The best model, only when the driver's
            callback receives it.
        load_error (Optional[str]): Why the best model could not be built, when
            the search completed but building it failed.
        results (Dict): The evaluation results of the best model.
        model_params (Dict): The best parameters and training iteration.
        timing_report (Dict): The row of the time report.
    """

    model_name: str
    outcome: TrainingOutcome
    built: bool = False
    model: Optional[Recommender] = None
    load_error: Optional[str] = None
    results: Dict = field(default_factory=dict)
    model_params: Dict = field(default_factory=dict)
    timing_report: Dict = field(default_factory=dict)


class HookMailbox:
    """Holds the Ray Tune hook calls the model tasks forward, until delivered.

    It runs as a Ray actor between the model tasks and the driver: a task
    posts without waiting for an answer, so that Ray Tune's loop is never held
    up, and the actor runs the calls of one task in the order they were made.
    """

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def post(self, call: Tuple[str, Dict[str, Any]]) -> None:
        """Keep one hook call.

        Args:
            call (Tuple[str, Dict[str, Any]]): The hook and its arguments.
        """
        self.calls.append(call)

    def take(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Hand over every call kept so far, oldest first.

        Returns:
            List[Tuple[str, Dict[str, Any]]]: The calls.
        """
        calls, self.calls = self.calls, []
        return calls


def _forwarding_hook(hook: str) -> Callable[..., None]:
    """Build the method a 'TuneHookForwarder' answers one Ray Tune hook with.

    Args:
        hook (str): The name of the hook.

    Returns:
        Callable[..., None]: The method, which forwards the call when the custom
            callback implements the hook.
    """

    def forward(self: "TuneHookForwarder", **info: Any) -> None:
        if hook in self.hooks:
            self.post((hook, info))
        # The last hook of a sweep: what was posted must have arrived before
        # the task returns and the driver collects the calls
        if hook == "on_experiment_end" and self.last_post is not None:
            ray.get(self.last_post)

    forward.__name__ = hook
    return forward


class TuneHookForwarder(WarpRecCallback):
    """Stands in for the custom callback in the Ray Tune sweep of a swarm model.

    In the swarm every model's sweep runs inside a Ray task, so the Ray Tune
    hooks of the custom callback would run on a copy of it in that task, and
    whatever they collected would never reach the callback the driver calls
    'on_training_complete' and 'on_evaluation_complete' on. This callback goes
    into the sweep in its place and forwards every call of a hook the custom
    callback implements, with its arguments, to the driver, which makes the
    same call on the custom callback.

    Args:
        mailbox (Any): Where the calls are posted: a 'HookMailbox' actor, or a
            'HookMailbox' itself when the run does not span processes.
        hooks (Tuple[str, ...]): The hooks the custom callback implements.
    """

    def __init__(self, mailbox: Any, hooks: Tuple[str, ...]):
        super().__init__()
        self.mailbox = mailbox
        self.hooks = hooks
        self.last_post: Optional[Any] = None

    def post(self, call: Tuple[str, Dict[str, Any]]) -> None:
        """Post one hook call to the driver.

        Args:
            call (Tuple[str, Dict[str, Any]]): The hook and its arguments.
        """
        if isinstance(self.mailbox, HookMailbox):
            self.mailbox.post(call)
        else:
            self.last_post = self.mailbox.post.remote(call)


for _hook in TUNE_HOOKS:
    setattr(TuneHookForwarder, _hook, _forwarding_hook(_hook))


class TuneHookRelay:
    """Delivers the Ray Tune hooks of the swarm's sweeps to the custom callback.

    Args:
        callback (WarpRecCallback): The custom callback of the run.
        mailbox (Optional[Any]): Where the calls are relayed through. Left out,
            a 'HookMailbox' actor is started when the callback implements any
            Ray Tune hook.
    """

    def __init__(self, callback: WarpRecCallback, mailbox: Optional[Any] = None):
        self.callback = callback
        self.hooks = tuple(
            hook
            for hook in TUNE_HOOKS
            if getattr(type(callback), hook) is not getattr(tune.Callback, hook)
        )
        if mailbox is None and self.hooks:
            mailbox = ray.remote(num_cpus=0)(HookMailbox).remote()
        self.mailbox = mailbox

    def forwarder(self) -> TuneHookForwarder:
        """The callback to put in the sweep of each model task.

        Returns:
            TuneHookForwarder: A forwarder to this relay.
        """
        return TuneHookForwarder(self.mailbox, self.hooks)

    def deliver(self) -> None:
        """Make on the custom callback every call forwarded so far, in order."""
        if self.mailbox is None:
            return
        calls = (
            self.mailbox.take()
            if isinstance(self.mailbox, HookMailbox)
            else ray.get(self.mailbox.take.remote())
        )
        for hook, info in calls:
            getattr(self.callback, hook)(**info)

    def close(self) -> None:
        """Deliver what is left and stop the mailbox."""
        self.deliver()
        if self.mailbox is not None and not isinstance(self.mailbox, HookMailbox):
            ray.kill(self.mailbox)
        self.mailbox = None


def swarm_pipeline(path: str):
    """Main function to start the distributed experiment.

    This method will start the highly parallelized train pipeline,
    launching all models concurrently on the Ray cluster.

    Args:
        path (str): Path to the configuration file.
    """
    logger.attention(
        "WARNING: Swarm pipeline is experimental. Please submit bug reports via GitHub Issues."
    )
    context = bootstrap_pipeline(path, "swarm", "Start experiment swarming.")
    with measure_emissions(context) as emissions:
        _swarm_models(context, emissions)


def _swarm_models(  # noqa: PLR0912, PLR0915
    context: PipelineContext, emissions: Optional["EmissionsSettings"]
) -> None:
    """Prepare the data, then run every model at once on the cluster.

    Args:
        context (PipelineContext): The context produced by the bootstrap.
        emissions (Optional[EmissionsSettings]): The run's CodeCarbon
            measurement, or None when the run is not measured.
    """
    config = context.config

    # Build the datasets on the cluster and persist the splits if asked to
    data = prepare_datasets(context)
    data_times = data.layout.times
    needs_model = callback_needs_model(context.callback)

    # List of models to train
    models = list(config.models.keys())

    # Check if statistical significance is requested
    requires_stat_significance = (
        config.evaluation.stat_significance.requires_stat_significance()
    )
    if requires_stat_significance:
        model_results: Dict[str, Any] = {}

    data_preparation_time = data_preparation_seconds(context.started_at, emissions)
    logger.positive(
        f"Data preparation completed in {data_preparation_time:.2f} seconds."
    )
    data_times.log()
    model_timing_report: List[Dict[str, Any]] = []

    # Starting the model swarming
    logger.msg(
        f"Launching the swarm of experiments. Number of different models: {len(models)}"
    )

    # Only launch the models that are not already finished
    pending_models = []
    resumable_models = set()
    for model_name in models:
        # Compared before the status is read, so that a model whose
        # configuration changed is optimized again whatever its status
        model_state, resumable = reconcile_model_state(
            context.state, model_name, config.models[model_name]
        )

        if model_state.status == ModelStatus.COMPLETED:
            logger.msg(f"Skipping {model_name}: already completed in this run.")
            if requires_stat_significance:
                stored_results = context.state_store.load_eval_results(model_name)
                if stored_results is not None:
                    model_results[model_name] = stored_results
                else:
                    logger.attention(
                        f"Evaluation results for {model_name} could not be loaded. "
                        "It will be excluded from the statistical significance tests."
                    )
            continue

        if model_state.status == ModelStatus.FAILED:
            logger.msg(f"Skipping {model_name}: it failed in a previous run.")
            continue

        if resumable:
            resumable_models.add(model_name)
        pending_models.append(model_name)

    context.state_store.save(context.state)

    # The sweeps run in the model tasks: their Ray Tune hooks are relayed to
    # the custom callback here, so that what it collects is on the driver
    relay = TuneHookRelay(context.callback)

    futures = []
    for model_name in pending_models:
        future = remote_model_pipeline.remote(
            model_name=model_name,
            config=config,
            data=data,
            callback=relay.forwarder(),
            data_preparation_time=data_preparation_time,
            data_times=data_times,
            run_name=context.run_name,
            writer_timestamp=context.state.writer_timestamp,
            resumable=model_name in resumable_models,
            emissions=emissions,
            model_state=context.state.model_state(model_name),
            return_model=needs_model,
            concurrent_searches=len(pending_models),
        )  # type: ignore[call-arg]
        futures.append(future)

    # Wait for the models to finish, staying responsive to a pause request.
    #
    # The tuning loops run inside the worker tasks, so a signal delivered to
    # this driver never reaches Ray Tune's own graceful handler. The best the
    # driver can do is cancel the tasks and rely on Ray Tune's periodic
    # experiment checkpoint, which makes a swarm pause coarser than a train one.
    paused = False
    completed_models = []

    with PauseController(enabled=config.run.pause_on_signal) as pause:
        pending = dict(zip(pending_models, futures))
        while pending:
            if pause.pause_requested:
                paused = True
                logger.attention(
                    "Pause requested. Cancelling the models still running. Their "
                    "Ray Tune experiments will resume from their last saved state."
                )
                for name, future in pending.items():
                    ray.cancel(future, force=False, recursive=True)
                    context.state.model_state(name).status = ModelStatus.INTERRUPTED
                context.state_store.save(context.state)
                break

            ready, _ = ray.wait(list(pending.values()), num_returns=1, timeout=5.0)
            relay.deliver()
            if not ready:
                continue

            finished = ready[0]
            name = next(n for n, f in pending.items() if f == finished)
            del pending[name]
            try:
                completed_models.append(ray.get(finished))
            except (
                ray.exceptions.RayTaskError,
                ray.exceptions.TaskCancelledError,
            ) as e:
                logger.negative(f"Model {name} did not complete: {e}")
                context.state.model_state(name).status = ModelStatus.INTERRUPTED
                context.state_store.save(context.state)

    # Every hook a finished task made is queued by now
    relay.close()

    # Final result logging of driver
    logger.msg("Swarming completed. Aggregating and saving results.")

    failed_loads: List[str] = []
    for completed in completed_models:
        model_name = completed.model_name
        outcome = completed.outcome
        model_state = context.state.model_state(model_name)
        if outcome.interrupted or outcome.failed:
            logger.attention(f"HPO for {model_name} returned no valid model.")
            model_state.status = (
                ModelStatus.INTERRUPTED if outcome.interrupted else ModelStatus.FAILED
            )
            context.state_store.save(context.state)
            continue

        # The search finished: recorded whether or not its best model was built,
        # so that resuming the run builds it again without searching again
        model_state.status = ModelStatus.HPO_COMPLETED
        model_state.best_params = outcome.best_params
        model_state.best_iter = outcome.best_iter
        model_state.best_checkpoint_path = outcome.best_checkpoint_path
        model_state.ray_report = outcome.report
        if not completed.built:
            logger.negative(
                f"The best model of {model_name} could not be built: "
                f"{completed.load_error}"
            )
            model_state.status = ModelStatus.FAILED_LOAD
            context.state_store.save(context.state)
            failed_loads.append(model_name)
            continue
        context.state_store.save(context.state)

        results = completed.results
        model_params = completed.model_params
        timing_report = completed.timing_report

        # Callbacks
        if needs_model:
            context.callback.on_training_complete(model=completed.model)
            context.callback.on_evaluation_complete(
                model=completed.model,
                params=model_params[model_name]["Best Params"],
                results=results,
            )

        # Log the results
        log_evaluation(results, "Test", config.evaluation.max_metric_per_row)

        # The same files a train run writes for a model
        write_model_outputs(
            context,
            model_name,
            results,
            model_params[model_name]["Best Params"],
            model_params[model_name]["Best Training Iteration"],
        )

        if config.general.time_report:
            model_timing_report.append(timing_report)

        # Collect for statistical significance
        if requires_stat_significance:
            model_results[model_name] = results
            context.state_store.save_eval_results(model_name, results)

        model_state.status = ModelStatus.COMPLETED
        model_state.timing = timing_report
        context.state_store.save(context.state)

    # Write aggregated time report (if requested)
    if config.general.time_report and model_timing_report:
        context.writer.write_time_report(model_timing_report)

    if paused:
        logger.attention(
            f"Swarm run '{context.run_name}' has been paused. Progress is saved at "
            f"{context.state_store.state_path}."
        )
        logger.msg(
            "Resume it by running the same command again with run.name set to "
            f"'{context.run_name}' and run.resume set to 'auto' or 'force'."
        )
    else:
        # Compute statistical significance (if requested)
        if requires_stat_significance:
            report_statistical_significance(context, model_results, models)

        logger.positive("Experiment swarming concluded. WarpRec is shutting down.")

    raise_failed_loads(failed_loads, context.run_name)


@ray.remote(num_cpus=0.05)  # Zero-Resource Orchestrator
def remote_model_pipeline(  # noqa: PLR0913
    model_name: str,
    config: TrainConfiguration,
    data: PreparedDatasets,
    callback: WarpRecCallback,
    data_preparation_time: float,
    data_times: Optional[DataPreparationTimes] = None,
    run_name: Optional[str] = None,
    writer_timestamp: Optional[str] = None,
    resumable: bool = False,
    emissions: Optional["EmissionsSettings"] = None,
    model_state: Optional[ModelState] = None,
    return_model: bool = False,
    concurrent_searches: int = 1,
) -> SwarmModelResult:
    """Orchestrates the entire lifecycle of a single model in parallel.

    This task acts as a lightweight manager. It launches HPO, Retraining,
    Evaluation, and Recommendation generation for a specific model.

    Args:
        model_name (str): The name of the model to process.
        config (TrainConfiguration): The configuration object.
        data (PreparedDatasets): The datasets of the run, by reference: this
            task never fetches them, the trials and the model task do.
        callback (WarpRecCallback): The callback object.
        data_preparation_time (float): Time taken for data prep (for reporting).
        data_times (Optional[DataPreparationTimes]): Where the data preparation
            spent its time (for reporting).
        run_name (Optional[str]): The identifier of the run, used to give the Ray
            Tune experiment a deterministic name so that it can be restored.
        writer_timestamp (Optional[str]): The timestamp pinned into the output
            file names of this run.
        resumable (bool): Whether the run state allows restoring the Ray Tune
            experiment this model left behind.
        emissions (Optional[EmissionsSettings]): The run's CodeCarbon
            measurement, or None when the run is not measured.
        model_state (Optional[ModelState]): What the run state records of the
            model, so that a search it finished is not run again.
        return_model (bool): Whether to bring the best model back, for the
            driver's callback.
        concurrent_searches (int): How many models the swarm searches at once,
            which share the cluster equally.

    Returns:
        SwarmModelResult: The outcome of the search and, when its best model
            was built, the model, its results and its time report row.
    """

    model_exploration_start_time = time.time()

    # The search names its trials after the model, so it has to be registered here
    load_custom_modules(config.general.custom_modules)

    # Retrieve storage path for Ray results
    # based on the writer configuration
    storage_path = config.get_storage_path()

    params = model_param_from_dict(model_name, config.models[model_name])

    trainer = Trainer(
        storage_path=storage_path,
        custom_callback=callback,
        custom_modules=config.general.custom_modules,
        dashboard_config=config.dashboard,
        run_name=run_name,
        errored_trials=config.run.errored_trials,
        resumable=resumable,
        emissions=emissions,
        gpu_deterministic=config.general.gpu_deterministic,
        concurrent_searches=concurrent_searches,
    )

    # Run the HPO, unless the run state records it finished
    outcome = resolve_model_outcome(
        model_name,
        params,
        trainer,
        config,
        data,
        model_state or ModelState(),
    )
    if outcome.interrupted or outcome.failed:
        return SwarmModelResult(model_name=model_name, outcome=outcome)

    search_seconds = time.time() - model_exploration_start_time

    # Built, evaluated and written by a task of its own, with the resources of
    # a trial: this one reserves almost none
    writer = WriterFactory.get_writer(config=config, timestamp=writer_timestamp)
    completion = complete_model(
        model_name, params, outcome, data, config, writer, return_model=return_model
    )
    if completion.error is not None:
        return SwarmModelResult(
            model_name=model_name, outcome=outcome, load_error=completion.error
        )
    outcome = completion.outcome

    # Prepare the report to return to the driver
    model_params = {
        model_name: {
            "Best Params": completion.best_params,
            "Best Training Iteration": outcome.best_iter,
        }
    }

    timing_report = model_timing_row(
        model_name=model_name,
        data_preparation_time=data_preparation_time,
        exploration_time=search_seconds + completion.build_seconds,
        ray_report=outcome.report,
        evaluation_time=completion.evaluation_seconds,
        inference_time=completion.inference_seconds,
        data_times=data_times,
    )

    return SwarmModelResult(
        model_name=model_name,
        outcome=outcome,
        built=True,
        model=completion.model,
        results=completion.results,
        model_params=model_params,
        timing_report=timing_report,
    )
