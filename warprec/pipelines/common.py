import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import ray

from warprec.errors import (
    CheckpointError,
    ClusterError,
    RunStateError,
)
import warprec
from warprec.common import (
    RunState,
    RunStateStore,
    resolve_run_name,
    run_fingerprint,
    warprec_version,
)
from warprec.data.timing import DataPreparationTimes
from warprec.evaluation.statistical_significance import compute_paired_statistical_test
from warprec.data.reader import ReaderFactory
from warprec.data.reader.base_reader import Reader
from warprec.data.writer import WriterFactory
from warprec.data.writer.base_writer import Writer
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config import (
    RecomModel,
    TrainConfiguration,
    load_callback,
    load_train_configuration,
)
from warprec.pipelines.remotes import DataLayout, remote_data_preparation
from warprec.utils.enums import ResumeMode
from warprec.utils.logger import logger

if TYPE_CHECKING:
    # The dashboard extra is optional
    from warprec.recommenders.trainer.emissions import EmissionsSettings


@dataclass
class PipelineContext:
    """Everything a pipeline needs before it starts doing its own work.

    The pipelines share an identical opening: parse the configuration, connect to
    Ray, build the I/O modules and resolve the identity of the run so that an
    interrupted one can be picked up again. This groups the results of that
    sequence so it can be produced once and handed over whole.

    Attributes:
        config (TrainConfiguration): The parsed configuration.
        callback (WarpRecCallback): The callback bound to this run.
        reader (Reader): The reader used to load the data.
        writer (Writer): The writer bound to this run's timestamp.
        state (RunState): The state of this run, new or resumed.
        state_store (RunStateStore): The store the state is persisted to.
        run_name (str): The resolved name of the run.
        started_at (float): The moment the experiment started, for timings.
    """

    config: TrainConfiguration
    callback: WarpRecCallback
    reader: Reader
    writer: Writer
    state: RunState
    state_store: RunStateStore
    run_name: str
    started_at: float


def initialise_ray(config: TrainConfiguration) -> None:
    """Connect to the Ray cluster the configuration points at.

    Args:
        config (TrainConfiguration): The configuration of the experiment.

    Raises:
        ClusterError: If unable to connect to the Ray cluster.
    """
    # WarpRec is shipped as the package that is imported here, wherever it is
    # installed, rather than as a path that exists only when WarpRec is run
    # from its own repository. The configured list is copied, not extended.
    py_modules: List[Any] = [*(config.general.custom_modules or []), warprec]

    try:
        ray.init(
            address=config.general.ray_address,
            runtime_env={"py_modules": py_modules},
        )
        logger.positive("Connected to existing Ray cluster.")
    except ConnectionError as e:
        raise ClusterError(
            "Unable to connect to Ray cluster. Please ensure Ray is running."
        ) from e


def resolve_run_state(
    config: TrainConfiguration, pipeline: str, run_name: str, probe_writer: Writer
) -> RunState:
    """Load the state of a previous run, or start a new one.

    A state is only reusable when the configuration that produced it still
    matches, because a different reader, splitter or set of models would make the
    recorded progress meaningless.

    Args:
        config (TrainConfiguration): The configuration of the experiment.
        pipeline (str): The name of the pipeline creating the state.
        run_name (str): The resolved name of the run.
        probe_writer (Writer): The writer used to look for an existing state.

    Returns:
        RunState: The resumed state, or a fresh one.

    Raises:
        RunStateError: If resume is set to 'force' and no usable state exists.
    """
    fingerprint = run_fingerprint(config)
    state_store = RunStateStore(probe_writer, run_name)
    previous_state = state_store.load()

    if previous_state is not None and config.run.resume == ResumeMode.NEVER:
        logger.attention(
            f"Run state for '{run_name}' exists but resume is set to 'never'. "
            "It will be discarded and the run will start from scratch."
        )
        previous_state = None
    elif (
        previous_state is not None and previous_state.config_fingerprint != fingerprint
    ):
        message = (
            f"Run state for '{run_name}' was produced by a different configuration "
            "(the reader, filtering, splitter, evaluation or the set of models "
            "changed). It cannot be resumed."
        )
        if config.run.resume == ResumeMode.FORCE:
            raise RunStateError(message)
        logger.attention(f"{message} The run will start from scratch.")
        previous_state = None

    if previous_state is None and config.run.resume == ResumeMode.FORCE:
        raise RunStateError(
            "Resume is set to 'force' but no resumable state was found for run "
            f"'{run_name}' at {state_store.state_path}."
        )

    if previous_state is None:
        return RunState(
            run_name=run_name,
            pipeline=pipeline,
            warprec_version=warprec_version(),
            writer_timestamp=probe_writer.timestamp,
            config_fingerprint=fingerprint,
        )

    logger.positive(
        f"Resuming run '{run_name}' created at {previous_state.created_at}."
    )
    return previous_state


def bootstrap_pipeline(path: str, pipeline: str, start_message: str) -> PipelineContext:
    """Run the opening sequence every pipeline shares.

    Args:
        path (str): Path to the configuration file.
        pipeline (str): The name of the pipeline, recorded in the run state.
        start_message (str): The message logged as the experiment starts.

    Returns:
        PipelineContext: The configuration, I/O modules and run state.
    """
    logger.msg(start_message)
    started_at = time.time()

    config = load_train_configuration(path)

    initialise_ray(config)

    callback: WarpRecCallback = load_callback(
        config.general.callback,
        *config.general.callback.args,
        **config.general.callback.kwargs,
    )

    reader = ReaderFactory.get_reader(config=config)

    # Resolve the identity of this run and load any state a previous,
    # interrupted run of the same experiment left behind.
    run_name = resolve_run_name(config)
    logger.msg(f"Run name: {run_name}")

    probe_writer = WriterFactory.get_writer(config=config)
    state = resolve_run_state(config, pipeline, run_name, probe_writer)

    # The writer reuses the timestamp of the original run so that a resumed run
    # keeps merging into the output files that run created.
    writer = WriterFactory.get_writer(config=config, timestamp=state.writer_timestamp)
    state_store = RunStateStore(writer, run_name)
    state_store.save(state)

    return PipelineContext(
        config=config,
        callback=callback,
        reader=reader,
        writer=writer,
        state=state,
        state_store=state_store,
        run_name=run_name,
        started_at=started_at,
    )


def data_preparation_seconds(
    started_at: float, emissions: Optional["EmissionsSettings"]
) -> float:
    """How long a run took to prepare its data, up to now.

    The run waits for its CodeCarbon trackers to measure before it prepares the
    data; that wait is not part of the preparation and is left out.

    Args:
        started_at (float): When the run started, as 'time.time()'.
        emissions (Optional[EmissionsSettings]): The run's CodeCarbon
            measurement, or None when the run is not measured.

    Returns:
        float: The seconds since the run started, without the wait.
    """
    waited = emissions.startup_seconds if emissions is not None else 0.0
    return time.time() - started_at - waited


def model_timing_row(
    model_name: str,
    data_preparation_time: float,
    exploration_time: float,
    ray_report: Dict[str, Any],
    evaluation_time: float,
    inference_time: float,
    data_times: Optional[DataPreparationTimes] = None,
) -> Dict[str, Any]:
    """The row of the time report for one model, as train and swarm write it.

    Args:
        model_name (str): The name of the model.
        data_preparation_time (float): The seconds the run took to prepare its
            data, shared by every model.
        exploration_time (float): The wall clock of the model's search, from
            building its trainer to loading its best model (and, under
            cross-validation, retraining it), Ray's scheduling included.
        ray_report (Dict[str, Any]): The summary of the search: memory,
            parameters and where the trials' time went.
        evaluation_time (float): The seconds the evaluation of the best model
            on the test set took on its worker.
        inference_time (float): The seconds of one prediction call, which
            the writer gives in milliseconds.
        data_times (Optional[DataPreparationTimes]): Where the data preparation
            spent its time, shared by every model. Its columns are NaN when not
            given.

    Returns:
        Dict[str, Any]: The row, by column.
    """
    return {
        "Model Name": model_name,
        "Data Preparation Time": data_preparation_time,
        **(data_times or DataPreparationTimes()).columns(),
        "Hyperparameter Exploration Time": exploration_time,
        **ray_report,
        "Evaluation Time": evaluation_time,
        "Inference Time": inference_time,
        "Total Time": data_preparation_time + exploration_time + evaluation_time,
    }


def measure_emissions(context: PipelineContext) -> AbstractContextManager[Any]:
    """The CodeCarbon measurement of a run, when the configuration asks for one.

    Entered as the run starts its work, before the data is prepared, and left
    when it ends, whether it completes, pauses or fails, so that the data
    preparation, the searches, the retraining and the test evaluation are all
    covered.

    Args:
        context (PipelineContext): The context produced by the bootstrap.

    Returns:
        AbstractContextManager[Any]: An 'EmissionsRecorder', which gives the
            'EmissionsSettings' the trainers take, or a context giving None
            when CodeCarbon is disabled or not installed.
    """
    codecarbon = context.config.dashboard.codecarbon
    if not codecarbon.enabled:
        return nullcontext()
    try:
        # the dashboard extra is optional
        from warprec.recommenders.trainer.emissions import EmissionsRecorder
    except ImportError:
        # The trainers say the dashboards are not available
        return nullcontext()
    return EmissionsRecorder(
        tracking_mode=codecarbon.tracking_mode,
        save_to_api=codecarbon.save_to_api,
        save_to_file=codecarbon.save_to_file,
        output_dir=context.config.get_storage_path(),
        project_name=context.run_name,
    )


@dataclass
class PreparedDatasets:
    """The datasets of a run, as references to the objects the cluster holds.

    The driver never fetches them: the trials, the model tasks and the swarm's
    orchestrators receive the references, and Ray hands each task the datasets
    it is given from the object store, without serializing them again.

    Attributes:
        main (ray.ObjectRef): The main dataset, the train/test split.
        validation (ray.ObjectRef): The validation dataset, a reference to None
            without one.
        folds (ray.ObjectRef): The list of cross-validation folds, empty
            without them.
        layout (DataLayout): How the data was split, and the preparation times.
    """

    main: ray.ObjectRef
    validation: ray.ObjectRef
    folds: ray.ObjectRef
    layout: DataLayout

    @property
    def cross_validated(self) -> bool:
        """Whether the models are searched over cross-validation folds.

        Returns:
            bool: True when there are folds and no validation split.
        """
        return not self.layout.has_validation and self.layout.n_folds > 0


def prepare_datasets(context: PipelineContext) -> PreparedDatasets:
    """Build the datasets on the Ray cluster, and write the splits if asked to.

    Args:
        context (PipelineContext): The context produced by the bootstrap.

    Returns:
        PreparedDatasets: References to the main dataset, the validation dataset
            and the cross-validation folds, and how the task preparing them
            split the data and spent its time.

    Raises:
        ClusterError: If Ray's memory monitor stops the data preparation.
    """
    config = context.config
    cpu_data_prep = config.general.cpu_data_prep
    custom_res_data_prep = config.general.custom_resources_data_prep
    label_selector_data_prep = config.general.label_selector_data_prep

    # Four returns, so that each dataset is an object of its own
    refs: List[ray.ObjectRef] = remote_data_preparation.options(  # type: ignore[assignment]
        num_cpus=cpu_data_prep,
        num_returns=4,
        # Ray retries a task its memory monitor kills without limit unless
        # the task is not retried at all: the preparation would be killed
        # and started again until the job's time ran out, without an error
        max_retries=0,
        resources=custom_res_data_prep if custom_res_data_prep else None,
        label_selector=label_selector_data_prep if label_selector_data_prep else None,
    ).remote(
        reader=context.reader,
        callback=context.callback,
        config=config,
        writer=context.writer,
    )  # type: ignore[call-arg]

    main_ref, val_ref, folds_ref, layout_ref = refs

    # Only the layout comes to the driver; the datasets stay where they are
    try:
        layout = ray.get(layout_ref)
    except ray.exceptions.OutOfMemoryError as error:
        raise ClusterError(
            "Ray's memory monitor stopped the data preparation, which needed "
            "more memory than its node had left. The datasets are built in "
            "the task's memory and then stored in Ray's object store, so on "
            "the node both count: leave the object store at its default size "
            "(it spills to disk), or prepare the data on a node with more "
            f"memory (general.label_selector_data_prep). Ray's message: {error}"
        ) from error
    return PreparedDatasets(
        main=main_ref, validation=val_ref, folds=folds_ref, layout=layout
    )


def model_task_options(
    params: RecomModel, config: TrainConfiguration
) -> Tuple[str, Dict[str, Any]]:
    """The device and the resources of the task that completes a model.

    Args:
        params (RecomModel): The parameters of the model.
        config (TrainConfiguration): The configuration of the experiment.

    Returns:
        Tuple[str, Dict[str, Any]]: The device, and the options of the Ray task.
    """
    general_device = config.general.device
    model_device = params.optimization.device
    device = general_device if model_device is None else model_device

    num_gpus = params.optimization.gpu_per_trial
    # A model on cuda without GPUs requested still needs a node that has one
    if device == "cuda" and num_gpus == 0:
        num_gpus = 1

    custom_res = params.optimization.custom_resources_per_trial or {}
    label_selector = params.optimization.label_selector or {}
    return device, {
        "num_cpus": params.optimization.cpu_per_trial,
        "num_gpus": num_gpus,
        "resources": custom_res if custom_res else None,
        "label_selector": label_selector if label_selector else None,
    }


def callback_needs_model(callback: WarpRecCallback) -> bool:
    """Whether a callback receives the best model, so the driver must fetch it.

    Only the hooks that receive the model count, and only when the callback
    overrides them: otherwise the model never leaves the cluster.

    Args:
        callback (WarpRecCallback): The callback of the run.

    Returns:
        bool: True when it overrides 'on_training_complete' or
            'on_evaluation_complete'.
    """
    return any(
        getattr(type(callback), hook) is not getattr(WarpRecCallback, hook)
        for hook in ("on_training_complete", "on_evaluation_complete")
    )


def write_model_outputs(
    context: PipelineContext,
    model_name: str,
    results: Dict[Any, Any],
    best_params: Dict[str, Any],
    best_iter: int,
) -> None:
    """Write what the driver keeps of one evaluated model.

    The train and swarm pipelines both call this once a model is evaluated, so
    that either of them leaves the same files behind: the overall results and
    the best parameters. The per-user results, the model and its
    recommendations are written on the cluster, by the task that evaluated it.

    Args:
        context (PipelineContext): The context produced by the bootstrap.
        model_name (str): The name of the model.
        results (Dict[Any, Any]): The evaluation results of the model.
        best_params (Dict[str, Any]): The hyperparameters of the best model.
        best_iter (int): The best training iteration.
    """
    config = context.config
    writer = context.writer

    writer.write_results(results, model_name, **config.writer.results.model_dump())
    writer.write_params(
        {
            model_name: {
                "Best Params": best_params,
                "Best Training Iteration": best_iter,
            }
        }
    )


def raise_failed_loads(model_names: List[str], run_name: str) -> None:
    """End a run whose searches finished but some best models could not be built.

    The run goes on to the other models first, and the searches are recorded,
    so resuming it builds those models again without repeating them.

    Args:
        model_names (List[str]): The models whose best model could not be built.
        run_name (str): The name of the run.

    Raises:
        CheckpointError: When any model is listed.
    """
    if not model_names:
        return
    raise CheckpointError(
        f"The best model of {', '.join(model_names)} could not be built from its "
        "finished search; the error is logged above. Run the same command again "
        f"with run.name set to '{run_name}' to build it without searching again."
    )


def report_statistical_significance(
    context: PipelineContext, model_results: dict, models: list
) -> None:
    """Run the configured significance tests and write their results.

    Args:
        context (PipelineContext): The context produced by the bootstrap.
        model_results (dict): The per-user results of every evaluated model.
        models (list): The models the run covered.
    """
    if len(model_results) < 2:
        logger.attention(
            "Statistical significance tests require at least two evaluated models. "
            "Skipping statistical significance computation."
        )
        return

    logger.msg(f"Computing statistical significance tests for {len(models)} models.")

    stat_significance = context.config.evaluation.stat_significance.model_dump(
        exclude=["corrections"]  # type: ignore[arg-type]
    )
    corrections = context.config.evaluation.stat_significance.corrections.model_dump()

    for stat_name, enabled in stat_significance.items():
        if enabled:
            test_results = compute_paired_statistical_test(
                model_results,
                stat_name,
                backend=context.config.general.backend,
                **corrections,
            )
            context.writer.write_statistical_significance_test(test_results, stat_name)

    logger.positive("Statistical significance tests completed successfully.")
