import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import ray

from warprec.common import (
    RunState,
    RunStateStore,
    resolve_run_name,
    run_fingerprint,
    warprec_version,
)
from warprec.data import Dataset
from warprec.evaluation.statistical_significance import compute_paired_statistical_test
from warprec.data.reader import ReaderFactory
from warprec.data.reader.base_reader import Reader
from warprec.data.writer import WriterFactory
from warprec.data.writer.base_writer import Writer
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config import (
    TrainConfiguration,
    load_callback,
    load_train_configuration,
)
from warprec.pipelines.remotes import remote_data_preparation
from warprec.utils.enums import ResumeMode
from warprec.utils.logger import logger


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
        ConnectionError: If unable to connect to the Ray cluster.
    """
    py_modules = (
        [] if config.general.custom_modules is None else config.general.custom_modules
    )
    py_modules.extend(["warprec"])  # type: ignore[union-attr]

    try:
        ray.init(
            address=config.general.ray_address,
            runtime_env={"py_modules": py_modules},
        )
        logger.positive("Connected to existing Ray cluster.")
    except ConnectionError as e:
        raise ConnectionError(
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
        ValueError: If resume is set to 'force' and no usable state exists.
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
            raise ValueError(message)
        logger.attention(f"{message} The run will start from scratch.")
        previous_state = None

    if previous_state is None and config.run.resume == ResumeMode.FORCE:
        raise ValueError(
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


def prepare_datasets(
    context: PipelineContext,
) -> Tuple[Dataset, Optional[Dataset], Optional[List[Dataset]]]:
    """Build the datasets on the Ray cluster, and write the splits if asked to.

    Args:
        context (PipelineContext): The context produced by the bootstrap.

    Returns:
        Tuple[Dataset, Optional[Dataset], Optional[List[Dataset]]]: The main
            dataset, the validation dataset and the cross-validation folds.

    Raises:
        ValueError: If the configured split file format is not supported.
    """
    config = context.config
    cpu_data_prep = config.general.cpu_data_prep
    custom_res_data_prep = config.general.custom_resources_data_prep
    label_selector_data_prep = config.general.label_selector_data_prep

    main_dataset, val_dataset, fold_dataset = ray.get(
        remote_data_preparation.options(
            num_cpus=cpu_data_prep,
            resources=custom_res_data_prep if custom_res_data_prep else None,
            label_selector=label_selector_data_prep
            if label_selector_data_prep
            else None,
        ).remote(
            reader=context.reader,
            callback=context.callback,
            config=config,
        )  # type: ignore[call-arg]
    )

    # Write split information if required
    if config.splitter and config.writer.save_split:
        file_format = config.writer.split.file_format

        match file_format:
            case "tabular":
                context.writer.write_tabular_split(
                    main_dataset,
                    val_dataset,
                    fold_dataset,
                    **config.writer.split.model_dump(),
                )
            case "parquet":
                context.writer.write_parquet_split(
                    main_dataset,
                    val_dataset,
                    fold_dataset,
                    **config.writer.split.model_dump(),
                )
            case _:
                raise ValueError(f"File format '{file_format}'not supported.")

    return main_dataset, val_dataset, fold_dataset


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
                model_results, stat_name, **corrections
            )
            context.writer.write_statistical_significance_test(test_results, stat_name)

    logger.positive("Statistical significance tests completed successfully.")
