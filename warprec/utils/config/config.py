import copy
import difflib
import os
import sys
import importlib
from pathlib import Path
from typing import ClassVar, Dict, Any, List

from pydantic import BaseModel, model_validator, Field
from warprec.errors import ConfigurationError, RegistryError
from warprec.utils.helpers import (
    is_python_module,
    load_custom_modules,
    validation_metric,
)
from warprec.data.filtering import Filter
from warprec.utils.config.common import read_configuration_file
from warprec.utils.config import (
    GeneralConfig,
    WarpRecCallbackConfig,
    ReaderConfig,
    WriterConfig,
    SplittingConfig,
    DashboardConfig,
    RecomModel,
    RerankConfig,
    TrainingConfig,
    EvaluationConfig,
    EstimateConfig,
    WarpRecRunConfig,
)
from warprec.utils.callback import WarpRecCallback
from warprec.utils.enums import ReadingMethods, SearchSpace, WritingMethods
from warprec.utils.registry import model_registry, params_registry, filter_registry
from warprec.utils.logger import logger


class WarpRecConfiguration(BaseModel):
    """Definition of WarpRec base configuration file structure.

    Attributes:
        section_schemas (ClassVar[Dict[str, Any]]): The type each model's raw
            parameters are checked against for keys no section reads: they
            are validated by the model's own class, so the field is loose.
        pipeline_sections (ClassVar[Dict[str, Any]]): Every top-level section
            any pipeline reads, with its type. A file may serve several
            pipelines, so a section only another one reads is not reported as
            unknown, and its keys are still checked against its type.
        reader (ReaderConfig): Configuration of the reading process.
        filtering (Dict[str, dict]): The dictionary containing filtering
            information in the format {filter_name: dict{param_1: value, param_2: value, ...}, ...}
        models (Dict[str, dict]): The dictionary containing model information
            in the format {model_name: dict{param_1: value, param_2: value, ...}, ...}
        general (GeneralConfig): General configuration of the experiment.
        training (TrainingConfig): The options that shape the training signal.
        rerank (RerankConfig): The re-ranking applied to every list produced.
    """

    section_schemas: ClassVar[Dict[str, Any]] = {"models": Dict[str, RecomModel]}
    pipeline_sections: ClassVar[Dict[str, Any]] = {
        "reader": ReaderConfig,
        "filtering": Dict[str, dict],
        "models": Dict[str, RecomModel],
        "general": GeneralConfig,
        "training": TrainingConfig,
        "rerank": RerankConfig,
        "writer": WriterConfig,
        "splitter": SplittingConfig,
        "dashboard": DashboardConfig,
        "evaluation": EvaluationConfig,
        "run": WarpRecRunConfig,
        "estimate": EstimateConfig,
    }

    reader: ReaderConfig
    filtering: Dict[str, dict] = None
    models: Dict[str, dict]
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)

    @model_validator(mode="before")
    @classmethod
    def load_custom_modules_first(cls, data: Any) -> Any:
        """Import the user's modules before any section is validated.

        Sections check the names they are given against the registries, so a
        model, metric or filter a custom module registers has to be there
        before them. A path that is not a module is left to the 'general'
        section, which rejects it with a message saying so.

        Args:
            data (Any): The raw configuration.

        Returns:
            Any: The raw configuration, unchanged.
        """
        if not isinstance(data, dict):
            return data
        general = data.get("general") or {}
        modules = general.get("custom_modules") if isinstance(general, dict) else None
        if isinstance(modules, str):
            modules = [modules]
        load_custom_modules(
            [m for m in modules or [] if isinstance(m, str) and is_python_module(m)]
        )
        return data

    @model_validator(mode="after")
    def migrate_reader_training_options(self) -> "WarpRecConfiguration":
        """Carry the training options still written under the reader across.

        Both were released under 'reader', so a configuration written against
        1.7 keeps working: the value is moved and the user is told where it
        belongs now. An explicit 'training' entry always wins.

        Returns:
            WarpRecConfiguration: The validated configuration.
        """
        for field in ("negative_sampling", "sequence_pooling"):
            legacy = getattr(self.reader, field)
            if legacy is None:
                continue

            logger.attention(
                f"'reader.{field}' has moved to the 'training' section and will "
                f"be removed in a future release. Write it as "
                f"'training.{field}: {legacy}' instead."
            )

            if field not in self.training.model_fields_set:
                setattr(self.training, field, legacy)

        return self

    @model_validator(mode="after")
    def config_validation(self) -> "WarpRecConfiguration":  # noqa: PLR0912
        """This method checks if everything in the configuration file is missing or incorrect.

        Returns:
            WarpRecConfiguration: The validated configuration.

        Raises:
            FileNotFoundError: If the local file has not been found. It stays
                the built-in, which pydantic lets through, so building the
                configuration directly still raises it; loading the file
                raises it as DataNotFoundError.
            RegistryError: If a filter is not registered.
            ConfigurationError: If any information between parts of the configuration file is inconsistent.
        """

        # Reading method specific checks
        if self.reader.reading_method == ReadingMethods.LOCAL:
            local_path: str = None
            if self.reader.local_path is not None:
                local_path = self.reader.local_path
            elif self.reader.split.local_path is not None:
                ext = self.reader.split.ext
                local_path = os.path.join(self.reader.split.local_path, "train" + ext)
            else:
                raise ConfigurationError(
                    "Unsupported local source or missing local path."
                )

            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Training file not at {local_path}")

        elif self.reader.reading_method == ReadingMethods.AZURE_BLOB:
            # Check if the Azure configuration is complete
            if self.general.azure is None:
                raise ConfigurationError(
                    "Azure configuration must be provided for Azure Blob reading method."
                )
            if (
                not self.general.azure.storage_account_name
                or not self.general.azure.container_name
            ):
                raise ConfigurationError(
                    "Both storage_account_name and container_name must be provided in Azure configuration."
                )

        # Data read already split never goes through the filters. Such
        # configurations ran before, so they are warned about, not refused.
        if self.filtering and self.reader.loading_strategy == "split":
            logger.attention(
                "The 'filtering' section is ignored with 'reader.loading_strategy: "
                "split': data read already split is not filtered. Filter the data "
                "before splitting it, or read it whole with 'loading_strategy: "
                "dataset' to have WarpRec filter and split it."
            )

        # Check if the filters have been set correctly
        if self.filtering is not None:
            labels = self.reader.labels.model_dump()
            for filter_name, filter_params in self.filtering.items():
                if filter_name.upper() not in filter_registry.list_registered():
                    raise RegistryError(
                        **filter_registry.unknown(
                            filter_name,
                            f"Filter '{filter_name}' is not registered. These are "
                            "the filters registered: "
                            f"{filter_registry.list_registered()}",
                        )
                    )
                try:
                    filter_registry.get(filter_name, **filter_params, **labels)
                except Exception as e:
                    raise ConfigurationError(
                        f"Error initializing filter '{filter_name}' with these params {filter_params}: {e}"
                    ) from e

        return self

    @model_validator(mode="after")
    def models_validation(self) -> "WarpRecConfiguration":
        """Validate every model against the registries and the reader.

        Every pipeline runs these checks, so a model missing what it reads is
        reported when the configuration is loaded rather than when the model is
        built. The values are only validated here: the models section is left
        as written, since the design and evaluation pipelines build the models
        from it. The training pipeline overrides this to also turn the values
        into search spaces.

        Returns:
            WarpRecConfiguration: The validated configuration.
        """
        for model_name, model_data in self.models.items():
            self.validate_model(model_name, model_data)
        return self

    def validate_model(self, model_name: str, model_data: dict) -> RecomModel:
        """Validate one model's parameters and what it reads.

        Args:
            model_name (str): The name the model is configured under.
            model_data (dict): The parameters written for it.

        Returns:
            RecomModel: The validated parameters.

        Raises:
            RegistryError: If the model is not registered.
            ConfigurationError: If it reads side information, context, a
                knowledge graph or multimodal features the reader does not
                provide.
        """
        if model_name.upper() not in model_registry.list_registered():
            raise RegistryError(
                **model_registry.unknown(
                    model_name,
                    f"The model '{model_name}' is not registered. Check its name "
                    "under 'models', or load the module that registers it through "
                    "'general.custom_modules'. These are the models registered: "
                    f"{model_registry.list_registered()}",
                )
            )

        if model_name.upper() not in params_registry.list_registered():
            logger.attention(
                f"The model {model_name} is registered in the model registry, but "
                "not in parameter registry. The model will be loaded but the "
                "hyperparameter will not be validated."
            )
            self._warn_unread_hyperparameters(model_name, model_data)
            return RecomModel(**copy.deepcopy(model_data))

        # The validators edit the lists they are handed, and the values as
        # written are what the design and evaluation pipelines build from.
        model_class = params_registry.get(model_name, **copy.deepcopy(model_data))

        if model_class.need_side_information and self.reader.side is None:
            raise ConfigurationError(
                f"The model {model_name} requires side information to be provided, "
                "but none have been provided. Check the configuration file."
            )

        if model_class.need_context and not self.reader.labels.context_labels:
            raise ConfigurationError(
                f"The model {model_name} requires contextual information to be provided, "
                "but no context_labels have been configured. Check the configuration file."
            )

        if model_class.need_knowledge and self.reader.knowledge is None:
            raise ConfigurationError(
                f"The model {model_name} scores from a knowledge graph, but none "
                "has been provided. Configure 'reader.knowledge' with the triples "
                "and the item alignment. Check the configuration file."
            )

        if model_class.need_multimodal and not self.reader.multimodal:
            raise ConfigurationError(
                f"The model {model_name} scores from multimodal item features, but "
                "none have been provided. Configure 'reader.multimodal' with one "
                "entry per modality. Check the configuration file."
            )

        self._check_modalities_exist(model_name, model_data)

        # Check if there is at least one valid combination
        model_class.validate_all_combinations()

        return model_class

    @staticmethod
    def _warn_unread_hyperparameters(model_name: str, model_data: dict) -> None:
        """Log the keys a model without a parameter class will not read.

        Such a model sets only the hyperparameters it annotates, so any other
        key is dropped without a word. A model with a parameter class refuses
        it instead, when the class validates the parameters.

        Args:
            model_name (str): The name the model is configured under.
            model_data (dict): The parameters written for it.
        """
        accepted = model_registry.get_class(model_name).hyperparameter_names()
        reserved = {"meta", "optimization", "early_stopping", "fold"}
        for key in model_data:
            if key in reserved or key in accepted:
                continue
            message = (
                f"The configuration key 'models.{model_name}.{key}' is not a "
                f"hyperparameter of {model_name} and is ignored."
            )
            close = difflib.get_close_matches(str(key), accepted, n=1)
            if close:
                message += f" Did you mean 'models.{model_name}.{close[0]}'?"
            logger.attention(message)

    def _check_modalities_exist(self, model_name: str, model_data: dict) -> None:
        """Validate that a model only asks for modalities the reader provides.

        A misspelled modality would otherwise surface as an empty feature block
        at training time, long after the run started.

        Args:
            model_name (str): The name of the model being validated.
            model_data (dict): The raw parameters written for that model.

        Raises:
            ConfigurationError: If the model names a modality that was not configured.
        """
        wanted = model_data.get("modalities")
        if not wanted or not self.reader.multimodal:
            return

        # A single name may be written unwrapped, as any other parameter may.
        if isinstance(wanted, str):
            wanted = [wanted]

        # The parameter is a search space, so it may hold several lists.
        names = {
            name
            for entry in wanted
            for name in (entry if isinstance(entry, list) else [entry])
            if isinstance(name, str)
        }

        missing = sorted(names - set(self.reader.multimodal))
        if missing:
            raise ConfigurationError(
                f"The model {model_name} asks for the modalities {missing}, which "
                f"'reader.multimodal' does not configure. The configured ones are "
                f"{sorted(self.reader.multimodal)}. Check the configuration file."
            )

    def get_filters(self) -> List[Filter]:
        """Returns the initialized filters based on the configuration.

        Returns:
            List[Filter]: A list of initialized filter instances.
        """
        if not self.filtering:
            return []
        labels = self.reader.labels.model_dump()
        return [
            filter_registry.get(filter_name, **filter_params, **labels)
            for filter_name, filter_params in self.filtering.items()
        ]


class TrainConfiguration(WarpRecConfiguration):
    """Definition of configuration, used to interact with the framework.

    This class defines the structure of the configuration file accepted by the framework.

    Attributes:
        writer (WriterConfig): Configuration of the writing process.
        splitter (SplittingConfig): Configuration of the splitting process.
        dashboard (DashboardConfig): Configuration of the dashboard process.
        evaluation (EvaluationConfig): Configuration of the evaluation process.
        run (WarpRecRunConfig): Configuration of the run identity and of the
            pause and resume behaviour.
    """

    writer: WriterConfig
    splitter: SplittingConfig = None
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    evaluation: EvaluationConfig
    run: WarpRecRunConfig = Field(default_factory=WarpRecRunConfig)

    @model_validator(mode="after")
    def train_validation(self) -> "TrainConfiguration":
        """This method checks if everything in the configuration file is missing or incorrect.

        Returns:
            TrainConfiguration: The validated configuration.

        Raises:
            ConfigurationError: If any information between parts of the configuration file is inconsistent.
        """

        # Writing method specific checks
        if self.writer.writing_method == WritingMethods.AZURE_BLOB:
            # Check if the Azure configuration is complete
            if self.general.azure is None:
                raise ConfigurationError(
                    "Azure configuration must be provided for Azure Blob writing method."
                )
            if (
                not self.general.azure.storage_account_name
                or not self.general.azure.container_name
            ):
                raise ConfigurationError(
                    "Both storage_account_name and container_name must be provided in Azure configuration."
                )

        # Check if evaluation has been set up correctly
        if self.evaluation.full_evaluation_on_report:
            # Check if the validation metric is in the
            # metrics to evaluate
            val_metric, val_k = validation_metric(self.evaluation.validation_metric)

            if (
                val_metric not in self.evaluation.metrics
                or val_k not in self.evaluation.top_k
            ):
                raise ConfigurationError(
                    "Full evaluation on report has been requested but the "
                    "validation metric it's not contained in the evaluation metrics. "
                    f"Validation metric: {self.evaluation.validation_metric} ."
                    f"Metrics to evaluate: {self.evaluation.metrics}. "
                    f"TopK to evaluate: {self.evaluation.top_k}."
                )

        # Check if dataset splitting has been correctly setup
        if self.reader.loading_strategy == "dataset" and self.splitter is None:
            raise ConfigurationError(
                "The 'dataset' loading strategy requires an explicit dataset splitting "
                "configuration, but none was provided. Please verify the splitter "
                "settings in the configuration file."
            )

        return self

    @model_validator(mode="after")
    def models_validation(self) -> "TrainConfiguration":
        """Validate every model and turn its values into search spaces.

        Returns:
            TrainConfiguration: The validated configuration.
        """
        self.models = self.parse_models()
        return self

    def parse_models(self) -> dict:
        """This method parses the models and creates the correct data structures.

        Returns:
            dict: The dictionary containing all the models and their parameters.
        """
        # Check if Context-Aware model is in the experiment
        from warprec.recommenders.base_recommender import ContextRecommenderUtils

        parsed_models = {}
        for model_name, model_data in self.models.items():
            model_class = self.validate_model(model_name, model_data)

            if issubclass(
                model_registry.get_class(model_name), ContextRecommenderUtils
            ):
                logger.attention(
                    f"The model {model_name} is a Context-Aware model. "
                    "Leave-One-Out splitting strategy is advised for a correct evaluation."
                )

            parsed_models[model_name] = model_class.model_dump()

        return parsed_models

    def get_storage_path(self) -> str:
        """Returns the storage path for the ray results.

        Returns:
            str: The storage path.
        """
        match self.writer.writing_method:
            case WritingMethods.LOCAL:
                # The local path will be ~/experiment_path/dataset_name/ray_results
                return os.path.join(
                    os.getcwd(),
                    self.writer.local_experiment_path,
                    self.writer.dataset_name,
                    "ray_results",
                )
            case WritingMethods.AZURE_BLOB:
                # The azure blob path will be az://<container_name>/<blob_experiment_container>/dataset_name/ray_results
                return os.path.join(
                    "az://",
                    self.general.azure.container_name,
                    self.writer.azure_blob_experiment_container,
                    self.writer.dataset_name,
                    "ray_results",
                )

        return os.path.join(os.getcwd(), "ray_results")


class DesignConfiguration(WarpRecConfiguration):
    """Definition of design pipeline configuration, used to test custom models.

    Attributes:
        splitter (SplittingConfig): Configuration of the splitting process.
        evaluation (EvaluationConfig): Configuration of the evaluation process.
    """

    splitter: SplittingConfig = None
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def models_validation(self) -> "DesignConfiguration":
        """Validate every model, and that each is given single values.

        The design pipeline builds each model once, from the values as written,
        so a search space would reach the model as a list and fail inside it.

        Returns:
            DesignConfiguration: The validated configuration.
        """
        for model_name, model_data in self.models.items():
            params = self.validate_model(model_name, model_data)

            # Without a parameter class the type of a value is unknown, so a
            # list cannot be told apart from a list-valued hyperparameter.
            if model_name.upper() in params_registry.list_registered():
                self._check_single_values(model_name, model_data, params)
        return self

    @staticmethod
    def _check_single_values(
        model_name: str, model_data: dict, params: RecomModel
    ) -> None:
        """Refuse a hyperparameter written as a search space.

        Validation turns every value into the list of alternatives a search
        would draw from, so a single value is one with exactly one alternative,
        written as that alternative rather than wrapped in a list.

        Args:
            model_name (str): The name the model is configured under.
            model_data (dict): The parameters as written.
            params (RecomModel): The same parameters, validated.

        Raises:
            ConfigurationError: If a hyperparameter is a list of values or a range.
        """
        validated = params.model_dump()
        spaces = [space.value for space in SearchSpace]
        for field, raw in model_data.items():
            if field in ("meta", "optimization", "early_stopping"):
                continue
            space = validated.get(field)
            if not isinstance(space, list) or not space:
                continue

            token = space[0] if isinstance(space[0], str) else None
            if token is not None and token.lower() not in spaces:
                token = None
            alternatives = space[1:] if token is not None else space
            is_range = token is not None and token.lower() not in (
                SearchSpace.GRID.value,
                SearchSpace.CHOICE.value,
            )

            if (
                is_range
                or len(alternatives) != 1
                or (isinstance(raw, list) and raw != alternatives[0])
            ):
                raise ConfigurationError(
                    f"The design pipeline trains {model_name} once, from single "
                    f"values, but 'models.{model_name}.{field}' is {raw}, which "
                    "it reads as a search space. Write the one value to train "
                    "with, without brackets unless the value is itself a list, or "
                    "run the train pipeline to search over several."
                )


class EvalConfiguration(WarpRecConfiguration):
    """Definition of eval pipeline configuration, used evaluate trained models.

    Attributes:
        writer (WriterConfig): Configuration of the writing process.
        splitter (SplittingConfig): Configuration of the splitting process.
        evaluation (EvaluationConfig): Configuration of the evaluation process.
    """

    writer: WriterConfig
    splitter: SplittingConfig = None
    evaluation: EvaluationConfig


class EstimateConfiguration(EvalConfiguration):
    """Definition of estimate pipeline configuration.

    Attributes:
        estimate (EstimateConfig): Estimate-specific execution settings.
    """

    estimate: EstimateConfig = Field(default_factory=EstimateConfig)


def load_train_configuration(path: str) -> TrainConfiguration:
    """This method reads the train configuration file and returns
        a TrainConfiguration object.

    A file that is not a valid train configuration raises one
    ConfigurationError listing the invalid keys; a missing file, or missing
    training data it names, raises DataNotFoundError (see
    read_configuration_file).

    Args:
        path (str): The path to the configuration file.

    Returns:
        TrainConfiguration: The configuration object created from the configuration file.
    """
    logger.msg(f"Reading train configuration file in: {path}")
    config = read_configuration_file(path, TrainConfiguration, "train")
    logger.msg("Reading process completed correctly.")
    return config


def load_design_configuration(path: str) -> DesignConfiguration:
    """This method reads the train configuration file and returns
        a DesignConfiguration object.

    A file that is not a valid design configuration raises one
    ConfigurationError listing the invalid keys; a missing file, or missing
    training data it names, raises DataNotFoundError (see
    read_configuration_file).

    Args:
        path (str): The path to the configuration file.

    Returns:
        DesignConfiguration: The configuration object created from the configuration file.
    """
    logger.msg(f"Reading design configuration file in: {path}")
    config = read_configuration_file(path, DesignConfiguration, "design")
    logger.msg("Reading process completed correctly.")
    return config


def load_eval_configuration(path: str) -> EvalConfiguration:
    """This method reads the train configuration file and returns
        a EvalConfiguration object.

    A file that is not a valid eval configuration raises one
    ConfigurationError listing the invalid keys; a missing file, or missing
    training data it names, raises DataNotFoundError (see
    read_configuration_file).

    Args:
        path (str): The path to the configuration file.

    Returns:
        EvalConfiguration: The configuration object created from the configuration file.
    """
    logger.msg(f"Reading eval configuration file in: {path}")
    config = read_configuration_file(path, EvalConfiguration, "eval")
    logger.msg("Reading process completed correctly.")
    return config


def load_estimate_configuration(path: str) -> EstimateConfiguration:
    """This method reads the estimate configuration file and returns
        a EstimateConfiguration object.

    A file that is not a valid estimate configuration raises one
    ConfigurationError listing the invalid keys; a missing file, or missing
    training data it names, raises DataNotFoundError (see
    read_configuration_file).

    Args:
        path (str): The path to the configuration file.

    Returns:
        EstimateConfiguration: The configuration object created from the configuration file.
    """
    logger.msg(f"Reading estimate configuration file in: {path}")
    config = read_configuration_file(path, EstimateConfiguration, "estimate")
    logger.msg("Reading process completed correctly.")
    return config


def load_callback(
    callback_config: WarpRecCallbackConfig, *args: Any, **kwargs: Any
) -> WarpRecCallback:
    """Dynamically loads and initializes a custom WarpRecCallback class
    based on the provided configuration.

    This function assumes that `callback_config` has already been validated
    via Pydantic, ensuring that the module path and class name are correct,
    and that the class exists and is a subclass of `WarpRecCallback`.

    Args:
        callback_config (WarpRecCallbackConfig): The Pydantic configuration object
            for the custom callback.
        *args (Any): Additional positional arguments to pass to the callback's constructor.
        **kwargs (Any): Additional keyword arguments to pass to the callback's constructor.

    Returns:
        WarpRecCallback: An instance of the custom callback, or None if no
            custom callback is specified in the configuration.

    Raises:
        RuntimeError: If an unexpected error occurs during loading or initialization,
            given that prior validation should have prevented most errors.
    """
    if (
        callback_config is None
        or callback_config.callback_path is None
        or callback_config.callback_name is None
    ):
        return WarpRecCallback()  # Empty callback used for consistency

    module_path = Path(callback_config.callback_path)
    class_name = callback_config.callback_name

    # Save the original sys.path to restore it afterwards
    original_sys_path = sys.path[:]

    try:
        # Add the module's directory to sys.path to allow for internal imports
        module_dir = module_path.parent
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))

        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None:
            raise RuntimeError(
                f"Could not load spec for module: {module_path}. "
                f"This should not happen after validation."
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the class from the module
        callback_class = getattr(module, class_name)

        # Initialize and return the callback instance
        return callback_class(*args, **kwargs)

    except Exception as e:
        # Catch any residual errors, though validation should prevent most
        raise RuntimeError(
            f"Unexpected error during initialization of callback '{class_name}' "
            f"from '{module_path}': {e}"
        ) from e
    finally:
        # Restore sys.path to avoid side-effects
        sys.path = original_sys_path
