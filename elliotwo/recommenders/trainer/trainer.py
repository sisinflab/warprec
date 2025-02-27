from typing import Tuple

import ray
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from elliotwo.recommenders.abstract_recommender import AbstractRecommender
from elliotwo.data.dataset import AbstractDataset
from elliotwo.evaluation.metrics import AbstractMetric
from elliotwo.utils.config import Configuration
from elliotwo.utils.dataclasses import TrainerConfig
from elliotwo.utils.logger import logger
from elliotwo.utils.registry import model_registry


class Trainer:
    """This class will be used to train a model and optimize the hyperparameters.

    Args:
        dataset (AbstractDataset): The dataset to use during training.
        train_config (TrainerConfig): The training configuration.
        config (Configuration): The configuration of the experiment.
    """

    def __init__(
        self,
        dataset: AbstractDataset,
        train_config: TrainerConfig,
        config: Configuration,
    ):
        self._train_config = train_config
        self._dataset = ray.put(dataset)
        self._config = config
        self._imp = config.models[train_config.model_name]["meta"]["implementation"]
        self._strategy = config.models[train_config.model_name]["optimization"][
            "strategy"
        ]
        self._mode = config.models[train_config.model_name]["optimization"]["mode"]
        self._num_samples = config.models[train_config.model_name]["optimization"][
            "num_samples"
        ]
        self._cpu = config.models[train_config.model_name]["optimization"][
            "cpu_per_trial"
        ]
        self._gpu = config.models[train_config.model_name]["optimization"][
            "gpu_per_trial"
        ]
        self._dgts = self._config.general.float_digits

    def train_and_evaluate(self) -> Tuple[AbstractRecommender, dict]:
        """Main method of the Trainer class.

        This method will execute the training of the model and evaluation,
        according to information passed through configuration.

        Returns:
            Tuple[AbstractRecommender, dict]:
                AbstractRecommender: The model trained.
                dict: A dictionary with the best params. The content depends on the model trained.
        """
        logger.separator()
        logger.msg(
            f"Starting hyperparameter tuning for {self._train_config.model_name} "
            f"with {self._strategy} strategy."
        )

        # Ray Tune parameters
        obj_function = tune.with_parameters(
            self._objective_function,
            dataset=self._dataset,
            model_name=self._train_config.model_name,
            metric=self._train_config.metric,
            top_k=self._train_config.top_k,
            implementation=self._imp,
            config=self._config,
        )

        # Selecting the correct search algorithm.
        # This might be done in a better way
        search_alg = None  # Grid search

        if self._strategy == "hopt":
            search_alg = HyperOptSearch(
                metric="score",
                mode=self._mode,
                random_state_seed=self._config.general.seed,
            )

        # Run the hyperparameter tuning
        analysis = tune.run(
            obj_function,
            resources_per_trial={"cpu": self._cpu, "gpu": self._gpu},
            config=self._train_config.param,
            metric="score",
            mode=self._mode,
            search_alg=search_alg,
            num_samples=self._num_samples,
            verbose=0,
        )

        # Train and retrieve results
        best_trial = analysis.best_trial

        best_params = analysis.best_config
        best_model = best_trial.last_result["model"]
        best_score = best_trial.last_result["score"]

        logger.msg(
            f"Best params combination: {best_params} with a score of "
            f"{self._train_config.metric.get_name()}@"
            f"{self._train_config.top_k}: "
            f"{best_score:.{self._dgts}f}."
        )
        logger.positive(
            f"Hyperparameter tuning for {self._train_config.model_name} ended successfully."
        )
        logger.separator()

        return best_model, best_params

    def _objective_function(
        self,
        params: dict,
        dataset: AbstractDataset,
        model_name: str,
        metric: AbstractMetric,
        top_k: int,
        implementation: str,
        config: Configuration,
    ) -> dict:
        """Objective function to optimize the hyperparameters.

        Args:
            params (dict): The dictionary with the parameters for the model.
            dataset (AbstractDataset): The dataset to use for training.
            model_name (str): The name of the model to optimize.
            metric (AbstractMetric): The metric to use for evaluation.
            top_k (int): The cutoff to use for evaluation.
            implementation (str): The implementation to use.
            config (Configuration): The configuration of the experiment.

        Returns:
            dict: A dictionary containing the score and the model trained.
        """
        if implementation == "latest":
            model = model_registry.get_latest(
                name=model_name, config=config, dataset=dataset, params=params
            )
        else:
            model = model_registry.get(
                name=model_name,
                implementation=implementation,
                config=config,
                dataset=dataset,
                params=params,
            )
        model.fit()
        score = self._evaluate(model, dataset, metric, top_k)
        return {"score": score, "model": model}

    def _evaluate(
        self,
        model: AbstractRecommender,
        dataset: AbstractDataset,
        metric: AbstractMetric,
        top_k: int,
    ) -> float:
        """This function will evaluate a metric on a trained model.

        Args:
            model (AbstractRecommender): The trained model to evaluate.
            dataset (AbstractDataset): The dataset on which evaluation will be executed.
            metric (AbstractMetric): The metric to evaluate.
            top_k (int): The cutoff to calculate metric.

        Returns:
            float: The value of the metric.
        """
        if self._config.splitter.validation:
            return metric.eval(model, dataset.val_set, top_k)
        return metric.eval(model, dataset.test_set, top_k)
