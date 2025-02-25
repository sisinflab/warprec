from typing import Tuple, Any

import numpy as np
import ray
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from elliotwo.recommenders.abstract_recommender import AbstractRecommender
from elliotwo.data.dataset import AbstractDataset
from elliotwo.evaluation.metrics import AbstractMetric
from elliotwo.utils.config import Configuration
from elliotwo.utils.logger import logger
from elliotwo.utils.registry import model_registry


class Trainer:
    """This class will be used to train a model and optimize the hyperparameters.

    Args:
        model_name (str): The name of the model to optimize.
        dataset (AbstractDataset): The dataset on wich optimize the model.
        param_space (dict): The param space to optimize using Hyperopt.
        metric (AbstractMetric): The metric to use as validation.
        top_k (int): The cutoff tu use as validation.
        config (Configuration): The configuration of the experiment.
    """

    def __init__(
        self,
        model_name: str,
        dataset: AbstractDataset,
        param_space: dict,
        metric: AbstractMetric,
        top_k: int,
        config: Configuration,
    ):
        self._model_name = model_name
        self._dataset = ray.put(dataset)
        self._param_space = param_space
        self._metric = metric
        self._top_k = top_k
        self._config = config
        self._imp = config.models[model_name]["meta"]["implementation"]
        self._dgts = self._config.general.float_digits
        self._rstate = np.random.default_rng(self._config.general.seed)

        # Track best evaluation
        self.best_params: dict[str, Any] = {}
        self.best_model = None
        self.best_score = -np.inf

    def train_and_evaluate(self) -> Tuple[AbstractRecommender, dict]:
        """Main method of the Trainer class.

        This method will execute the training of the model and evaluation, \
            according to informations passed through configuration.

        Returns:
            Tuple[AbstractRecommender, dict]:
                AbstractRecommender: The model trained.
                dict: A dictionary with the best params. The content depends on the model trained.
        """
        logger.separator()
        logger.msg(f"Starting hyperparameter tuning for {self._model_name}")

        # Using HyperOpt for TPE search algo
        search_alg = HyperOptSearch(
            metric="score",
            mode="max",
            random_state_seed=self._config.general.seed,
        )

        # Run the hyperparameter tuning
        analysis = tune.run(
            tune.with_parameters(
                self._objective_function,
                dataset=self._dataset,
                model_name=self._model_name,
                metric=self._metric,
                top_k=self._top_k,
                implementation=self._imp,
                config=self._config,
            ),
            resources_per_trial={"cpu": 1},
            config=self._param_space,
            metric="score",
            mode="max",
            search_alg=search_alg,
            num_samples=10,
            verbose=0,
        )

        # Train and retrieve results
        best_trial = analysis.best_trial

        self.best_params = analysis.best_config
        self.best_model = best_trial.last_result["model"]
        self.best_score = best_trial.last_result["score"]

        logger.msg(
            f"Best params combination: {self.best_params} with a score of \
                {self._metric.get_name()}@{self._top_k}: {self.best_score:.{self._dgts}f}"
        )
        logger.positive(
            f"Hyperparameter tuning for {self._model_name} ended successfully."
        )
        logger.separator()

        return self.best_model, self.best_params

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
        logger.msg(f"Training with current parameters: {params}")
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
            dataset (AbstractDataset): The dataset on wich evaluation will be executed.
            metric (AbstractMetric): The metric to evaluate.
            top_k (int): The cutoff to calculate metric.

        Returns:
            float: The value of the metric.
        """
        if self._config.splitter.validation:
            return metric.eval(model, dataset.val_set, top_k)
        return metric.eval(model, dataset.test_set, top_k)
