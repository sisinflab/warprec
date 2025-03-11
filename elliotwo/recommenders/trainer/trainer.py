import os
import tempfile
from typing import Tuple

import ray
import torch
import shutil
from ray import tune
from ray.tune import Checkpoint
from elliotwo.recommenders.abstract_recommender import AbstractRecommender
from elliotwo.data.dataset import AbstractDataset
from elliotwo.evaluation.evaluator import Evaluator
from elliotwo.recommenders.trainer.search_algorithm_wrapper import (
    BaseSearchWrapper,
)
from elliotwo.recommenders.trainer.scheduler_wrapper import (
    BaseSchedulerWrapper,
)
from elliotwo.utils.config import Configuration, RecomModel
from elliotwo.utils.logger import logger
from elliotwo.utils.registry import (
    model_registry,
    params_registry,
    search_algorithm_registry,
    scheduler_registry,
)


class Trainer:
    """This class will be used to train a model and optimize the hyperparameters.

    Args:
        model_name (str): The name of the model to optimize.
        param (dict): The parameters of the model already in
            Ray Tune format.
        dataset (AbstractDataset): The dataset to use during training.
        metric_name (str): The name of the metric that will be used
            as validation.
        top_k (int): The cutoff tu use as validation.
        config (Configuration): The configuration of the experiment.
    """

    def __init__(
        self,
        model_name: str,
        param: dict,
        dataset: AbstractDataset,
        metric_name: str,
        top_k: int,
        config: Configuration,
    ):
        self.infos = dataset.info()
        self._model_params: RecomModel = params_registry.get(
            model_name, **config.models[model_name]
        )
        self._model = ray.put(
            model_registry.get(
                model_name, self._model_params.meta.implementation, **self.infos
            )
        )
        self.model_name = model_name
        self._evaluator = Evaluator([metric_name], [top_k])
        self._train_param = param
        self._metric_name = metric_name
        self._top_k = top_k
        self._dataset = ray.put(dataset)
        self._config = config

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
            f"Starting hyperparameter tuning for {self.model_name} "
            f"with {self._model_params.optimization.strategy.name} strategy "
            f"and with {self._model_params.optimization.scheduler.name} scheduler."
        )

        properties = self._model_params.optimization.properties.model_dump()

        # Ray Tune parameters
        obj_function = tune.with_parameters(
            self._objective_function,
            model=self._model,
            dataset=self._dataset,
            mode=properties["mode"],
            evaluator=self._evaluator,
        )

        search_alg: BaseSearchWrapper = search_algorithm_registry.get(
            self._model_params.optimization.strategy, **properties
        )
        scheduler: BaseSchedulerWrapper = scheduler_registry.get(
            self._model_params.optimization.scheduler, **properties
        )

        best_checkpoint_callback = BestCheckpointCallback(
            "score", self._model_params.optimization.properties.mode
        )

        # Run the hyperparameter tuning
        analysis = tune.run(
            obj_function,
            resources_per_trial={
                "cpu": self._model_params.optimization.cpu_per_trial,
                "gpu": self._model_params.optimization.gpu_per_trial,
            },
            config=self._train_param,
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=self._model_params.optimization.num_samples,
            verbose=1,
            callbacks=[best_checkpoint_callback],
        )

        # Train and retrieve results
        best_trial = analysis.get_best_trial(
            "score", self._model_params.optimization.properties.mode
        )
        best_params = analysis.get_best_config(
            "score", self._model_params.optimization.properties.mode
        )

        # Retrieve best model from checkpoint
        checkpoint = torch.load(
            os.path.join(
                best_checkpoint_callback.get_best_checkpoint(), "checkpoint.pt"
            )
        )
        model_state = checkpoint["model_state"]
        best_model = model_registry.get(
            self.model_name, self._model_params.meta.implementation, **self.infos
        )
        best_model.load_state_dict(model_state)

        # Best score obtained during hyperparameter optimization
        best_score = best_trial.last_result["score"]

        logger.msg(
            f"Best params combination: {best_params} with a score of "
            f"{self._metric_name}@"
            f"{self._top_k}: "
            f"{best_score:.{self._config.general.float_digits}f}."
        )
        logger.positive(
            f"Hyperparameter tuning for {self.model_name} ended successfully."
        )

        ray.shutdown()

        return (
            best_model,
            best_params,
        )

    def _objective_function(
        self,
        params: dict,
        model: AbstractRecommender,
        dataset: AbstractDataset,
        mode: str,
        evaluator: Evaluator,
    ) -> dict:
        """Objective function to optimize the hyperparameters.

        Args:
            params (dict): The parameter to train the model.
            model (AbstractRecommender): The model to train.
            dataset (AbstractDataset): The dataset to train the model on.
            mode (str): Wether or not to maximize or minimize the metric.
            evaluator (Evaluator): The evaluator that will calculate the
                validation metric.

        Returns:
            dict: A dictionary containing the score and the model trained.
        """
        try:
            model.fit(dataset.train_set, params)
        except Exception as e:
            logger.negative(
                f"The fitting of the model {model.name}, failed with parameters: {params}. Error: {e}"
            )
            if mode == "max":
                return {"score": -torch.inf}
            return {"score": torch.inf}

        evaluator.evaluate(model, dataset, test_set=False)
        results = evaluator.compute_results()
        score = results[self._top_k][self._metric_name]

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(
                {"model_state": model.state_dict()},
                os.path.join(tmpdir, "checkpoint.pt"),
            )
            tune.report(
                metrics={"score": score}, checkpoint=Checkpoint.from_directory(tmpdir)
            )

        return {"score": score}


class BestCheckpointCallback(tune.Callback):
    def __init__(self, metric: str, mode: str):
        self.metric = metric
        self.mode = mode
        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.best_checkpoint = None

    def on_trial_complete(self, iteration, trials, trial, **info):
        score = trial.last_result.get(self.metric, None)
        if score is None:
            return

        is_better = (self.mode == "max" and score > self.best_score) or (
            self.mode == "min" and score < self.best_score
        )

        if is_better:
            # Delete previous best checkpoint
            if self.best_checkpoint and os.path.exists(self.best_checkpoint):
                shutil.rmtree(self.best_checkpoint)

            # Update best score and checkpoint
            self.best_score = score
            self.best_checkpoint = trial.checkpoint.path

        elif os.path.exists(trial.checkpoint.path):
            shutil.rmtree(trial.checkpoint.path)

    def get_best_checkpoint(self):
        return self.best_checkpoint
