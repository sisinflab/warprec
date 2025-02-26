from tqdm import tqdm
from tabulate import tabulate
from elliotwo.data.dataset import AbstractDataset
from elliotwo.utils.config import Configuration
from elliotwo.evaluation.metrics import AbstractMetric
from elliotwo.recommenders.abstract_recommender import AbstractRecommender
from elliotwo.utils.logger import logger
from elliotwo.utils.registry import metric_registry


class Evaluator:
    """Evaluator class will evaluate a trained model on a given
    set of metrics, taking into account the cutoff.

    If a validation set has been provided in the dataset, then this
    class will provide results on validation too.

    Args:
        dataset (AbstractDataset): The dataset to be evaluate.
        config (Configuration): The configuration where all
            information about evaluation is stored.
    """

    def __init__(self, dataset: AbstractDataset, config: Configuration):
        self._dataset = dataset
        self._config = config
        self._metrics = self._config.evaluation.metrics
        self._top_k = self._config.evaluation.top_k
        self._result_dict: dict[str, dict] = {}
        if self._config.splitter.validation:
            self._result_dict["Validation"] = {}
        self._result_dict["Test"] = {}

    def run(self, model: AbstractRecommender) -> dict:
        """Use this method to start the evaluation.

        Args:
            model (AbstractRecommender): The model already trained to be evaluated.

        Returns:
            dict: The results in dict format using this
                criteria {'Test' or 'Validation': dict{metric_name@top_k: value}}
        """
        logger.separator()
        logger.msg(f"Starting evaluation for model {model.name}")

        total_evaluation = len(self._metrics) * len(self._top_k)
        with tqdm(total=total_evaluation, desc="Evaluating metrics") as pbar:
            for metric_name in self._metrics:
                for k in self._top_k:
                    result = self.evaluate_metric(
                        metric_name, model, self._dataset, self._config, k
                    )

                    if self._config.splitter.validation:
                        self._result_dict["Validation"][f"{metric_name}@{k}"] = result[
                            "val"
                        ]
                    self._result_dict["Test"][f"{metric_name}@{k}"] = result["test"]
                    pbar.update()

        self._print_console()
        logger.positive(f"Evaluation completed for model {model.name}")

        return self._result_dict

    def evaluate_metric(
        self,
        metric_name: str,
        model: AbstractRecommender,
        dataset: AbstractDataset,
        config: Configuration,
        top_k: int,
    ) -> dict:
        """This function will calculate a given metric on data provided.

        Args:
            metric_name (str): The string name of the metric that will be calculated.
            model (AbstractRecommender): The trained model.
            dataset (AbstractDataset): The dataset to evaluate.
            config (Configuration): Configuration file.
            top_k (int): Cutoff to be used during evaluation.

        Returns:
            dict: The results of the validation in dict
                format using this criteria {'test' or 'val': value}
        """

        results = {}
        metric: AbstractMetric = metric_registry.get(metric_name, config)
        if config.splitter.validation:
            results["val"] = metric.eval(model, dataset.val_set, top_k)
        results["test"] = metric.eval(model, dataset.test_set, top_k)
        return results

    def _print_console(self):
        """Utility function to print results using tabulate."""
        logger.msg("")
        if self._config.splitter.validation:
            self.__pprint_result_set(
                self._result_dict["Validation"], "Validation Set Evaluation"
            )
            logger.msg("")
        self.__pprint_result_set(self._result_dict["Test"], "Test Set Evaluation")

    def __pprint_result_set(self, res_dict: dict, header: str):
        """Utility function to print results using tabulate.

        Args:
            res_dict (dict): The dictionary containing all the results.
            header (str): The header of the evaluation grid,
                usually set with the name of evaluation.
        """
        _tab = []
        for k in self._top_k:
            _metric_tab = ["Top@" + str(k)]
            for metric_name in self._metrics:
                key = metric_name + "@" + str(k)
                score = res_dict[key]
                _metric_tab.append(score)
            _tab.append(_metric_tab)
        table = tabulate(_tab, headers=self._metrics, tablefmt="grid")
        _rlen = len(table.split("\n", maxsplit=1)[0])
        logger.msg(header.center(_rlen, "-"))
        for row in table.split("\n"):
            logger.msg(row)
