"""The extension points a user writes against: models, parameters, metrics, stash.

Each test pins down a trap a user fell into: something that silently did the
wrong thing, or failed far from its cause, now either works or says what to do.
"""

import sys
import uuid
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Set

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor

import warprec.evaluation  # noqa: F401  (populates the metric registry)
from warprec.data.dataset import Dataset
from warprec.data.stash import METRIC_ARGUMENTS
from warprec.evaluation import Evaluator
from warprec.evaluation.metrics.accuracy.recall import Recall
from warprec.evaluation.metrics.base_metric import TopKMetric, UserAverageTopKMetric
from warprec.evaluation.metrics.bias.aplt import APLT
from warprec.pipelines.eval import _load_or_build_model
from warprec.recommenders.base_recommender import Recommender
from warprec.recommenders.trainer.trainer import Trainer
from warprec.utils.config.evaluation_configuration import ComplexMetricConfig
from warprec.utils.config.model_configuration import FLOAT_FIELD, INT_FIELD, RecomModel
from warprec.utils.enums import MetricBlock, RatingType, SearchSpace
from warprec.utils.helpers import load_custom_modules
from warprec.utils.registry import metric_registry, model_registry, params_registry


def unique_name(prefix: str) -> str:
    """A registry name no other test uses.

    Args:
        prefix (str): The readable start of the name.

    Returns:
        str: The prefix followed by a random suffix.
    """
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def test_a_parameter_class_may_be_named_unlike_its_model():
    """The class is found by the name it is registered under, not its own."""
    name = unique_name("MyBPR")

    @model_registry.register(name)
    class MyModel(Recommender):  # pylint: disable=unused-variable
        """A model the parameter class below describes."""

        embedding_size: int

    @params_registry.register(name)
    class MyModelParams(RecomModel):
        """Parameters named after nothing in particular."""

        embedding_size: INT_FIELD

    params = MyModelParams(embedding_size=[8, 16])

    assert params.embedding_size == [SearchSpace.GRID, 8, 16]
    space = Trainer._parse_params(None, params)  # type: ignore[arg-type]
    assert list(space["embedding_size"]["grid_search"]) == [8, 16]


def test_a_parameter_class_named_like_its_model_still_validates():
    """The documented naming keeps working."""
    name = unique_name("Named")

    @model_registry.register(name)
    class Model(Recommender):  # pylint: disable=unused-variable
        """A model the parameter class below describes."""

        embedding_size: int

    params_class = type(
        name, (RecomModel,), {"__annotations__": {"embedding_size": INT_FIELD}}
    )
    params_registry.register(name)(params_class)

    params = params_class(embedding_size=[8, 16])

    assert params.embedding_size == [SearchSpace.GRID, 8, 16]


class _TiedScoreRecommender(Recommender):
    """Scores with many ties, so the ranking depends on how ties are broken."""

    def __init__(self, params: dict, info: dict, **kwargs: Any):
        super().__init__(params, info, **kwargs)
        generator = torch.Generator().manual_seed(3)
        self.scores = torch.randint(
            0, 4, (info["n_users"], info["n_items"]), generator=generator
        ).float()

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Score the requested users over the catalogue.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): Unused.
            item_indices (Optional[Tensor]): Unused; the tests rank in full.
            **kwargs (Any): Unused.

        Returns:
            Tensor: The score matrix for those users.
        """
        return self.scores[user_indices].clone()

    def forward(self, *args: Any, **kwargs: Any):
        """Unused by the evaluation path."""
        raise NotImplementedError


@pytest.fixture(name="ranking_dataset", scope="module")
def ranking_dataset_fixture() -> Dataset:
    """Eight users over forty items, each with two held-out items.

    Returns:
        Dataset: The dataset under test.
    """
    rng = np.random.default_rng(5)
    train_rows, eval_rows = [], []
    for user in range(8):
        seen = rng.choice(40, 8, replace=False)
        train_rows += [(user, int(item), 5.0) for item in seen[:6]]
        eval_rows += [(user, int(item), 5.0) for item in seen[6:]]
    columns = ["user_id", "item_id", "rating"]
    return Dataset(
        train_data=pd.DataFrame(train_rows, columns=columns),
        eval_data=pd.DataFrame(eval_rows, columns=columns),
        rating_type=RatingType.EXPLICIT,
        rating_label="rating",
        batch_size=3,
    )


def evaluate_alone(metric_name: str, dataset: Dataset) -> Dict[str, Any]:
    """Evaluate one metric at k=5, with no other metric next to it.

    Args:
        metric_name (str): The registered name of the metric.
        dataset (Dataset): The dataset to evaluate on.

    Returns:
        Dict[str, Any]: The results at k=5.
    """
    evaluator = Evaluator([metric_name], [5], train_set=dataset.train_set.get_sparse())
    evaluator.evaluate(
        model=_TiedScoreRecommender({}, dataset.info()),
        dataloader=dataset.get_evaluation_dataloader(),
        strategy="full",
        dataset=dataset,
    )
    return evaluator.compute_results()[5]


def test_an_undeclared_user_average_metric_evaluates_alone(ranking_dataset: Dataset):
    """It is given the blocks it reads, and ranks exactly as the built-ins do."""
    name = unique_name("MyRecall")

    @metric_registry.register(name)
    class MyRecall(UserAverageTopKMetric):  # pylint: disable=unused-variable
        """Recall, written without declaring the blocks it reads."""

        def compute_scores(
            self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
        ) -> Tensor:
            """The share of each user's relevant items in the top k.

            Args:
                preds (Tensor): The prediction tensor.
                target (Tensor): The binary relevance.
                top_k_rel (Tensor): The binary relevance of the top k.
                **kwargs (Any): Unused.

            Returns:
                Tensor: The recall of each user.
            """
            return top_k_rel.sum(dim=1) / target.sum(dim=1).clamp(min=1)

    mine = evaluate_alone(name, ranking_dataset)[name]
    builtin = evaluate_alone("Recall", ranking_dataset)["Recall"]

    torch.testing.assert_close(mine, builtin, equal_nan=True)


def test_an_undeclared_metric_reading_the_top_k_evaluates_alone(
    ranking_dataset: Dataset,
):
    """A metric reading the top-k indices straight from the keywords gets them."""
    name = unique_name("MyCoverage")

    @metric_registry.register(name)
    class MyCoverage(TopKMetric):  # pylint: disable=unused-variable
        """Item coverage, written without declaring the blocks it reads."""

        seen: Tensor

        def __init__(self, k: int, num_items: int, **kwargs: Any):
            super().__init__(k)
            self.add_state(
                "seen", default=torch.zeros(num_items + 1), dist_reduce_fx="sum"
            )

        def update(self, preds: Tensor, **kwargs: Any):
            """Count the items recommended in this batch.

            Args:
                preds (Tensor): The prediction tensor.
                **kwargs (Any): The blocks the evaluator computed.
            """
            indices = kwargs[f"top_{self.k}_indices"].flatten()
            self.seen.index_add_(0, indices, torch.ones_like(indices).float())

        def compute(self) -> Dict[str, float]:
            """The number of distinct items recommended.

            Returns:
                Dict[str, float]: The coverage under the metric's name.
            """
            return {self.name: float((self.seen > 0).sum())}

    mine = evaluate_alone(name, ranking_dataset)[name]
    builtin = evaluate_alone("ItemCoverage", ranking_dataset)["ItemCoverage"]

    assert mine == float(builtin)


def test_a_metric_reports_under_the_name_it_is_registered_under(
    ranking_dataset: Dataset,
):
    """A class Foo registered as 'MyName' used to report 'Foo@k'."""
    name = unique_name("MyName")

    @metric_registry.register(name)
    class Foo(Recall):  # pylint: disable=unused-variable
        """Recall, under a name of its own."""

    results = evaluate_alone(name.lower(), ranking_dataset)

    assert set(results) == {name}
    torch.testing.assert_close(
        results[name],
        evaluate_alone("Recall", ranking_dataset)["Recall"],
        equal_nan=True,
    )


def test_a_parametrised_name_starts_with_the_registered_name(
    ranking_dataset: Dataset,
):
    """A metric naming itself after its parameters builds on the registered name."""
    name = unique_name("MyLongTail")

    @metric_registry.register(name)
    class Tail(APLT):  # pylint: disable=unused-variable
        """APLT, under a name of its own."""

    evaluator = Evaluator(
        [],
        [5],
        train_set=ranking_dataset.train_set.get_sparse(),
        complex_metrics=[ComplexMetricConfig(name=name, params={"pop_ratio": 0.5})],
    )
    evaluator.evaluate(
        model=_TiedScoreRecommender({}, ranking_dataset.info()),
        dataloader=ranking_dataset.get_evaluation_dataloader(),
        strategy="full",
        dataset=ranking_dataset,
    )

    assert set(evaluator.compute_results()[5]) == {f"{name}[Pop50%]"}


@pytest.mark.parametrize("key", metric_registry.list_registered())
def test_a_built_in_metric_reports_under_its_class_name(key: str):
    """Every built-in is registered under its class name, which stays its key."""
    cls = metric_registry.get_class(key)

    assert cls.registered_name() == cls.__name__


def test_a_metric_updated_without_its_blocks_computes_them():
    """Called directly with blocks missing or None, a metric derives them."""
    ground = torch.tensor([[0.0, 4.0, 0.0, 5.0], [3.0, 0.0, 0.0, 0.0]])
    preds = torch.tensor([[0.1, 0.9, 0.2, 0.3], [0.8, 0.1, 0.2, 0.3]])
    users = torch.tensor([0, 1])

    absent = metric_registry.get("Recall", k=1, num_users=2)
    absent.update(preds, user_indices=users, ground=ground)
    none = metric_registry.get("Recall", k=1, num_users=2)
    none.update(
        preds,
        user_indices=users,
        ground=ground,
        binary_relevance=None,
        valid_users=None,
        top_1_binary_relevance=None,
    )

    expected = torch.tensor([0.5, 1.0])
    torch.testing.assert_close(absent.compute()["Recall"], expected)
    torch.testing.assert_close(none.compute()["Recall"], expected)


def test_a_different_class_cannot_take_a_built_in_name():
    """Registering a custom model as 'ease' used to replace EASE without a word."""
    builtin = model_registry.get_class("EASE")

    with pytest.raises(ValueError) as error:

        @model_registry.register("ease")
        class MyEASE(Recommender):  # pylint: disable=unused-variable
            """A model that tries to take EASE's name."""

    message = str(error.value)
    assert "MyEASE" in message and __name__ in message
    assert builtin.__qualname__ in message and builtin.__module__ in message
    assert model_registry.get_class("EASE") is builtin


def test_every_registry_refuses_a_taken_name():
    """The rule is the registry's, so it holds for metrics and parameters too."""
    for registry, base in (
        (metric_registry, TopKMetric),
        (params_registry, RecomModel),
    ):
        name = unique_name("Taken")
        first = type("First", (base,), {})
        registry.register(name)(first)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(name)(type("Second", (base,), {}))
        assert registry.get_class(name) is first


def test_the_same_class_may_be_registered_again():
    """A notebook cell run twice or a Ray worker re-imports the module."""
    name = unique_name("Again")

    def define() -> type:
        """Define the class afresh, as re-running its module does.

        Returns:
            type: A new class object with the same module and qualified name.
        """

        class Again(Recommender):
            """A model whose module is imported twice."""

        return Again

    first, second = define(), define()
    model_registry.register(name)(first)
    model_registry.register(name)(second)

    assert model_registry.get_class(name) is second


def write_module(directory: Path, name: str, body: str) -> Path:
    """Write a module file.

    Args:
        directory (Path): Where to write it.
        name (str): The module name.
        body (str): The source.

    Returns:
        Path: The module's path.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.py"
    path.write_text(body)
    return path


def test_a_second_module_with_a_taken_name_is_refused(tmp_path: Path):
    """Two files called alike used to load the first and drop the second silently."""
    name = unique_name("models_").lower()
    first = write_module(tmp_path / "one", name, "VALUE = 1\n")
    second = write_module(tmp_path / "two", name, "VALUE = 2\n")
    load_custom_modules([str(first)])

    with pytest.raises(ImportError) as error:
        load_custom_modules([str(second)])

    message = str(error.value)
    assert str(first.resolve()) in message and str(second) in message


def test_a_module_named_like_an_imported_one_is_refused(tmp_path: Path):
    """A file called json.py would silently be the standard library's json."""
    path = write_module(tmp_path / "user_code", "json", "VALUE = 1\n")

    with pytest.raises(ImportError, match="json"):
        load_custom_modules([str(path)])

    # Nor is its folder left first on the import path, where every process
    # started afterwards (a spawned DataLoader worker) would take its 'json'
    assert str(path.parent.resolve()) not in sys.path


def test_the_same_module_may_be_loaded_again_or_from_a_copy(tmp_path: Path):
    """Each pipeline loads the modules again, and Ray ships a copy to its workers."""
    name = unique_name("metrics_").lower()
    path = write_module(tmp_path / "one", name, "VALUE = 1\n")
    copy = write_module(tmp_path / "ray_copy", name, "VALUE = 1\n")

    load_custom_modules([str(path)])
    load_custom_modules([str(path), str(copy)])

    assert sys.modules[name].VALUE == 1


def test_a_stash_key_named_like_a_metric_argument_is_refused(ranking_dataset: Dataset):
    """'num_items' in the stash used to replace the catalogue size of every metric."""
    for key in ("num_items", "k"):
        with pytest.raises(ValueError, match=f"'{key}'"):
            Evaluator(
                ["Recall"],
                [5],
                train_set=ranking_dataset.train_set.get_sparse(),
                additional_data={key: 3},
            )


def test_a_stash_key_named_like_a_model_argument_is_refused(ranking_dataset: Dataset):
    """A stash written to directly is checked where models are built."""
    stash = ranking_dataset.get_stash()
    stash["seed"] = 3
    try:
        with pytest.raises(ValueError, match="'seed'"):
            _load_or_build_model("EASE", {"l2": 10.0}, ranking_dataset, 50, 4096)
    finally:
        del stash["seed"]


@pytest.mark.parametrize("key", ["seed", "params", "info", "num_items", "k"])
def test_add_to_stash_refuses_a_reserved_name(key: str, ranking_dataset: Dataset):
    """The name is refused where it is added, before anything is built."""
    with pytest.raises(ValueError, match=f"'{key}'"):
        ranking_dataset.add_to_stash(key, 1)
    assert key not in ranking_dataset.get_stash()


def test_a_stash_entry_of_its_own_reaches_models_and_metrics(
    ranking_dataset: Dataset,
):
    """Every other name still works."""
    ranking_dataset.add_to_stash("item_year", torch.zeros(40))
    try:
        Evaluator(
            ["Recall"],
            [5],
            train_set=ranking_dataset.train_set.get_sparse(),
            additional_data=ranking_dataset.get_stash(),
        )
        model = _load_or_build_model("EASE", {"l2": 10.0}, ranking_dataset, 50, 4096)
        assert model is not None
    finally:
        del ranking_dataset.get_stash()["item_year"]


def test_the_reserved_metric_names_are_those_the_evaluator_passes(
    ranking_dataset: Dataset,
):
    """add_to_stash refuses what the evaluator would refuse, no more, no less."""
    evaluator = Evaluator(
        ["Recall"], [5], train_set=ranking_dataset.train_set.get_sparse()
    )

    assert set(evaluator.common_params) | {"k"} == METRIC_ARGUMENTS


def register_constant_metric(value: Any) -> str:
    """Register a metric whose result is the given value, whatever it ranks.

    Args:
        value (Any): What compute() returns under the metric's name.

    Returns:
        str: The name the metric is registered under.
    """
    name = unique_name("Constant")

    @metric_registry.register(name)
    class Constant(TopKMetric):
        """A metric reporting a fixed value."""

        _REQUIRED_COMPONENTS: Set[MetricBlock] = set()

        def update(self, preds: Tensor, **kwargs: Any):
            """Count nothing.

            Args:
                preds (Tensor): Unused.
                **kwargs (Any): Unused.
            """

        def compute(self) -> Dict[str, Any]:
            """The fixed value.

            Returns:
                Dict[str, Any]: The value under the metric's name.
            """
            return {"Constant": value}

    return name


@pytest.mark.parametrize(
    "value", [torch.tensor(2.5), np.float32(2.5), 2.5], ids=["0-d", "numpy", "float"]
)
def test_a_global_value_is_reported_as_a_float(value: Any, ranking_dataset: Dataset):
    """A 0-d tensor used to pass for a per-user one, and break the writer."""
    result = evaluate_alone(register_constant_metric(value), ranking_dataset)

    assert type(result["Constant"]) is float and result["Constant"] == 2.5


def test_a_value_of_another_shape_is_refused(ranking_dataset: Dataset):
    """A tensor that is not one value per user cannot be averaged or written."""
    name = register_constant_metric(torch.zeros(2, 2))

    with pytest.raises(ValueError, match="'Constant'.*one value per user"):
        evaluate_alone(name, ranking_dataset)


@pytest.mark.parametrize("model_name", ["Pop", "Random"])
def test_a_model_without_hyperparameters_reports_none(
    model_name: str, ranking_dataset: Dataset
):
    """Pop and Random used to report the base class's DATALOADER_TYPE as one."""
    model = model_registry.get(
        model_name,
        params={},
        info=ranking_dataset.info(),
        interactions=ranking_dataset.train_set,
    )

    assert not model.get_params()


def test_class_constants_are_not_hyperparameters():
    """A ClassVar or a name in capitals is a constant of the class."""

    class WithConstants(Recommender):
        """A model with one hyperparameter and two constants."""

        LOADER: str = "pairs"
        flag: ClassVar[bool] = True
        alpha: float

        def predict(self, *args: Any, **kwargs: Any):
            """Unused.

            Args:
                *args (Any): Unused.
                **kwargs (Any): Unused.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError

    model = WithConstants(
        {"alpha": 0.5, "LOADER": "x", "flag": False}, {"n_users": 2, "n_items": 2}
    )

    assert model.get_params() == {"alpha": 0.5}
    assert WithConstants.LOADER == "pairs" and WithConstants.flag is True


def test_a_subclass_of_a_model_keeps_its_hyperparameters(ranking_dataset: Dataset):
    """A model that only overrides behaviour reads its parent's hyperparameters."""
    ease = model_registry.get_class("EASE")

    class MyEASE(ease):  # type: ignore[valid-type,misc]
        """EASE, unchanged."""

    model = MyEASE(
        {"l2": 10.0}, ranking_dataset.info(), interactions=ranking_dataset.train_set
    )

    assert model.get_params() == {"l2": 10.0}


BPR_VALUES: Dict[str, Any] = {
    "embedding_size": [8, 16],
    "reg_weight": 0.1,
    "batch_size": 32,
    "epochs": 1,
    "learning_rate": ["uniform", 0.001, 0.01],
    "optimization": {"strategy": "random"},
}


def register_bpr_variant(params_class: type) -> str:
    """Register a copy of BPR described by the given parameter class.

    Args:
        params_class (type): The parameter class to register for it.

    Returns:
        str: The name both are registered under.
    """
    name = unique_name("VariantBPR")
    model_registry.register(name)(type(name, (model_registry.get_class("BPR"),), {}))
    params_registry.register(name)(params_class)
    return name


def test_a_parameter_class_extending_a_built_in_one_validates_every_field():
    """Adding one field used to lose the types of the inherited ones (KeyError)."""
    builtin = params_registry.get_class("BPR")

    class ExtendedBPRParams(builtin):  # type: ignore[valid-type,misc]
        """BPR's parameters and one more."""

        dropout: FLOAT_FIELD = 0.0

    register_bpr_variant(ExtendedBPRParams)
    params = ExtendedBPRParams(**BPR_VALUES, dropout=[0.1, 0.2])

    assert params.embedding_size == ["choice", 8, 16]
    assert params.learning_rate == ["uniform", 0.001, 0.01]
    assert params.dropout == ["choice", 0.1, 0.2]


def test_a_parameter_class_reusing_a_built_in_one_validates_its_fields():
    """With no annotations of its own, the class reads its parent's."""
    builtin = params_registry.get_class("BPR")

    class SameBPRParams(builtin):  # type: ignore[valid-type,misc]
        """BPR's parameters, unchanged."""

    register_bpr_variant(SameBPRParams)
    params = SameBPRParams(**BPR_VALUES)

    assert params.learning_rate == ["uniform", 0.001, 0.01]
