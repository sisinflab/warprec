"""A sequential model must say whether a session alone is enough to score it.

Serving answers anonymous sessions - a history with no known user - only for
models that ignore the user index. The flag is checked against what each
model's predict actually reads, so a new model cannot silently get it wrong.
"""

import inspect
import textwrap

import pytest

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.recommenders.base_recommender import SequentialRecommenderUtils
from warprec.utils.registry import model_registry

from conftest import make_model

SEQUENTIAL = sorted(
    name
    for name in model_registry.list_registered()
    if issubclass(model_registry.get_class(name), SequentialRecommenderUtils)
)


def test_sequential_models_are_discovered():
    """Guards against the discovery above silently matching nothing."""
    assert "SASREC" in SEQUENTIAL and "CASER" in SEQUENTIAL


@pytest.mark.parametrize("model_name", SEQUENTIAL)
def test_the_flag_matches_what_predict_reads(model_name: str):
    """needs_user is True exactly when predict's body reads user_indices."""
    model_class = model_registry.get_class(model_name)
    source = textwrap.dedent(inspect.getsource(model_class.predict))
    # The docstring names every argument; only the code after it counts.
    body = source.split('"""')[-1]
    assert model_class.needs_user == ("user_indices" in body), model_name


@pytest.mark.parametrize("model_name", ["CASER", "SASREC"])
def test_the_flag_is_not_a_hyperparameter(model_name: str, dataset):
    """get_params reads annotations; the flag must stay out of the run's name."""
    assert "needs_user" not in make_model(model_name, dataset).get_params()
