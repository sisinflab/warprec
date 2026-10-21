"""A search never launches a trial for a combination its model refuses."""

from typing import Dict, List, Optional

from ray.tune.search import Searcher

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.recommenders.trainer.search_algorithm_wrapper import ValidSuggestions

from conftest import build_params


class ScriptedSearcher(Searcher):
    """Suggests the given combinations in order, and records what it is told."""

    def __init__(self, combinations: List[Dict]):
        super().__init__(metric="nDCG@10", mode="max")
        self.combinations = combinations
        self.suggested: List[str] = []
        self.completed: Dict[str, bool] = {}

    def suggest(self, trial_id: str) -> Optional[Dict]:
        self.suggested.append(trial_id)
        return self.combinations.pop(0)

    def on_trial_complete(self, trial_id, result=None, error=False):
        self.completed[trial_id] = error


def test_a_refused_combination_is_reported_failed_and_replaced():
    base = build_params("SASRec")
    invalid = {**base, "embedding_size": 10, "n_heads": 3}
    valid = {**base, "embedding_size": 12, "n_heads": 3}
    inner = ScriptedSearcher([invalid, valid])
    searcher = ValidSuggestions(inner, "SASRec")

    assert searcher.suggest("t1") == valid
    # The refused one taught the searcher, and the trial runs under a new id
    assert inner.completed == {"t1": True}
    assert inner.suggested == ["t1", "t1_retry1"]

    searcher.on_trial_complete("t1", result={"nDCG@10": 0.1})
    assert inner.completed["t1_retry1"] is False
