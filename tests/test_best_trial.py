"""The best model of a search is built from a trial that trained.

A failed trial reports an infinite score instead of raising, so Ray Tune counts
it as a success; picking it left the best model without a checkpoint to load.
"""

from types import SimpleNamespace

import pandas as pd

from warprec.recommenders.trainer import Trainer


def trial(scores, checkpoints=None, error=None):
    """A finished trial, as the search's result grid holds it."""
    history = pd.DataFrame({"nDCG@10": scores})
    if checkpoints is not None:
        history["checkpoint_path"] = checkpoints
    metrics = {"trial_error": error} if error else {"nDCG@10": scores[-1]}
    return SimpleNamespace(metrics=metrics, error=None, metrics_dataframe=history)


def best(results, needs_checkpoint=True):
    trainer = Trainer.__new__(Trainer)
    trainer._trial_history = lambda result: result.metrics_dataframe
    return trainer._best_usable_trial(results, "nDCG@10", "max", needs_checkpoint)


def test_a_search_whose_trials_all_failed_has_no_best_trial():
    results = [
        trial([-float("inf")], error="RuntimeError: out of memory"),
        trial([-float("inf")], error="ValueError: bad input"),
    ]

    assert best(results) is None
    assert Trainer._first_trial_error(results) == "RuntimeError: out of memory"


def test_the_best_trial_is_the_best_one_that_can_be_rebuilt():
    diverged = trial([float("nan"), 0.9], checkpoints=["a", None])
    trained = trial([0.2, 0.3], checkpoints=["b", "c"])

    result, row = best([diverged, trained])
    assert result is trained
    assert row["checkpoint_path"] == "c"

    # A closed-form model is refitted, so a report without a checkpoint is enough
    result, _ = best([diverged, trained], needs_checkpoint=False)
    assert result is diverged
