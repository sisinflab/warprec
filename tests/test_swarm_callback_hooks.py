"""The Ray Tune hooks of a custom callback reach the driver in a swarm run.

In the train pipeline Ray Tune runs on the driver, so a custom callback's
hooks ('on_trial_result' and the like) collect their state on the very object
the driver later calls 'on_training_complete' on. The swarm runs each model's
sweep inside a Ray task instead, and the hooks used to run on a copy of the
callback shipped to that task: whatever they collected stayed there, and the
driver's callback ended the run having seen nothing.
"""

import pickle
from typing import Any, List, Tuple

from warprec.pipelines.swarm import HookMailbox, TuneHookForwarder, TuneHookRelay
from warprec.utils.callback import WarpRecCallback


class RecordingCallback(WarpRecCallback):
    """A custom callback collecting the reports of the trials."""

    def __init__(self) -> None:
        super().__init__()
        self.reports: List[Tuple[str, float]] = []
        self.completed: List[str] = []

    def on_trial_result(
        self, iteration: int, trials: list, trial: Any, result: dict, **info: Any
    ) -> None:
        """Record a report.

        Args:
            iteration (int): The step of the Tune loop.
            trials (list): Every trial of the sweep.
            trial (Any): The trial that reported.
            result (dict): The report.
            **info (Any): Anything else Ray Tune passes.
        """
        self.reports.append((trial, result["nDCG@10"]))

    def on_trial_complete(
        self, iteration: int, trials: list, trial: Any, **info: Any
    ) -> None:
        """Record a finished trial.

        Args:
            iteration (int): The step of the Tune loop.
            trials (list): Every trial of the sweep.
            trial (Any): The trial that finished.
            **info (Any): Anything else Ray Tune passes.
        """
        self.completed.append(trial)


def test_the_hooks_of_a_model_task_reach_the_driver_callback():
    """What the hooks collect inside a task is on the driver's callback."""
    driver_callback = RecordingCallback()
    relay = TuneHookRelay(driver_callback, mailbox=HookMailbox())

    # What the model task receives, pickled: a forwarder, not the callback
    shipped = pickle.dumps(relay.forwarder())
    assert b"RecordingCallback" not in shipped
    in_task = pickle.loads(shipped)
    assert isinstance(in_task, TuneHookForwarder)

    # The mailbox actor is one object both sides talk to
    in_task.mailbox = relay.mailbox

    # Ray Tune calls the hooks with keyword arguments, on the task's copy
    in_task.on_step_begin(iteration=0, trials=[])
    in_task.on_trial_result(
        iteration=1, trials=["a"], trial="a", result={"nDCG@10": 0.5}
    )
    in_task.on_trial_result(
        iteration=2, trials=["a"], trial="a", result={"nDCG@10": 0.7}
    )
    in_task.on_trial_complete(iteration=3, trials=["a"], trial="a")
    in_task.on_experiment_end(trials=["a"])
    relay.close()

    assert driver_callback.reports == [("a", 0.5), ("a", 0.7)]
    assert driver_callback.completed == ["a"]


def test_a_callback_without_tune_hooks_needs_no_mailbox():
    """A callback implementing no Ray Tune hook costs the run nothing."""
    relay = TuneHookRelay(WarpRecCallback())

    assert relay.hooks == ()
    assert relay.mailbox is None
    relay.forwarder().on_trial_result(iteration=1, trials=[], trial="a", result={})
    relay.close()
