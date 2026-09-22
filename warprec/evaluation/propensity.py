from typing import Optional

import numpy as np
import torch
from torch import Tensor

UNIFORM = "uniform"
POPULARITY = "popularity"


def build_propensity(
    item_counts: np.ndarray,
    estimator: str = POPULARITY,
    power: float = 0.5,
    clip: float = 0.1,
) -> Optional[Tensor]:
    """The probability that each item was observed, used to debias a metric.

    An offline test set is missing not at random: an item is in it partly because
    it was shown. The debiased estimators divide an item's contribution by this
    probability, so that a hit on an item nobody was ever shown counts for more
    than a hit on one everybody saw.

    The popularity estimator follows Saito et al., *Unbiased Recommender Learning
    from Missing-Not-At-Random Implicit Feedback* (WSDM 2020): the propensity is
    read off the interaction counts, ``p_i = (n_i / max_j n_j) ** power``, and
    floored at ``clip``. The estimators that consume it are defined in Yang et
    al., *Unbiased Offline Recommender Evaluation for Missing-Not-At-Random
    Implicit Feedback* (RecSys 2018).

    The floor is not cosmetic: the estimator divides by ``p``, so a single item
    with a near-zero propensity would otherwise dominate the whole result, and an
    item with no interactions at all would divide by zero.

    Args:
        item_counts (np.ndarray): How many training interactions each item has.
        estimator (str): 'popularity' to read the propensity off the counts, or
            'uniform' to apply no correction at all.
        power (float): The exponent applied to the normalised counts. Lower values
            flatten the correction, 0 removes it.
        clip (float): The smallest propensity any item may carry.

    Returns:
        Optional[Tensor]: One propensity per item, or None when the estimator is
            'uniform' and there is nothing to correct.

    Raises:
        ValueError: If the estimator is not one WarpRec knows.
    """
    if estimator == UNIFORM:
        return None

    if estimator != POPULARITY:
        raise ValueError(
            f"Propensity estimator '{estimator}' is not supported. "
            f"Use '{POPULARITY}' or '{UNIFORM}'."
        )

    counts = torch.as_tensor(np.asarray(item_counts), dtype=torch.float)
    largest = float(counts.max()) if counts.numel() else 0.0

    # A catalogue nobody has touched carries no popularity signal, so there is
    # no correction to make and every item is treated as equally observable.
    if largest <= 0:
        return torch.ones_like(counts)

    return (counts / largest).pow(power).clamp(min=clip, max=1.0)
