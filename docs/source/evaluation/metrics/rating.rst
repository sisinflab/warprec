########
Rating
########

.. py:module:: warprec.evaluation.metrics.rating

Summary
~~~~~~~

.. autosummary::

    mae.MAE
    mse.MSE
    rmse.RMSE

**Rating metrics** are specifically designed for recommender systems that predict **explicit user ratings** (e.g., 1-5 stars). These metrics quantify the **accuracy of these numerical predictions** by comparing them to the actual user ratings.

MAE (Mean Absolute Error)
=========================

.. module:: warprec.evaluation.metrics.rating.mae
.. autoclass:: MAE
    :members:
    :undoc-members:
    :show-inheritance:

The **average absolute difference** between predicted and actual ratings.

.. math::

    \text{MAE} = \frac{1}{N} \sum_{(u,i)} |\hat{r}_{ui} - r_{ui}|

For further details, please refer to this `link <https://en.wikipedia.org/wiki/Mean_absolute_error>`_.

.. code-block:: yaml

    evaluation:
        metrics: [MAE]

MSE (Mean Squared Error)
========================

.. module:: warprec.evaluation.metrics.rating.mse
.. autoclass:: MSE
    :members:
    :undoc-members:
    :show-inheritance:

The **average of the squared differences** between predicted and actual ratings.

.. math::

    \text{MSE} = \frac{1}{N} \sum_{(u,i)} (\hat{r}_{ui} - r_{ui})^2

For further details, please refer to this `link <https://en.wikipedia.org/wiki/Mean_squared_error>`_.

.. code-block:: yaml

    evaluation:
        metrics: [MSE]

RMSE (Root Mean Squared Error)
==============================

.. module:: warprec.evaluation.metrics.rating.rmse
.. autoclass:: RMSE
    :members:
    :undoc-members:
    :show-inheritance:

The **square root of the MSE**, providing an error measure in the same units as the ratings.

.. math::

    \text{RMSE} = \sqrt{\frac{1}{N} \sum_{(u,i)} (\hat{r}_{ui} - r_{ui})^2}

For further details, please refer to this `link <https://en.wikipedia.org/wiki/Root_mean_square_deviation>`_.

.. code-block:: yaml

    evaluation:
        metrics: [RMSE]
