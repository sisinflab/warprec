########
Rating
########

**Rating metrics** are specifically designed for recommender systems that predict **explicit user ratings** (e.g., 1-5 stars). These metrics quantify the **accuracy of these numerical predictions** by comparing them to the actual user ratings.

MAE (Mean Absolute Error)
=========================

The **average absolute difference** between predicted and actual ratings.

For further details, please refer to this `link <https://en.wikipedia.org/wiki/Mean_absolute_error>`_.

.. code-block:: yaml

    evaluation:
        metrics: [MAE]

MSE (Mean Squared Error)
========================

The **average of the squared differences** between predicted and actual ratings.

For further details, please refer to this `link <https://en.wikipedia.org/wiki/Mean_squared_error>`_.

.. code-block:: yaml

    evaluation:
        metrics: [MSE]

RMSE (Root Mean Squared Error)
==============================

The **square root of the MSE**, providing an error measure in the same units as the ratings.

For further details, please refer to this `link <https://en.wikipedia.org/wiki/Root_mean_square_deviation>`_.

.. code-block:: yaml

    evaluation:
        metrics: [RMSE]
