#########################
Filtering Configuration
#########################

The **Filtering Configuration** module defines the preprocessing strategies applied to the dataset
*before* the splitting phase.
Filtering is a fundamental step when the dataset contains redundant or low-quality interactions,
or when its size exceeds the available computational resources.

By applying filters, WarpRec ensures that the resulting dataset is both computationally manageable
and more representative of the target recommendation task.

.. note::
   Filtering strategies are applied **sequentially** in the order they appear in the configuration file.
   This execution order may significantly affect the resulting dataset, so it must be carefully designed.

.. warning::
    Some strategies (e.g., ``MinRating`` and ``UserAverage``) are **incompatible** with implicit feedback datasets.
    Attempting to apply them in such contexts will raise an error.

    Trying to filter an implicit dataset with these strategies may result in unexpected behaviors.

-----------------------------
General Configuration Format
-----------------------------

Filtering strategies must be declared under the ``filtering`` section of the configuration file.
Each strategy is specified by name, followed by its parameters (if required).

.. code-block:: yaml

   filtering:
       strategy_name_1:
           arg_name_1: value_1
       strategy_name_2:
           arg_name_1: value_1
           arg_name_2: value_2
   ...

.. important::
   - Strategies are executed **top to bottom** in the exact order they are listed.
   - Incorrect strategy names or invalid parameter keys will cause WarpRec to raise an error.

-----------------------------
Supported Filtering Strategies
-----------------------------

WarpRec currently supports the following filtering strategies:

**1. MinRating**

Removes all interactions where the rating value is strictly below the specified threshold.
Not compatible with implicit feedback datasets.

.. code-block:: yaml

   filtering:
       MinRating:
           min_rating: 3.0

**2. UserAverage**

Removes all interactions for which the rating is below the corresponding user’s average rating.
Not applicable to implicit feedback scenarios.

.. code-block:: yaml

   filtering:
       UserAverage: {}   # No parameters required

**3. UserMin**

Removes all interactions involving users with fewer interactions than the given threshold.

.. code-block:: yaml

   filtering:
       UserMin:
           min_interactions: 5

**4. UserMax**

Removes all interactions involving users with more interactions than the given threshold.
This is particularly useful for **cold-start user analysis**.

.. code-block:: yaml

   filtering:
       UserMax:
           max_interactions: 2

**5. ItemMin**

Removes all interactions involving items with fewer interactions than the given threshold.

.. code-block:: yaml

   filtering:
       ItemMin:
           min_interactions: 5

**6. ItemMax**

Removes all interactions involving items with more interactions than the given threshold.
Useful for analyzing **cold-start item scenarios**.

.. code-block:: yaml

   filtering:
       ItemMax:
           max_interactions: 2

**7. IterativeKCore**

Applies ``UserMin`` and ``ItemMin`` iteratively until no further interactions can be removed
(i.e., until a stable state is reached).

.. code-block:: yaml

   filtering:
       IterativeKCore:
           min_interactions: 5

**8. NRoundsKCore**

Applies ``UserMin`` and ``ItemMin`` for a fixed number of iterations.
This is a simplified variant of ``IterativeKCore`` that does not require full convergence.

.. code-block:: yaml

   filtering:
       NRoundsKCore:
           rounds: 3
           min_interactions: 5

.. tip::
   ``IterativeKCore`` ensures dataset stability, but may be computationally expensive.
   ``NRoundsKCore`` is recommended when deterministic runtime is preferred over convergence.

-----------------------------
Example Filtering Pipeline
-----------------------------

The following example demonstrates a pipeline where:

1. All ratings below 3.0 are removed.
2. Users with fewer than 10 interactions are filtered out.

.. code-block:: yaml

   filtering:
       MinRating:
           min_rating: 3.0
       UserMin:
           min_interactions: 10
