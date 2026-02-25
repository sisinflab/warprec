###########################
Unpersonalized Recommenders
###########################

.. py:module:: warprec.recommenders.unpersonalized_recommender

.. autosummary::
   :nosignatures:

   pop.Pop
   random.Random
   proxy.ProxyRecommender

The **Unpersonalized Recommenders** module of WarpRec serves as a collection of simple baselines.
These models do not learn from individual user interactions but instead rely on global statistics or random selection.
They are essential for producing reference metric scores against which the performance of actual (personalized) models can be compared.

In the following sections, you will find the list of available unpersonalized models within WarpRec.

.. autoclass:: warprec.recommenders.unpersonalized_recommender.pop.Pop
   :members:
   :undoc-members:
   :show-inheritance:

- Pop:
  Recommends the most popular items overall. This model helps assess whether other recommenders are biased towards popularity.

  .. code-block:: yaml

      models:
          Pop: {}

.. autoclass:: warprec.recommenders.unpersonalized_recommender.random.Random
   :members:
   :undoc-members:
   :show-inheritance:

- Random:
  Recommends items at random. This model defines a lower bound for performance metrics, serving as a sanity check during evaluation.

  .. code-block:: yaml

      models:
          Random: {}

.. autoclass:: warprec.recommenders.unpersonalized_recommender.proxy.ProxyRecommender
   :members:
   :undoc-members:
   :show-inheritance:

- ProxyRecommender:
  A recommender that uses precomputed recommendations from an external source. For detailed documentation, configuration examples, and usage scenarios, see :doc:`/evaluation/proxy_evaluation`.
