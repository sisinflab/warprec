######################
Hybrid Recommenders
######################

.. py:module:: warprec.recommenders.hybrid_recommender

.. autosummary::
   :nosignatures:

   addease.AddEASE
   cease.CEASE
   attributeitemknn.AttributeItemKNN
   attributeuserknn.AttributeUserKNN

The **Hybrid Recommenders** module of WarpRec contains models that combine collaborative filtering signals with side information (content).
By leveraging both user-item interactions and item/user attributes, these models aim to overcome limitations such as data sparsity and the cold-start problem, providing more robust recommendations than pure collaborative or pure content-based approaches.

In the following sections, you will find the list of available hybrid models within WarpRec, together with their respective parameters.

===================
Hybrid Autoencoders
===================

Hybrid Autoencoders extend standard autoencoder architectures by injecting side information into the learning process. This allows the model to reconstruct interactions not just based on historical data, but also influenced by item or user features.

.. autoclass:: warprec.recommenders.hybrid_recommender.addease.AddEASE
   :members:
   :undoc-members:
   :show-inheritance:

- AddEASE:
  An extension of the EASE model using side information. It solves two linear problems, increasing the complexity of the underlying task. **This model requires side information to function properly**.

  For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3383313.3418480>`_.

  .. code-block:: yaml

      models:
        AddEASE:
          l2: 10
          alpha: 0.1

.. autoclass:: warprec.recommenders.hybrid_recommender.cease.CEASE
   :members:
   :undoc-members:
   :show-inheritance:

- CEASE:
  An extension of the EASE model using side information. Extends the EASE problem without adding more complexity. **This model requires side information to function properly**.

  For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3383313.3418480>`_.

  .. code-block:: yaml

      models:
        CEASE:
          l2: 10
          alpha: 0.1

===========
Hybrid KNN
===========

Hybrid KNN models enhance standard Nearest Neighbor approaches by incorporating content data into the similarity computation. Instead of relying solely on interaction overlap, these models use attributes to find similar items or users.

.. autoclass:: warprec.recommenders.hybrid_recommender.attributeitemknn.AttributeItemKNN
   :members:
   :undoc-members:
   :show-inheritance:

- AttributeItemKNN:
  An item-based KNN variant that incorporates item content to compute similarities. **This model requires side information to function properly**.

  For further details, please refer to the `paper <https://www.researchgate.net/publication/221141162_MyMediaLite_A_free_recommender_system_library>`_.

  .. code-block:: yaml

      models:
        AttributeItemKNN:
          k: 10
          similarity: cosine

.. autoclass:: warprec.recommenders.hybrid_recommender.attributeuserknn.AttributeUserKNN
   :members:
   :undoc-members:
   :show-inheritance:

- AttributeUserKNN:
  A user-based KNN model that uses content-based profiles (e.g., TF-IDF) to define user similarity. **This model requires side information to function properly**.

  For further details, please refer to the `paper <https://www.researchgate.net/publication/221141162_MyMediaLite_A_free_recommender_system_library>`_.

  .. code-block:: yaml

      models:
        AttributeUserKNN:
          k: 10
          similarity: cosine
          user_profile: tfidf

==================================
Summary of Available Hybrid Models
==================================

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Category
     - Model
     - Description
   * - Hybrid Autoencoders
     - AddEASE
     - EASE extension solving two linear problems to incorporate side info.
   * -
     - CEASE
     - EASE extension incorporating side info without added complexity.
   * - Hybrid KNN
     - AttributeItemKNN
     - Item-based KNN using content features for similarity.
   * -
     - AttributeUserKNN
     - User-based KNN using content-derived user profiles.
