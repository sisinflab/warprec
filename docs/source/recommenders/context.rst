######################
Context-Aware Recommenders
######################

The **Context Recommenders** module of WarpRec contains models designed to incorporate contextual information (e.g., time, location, device, session data) into the recommendation process.
Unlike collaborative filtering recommenders that rely solely on User-Item interactions, these models leverage additional dimensions to improve prediction accuracy in specific situations.

.. important::
  Context-Aware recommenders assume a Leave-One-Out strategy has been used to create the test/validation set. In any other case the framework will not raise an error but it will yield incorrect results.

In the following sections, you will find the list of available context-aware models within WarpRec, together with their respective parameters.

===================
Factorization-Based
===================

Factorization-Based context models extend standard matrix factorization techniques to handle multidimensional data (tensors) or feature vectors that include contextual variables.

- AFM (Attentional Factorization Machines):
    An extension of Factorization Machines that introduces an attention network to learn the importance of each feature interaction. Unlike standard FM, where all interactions are weighted equally, AFM focuses more on informative interactions and less on useless ones. **This model requires contextual information to function properly.**

    For further details, please refer to the `paper <https://arxiv.org/abs/1708.04617>`_.

.. code-block:: yaml

    models:
      AFM:
        embedding_size: 64
        attention_size: 64
        dropout: 0.3
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
        neg_samples: 2

- DCN (Deep & Cross Network):
    A model designed to learn explicit and bounded-degree feature interactions effectively. It consists of a Cross Network, which applies explicit feature crossing at each layer, and a Deep Network that captures implicit high-order interactions. **This model requires contextual information to function properly.**

    For further details, please refer to the `paper <https://arxiv.org/abs/1708.05123>`_.

.. code-block:: yaml

    models:
      DCN:
        embedding_size: 64
        mlp_hidden_size: [64, 32]
        cross_layer_num: 2
        dropout: 0.3
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
        neg_samples: 2

- DCNv2 (Deep & Cross Network V2):
    An improved version of DCN that introduces a Mixture-of-Experts (MoE) architecture in the Cross Network to learn feature interactions more effectively. It supports both "parallel" and "stacked" structures and utilizes low-rank techniques to reduce computational complexity while maintaining expressiveness. **This model requires contextual information to function properly.**

    For further details, please refer to the `paper <https://arxiv.org/abs/2008.13535>`_.

.. code-block:: yaml

    models:
      DCNv2:
        embedding_size: 64
        mlp_hidden_size: [64, 32]
        cross_layer_num: 2
        dropout: 0.3
        model_structure: parallel
        use_mixed: True
        expert_num: 2
        low_rank: 32
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
        neg_samples: 2

- DeepFM (Deep Factorization Machines):
    A neural architecture that integrates a Factorization Machine component to model low-order feature interactions and a Deep Neural Network to capture high-order interactions. Both components share the same input embedding layer and operate in parallel to predict the final score. **This model requires contextual information to function properly.**

    For further details, please refer to the `paper <https://arxiv.org/abs/1703.04247>`_.

.. code-block:: yaml

    models:
      DeepFM:
        embedding_size: 64
        mlp_hidden_size: [64, 32]
        dropout: 0.3
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
        neg_samples: 2

- FM (Factorization Machines):
    A general predictor that models all nested interactions between input variables using factorized parameters. It explicitly captures second-order interactions between users, items, and contextual features, making it effective for sparse datasets with categorical variables. **This model requires contextual information to function properly.**

    For further details, please refer to the `paper <https://ieeexplore.ieee.org/document/5694074>`_.

.. code-block:: yaml

    models:
      FM:
        embedding_size: 64
        reg_weight: 0.001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
        neg_samples: 2

- NFM (Neural Factorization Machines):
    An extension of Factorization Machines that replaces the standard second-order interaction term with a "Bi-Interaction Pooling" layer followed by a Multi-Layer Perceptron (MLP). This architecture allows the model to capture complex, non-linear, and higher-order dependencies between features. **This model requires contextual information to function properly.**

    For further details, please refer to the `paper <https://arxiv.org/abs/1708.05027>`_.

.. code-block:: yaml

    models:
      NFM:
        embedding_size: 64
        mlp_hidden_size: [64, 32]
        dropout: 0.3
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
        neg_samples: 2

- WideAndDeep (Wide & Deep Learning):
    A framework that jointly trains a wide linear model for memorization and a deep neural network for generalization. It combines a generalized linear model (Wide) with a feed-forward neural network (Deep) to capture both low-order and high-order feature interactions. **This model requires contextual information to function properly.**

    For further details, please refer to the `paper <https://arxiv.org/abs/1606.07792>`_.

.. code-block:: yaml

    models:
      WideAndDeep:
        embedding_size: 64
        mlp_hidden_size: [64, 32]
        dropout: 0.3
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
        neg_samples: 2

- xDeepFM (eXtreme Deep Factorization Machine):
    A model that generates explicit high-order feature interactions at the vector-wise level using a Compressed Interaction Network (CIN). It combines the CIN with a linear part and a plain DNN to learn explicit and implicit interactions simultaneously. **This model requires contextual information to function properly.**

    For further details, please refer to the `paper <https://arxiv.org/abs/1803.05170>`_.

.. code-block:: yaml

    models:
      xDeepFM:
        embedding_size: 64
        mlp_hidden_size: [64, 32]
        cin_layer_size: [64, 64]
        dropout: 0.3
        direct: False
        reg_weight: 0.001
        weight_decay: 0.0001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
        neg_samples: 2

===============================
Summary of Available Context-Aware Models
===============================

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Category
     - Model
     - Description
   * - Factorization-Based
     - AFM
     - Attentional Factorization Machine using an attention network to weigh feature interactions.
   * -
     - DCN
     - Deep & Cross Network using a Cross Network for explicit bounded-degree feature interactions.
   * -
     - DCNv2
     - Improved DCN with Mixture-of-Experts (MoE) and low-rank cross network.
   * -
     - DeepFM
     - Parallel combination of FM and DNN to capture both low- and high-order feature interactions.
   * -
     - FM
     - Factorization Machine modeling second-order interactions between user, item, and context.
   * -
     - NFM
     - Neural Factorization Machine using MLP to model higher-order interactions between features.
   * -
     - WideAndDeep
     - Joint training of a linear model (Wide) and a Deep Neural Network (Deep) for memorization and generalization.
   * -
     - xDeepFM
     - eXtreme DeepFM using Compressed Interaction Network (CIN) for vector-wise explicit interactions.
