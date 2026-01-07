.. _configuration:

#################
Configuration
#################

This section provides a comprehensive overview of WarpRec's configuration system.
WarpRec configurations allow users to customize data loading, preprocessing, model training, evaluation, and general experiment settings.
Proper configuration ensures reproducibility, optimal performance, and easy management of multiple experiments.

Overview
--------

WarpRec's configuration is divided into several main sections, each one has its dedicated description with all the keywords available and their behavior.

.. toctree::
   :maxdepth: 1

   reader
   writer
   filtering
   splitter
   dashboard
   models
   evaluation
   general

Pipeline Configurations
-----------------------

WarpRec supports multiple pipelines that leverage the configuration system differently depending on the purpose of the experiment:

1. **Training Pipeline**: Executes full workflow including hyperparameter optimization (HPO), model training, evaluation, and result saving.
2. **Design Pipeline**: Focuses on testing and evaluating models without HPO. This pipeline is ideal for rapid prototyping and design experiments.

A single configuration file can be used across multiple pipelines; WarpRec ensures that workflows remain interchangeable, with some sections being ignored or interpreted differently depending on the pipeline.

Training Pipeline
~~~~~~~~~~~~~~~~~

The **training pipeline** is the core of the framework. It executes a complete experiment with all the provided models, performs **hyperparameter optimization (HPO)**, and generates reports on model performance. The workflow requires the following sections:

- reader
- writer
- splitter
- models
- evaluation

Optionally, you can also provide:

- filtering
- dashboard
- general

A minimal training configuration example:

.. code-block:: yaml

    reader:
        loading_strategy: dataset
        data_type: transaction
        reading_method: local
        local_path: path/to/your/dataset.csv
        rating_type: implicit
        sep: ','
    writer:
        dataset_name: MyDataset
        writing_method: local
        local_experiment_path: experiment/test/
    splitter:
        test_splitting:
            strategy: temporal_holdout
            ratio: 0.1
    models:
        ItemKNN:
            k: 10
            similarity: cosine
    evaluation:
        top_k: [10, 20, 50]
        metrics: [nDCG, Precision, Recall, HitRate]

Run the training pipeline with:

.. code-block:: bash

    python -m warprec.run --config path/to/the/config.yml --pipeline train

Design Pipeline
~~~~~~~~~~~~~~~

The **design pipeline** is used for rapid evaluation and testing of models. It does **not execute HPO**, and requires models to have **single-value hyperparameters**. The workflow requires the following sections:

- reader
- splitter
- models
- evaluation

Optionally, you can also provide:

- filtering
- general

An example use of the design pipeline is testing a custom implementation. Here is a configuration example:

.. code-block:: yaml

    reader:
        loading_strategy: dataset
        data_type: transaction
        reading_method: local
        local_path: tests/test_dataset/movielens.csv
        rating_type: explicit
        sep: ','
    splitter:
        test_splitting:
            strategy: temporal_holdout
            ratio: 0.1
    models:
        # Models in the design pipeline must have single-value hyperparameters
        CustomBPR:
            embedding_size: 32
            weight_decay: 0.
            batch_size: 1024
            epochs: 10
            learning_rate: 0.0001
    evaluation:
        top_k: [10, 20, 50]
        batch_size: 1024
        metrics: [nDCG, Precision, Recall, HitRate]
    general:
        custom_models: [my_custom_model.py]

Run the design pipeline with:

.. code-block:: bash

    python -m warprec.run --config path/to/the/config.yml --pipeline design
