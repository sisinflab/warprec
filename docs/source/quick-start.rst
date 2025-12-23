.. _quick-start:

#################
Quick Start
#################

WarpRec provides a modular and extensible environment designed to support both advanced practitioners and newcomers.
This quick-start guide demonstrates the minimal steps required to execute a first end-to-end experiment, from dataset preparation to model evaluation.

Dataset Preparation
-------------------

The first prerequisite is to structure the dataset in a **WarpRec-compatible input format**.
By default, WarpRec expects a **tab-separated values (.tsv)** file, where:

- the **first row** specifies the column headers,
- each **subsequent row** encodes a single user–item interaction event.

At minimum, the dataset must include the following fields:

- **user_id**: unique identifier of the user.
- **item_id**: unique identifier of the item.

In this minimal configuration, each row corresponds to a **binary implicit interaction**, all assigned equal weight.

Optionally, the dataset may also include:

- **rating**: a real-valued score representing the strength or relevance of the interaction.
  When present, WarpRec leverages this value as a **per-event weight** during training and evaluation.

- **timestamp**: a temporal marker (e.g., Unix epoch) specifying when the interaction occurred.
  This field is **mandatory** for temporal-based splitting strategies (e.g., holdout) and is also exploited by sequential or time-aware models.

.. csv-table:: Example Dataset (with rating and timestamp)
   :header: "user_id", "item_id", "rating", "timestamp"
   :widths: 15, 15, 10, 30
   :align: left

   1, 1193, 5, 978300760
   1, 661, 3, 978302109
   2, 1357, 5, 978298709
   3, 2393, 4, 978297054

The dataset file can be stored in any location. For simplicity, in this example we assume it is placed in the root directory of your WarpRec clone.

.. note::

    WarpRec provides multiple I/O backends through its **Reader** and **Writer** modules (e.g., local filesystem, remote storage, databases).
    In this guide, we demonstrate the simplest **local I/O workflow**.

    For a complete reference, see ... .

Experiment Configuration
------------------------

Once the dataset is available, WarpRec offers two modes of interaction:

- **Configuration files**: centralize all experimental settings (data loading, preprocessing, models, and evaluation) into a single YAML file, enabling **reproducibility** and **version control**.
- **Python scripting**: directly use WarpRec APIs to build custom pipelines and integrate the framework into external workflows.

In this example, we provide a **predefined configuration file** located at *config/quick_start.yml*.
This file defines a complete pipeline: dataset loading, splitting, model training, and evaluation.

Below is the configuration:

.. code-block:: yaml

    reader:
        loading_strategy: dataset
        data_type: transaction
        reading_method: local
        local_path: path/to/your/dataset.tsv
        rating_type: explicit
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
            k: 100
            similarity: cosine
    evaluation:
        top_k: [5, 10]
        metrics: [Precision, Recall, nDCG]

This configuration executes the following workflow:

1. **Reader**: loads the dataset from the local path, interpreting it as explicit feedback.
2. **Splitter**: partitions the dataset into training and test sets using a temporal holdout strategy (90% train, 10% test).
3. **Model**: trains an **ItemKNN** recommender with `k=100` neighbors and cosine similarity.
4. **Evaluator**: computes ranking metrics at different cutoff values.
5. **Writer**: stores all experiment artifacts (logs, splits, and results) in the specified output directory.

This example provides a **minimal yet complete workflow** covering the essential functionalities of WarpRec.
From here, you can extend the configuration by:

- Integrating multiple models for comparative evaluation.
- Using advanced data splitting strategies (e.g., k-fold, user-based splits).
- Incorporating side information (e.g., item metadata, user demographics).
- Enabling distributed training and large-scale evaluation.

For a detailed explanation of the configuration system, see :ref:`Configuration Files <configuration>`.

If you wish to start a training experiment, a Ray cluster must be initialized. You can do so using the following command:

.. code-block:: bash

    ray start --head

If you wish to customize available resources, you can start your head node manually using the following command:

.. code-block:: bash

    ray start --head --num-cpus <NUM_CPUS> --num-gpus <NUM_GPUS>

Also, in scenarios where multiple users share the same machine, you might want to specify which GPU devices to use by setting the ``CUDA_VISIBLE_DEVICES`` environment variable. For example, to use only GPU devices 0 and 2, you can run the following command:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0,2 ray start --head --num-cpus=<NUM_CPUS> --num-gpus=2

Finally you can run your experiment using the following command:

.. code-block:: bash

    python warprec/train.py -c config/quick_start.yml
