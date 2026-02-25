.. _green_ai:

######################
Green AI & CodeCarbon
######################

WarpRec is the first Recommender Systems framework to enforce ecological accountability by natively integrating `CodeCarbon <https://mlco2.github.io/codecarbon/>`_ for real-time energy tracking and carbon-emission profiling at the trial level.

.. contents:: On this page
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

-----

Why Green AI Matters
====================

The field has pivoted from accuracy-maximizing "Red AI" to "Green AI," elevating energy efficiency and carbon transparency to first-class metrics. This is particularly critical in Recommender Systems, where the combination of billion-parameter embedding tables and exhaustive hyperparameter optimization grids leads to extreme energy consumption. Recent research has shown that marginal gains in performance often require exponential increases in carbon emissions.

Despite this critical importance, **existing RS frameworks remain fundamentally energy-blind**. WarpRec addresses this gap by providing native mechanisms to quantify carbon emissions and power consumption within the experimental pipeline.

-----

Configuration
=============

CodeCarbon tracking is controlled via the ``dashboard`` section of the configuration file.

.. code-block:: yaml

    dashboard:
        codecarbon:
            enabled: true
            save_to_file: true
            output_dir: ./carbon_reports/
            tracking_mode: machine

.. list-table:: CodeCarbon Configuration Keys
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Default
     - Description
   * - ``enabled``
     - ``false``
     - Enable or disable CodeCarbon tracking.
   * - ``save_to_api``
     - ``false``
     - Upload emissions data to the CodeCarbon cloud API.
   * - ``save_to_file``
     - ``false``
     - Save emissions data to a local JSON file.
   * - ``output_dir``
     - ``./``
     - Directory for emission output files.
   * - ``tracking_mode``
     - ``machine``
     - Tracking granularity (``machine`` tracks the full system).

.. note::

    CodeCarbon requires the ``dashboard`` dependency group. Install it with:

    .. code-block:: bash

        poetry install --only main dashboard

-----

How It Works
============

WarpRec integrates CodeCarbon at the **Ray Tune trial level** via a dedicated ``CodeCarbonCallback`` (see :ref:`pipeline-train` for the full Training Pipeline documentation):

1. When a trial starts, an ``EmissionsTracker`` is initialized and begins monitoring CPU power, GPU power, and RAM energy consumption.
2. Throughout training, the tracker records cumulative energy draw and carbon emissions based on the local electricity grid's carbon intensity.
3. When the trial completes (or fails), the tracker stops and persists the recorded data.

Because tracking occurs per trial, you get granular visibility into the environmental cost of each hyperparameter configuration, not just the aggregate experiment.

-----

Output Metrics
==============

The following metrics are captured per trial:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Metric
     - Unit
     - Description
   * - Emissions
     - kg CO\ :sub:`2`\ eq
     - Total carbon dioxide equivalent emitted.
   * - Emissions Rate
     - kg CO\ :sub:`2`\ eq/h
     - Carbon emission rate over time.
   * - CPU Power
     - W
     - Average CPU power draw.
   * - GPU Power
     - W
     - Average GPU power draw.
   * - CPU Energy
     - kWh
     - Total CPU energy consumed.
   * - GPU Energy
     - kWh
     - Total GPU energy consumed.
   * - RAM Energy
     - kWh
     - Total RAM energy consumed.
   * - Energy Consumed
     - kWh
     - Total energy consumed (CPU + GPU + RAM).
   * - Peak RAM Usage
     - GB
     - Maximum RAM usage during the trial.

-----

Benchmark: Green AI Profiling
=============================

The following table shows the environmental impact of training five representative algorithms on the NetflixPrize-100M dataset (100M interactions), profiled via CodeCarbon on NVIDIA A100 GPUs.

.. list-table:: Green AI Profiling on NetflixPrize-100M
   :header-rows: 1
   :widths: 22 14 14 14 14 14

   * - Metric
     - ItemKNN
     - EASE\ :sup:`R`
     - NeuMF
     - LightGCN
     - SASRec
   * - Emissions (kg CO\ :sub:`2`\ eq)
     - **0.0002**
     - 0.0005
     - 0.0004
     - 0.0095
     - 0.0012
   * - Emissions Rate (kg CO\ :sub:`2`\ eq/h)
     - **2.96e-7**
     - 3.09e-7
     - 4.97e-7
     - 4.00e-7
     - 5.33e-7
   * - CPU Power (W)
     - **137.8**
     - 155.3
     - 219.4
     - 154.1
     - 152.7
   * - GPU Power (W)
     - 75.9
     - **71.8**
     - 177.3
     - 157.4
     - 278.6
   * - CPU Energy (kWh)
     - **0.025**
     - 0.064
     - 0.053
     - 1.024
     - 0.097
   * - GPU Energy (kWh)
     - **0.013**
     - 0.029
     - 0.042
     - 1.037
     - 0.185
   * - RAM Energy (kWh)
     - **0.010**
     - 0.022
     - 0.013
     - 0.358
     - 0.035
   * - Energy Consumed (kWh)
     - **0.048**
     - 0.115
     - 0.108
     - 2.419
     - 0.316
   * - Peak RAM Usage (GB)
     - 74.8
     - 67.6
     - 99.4
     - **53.6**
     - 86.1

**Key insights:**

- **Total energy consumption is driven more by training duration than by instantaneous power draw.** While SASRec exhibits the highest peak GPU power (278.6 W), its relatively rapid convergence results in a moderate total energy footprint (0.32 kWh).
- **LightGCN**, despite a lower average power draw (157.4 W), requires an extensive training phase to converge, resulting in the highest aggregate consumption (2.42 kWh) and carbon emissions (0.0095 kg CO\ :sub:`2`\ eq).
- **Shallow architectures demonstrate a superior efficiency-effectiveness balance.** EASE\ :sup:`R` consumes approximately 95% less energy than deep graph-based baselines.

-----

Full Configuration Example
===========================

.. code-block:: yaml

    reader:
        loading_strategy: dataset
        reading_method: local
        local_path: data/netflix.tsv
        rating_type: implicit
    writer:
        dataset_name: GreenAI-Benchmark
        writing_method: local
        local_experiment_path: experiments/green/
    splitter:
        test_splitting:
            strategy: temporal_holdout
            ratio: 0.1
    models:
        LightGCN:
            optimization:
                strategy: hopt
                scheduler: asha
                device: cuda
                cpu_per_trial: 4
                gpu_per_trial: 1
                num_samples: 10
            embedding_size: [64, 128]
            n_layers: [2, 3]
            reg_weight: [uniform, 0.0001, 0.01]
            batch_size: 4096
            epochs: 100
            learning_rate: [uniform, 0.0001, 0.01]
    evaluation:
        top_k: [10, 20]
        metrics: [nDCG, Precision, Recall]
        validation_metric: nDCG@10
    dashboard:
        codecarbon:
            enabled: true
            save_to_file: true
            output_dir: ./carbon_reports/
        wandb:
            enabled: true
            project: GreenAI-Benchmark
