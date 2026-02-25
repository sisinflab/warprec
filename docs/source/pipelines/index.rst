.. _pipelines:

##########
Pipelines
##########

WarpRec abstracts complex workflows into three standardized execution pipelines, all controlled via declarative YAML configuration files. Each pipeline serves a distinct purpose in the experimentation lifecycle:

.. list-table::
   :header-rows: 1
   :widths: 18 15 67

   * - Pipeline
     - Command
     - Purpose
   * - **Design**
     - ``-p design``
     - Rapid prototyping and model debugging. Runs locally without Ray or HPO.
   * - **Training**
     - ``-p train``
     - Full-scale experiments with distributed HPO, cross-validation, and statistical testing via Ray.
   * - **Evaluation**
     - ``-p eval``
     - Evaluate pre-trained checkpoints or external recommendation files without retraining.

All pipelines are invoked with the same command structure:

.. code-block:: bash

    python -m warprec.run -c <config_file>.yml -p <pipeline>

.. contents:: On this page
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

-----

Choosing the Right Pipeline
============================

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Use Case
     - Pipeline
     - Ray Required?
     - Writer Required?
   * - Debug a new model implementation
     - Design
     - No
     - No
   * - Validate a configuration before full HPO
     - Design
     - No
     - No
   * - Run a full benchmark with HPO
     - Training
     - Yes
     - Yes
   * - Compare models with statistical testing
     - Training
     - Yes
     - Yes
   * - Evaluate a saved checkpoint on new metrics
     - Evaluation
     - No
     - Optional
   * - Evaluate recommendations from another framework
     - Evaluation
     - No
     - Optional

-----

.. toctree::
   :maxdepth: 2

   design
   train
   eval
   guide
