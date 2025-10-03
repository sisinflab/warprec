#################
Introduction
#################

.. toctree::
   :hidden:
   :maxdepth: 2

   install
   quick-start

WarpRec is a comprehensive, efficient, and fully reproducible framework specifically engineered to support the testing, evaluation, design, and management of Recommender Systems across a wide range of experimental settings. The framework provides a structured environment that integrates all the essential components required for end-to-end experimentation, including standardized interfaces for data ingestion, configurable pipelines for preprocessing, and flexible modules for model training and evaluation.

From a usability perspective, WarpRec has been developed with a strong focus on accessibility: the framework ensures that new users can rapidly set up and execute experiments thanks to its intuitive configuration mechanisms, clear abstractions, and well-documented workflows. This lowers the entry barrier for practitioners who are approaching recommender systems for the first time.

At the same time, WarpRec is designed with advanced customization capabilities in mind, offering a high degree of modularity that allows expert users to tailor every aspect of the experimentation process. Researchers and developers can easily extend or replace existing components, fine-tune experimental protocols, and integrate custom models, metrics, or optimization strategies. This dual nature—combining ease of use with deep configurability—makes WarpRec suitable both for educational purposes and for advanced research and production-grade experimentation, where reproducibility, scalability, and flexibility are critical requirements.

WarpRec is not only a framework for building and evaluating recommender systems, but also a **powerful experimentation platform** equipped with a rich set of advanced features:

- **Comprehensive Hyperparameter Optimization (HPO) engine**, with support for:

  - Grid search
  - Bayesian optimization strategies with **Hyperopt**, **Optuna**, and other libraries

- **Innovative and extensive evaluation module**, capable of computing a wide range of metrics with **GPU acceleration** for large-scale experiments

- **Integrated statistical testing tools**, including multiple comparison corrections, to enable rigorous and reliable analysis of results

- **Detailed performance reporting**, providing insights into model behavior both:

  - **Spatially**, by monitoring **memory consumption (RAM and VRAM)** of models
  - **Temporally**, by analyzing **inference times** across different conditions


- **Seamless MLOps integration**, with ready-to-use connectors for widely adopted dashboards such as **Weights & Biases (W&B)** and **MLflow**, enabling effective experiment tracking and visualization

- And much more!
