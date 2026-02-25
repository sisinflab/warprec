.. _introduction:

#################
Why WarpRec?
#################

The Fragmented Landscape
========================

The landscape of recommender frameworks is fractured into distinct silos, none capable of satisfying the demands of modern research and deployment simultaneously.

The Academic Silo
-----------------

Academic frameworks have evolved from early tools for classic and deep learning models into comprehensive libraries offering hundreds of models and frameworks that prioritize reproducibility and scientific rigor through automated experimental lifecycles. Frameworks like RecBole, Elliot, Cornac, and DaisyRec are excellent for algorithmic exploration.

However, they remain **confined to a single-node paradigm**. Their reliance on eager execution engines (e.g., Pandas) means they fail to scale beyond single-node memory limits. Distributed hyperparameter optimization, multi-GPU training, and elastic cloud deployment require manual infrastructure that these tools do not provide. Even rigorous frameworks often omit multiple hypothesis testing corrections, leaving them vulnerable to p-hacking.

The Industrial Silo
-------------------

Industrial-grade frameworks such as NVIDIA Merlin, Apache Spark MLlib, and Microsoft Recommenders are architected for production-scale environments. These systems employ distributed dataframes and GPU offloading to train complex architectures on billions of data points.

However, they **prioritize serving over science**. Industrial frameworks typically provide limited data-splitting and evaluation metrics, and usually exclude significance testing and beyond-accuracy dimensions such as bias, fairness, and diversity.

The Green AI Gap
-----------------

The field has pivoted from accuracy-maximizing "Red AI" to "Green AI," elevating energy efficiency and carbon transparency to first-class metrics. This is particularly critical in Recommender Systems, where the combination of billion-parameter embedding tables and exhaustive hyperparameter optimization grids leads to extreme energy consumption. Recent works have shown that marginal gains in performance often require exponential increases in carbon emissions.

Despite this, **current RS frameworks remain fundamentally energy-blind**. Leading platforms provide no native mechanisms to quantify carbon emissions or power consumption within the experimental pipeline.

The Agentic AI Gap
-------------------

Artificial intelligence is shifting from monolithic models to agentic workflows, where autonomous agents call external tools and interleave reasoning with actions. In this paradigm, the recommender becomes a callable tool within an agent's decision-making process. This new role requires interactive dialogue to iteratively refine results, yet **current frameworks lack the standardized interfaces** to enable this.

-----

How WarpRec Bridges the Gap
=============================

WarpRec resolves these four gaps through a unified, backend-agnostic architecture:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Gap
     - WarpRec Solution
   * - **Academic / Industrial**
     - Backend-agnostic design via Narwhals. Models transition seamlessly from local debugging to distributed training on Ray clusters.
   * - **Green AI**
     - Native CodeCarbon integration for real-time energy tracking and carbon-emission profiling at the trial level.
   * - **Agentic AI**
     - Native Model Context Protocol (MCP) server that exposes trained models as queryable tools for LLMs and autonomous agents.
   * - **Scientific Rigor**
     - Automated hypothesis testing (Wilcoxon, t-test, Mann-Whitney U) with Bonferroni, Holm-Bonferroni, and FDR corrections.

WarpRec is not only a framework for building and evaluating recommender systems, but also a **powerful experimentation platform** equipped with:

- **55 built-in algorithms** spanning unpersonalized, content-based, collaborative filtering, context-aware, sequential, and hybrid paradigms
- **40 GPU-accelerated metrics** across accuracy, rating, coverage, novelty, diversity, bias, fairness, and multi-objective families
- **19 filtering and splitting strategies** for rigorous, leak-free experimental protocols
- **Comprehensive HPO engine** with Grid Search, Bayesian optimization (HyperOpt, Optuna), and bandit-based strategies (BOHB)
- **Integrated dashboarding** via Weights & Biases, MLflow, and TensorBoard
- **Event-driven callback system** for injecting custom logic at any pipeline stage

For a detailed walkthrough of the architecture, see :ref:`Architecture <architecture>`.

To start your first experiment, proceed to the :ref:`Installation <install_guide>` guide or jump directly to the :ref:`Quick Start <quick-start>`.
