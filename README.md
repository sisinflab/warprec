# 🚀 WarpRec

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Documentation Status](https://readthedocs.org/projects/warprec/badge/?version=latest)](https://warprec.readthedocs.io/en/latest/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CodeCarbon](https://img.shields.io/badge/carbon%20tracked-CodeCarbon-brightgreen?logo=leaflet&logoColor=white)](https://codecarbon.io/)
[![MCP Powered](https://img.shields.io/badge/MCP-powered-blueviolet?logo=anthropic&logoColor=white)](https://modelcontextprotocol.io/)
[![GitHub Stars](https://img.shields.io/github/stars/sisinflab/warprec?style=social)](https://github.com/sisinflab/warprec)

<p align="center">
  <a href="https://warprec.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/badge/📖%20Read%20the%20Docs-warprec-blue?style=for-the-badge" alt="Read the Docs"/>
  </a>
</p>

WarpRec is a flexible and efficient framework designed for building, training, and evaluating recommendation models. It supports a wide range of configurations, customizable pipelines, and powerful optimization tools to enhance model performance and usability.

WarpRec is designed for both beginners and experienced practitioners. For newcomers, it offers a simple and intuitive interface to explore and experiment with state-of-the-art recommendation models. For advanced users, WarpRec provides a modular and extensible architecture that allows rapid prototyping, complex experiment design, and fine-grained control over every step of the recommendation pipeline.

Whether you're learning how recommender systems work or conducting high-performance research and development, WarpRec offers the right tools to match your workflow.

## 📚 Table of Contents

- [✨ Key Features](#-key-features)
- [⚙️ Installation](#️-installation)
  - [📋 Prerequisites](#-prerequisites)
  - [🛠️ Setup Guide](#️-setup-guide)
- [🚂 Usage](#-usage)
  - [🏋️ Training a model](#️-training-a-model)
  - [✏️ Design a model](#️-design-a-model)
  - [🔍 Evaluate a model](#-evaluate-a-model)
  - [🧰 Makefile Commands](#-makefile-commands)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [📖 Citation](#-citation)
- [📧 Contact](#-contact)

## ✨ Key Features

- **55 Built-in Algorithms**: WarpRec ships with 55 state-of-the-art recommendation models spanning 6 paradigms — Unpersonalized, Content-Based, Collaborative Filtering (e.g., `LightGCN`, `EASE`$^R$, `MultiVAE`), Context-Aware (e.g., `DeepFM`, `xDeepFM`), Sequential (e.g., `SASRec`, `BERT4Rec`, `GRU4Rec`), and Hybrid. All models are fully configurable and extend a standardized base class, making it easy to prototype custom architectures within the same pipeline.
- **Backend-Agnostic Data Engine**: Built on [Narwhals](https://narwhals-dev.github.io/narwhals/), WarpRec operates over Pandas, Polars, and Spark without code changes — enabling a true "write-once, run-anywhere" workflow from laptop to distributed cluster. Data ingestion supports both local filesystems and cloud object storage (Azure Blob Storage).
- **Comprehensive Data Processing**: The data module provides 13 filtering strategies (filter-by-rating, k-core, cold-start heuristics) and 6 splitting protocols (random/temporal Hold-Out, Leave-k-Out, Fixed Timestamp, k-fold Cross-Validation), for a total of 19 configurable strategies to ensure rigorous and reproducible experimental setups.
- **40 GPU-Accelerated Metrics**: The evaluation suite covers 40 metrics across 7 families — Accuracy, Rating, Coverage, Novelty, Diversity, Bias, and Fairness — including multi-objective metrics for simultaneous optimization of competing goals. All metrics are computed with full GPU acceleration for large-scale experiments.
- **Statistical Rigor**: WarpRec automates hypothesis testing with paired (Student's t-test, Wilcoxon signed-rank) and independent-group (Mann-Whitney U) tests, and applies multiple comparison corrections via **Bonferroni** and **FDR (Benjamini-Hochberg)** to prevent p-hacking and ensure statistically robust conclusions.
- **Distributed Training & HPO**: Seamless vertical and horizontal scaling from single-GPU to multi-node Ray clusters. Hyperparameter optimization supports Grid, Random, Bayesian, HyperOpt, Optuna, and BoHB strategies, with ASHA pruning and model-level early stopping to maximize computational efficiency.
- **Green AI & Carbon Tracking**: WarpRec is the first recommendation framework with native [CodeCarbon](https://codecarbon.io/) integration, automatically quantifying energy consumption and CO₂ emissions for every experiment and persisting carbon footprint reports alongside standard results.
- **Agentic AI via MCP**: WarpRec natively implements a [Model Context Protocol](https://modelcontextprotocol.io/) server (`infer-api/mcp_server.py`), exposing trained recommenders as callable tools within LLM and autonomous agent workflows — transforming the framework from a static predictor into an interactive, agent-ready component.
- **REST API & Model Serving**: Trained models are instantly deployable as RESTful microservices via the built-in FastAPI server (`infer-api/server.py`), decoupling the modeling core from serving infrastructure with zero additional engineering effort.
- **Experiment Tracking**: Native integrations with `TensorBoard`, `Weights & Biases`, and `MLflow` for real-time monitoring of metrics, training dynamics, and multi-run management.
- **Custom Pipelines & Callbacks**: Beyond the three standard pipelines (Training, Design, Evaluation), WarpRec exposes an event-driven Callback system for injecting custom logic at any stage — enabling complex experiments without modifying framework internals.

## ⚙️ Installation

WarpRec is designed to be easily installed and reproducible using **Conda**. This ensures that all dependencies and the Python environment are managed consistently.

### 📋 Prerequisites

- **Git**: To clone the repository.
- **Conda**: You need either [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

### 🛠️ Setup Guide

Follow these steps to clone the project and set up the environment:

1. **Clone the repository**
   Open your terminal and clone the WarpRec repository:
   ```bash
   git clone <repository_url>
   cd warprec

2. **Create the Conda environment**
    Use the provided environment.yml file to create the virtual environment. This will install Python 3.12 and the necessary core dependencies.
    ```bash
   conda env create --file environment.yml

## 🚂 Usage

### 🏋️‍♂️ Training a model

To train a model, use the `train` pipeline. Here's an example:

1. Prepare a configuration file (e.g. `config/train_config.yml`) with details
    about the model, dataset and training parameters.
2. Run the following command:
    ```bash
    python -m warprec.run -c config/train_config.yml -p train

This command starts the training process using the specified configuration file.

### ✏️ Design a model

To implement a custom model, WarpRec provides a dedicated design interface via the `design` pipeline. The recommended workflow is as follows:

1. Prepare a configuration file (e.g. `config/design_config.yml`) with details
    about the custom models, dataset and training parameters.
2. Run the following command:
    ```bash
    python -m warprec.run -c config/design_config.yml -p design

This command initializes a lightweight training pipeline, specifically intended for rapid prototyping and debugging of custom architectures within the framework.

### 🔍 Evaluate a model

To run only evaluation on a model, use the `eval` pipeline. Here's an example:

1. Prepare a configuration file (e.g. `config/eval_config.yml`) with details
    about the model, dataset and training parameters.
2. Run the following command:
    ```bash
    python -m warprec.run -c config/eval_config.yml -p eval

This command starts the evaluation process using the specified configuration file.

### 🧰 Makefile Commands

The project includes a Makefile to simplify common operations:

- 🧹 Run linting:
    ```bash
    make lint
- 🧑‍🔬 Run tests:
    ```bash
    make test

## 🤝 Contributing
We welcome contributions from the community! Whether you're fixing bugs, improving documentation, or proposing new features, your input is highly valued.

To get started:

1. Fork the repository and create a new branch for your feature or fix.
2. Follow the existing coding style and conventions.
3. Make sure the code passes all checks by running `make lint`.
4. Open a pull request with a clear description of your changes.

If you encounter any issues or have questions, feel free to open an issue in the Issues section of the repository.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📖 Citation
Citation details will be provided in an upcoming release. Stay tuned!

## 📧 Contact
For questions or suggestions, feel free to contact us at:

* **Marco Avolio** - marco.avolio@wideverse.com
* **Potito Aghilar** - potito.aghilar@wideverse.com
* **Sabino Roccotelli** - sabino.roccotelli@wideverse.com
* **Vito Walter Anelli** - vitowalter.anelli@poliba.it
* **Joseph Trotta** - joseph.trotta@ovs.it
