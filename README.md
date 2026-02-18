# 🚀 WarpRec

[Docs]

[Docs]: https://warprec.readthedocs.io/en/latest/

WarpRec is a flexible and efficient framework designed for building, training, and evaluating recommendation models. It supports a wide range of configurations, customizable pipelines, and powerful optimization tools to enhance model performance and usability.

WarpRec is designed for both beginners and experienced practitioners. For newcomers, it offers a simple and intuitive interface to explore and experiment with state-of-the-art recommendation models. For advanced users, WarpRec provides a modular and extensible architecture that allows rapid prototyping, complex experiment design, and fine-grained control over every step of the recommendation pipeline.

Whether you're learning how recommender systems work or conducting high-performance research and development, WarpRec offers the right tools to match your workflow.

## ✨ Key Features

- **Model Training**: WarpRec includes out-of-the-box support for a variety of recommendation algorithms, including classic models like `ItemKNN` and `EASE`, as well as deep learning approaches such as `MultiDAE`. Each model can be easily configured, trained, and extended, making the framework suitable for both simple baselines and advanced research.
- **Model Design**: WarpRec provides a flexible API for designing and integrating custom recommendation models. Developers can implement their own architectures by extending standardized base classes, ensuring compatibility with the training, evaluation, and optimization modules of the framework. This feature enables rapid prototyping and experimentation, allowing researchers and practitioners to seamlessly test novel approaches alongside existing baselines.
- **Evaluation**: The evaluation module offers a wide range of metrics, all of which are configurable and easy to extend. Metrics are computed in batches to ensure scalability and memory efficiency, and GPU acceleration is supported to speed up the evaluation process in large-scale experiments.
- **Custom Pipelines**: WarpRec allows you to build your own training and evaluation pipelines directly in Python, without relying on external configuration files. This feature is particularly useful for advanced users who want full control over the logic and flow of experiments, enabling faster iterations and experiments.
- **Hyperparameter Optimization**: The framework integrates seamlessly with Ray Tune, providing access to advanced search and scheduling algorithms. Whether you're running a basic grid search or a complex multi-trial optimization, WarpRec automates and accelerates the tuning process.
- **Data Management**: WarpRec streamlines data handling with built-in tools for loading, preprocessing, splitting, and exporting datasets. The system supports standard formats and is designed to work smoothly with both small-scale test sets and large real-world datasets.
- **Experiment Tracking and Visualization**: WarpRec integrates with popular tracking tools such as `TensorBoard`, `MLflow`, and `Weights & Biases`, allowing you to monitor metrics, visualize training dynamics, and manage multiple runs with ease. Additionally, the framework supports `CodeCarbon` to track the environmental impact of your experiments.

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
