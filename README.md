# 🚀 WarpRec

WarpRec is a flexible and efficient framework designed for building, training, and evaluating recommendation models. It supports a wide range of configurations, customizable pipelines, and powerful optimization tools to enhance model performance and usability.

WarpRec is built with both beginners and experienced practitioners in mind. For newcomers, it offers a simple and intuitive interface to explore and experiment with state-of-the-art recommendation models. For advanced users, WarpRec provides a modular and extensible architecture that allows rapid prototyping, complex experiment design, and fine-grained control over every step of the recommendation pipeline.

Whether you're learning how recommender systems work or conducting high-performance research and development, WarpRec offers the right tools to match your workflow.

## ✨ Key Features

- **Model Training**: WarpRec includes out-of-the-box support for a variety of recommendation algorithms, including classic models like `ItemKNN` and `EASE`, as well as deep learning approaches such as `MultiDAE`. Each model can be easily configured, trained, and extended, making the framework suitable for both simple baselines and advanced research.
- **Evaluation**: The evaluation module offers a wide range of metrics, all of which are configurable and easy to extend. Metrics are computed in batches to ensure scalability and memory efficiency, and GPU acceleration is supported to speed up the evaluation process in large-scale experiments.
- **Custom Pipelines**: WarpRec allows you to build your own training and evaluation pipelines directly in Python, without relying on external configuration files. This feature is particularly useful for advanced users who want full control over the logic and flow of experiments, enabling faster iteration and experimentation.
- **Hyperparameter Optimization**: The framework integrates seamlessly with Ray Tune, providing access to advanced search algorithms and scheduling strategies. Whether you're performing a basic grid search or a complex multi-trial optimization, WarpRec helps you automate and accelerate the tuning process efficiently.
- **Data Management**: Data handling is streamlined with built-in tools for loading, preprocessing, splitting, and exporting datasets. The system supports standard formats and is designed to work smoothly with both small-scale test sets and large real-world datasets.
- **Experiment Tracking and Visualization**: WarpRec integrates with popular tracking tools such as `TensorBoard`, `MLflow`, and `Weights & Biases`, allowing you to monitor metrics, visualize training dynamics, and manage multiple runs with ease. Additionally, the framework supports `CodeCarbon` to track the environmental impact of your experiments.

## ⚙️ Installation

### 📋 Prerequisites

- Python 3.12
- [Poetry](https://python-poetry.org/) for dependency management.

### 🛠️ Installation Guide

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
2. Install dependencies using poetry:
    ```bash
    poetry env use python3.12
    poetry install
3. (Optional) Activate the virtual environment:
    ```bash
    source $(poetry env info --path)/bin/activate
4. Verify the installation:
    ```bash
    poetry run python --version

### 🧰 Makefile Commands

The project includes a Makefile to simplify common operations:

- Install dependencies:
    ```bash
    make install
- Run linting:
    ```bash
    make lint
- Run tests:
    ```bash
    make test

## 🚂 Usage

### 🏋️‍♂️ Training a model

To train a model, use the `train.py` script. Here's an example:

1. Prepare a configuration file (e.g. `config/train_config.yml`) with details
    about the model, dataset and training parameters.
2. Run the following command:
    ```bash
    poetry run python warprec/train.py --config config/train_config.yml

This command will start the training process using the specified configuration file.

### 🔍 Infer a model

To do inference on a model, use the `infer.py` script. Here's an example:

1. Prepare a configuration file (e.g. `config/infer_config.yml`) with details
    about the model, dataset and training parameters.
2. Run the following command:
    ```bash
    poetry run python warprec/infer.py --config config/infer_config.yml

This command will start the inference process using the specified configuration file.

## 🤝 Contributing
We welcome contributions from the community! Whether you're fixing bugs, improving documentation, or proposing new features, your input is highly valued.

To get started:

1. Fork the repository and create a new branch for your feature or fix.
2. Follow the existing coding style and conventions.
3. Open a pull request with a clear description of your changes.

If you encounter any issues or have questions, feel free to open an issue in the Issues section of the repository.

## 📜 License
License information will be added soon.

## 📖 Citation
Citation details will be provided in an upcoming release. Stay tuned!

## 📧 Contact
For questions, suggestions, or inquiries, feel free to reach out:

**Marco Avolio** - marco@wideverse.com
**Joseph Trotta** - joseph.trotta@ovs.it
