# 🚀 WarpRec

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

WarpRec is designed with a modular dependency structure leveraging Poetry to manage a lean core installation and optional extras for advanced functionalities (like experiment tracking or cloud I/O).

### 📋 Prerequisites

- Python 3.12
- [Poetry 2.1.2](https://python-poetry.org/) for development and dependency management (highly recommended).

### 🛠️ Installing WarpRec for different environments

**PyG is only required if you intend to use or develop Graph-based recommendation models.** Due to strict compatibility constraints, you must install PyG manually, ensuring the versions match your specific CUDA setup. To ensure the correct installation, follow the [PyG Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

#### 📦 Using Poetry (`pyproject.toml`) - Preferred for Development

Poetry is the recommended tool, as it installs only the core dependencies by default, allowing you to selectively add specialized dependency groups (extras).

1. Create and activate environment:
    ```bash
    poetry env use python3.12

    # If you want to install only the core
    poetry install --only main

    # If you want to install all dependencies
    poetry install

    # If you want to install a specific subset of utilities
    poetry install --only main remote-io

2. Install PyTorch Geometric (PyG) - Optional for Graph Models:

    Example (replace with your CUDA version):
    ```bash
    # Example for CUDA 11.8
    poetry run pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
    poetry run pip install torch-geometric

#### 🧪 Using venv (`requirements.txt`)

When installing via `pip` and `requirements.txt`, all dependencies from the core and all extra groups are installed simultaneously.

1. Create the virtual environment and activate it:
    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate

2. Install base dependencies:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt

3. Manually install compatible versions of PyTorch and PyG (if needed for graph models, see step 3 in the Poetry section).

#### 🐍 Using Conda/Mamba (`environment.yml`)

Similar to `venv`, the Conda environment file typically includes all dependencies (core and extras) for maximum convenience and broader compatibility.

1. Create or update the environment:
    ```bash
    conda env create --file environment.yml --name warprec
    # or, if the env already exists
    conda env update --file environment.yml --name warprec

2. Activate the environment:
    ```bash
    conda activate warprec

3. Manually install compatible PyTorch and PyG (if needed for graph models, see step 3 in the Poetry section). Conda may not always provide the latest compatible versions; for full compatibility, installing with `pip` inside the Conda environment is often required.

### ⚠️ Important Notes

- **PyG is Optional:** Only install PyG (PyTorch Geometric) if you need **Graph Neural Network (GNN)** models.

- **PyG Compatibility:** PyG is highly sensitive to the version of **PyTorch** and **CUDA**. Incorrect combinations may lead to runtime errors or failed builds. Always check the official compatibility matrix before installing:
    - [PyTorch CUDA Support Matrix](https://pytorch.org/get-started/previous-versions/)
    - [PyG CUDA Compatibility](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

If you're unsure about your system's CUDA version, run:
```bash
nvcc --version
```

Or fall back to the CPU version of all packages by omitting the CUDA suffixes. Also while these environments are made available for convenience and broader compatibility, **Poetry remains the preferred tool for development**, ensuring consistency with the project's setup.

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
License information will be added soon.

## 📖 Citation
Citation details will be provided in an upcoming release. Stay tuned!

## 📧 Contact
For questions or suggestions, feel free to contact us at:

**Marco Avolio** - marco@wideverse.com
**Joseph Trotta** - joseph.trotta@ovs.it
