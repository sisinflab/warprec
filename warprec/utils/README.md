# 🛠️ [WarpRec](../../README.md) Utils

The `utils` module of WarpRec provides a collection of essential utilities designed to simplify the definition and execution of recommendation experiments via configuration files. These tools help make experiments reproducible, maintainable, and extensible.

## 📚 Table of Contents
- ✨ [Main Functionalities](#✨-main-functionalities)
- 📂 [Module Structure](#📂-module-structure)
- ⚙️ [Config](#️⚙️-config)
    - 🚀 [Quick Start](#🚀-quick-start)
- 🧾 [Logger](#🧾-logger)
    - 🔧 [Key Features](#🔧-key-features)
- 🧩 [Enums](#🧩-enums)
- 📦 [Registry](#📦-registry)



## ✨ Main Functionalities

- **⚙️ Configuration**: Defines the foundation for experiment reproducibility and modularity. Configuration files allow users to control all aspects of the pipeline (e.g., model selection, data splitting, hyperparameters) without modifying the underlying code.
- **🧾 Logger**: A flexible and extensible logging utility designed to provide consistent output across modules. It supports both console and file-based logging, with customizable formats and verbosity levels.
- **🧩 Enums**: A centralized collection of enumerations used throughout the framework, which ensures readability, type safety, and code consistency.
- **📦 Registry**: A core component enabling dynamic access and instantiation of key objects—such as models, metrics, splitters, or optimizers—based on configuration references. This design supports decoupling and extensibility across the framework.

## 📂 Module Structure

The module `utils` is structured as follows:

- **[`config`](config/README.md)**: Contains classes and functions to parse, validate, and manage configuration files. It supports default values, nested configurations, and integration with environment variables.
- **[`logger`](logger/logger.py)**: Provides a customizable logging interface used throughout WarpRec. It enables seamless tracking of experiments, debugging information, and system warnings/errors.
- **[`enums`](enums.py)**: Defines common enumerations used in WarpRec, including model types, dataset formats, splitter strategies, and more. These enums are used to validate user inputs and standardize behavior.
- **[`registry`](registry.py)**: Implements a lightweight registry pattern to manage dynamically loadable components. It simplifies the addition of custom elements (e.g., new loss functions or metrics) by automatically registering them for discovery at runtime.

## 📌 Example Usage

### ⚙️ Config

The `config` module provides all the utilities needed to define a **configuration file** for your experiments. In the quick start section, we explain how to run a simple experiment with your data.

#### 🚀 Quick Start

First of all, let's create a simple configuration file. Here's how you can do it:

```yaml
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
    strategy: temporal_holdout
    test_ratio: 0.1
models:
    ItemKNN:
        k: 10
        similarity: cosine
        normalize: False
evaluation:
    top_k: [10, 20, 50]
    metrics: [nDCG, Precision, Recall, HitRate]
```

To run an experiment with this configuration, just run the following:

```bash
python warprec/train.py --config path/to/the/config.yml
```

That's it! You're ready to go. This will get you started with the WarpRec configuration files. For a more in-depth description of all the possible configurations you can make, use **[`config`](config/README.md)** as reference.

### 🧾 Logger

The `logger` module provides a unified interface to log messages across your entire experiment pipeline.

You can easily initialize a logger with your preferred configuration:

```python
from warprec.utils.logger import get_logger

logger = get_logger(name="train", level="INFO")

logger.info("Training started")
logger.warning("This is a warning")
logger.error("An error occurred")
```

Or you can just use the ready-to-go logger used by WarpRec:

```python
from warprec.utils.logger import logger

logger.msg("This is a logger message")
logger.attention("Something might be wrong")
logger.error("Something is wrong")
```

#### 🔧 Key Features

- Console logging.
- Configurable logging levels.
- Timestamped and formatted output for easier debugging.

### 🧩 Enums

`Enums` are used throughout the framework to enforce valid input values and enhance code readability. Here’s an example usage:

```python
from warprec.utils.enums import RatingType, Similarities

rating_type = RatingType.IMPLICIT
similarity = Similarities.COSINE

print(f"Rating type: {rating_type}, Similarity: {similarity}")
```

All enums subclass both `str` and `Enum`, making them human-readable and type-safe. You can use them in configs like:

```yaml
reader:
    rating_type: implicit
...
models:
    ItemKNN:
        similarity: cosine
```

For a full list of supported enums, check the **[`enums`](enums.py)** module.

### 📦 Registry

The `registry` module enables dynamic instantiation of components via configuration, following a plugin-like architecture.

You can register and retrieve classes easily:

```python
from warprec.utils.registry import metric_registry

@metric_registry.register(name="MyCustomMetric")
class MyCustomMetric:
    def compute(self):
        return "custom result"

# Instantiate via name
metric = metric_registry.get("MyCustomMetric")
print(metric.compute())  # Outputs: custom result
```

Registered classes can then be referenced from a config like:

```yaml
evaluation:
    metrics: [MyCustomMetric]
```

Use `.list_registered()` to see the currently available components.
