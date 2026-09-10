# Install WarpRec

WarpRec is designed to be easily installed via **pip** or via **Conda**. This ensures that all dependencies and the Python environment are managed consistently.

### Quick Install (PyPI)
The easiest way to get started is using pip:
```bash
pip install warprec
```

WarpRec provides extra dependencies for specific use cases:

| extra | usage |
|---|---|
| dashboard | Dashboard functionalities like MLflow and Weights & Biases. |
| remote-io | Remote communication with cloud services like Azure. |
| serving | Optional dependencies to serve your recommendation models. |
| bohb | Dependencies required by the `bohb` search strategy and scheduler. |
| graph | PyTorch Geometric, required by the graph-based recommenders. |
| all | All of the above. |

You can install them at any moment using the following command:
```bash
pip install "warprec[dashboard, remote-io]"
```

### Graph-Based Recommenders

The graph-based recommenders (`LightGCN`, `NGCF`, `SGL`, `LightGCL` and the rest of the family) need [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/). It is an optional dependency, so install the `graph` extra:

```bash
pip install "warprec[graph]"
```

Without it those models remain registered but raise an `ImportError` explaining what to install as soon as they are instantiated. Every other model works normally.

!!! note
    WarpRec needs only the pure-Python `torch-geometric` package. The compiled companions (`torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`) are **not** required, so there is no wheel index to configure and no need to match a CUDA version.

### Install via Poetry

If you use [Poetry](https://python-poetry.org/) for dependency management, you can easily install WarpRec and its dependencies directly from the source:

1. **Clone the repository**
   Open your terminal and clone the WarpRec repository:
   ```bash
   git clone <repository_url>
   cd warprec
   ```

2. **Install the project**
    ```
    poetry install
    # Or you can install all extra dependencies
    poetry install --extras all
    ```

### Development Setup (Conda)

If you want to contribute, we recommend using Conda. The environment installs WarpRec with all extra dependencies:

1. **Clone the repository**
   Open your terminal and clone the WarpRec repository:
   ```bash
   git clone <repository_url>
   cd warprec
   ```

2. **Create the Conda environment**
    Use the provided `environment.yml` file. It installs Python 3.12 and then WarpRec itself with all extras, so the dependency set always matches `pyproject.toml`.
    ```bash
    conda env create --file environment.yml
    ```

3.  **Activate the environment:**

    ```bash
    conda activate warprec
    ```

4.  **CPU-only machines (optional)**

    The environment installs the default PyTorch build, which is CUDA-enabled on Linux. On a machine without a GPU you can replace it with the smaller CPU build:

    ```bash
    pip install torch==2.7.* --index-url https://download.pytorch.org/whl/cpu
    ```
