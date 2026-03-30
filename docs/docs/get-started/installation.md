# Install WarpRec

WarpRec is designed to be easily installed via **pip** or via **Conda**. This ensures that all dependencies and the Python environment are managed consistently. Conda environment is available both for CPU and GPU.

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
| all | All of the above. |

You can install them at any moment using the following command:
```bash
pip install "warprec[dashboard, remote-io]"
```

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

If you want to contribute or need a specific environment (CPU/GPU), we recommend using Conda. The conda environment already contains all the extra dependencies:

1. **Clone the repository**
   Open your terminal and clone the WarpRec repository:
   ```bash
   git clone <repository_url>
   cd warprec
   ```

2. **Create the Conda environment**
    Use the provided environment.gpu.yml (or environment.cpu.yml) file to create the virtual environment. This will install Python 3.12 and the necessary core dependencies.
    ```bash
    # For GPU support
    conda env create --file environment.gpu.yml
    # Or for CPU only
    conda env create --file environment.cpu.yml
    ```

3.  **Activate the environment:**

    ```bash
    conda activate warprec
    ```
