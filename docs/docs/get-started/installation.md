# Install WarpRec

The procedure to install WarpRec is streamlined using Conda or Mamba.

> **Stay tuned!** WarpRec will be available on **PyPI** in a future release.

## Prerequisites

- **Git**: To clone the repository.
- **Conda**: You need either [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

## Installation guide

Follow these steps to clone the project and set up the environment:

1. **Clone the repository**
   Open your terminal and clone the WarpRec repository:
   ```bash
   git clone <repository_url>
   cd warprec
   ```

2. **Create the Conda environment**
    Use the provided environment.yml file to create the virtual environment. This will install Python 3.12 and the necessary core dependencies.
    ```bash
    conda env create --file environment.yml
    ```

3.  **Activate the environment:**

    ```bash
    conda activate warprec
    ```
