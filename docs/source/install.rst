#################
Install WarpRec
#################

The procedure to install WarpRec its very simple, there are not many pre-requisites:

- Python 3.12
- `Poetry 2.1.2 <https://python-poetry.org/>`_ for dependency management. Poetry is required only for development.

Install with MakeFile
---------------------

WarpRec provides a **Makefile** to simplify common setup and development tasks.
Using these commands is **highly recommended for newcomers**, as they perform all the necessary checks to ensure a clean and consistent installation of the development environment:

- Install dependencies with Poetry:
.. code-block:: bash

    make install-poetry
- Install dependencies with venv:
.. code-block:: bash

    make install-venv
- Install dependencies with Conda/Mamba:
.. code-block:: bash

    make install-conda
- Run linting:
.. code-block:: bash

    make lint
- Run tests:
.. code-block:: bash

    make test

Manual Installation
---------------------

While WarpRec supports quick setup via `make install-*` commands, you may want to manually create and customize your environment using your preferred tool. Here are three supported approaches, depending on your workflow.

.. note::

   PyG (PyTorch Geometric) is highly sensitive to the version of PyTorch and CUDA. Incorrect combinations may lead to runtime errors or failed builds.

   Always check the official compatibility matrix before installing PyTorch and PyG:
      - `PyTorch CUDA Support Matrix <https://pytorch.org/get-started/previous-versions/>`_
      - `PyG CUDA Compatibility <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_

   If you're unsure about your system's CUDA version, run:

   .. code-block:: bash

     nvcc --version

.. important::
   While these environments are made available for convenience and broader compatibility, **Poetry remains the preferred tool for development**, ensuring consistency with the project's setup.

Using Poetry (`pyproject.toml`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Create and activate the environment:**

   .. code-block:: bash

      poetry env use python3.12
      poetry install

2. **Install PyTorch and PyG manually:**

   Due to compatibility constraints, PyG must be installed with the correct PyTorch and CUDA version. Refer to the official guides for the latest instructions:

   - `PyTorch Installation Guide <https://pytorch.org/get-started/locally/>`_
   - `PyG Installation Guide <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_

   Example (replace with your CUDA version):

   .. code-block:: bash

      # Example for CUDA 12.1
      poetry run pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
      poetry run pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
      -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
      poetry run pip install torch-geometric torchmetrics

Using venv (`requirements.txt`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Create and activate a virtual environment:**

   .. code-block:: bash

      python3.12 -m venv .venv
      source .venv/bin/activate

2. **Install base dependencies:**

   .. code-block:: bash

      pip install --upgrade pip
      pip install -r requirements.txt

3. **Install compatible versions of PyTorch and PyG:**

   .. code-block:: bash

      # Make sure to install the correct versions matching your CUDA setup
      pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
      pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
      -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
      pip install torch-geometric torchmetrics

Using Conda/Mamba (`environment.yml`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Create or update the environment:**

   .. code-block:: bash

      conda env create --file environment.yml --name warprec
      # or, if the env already exists
      conda env update --file environment.yml --name warprec

2. **Activate the environment:**

   .. code-block:: bash

      conda activate warprec

3. **Manually install compatible PyTorch and PyG:**

   Conda may not always provide the latest compatible versions. For full compatibility, refer to the installation links above and install with `pip` inside the Conda environment.

.. note::

   On some Linux systems, it has been observed that the `grpcio` library may need to be upgraded manually.
   This is typically required if you encounter errors related to gRPC during installation or runtime.

   You can upgrade `grpcio` using `pip` as follows:

   .. code-block:: bash

      # Upgrade grpcio to the latest version
      pip install --upgrade grpcio

   If you are using a virtual environment or Poetry, make sure the command is executed **inside the environment**.
