#################
Install WarpRec
#################

The procedure to install WarpRec its very simple, there are not many pre-requisites:

- Python 3.12
- `Poetry 2.1.2 <https://python-poetry.org/>`_ for dependency management. Poetry is required only for development.

WarpRec is designed with a **modular dependency structure** leveraging Poetry. This allows you to install a lean **core** and selectively add optional *extras* for advanced functionalities (like experiment tracking or cloud I/O).

.. list-table:: **Dependency Groups**
   :widths: 20 60
   :header-rows: 1

   * - Group
     - Description
   * - ``core`` (main)
     - Essential functionalities, including base models (ItemKNN, EASE, MultiDAE), data handling, and evaluation metrics.
   * - ``dashboard``
     - Remote logging and experiment tracking tools (`Weights & Biases`, `MLflow`, `CodeCarbon`).
   * - ``remote-io``
     - Functionality for reading and writing data to/from cloud storage (e.g., Azure Blob Storage).


Also the core of WarpRec supports a wide variety of recommendation models, Graph-based models requires manual installation of **PyTorch Geometric (PyG)** due to strict CUDA/PyTorch version constraints. It enables the use of Graph Neural Network (GNN) models.

.. _install_guide:

Installation guide
---------------------

In the following sections you will be guided on how to install WarpRec. Here are three supported approaches, depending on your workflow.

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

Poetry is the recommended tool, as it allows you to selectively install dependencies.

1. **Create and activate the environment:**

   .. code-block:: bash

      poetry env use python3.12

2. **Install core and optional extras:**

   To install only the **core** dependencies (``main``):

   .. code-block:: bash

      poetry install --only main

   To install the **core plus specific extras** (e.g., ``remote-io``):

   .. code-block:: bash

      poetry install --only main remote-io

   To install the **core plus ALL extras** (``dashboard``, ``remote-io``, etc.):

   .. code-block:: bash

      poetry install

3. **Install PyTorch Geometric (PyG) - Optional for Graph Models:**

   **PyG is only required if you intend to use or develop Graph-based recommendation models.** It must be installed manually with the correct PyTorch and CUDA version. Refer to the official guides for the latest instructions:

   - `PyTorch Installation Guide <https://pytorch.org/get-started/locally/>`_
   - `PyG Installation Guide <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_

   Example (replace with your CUDA version):

   .. code-block:: bash

      # Example for CUDA 11.8
      poetry run pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
      -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
      poetry run pip install torch-geometric

Using venv (`requirements.txt`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When installing via `pip` and `requirements.txt`, **all dependencies (core and all extras) are installed simultaneously** for maximum functionality.

1. **Create and activate a virtual environment:**

   .. code-block:: bash

      python3.12 -m venv .venv
      source .venv/bin/activate

2. **Install all dependencies:**

   .. code-block:: bash

      pip install --upgrade pip
      pip install -r requirements.txt

3. **Install compatible versions of PyTorch and PyG (if needed):**

   .. code-block:: bash

      # Make sure to install the correct versions matching your CUDA setup
      pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
      -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
      pip install torch-geometric

Using Conda/Mamba (`environment.yml`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similar to ``venv``, the Conda environment file typically includes **all dependencies (core and extras)**.

1. **Create or update the environment:**

   .. code-block:: bash

      conda env create --file environment.yml --name warprec
      # or, if the env already exists
      conda env update --file environment.yml --name warprec

2. **Activate the environment:**

   .. code-block:: bash

      conda activate warprec

3. **Manually install compatible PyTorch and PyG (if needed):**

   Conda may not always provide the latest compatible versions. For full compatibility, refer to the installation links above and install with `pip` inside the Conda environment.

.. note::

   On some Linux systems, it has been observed that the ``grpcio`` library may need to be upgraded manually.
   This is typically required if you encounter errors related to gRPC during installation or runtime.

   You can upgrade ``grpcio`` using ``pip`` as follows:

   .. code-block:: bash

      # Upgrade grpcio to the latest version
      pip install --upgrade grpcio

   If you are using a virtual environment or Poetry, make sure the command is executed **inside the environment**.

-----

Enabling Advanced Features
--------------------------

Dashboard Extras (Green AI, Experiment Tracking)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``dashboard`` group installs the tools needed for experiment tracking and Green AI profiling:

- **Weights & Biases** — real-time experiment tracking and visualization
- **MLflow** — experiment lifecycle management and model registry
- **CodeCarbon** — energy consumption and carbon emission monitoring

To install with Poetry:

.. code-block:: bash

    poetry install --only main dashboard

Once installed, enable these features in your configuration:

.. code-block:: yaml

    dashboard:
        wandb:
            enabled: true
            project: MyProject
        codecarbon:
            enabled: true
            save_to_file: true
            output_dir: ./carbon_reports/

For full details on dashboard configuration, see :ref:`Configuration > Dashboard <configuration>`.

Ray Cluster Setup (Distributed Training)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ray is included in the core dependencies. For the **Training Pipeline**, a Ray cluster must be initialized before running experiments:

.. code-block:: bash

    ray start --head

To customize available resources:

.. code-block:: bash

    ray start --head --num-cpus <NUM_CPUS> --num-gpus <NUM_GPUS>

For granular per-trial resource control (RAM and VRAM limits), initialize the cluster with custom resources:

.. code-block:: bash

    ray start --head --resources='{"ram_gb": <RAM_VALUE>, "vram_gb": <VRAM_VALUE>}'

For multi-node cluster setup, see the :doc:`clustering guide <clustering>`.

.. note::

   The **Design Pipeline** (``-p design``) does not require a Ray cluster and can run locally without any additional setup.
