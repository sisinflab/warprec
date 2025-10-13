#########################
Dashboard Configuration
#########################

The **Dashboard Configuration** module allows you to select which dashboards are activated during model training.
By default, the **TensorBoard dashboard** is always enabled and cannot be disabled.

WarpRec integrates with multiple dashboards through **Ray**, enabling flexible monitoring and logging of experiments.

-----------------------------
Available Dashboards
-----------------------------

The following nested sections can be used to configure dashboards:

- **wandb**: Configuration for **Weights & Biases**.
- **mlflow**: Configuration for **MLFlow**.
- **codecarbon**: Configuration for **CodeCarbon**.

.. note::
   Each dashboard section can be independently enabled or disabled.
   Default state for all optional dashboards is **disabled**.

.. important::
   - TensorBoard is **always active** and can be accessed locally.
   - Ensure all required credentials and API keys are valid; otherwise, dashboard logging may fail.


Weights & Biases (wandb)
-----------------------------

WarpRec supports **Weights & Biases** via Ray for experiment tracking.

- **enabled**: Boolean flag to activate the dashboard. Defaults to ``False``.
- **team**: Name of the team. If ``None``, the first team is used. Defaults to ``None``.
- **project**: Name of the W&B project.
- **group**: Name of the experiment group.
- **api_key_file**: Path to a file containing the API key.
- **api_key**: API key string.
- **excludes**: List of parameters to exclude from logging.
- **log_config**: Whether to log the experiment configuration. Defaults to ``False``.
- **upload_checkpoints**: Whether to upload model checkpoints. Defaults to ``False``.

MLFlow (mlflow)
-----------------

WarpRec supports **MLFlow** via Ray for experiment tracking and artifact logging.

- **enabled**: Boolean flag to activate the dashboard. Defaults to ``False``.
- **tracking_uri**: URI of the MLFlow tracking server.
- **registry_uri**: URI of the MLFlow model registry.
- **experiment_name**: Name of the MLFlow experiment.
- **tags**: Tags to attach to the MLFlow run.
- **tracking_token**: Token for authentication with the MLFlow server.
- **save_artifacts**: Whether to save artifacts to the MLFlow server. Defaults to ``False``.

CodeCarbon (codecarbon)
-----------------------

WarpRec supports **CodeCarbon** via Ray to monitor energy consumption and carbon footprint.

- **enabled**: Boolean flag to activate the dashboard. Defaults to ``False``.
- **save_to_api**: Whether to send results to the CodeCarbon API. Defaults to ``False``.
- **save_to_file**: Whether to save results locally to a file. Defaults to ``False``.
- **output_dir**: Directory where results are stored.
- **tracking_mode**: Tracking mode for CodeCarbon. Options are ``machine`` or ``process``.

-----------------------------
Example Dashboard Configuration
-----------------------------

The following example shows a dashboard configuration enabling **MLFlow** tracking:

.. code-block:: yaml

   dashboard:
       mlflow:
           enabled: true
           experiment_name: MyExperiment

.. note::
   Other dashboards (wandb, codecarbon) can be similarly enabled by specifying their corresponding nested sections.

.. warning::
   When using remote dashboards (e.g., W&B) with artifact logging, ensure that Ray checkpoints are **not deleted** during training.
