# Dashboard Configuration

The **Dashboard Configuration** module allows you to select which dashboards are activated during model training.

WarpRec integrates with multiple dashboards through **Ray**, enabling flexible monitoring and logging of experiments.

## Available Dashboards

The following nested sections can be used to configure dashboards:

- **wandb**: Configuration for **Weights & Biases**.
- **mlflow**: Configuration for **MLFlow**.
- **codecarbon**: Configuration for **CodeCarbon**.

!!! important
    - Every dashboard is **disabled** by default.
    - Ensure all required credentials and API keys are valid; otherwise, dashboard logging may fail.

## Weights & Biases (wandb)

WarpRec supports **Weights & Biases** via Ray for experiment tracking.

- **enabled**: Boolean flag to activate the dashboard. Defaults to `False`.
- **team**: Name of the team. If `None`, the first team is used. Defaults to `None`.
- **project**: Name of the W&B project.
- **group**: Name of the experiment group.
- **api_key_file**: Path to a file containing the API key.
- **api_key**: API key string.
- **excludes**: List of parameters to exclude from logging.
- **log_config**: Whether to log the experiment configuration. Defaults to `False`.
- **upload_checkpoints**: Whether to upload model checkpoints. Defaults to `False`.

## MLFlow (mlflow)

WarpRec supports **MLFlow** via Ray for experiment tracking and artifact logging.

- **enabled**: Boolean flag to activate the dashboard. Defaults to `False`.
- **tracking_uri**: URI of the MLFlow tracking server.
- **registry_uri**: URI of the MLFlow model registry.
- **experiment_name**: Name of the MLFlow experiment.
- **tags**: Tags to attach to the MLFlow run.
- **tracking_token**: Token for authentication with the MLFlow server.
- **save_artifacts**: Whether to save artifacts to the MLFlow server. Defaults to `False`.

!!! note
    MLFlow does not accept `@` in a metric name, so metrics appear in MLFlow as `<metric>/<k>` (for example `nDCG@10` is logged as `nDCG/10`). Everywhere else, WarpRec keeps the `<metric>@<k>` names.

## CodeCarbon (codecarbon)

WarpRec measures the energy its train and swarm runs consume, and the carbon they emit, with **CodeCarbon**.

- **enabled**: Boolean flag to activate the dashboard. Defaults to `False`.
- **save_to_api**: Whether to send results to the CodeCarbon API. Defaults to `False`. Each tracker sends its own emissions from the node it runs on, so the API settings must be readable there: put them in `~/.codecarbon.config` on every node, or in the `CODECARBON_EXPERIMENT_ID` and `CODECARBON_API_KEY` environment variables, rather than in a `.codecarbon.config` next to the configuration.
- **save_to_file**: Whether to save results locally to a file. Defaults to `False`. The driver appends the rows to `emissions.csv` in the run's Ray results folder (`<local_experiment_path>/<dataset_name>/ray_results/`); a resumed run appends its own set of rows.
- **tracking_mode**: Tracking mode for CodeCarbon. Options are `machine` or `process`. Defaults to `machine`.
    - `machine` measures every Ray node for the whole run: one tracker per node, started when the run starts its work, alongside the data preparation (a tracker measures from the moment it is up on its node, a few seconds in), and stopped when the run ends, whether it completes, pauses or fails. The data preparation, the searches, the cross-validation retraining and the test evaluation are all covered, and `emissions.csv` gets **one row per node** per run. A node that joins the cluster during the run is measured from the moment WarpRec sees it (it looks every 10 seconds); a node that leaves the cluster before the run ends loses its row, and WarpRec logs it. A tracker measures **everything that runs on its node**, not only WarpRec.
    - `process` measures the process the trial trains in (the Ray Train worker), not the Ray Tune driver that waits for it. The trial measures itself and the driver appends its row to `emissions.csv`, one row per trial and session. With `save_to_api`, the trial sends its emissions to the API itself, so the API settings must be readable there: put them in `~/.codecarbon.config` or in the `CODECARBON_EXPERIMENT_ID` and `CODECARBON_API_KEY` environment variables rather than in a `.codecarbon.config` next to the configuration. A session cut short by a pause writes no row under `process`.

### Rows of emissions.csv

`emissions.csv` keeps CodeCarbon's columns. WarpRec labels each row with the run and the node it measured in two of them:

| Column | `machine` |
| --- | --- |
| `project_name` | The run's name (`run.name`, or the one WarpRec derived). |
| `experiment_id` | The node: `host=<host name> ip=<IP address> node=<Ray node id>`. |
| `run_id` | CodeCarbon's id of the tracker, unique to the row. |

Only the file is labelled this way: what a tracker sends to the CodeCarbon API keeps the API's experiment. Several Ray nodes on one machine (a local test cluster, or several containers on one host) each measure that whole machine.

!!! note "Utilization columns"
    `cpu_utilization_percent`, `ram_utilization_percent` and `ram_used_gb` are CodeCarbon's averages of samples it takes once a second, with `psutil`, of the **whole machine**, under either tracking mode: under `process` the energy is the trial's process, the utilization is still the machine's. A trial shorter than a second has no sample, and its row reads 0 for the three. The CPU utilization of a sample is measured since the previous one taken in the same thread, and CodeCarbon takes each sample in a new thread, so its first two samples, and an occasional later one, read 0: on a short trial the CPU column is low or 0 whatever the load. Both are CodeCarbon's behaviour (3.3), not WarpRec's.

!!! warning "Location in emissions.csv"
    CodeCarbon looks up where the machine is from its IP address and writes it to every row of `emissions.csv`: country, region and, on a machine outside a cloud provider, latitude and longitude. WarpRec cannot leave these columns out (the file's columns are CodeCarbon's), so clear them before sharing the file, for example with `pandas`: `df[["country_name", "country_iso_code", "region", "longitude", "latitude"]] = None`.

When using the **save_to_api** feature, the main to correctly track the experiment is to create a `.codecarbon.config` file which contains the following information:

```yaml
[codecarbon]
experiment_id = YOUR_EXPERIMENT_ID
api_key = YOUR_API_KEY
```

!!! important
    You can find the API key and the experiment ID inside the codecarbon official dashboard.

## Example Dashboard Configuration

The following example shows a dashboard configuration enabling **MLFlow** tracking:

```yaml
dashboard:
    mlflow:
        enabled: true
        experiment_name: MyExperiment
```

!!! note
    Other dashboards (wandb, codecarbon) can be similarly enabled by specifying their corresponding nested sections.
    When using remote dashboards (e.g., W&B) with artifact logging, ensure that Ray checkpoints are **not deleted** during training.
