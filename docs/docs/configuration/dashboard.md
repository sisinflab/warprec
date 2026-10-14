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
    - `process` measures each trial on its own: every Ray Train worker of a trial measures the process it trains in, from the start of its training to its end, charging only the GPUs Ray assigned to it, and hands its row to the driver through Ray, so a worker on a node that cannot see the driver's storage delivers its row all the same. `emissions.csv` gets **one row per trial worker** and session. Only the searches are covered: the data preparation, the cross-validation retraining and the test evaluation are not. A trial that completes or fails with fewer rows than it has workers (a worker process that was killed delivers none) is logged. A session cut short by a pause writes no row.

### Rows of emissions.csv

`emissions.csv` keeps CodeCarbon's columns. WarpRec labels each row with the run and with what it measured in two of them:

| Column | `machine` | `process` |
| --- | --- | --- |
| `project_name` | The run's name (`run.name`, or the one WarpRec derived). | The run's name. |
| `experiment_id` | The node: `host=<host name> ip=<IP address> node=<Ray node id>`. | The node, the trial and the worker's rank in it: `host=<host name> ip=<IP address> node=<Ray node id> trial=<trial id> rank=<rank>`. |
| `run_id` | CodeCarbon's id of the tracker, unique to the row. | CodeCarbon's id of the tracker, unique to the row. |

Only the file is labelled this way: what a tracker sends to the CodeCarbon API keeps the API's experiment. Several Ray nodes on one machine (a local test cluster, or several containers on one host) each measure that whole machine.

### What is measured

The figures are CodeCarbon's (3.3); WarpRec decides where the trackers run and what they are told to measure.

| | `machine` | `process` |
| --- | --- | --- |
| **What a row covers** | One Ray node, from the start of the run to its end: data preparation, searches, cross-validation retraining, test evaluation. Everything that runs on the node is counted, not only WarpRec. | One Ray Train worker of one trial, while it trains and validates. |
| **CPU, Linux with RAPL** | The CPU package's energy counters: the whole machine. | The same counters, still the whole machine: the rows of workers sharing a node overlap, each counting the whole package. |
| **CPU without RAPL, and on Apple silicon** | The CPU model's TDP, from CodeCarbon's database, times the machine's CPU load. | The TDP times the process's share of the CPU. |
| **RAM** | A model of the machine's memory size (an estimated DIMM count), with a floor of 10 W on x86 and 3 W on ARM. | The same model applied to the process's memory, with the same floor: an idle worker is still charged 3 or 10 W. |
| **GPU** | Every GPU on the node, whole devices. | The GPUs Ray assigned to the worker, whole devices (two workers sharing a GPU through a fractional `gpu_per_trial` are each charged for all of it). `gpu_count` and `gpu_model` still describe every GPU of the node. |

Summing the rows of one `process` run therefore counts the machine's CPU (with RAPL) and the RAM floor once per worker.

!!! note "Utilization columns"
    `cpu_utilization_percent`, `ram_utilization_percent` and `ram_used_gb` are CodeCarbon's averages of samples it takes once a second, with `psutil`, of the **whole machine**, under either tracking mode: under `process` the energy is the worker's process, the utilization is still the machine's. `gpu_utilization_percent` covers the GPUs the row measures. A row covering less than a second has no sample, and reads 0 for the three. The CPU utilization of a sample is measured since the previous one taken in the same thread, and CodeCarbon takes each sample in a new thread, so its first two samples, and an occasional later one, read 0: on a short row the CPU column is low or 0 whatever the load. Both are CodeCarbon's behaviour (3.3), not WarpRec's.

!!! warning "Location in emissions.csv"
    CodeCarbon looks up where the machine is from its IP address and writes it to every row of `emissions.csv`: country, region and, on a machine outside a cloud provider, latitude and longitude. WarpRec cannot leave these columns out (the file's columns are CodeCarbon's). `experiment_id` also names each host and its IP address. Clear them before sharing the file, for example with `pandas`: `df[["country_name", "country_iso_code", "region", "longitude", "latitude"]] = None` and `df["experiment_id"] = df["experiment_id"].str.replace(r"host=\S+ ip=\S+ ", "", regex=True)`.

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
