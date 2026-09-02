# Cloud Clustering

WarpRec supports clustering of cloud resources to optimize distributed training and hyperparameter optimization tasks. By grouping similar cloud instances together, WarpRec can efficiently allocate resources based on workload requirements, leading to improved performance and cost savings.

This process is facilitated through the integration of Ray, a distributed computing framework that enables seamless scaling of machine learning workloads across multiple nodes. With Ray, WarpRec can dynamically manage clusters, monitor resource utilization, and ensure that tasks are executed in an optimal manner.

In this guide, we will explore how to set up and utilize cloud clustering using WarpRec. For this example, we will use GCP, but similar principles apply to other cloud providers supported by Ray.

## Setting Up Your Project

Before you begin, ensure that you have a Google Cloud Platform (GCP) account and have set up a project. You will need to enable the Compute Engine API and create a service account with the necessary permissions to manage resources. The IAM API should be enabled as well.

Once your project is set up, you will need to authenticate your local environment with GCP. First of all, install the Ray GCP dependencies and the Google Cloud CLI if you haven't already:

```bash
pip install google-api-python-client
```

Then, authenticate using the following command:

```bash
gcloud auth application-default login
```

After the authentication is complete, you will need to set up a Blob Storage bucket to store your WarpRec experiment data. You also must upload your dataset to a GCP Storage bucket to be accessible by the cloud instances. In order to do so, you can use the following commands:

```bash
gcloud storage buckets create gs://name-of-your-bucket --location=europe-west3
gcloud storage cp -r ./datasets gs://name-of-your-bucket/dataset
```

!!! note

    Make sure to replace `name-of-your-bucket` with a unique name for your storage bucket. Location can be adjusted based on your preferences.

Now everything is set up to start using WarpRec with cloud clustering.

## Creating a Cluster

To create a cluster using WarpRec and Ray, you will need to define the cluster configuration in a YAML file. This configuration will specify the number of nodes, machine types, Docker images, and setup commands (including environment creation).

You can find the example configuration file `guides/gcp_clustering/ray_cluster.yml` in the WarpRec repository.

The available settings for Ray Clustering can be found in the [Ray Cluster Configuration Documentation](https://docs.ray.io/en/latest/cluster/getting-started.html).

To start a cluster using a Ray Cluster configuration file, use the following command:

```bash
ray up path/to/your/cluster_config.yml -y
```

This command will initiate the cluster creation process based on the specifications provided in your configuration file. The `-y` flag automatically confirms the creation without prompting for user input. This process may take a few minutes as the necessary resources are provisioned and configured.

Once the cluster is up and running, you **must** open the Ray dashboard connection. This creates a secure tunnel between your local machine and the cluster, which is required for submitting jobs. Use the following command in a dedicated terminal:

```bash
ray dashboard path/to/your/cluster_config.yml
```

This will open the Ray dashboard in your default web browser (usually at `http://localhost:8265`), allowing you to visualize the cluster's performance and manage tasks effectively. **Keep this terminal open.**

To shut down the cluster when you are done, use the following command:

```bash
ray down path/to/your/cluster_config.yml -y
```

## Submitting WarpRec Jobs to the Cluster

You can submit WarpRec training or hyperparameter optimization jobs to the Ray cluster using the `ray job submit` command. This allows you to leverage the distributed computing capabilities of the cluster for your recommender system tasks.

**Important Configuration Note:** Since the cluster uses GCS Fuse to mount your bucket, you should not use `gs://` paths in your WarpRec configuration file. Instead, point to the shared mount point on the cluster nodes:

*   **Dataset Path:** `/home/ray/shared/dataset/your_file.csv`
*   **Experiment Output:** `/home/ray/shared/experiments/your_experiment_name/`

To submit a WarpRec job to the cluster, ensure the dashboard tunnel is running, then use the following command:

```bash
ray job submit \
    --address http://localhost:8265 \
    --working-dir . \
    -- \
    /usr/local/bin/micromamba run -r /home/ray/micromamba -n warprec \
    python -m warprec.run -c path/to/config.yml -p train
```

Make sure to replace `path/to/config.yml` with the path to your WarpRec configuration file (relative to your current directory).

## Surviving Preemption

Cloud nodes get reclaimed, and a spot instance can disappear mid-experiment. WarpRec runs can be **paused and resumed**, so a preemption need not cost the whole experiment.

Give the driver a stable run name and make a missing state a hard error rather than a silent restart:

```yaml
run:
    name: ml1m_baselines
    resume: force
```

Arrange for the pod or job to receive `SIGTERM` before it is reclaimed, with enough grace time for WarpRec to reach a safe point. On Kubernetes this is `terminationGracePeriodSeconds`; the run then pauses cleanly, saves its state alongside the other experiment artifacts, and exits with status 0.

Resubmitting the same job resumes the experiment: models that already finished are skipped, and interrupted hyperparameter searches continue from their last Ray Tune checkpoint. Because resource requests are deliberately excluded from the compatibility check, a paused run can be resumed on a cluster of a different size.

!!! important
    The run state and the Ray Tune checkpoints must outlive the node. Use a storage backend that persists beyond the pod, such as Azure Blob Storage via the writer configuration, or a shared volume mounted at the experiment path.

See [Pause & Resume](../pipelines/pause-resume.md) for the full workflow.

And with this, you are all set to run WarpRec experiments using cloud clustering! Enjoy the scalability and efficiency that comes with distributed training on the cloud.
