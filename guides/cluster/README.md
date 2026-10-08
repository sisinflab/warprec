# Running WarpRec on a cluster

This folder holds configurations rather than a notebook: a cluster cannot be
started from a laptop guide.

| File | What it is |
|---|---|
| [`ray_cluster.yml`](ray_cluster.yml) | A Ray cluster on Google Cloud: one head and one worker (`e2-standard-4`), each running `warprec[all]` inside the Ray Docker image, with a Cloud Storage bucket mounted at `/home/ray/shared` for data and results. Replace `YOUR-PROJECT-ID` and `YOUR-BLOB-STORAGE-NAME` before use. |

Start it with `ray up guides/cluster/ray_cluster.yml -y`, then point an
experiment at it with `general.ray_address: auto` from the head node.

- [Cluster Management](https://warprec.readthedocs.io/en/latest/cloud/cluster-management/)
  walks through the GCP setup this file assumes.
- [Deploying Served Models](https://warprec.readthedocs.io/en/latest/cloud/deployment/)
  covers serving on Docker, Kubernetes, Ray clusters and KubeRay.
