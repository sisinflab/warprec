# Deploying Served Models

A model saved by the training pipeline is served by [`warprec.serve`](../serving/index.md) on [Ray Serve](https://docs.ray.io/en/latest/serve/index.html). This page covers every way to run that server: one process on one machine, a container, a Kubernetes `Deployment`, a Ray cluster on virtual machines, and Ray on Kubernetes with [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html).

Whatever the target, a deployment needs the same three ingredients:

- **the `warprec[serving]` package** (`warprec[mcp]` for the MCP endpoint, plus `graph` for graph-based models such as LightGCN);
- **the checkpoints**, the `.pth` files written with `meta.save_model: true`;
- **a [serving configuration](../configuration/serving.md)**, `serve.yml`, that points each endpoint at its checkpoint.

## Choosing a Deployment

| Option | Good for | How it scales | Kubernetes needed |
|---|---|---|---|
| [One machine](#one-machine) | Development, demos, a single server | Replicas, batching and GPUs inside the machine | No |
| [Docker](#docker) | Shipping one self-contained image | One container per machine | No |
| [Kubernetes Deployment](#kubernetes-deployment) | Clusters without KubeRay; simple, stateless pods | More pods behind a Service | Yes |
| [Ray cluster on VMs](#ray-cluster-on-virtual-machines) | Teams already running Ray clusters with `ray up` | Ray Serve replicas across the cluster's nodes | No |
| [KubeRay `RayService`](#kuberay-rayservice) | Production on Kubernetes | Ray Serve autoscaling plus Ray's node autoscaler; zero-downtime upgrades | Yes, plus the KubeRay operator |
| [KubeRay `RayCluster` + `serve deploy`](#kuberay-raycluster-with-serve-deploy) | Iterating on a Kubernetes Ray cluster | As above, managed by hand | Yes, plus the KubeRay operator |

The first three run Ray inside the process. In the last three, the models are replicas spread over a Ray cluster.

## Building Blocks

These apply to every option below.

### Checkpoints Must Be Readable Where Replicas Run

Every model replica opens its checkpoint from the local file system, so on a multi-node cluster **each node must see the file at the same path**. Three ways to get there:

- **Bake the checkpoints into the image.** Simplest and fully reproducible: the image is the model release.
- **Mount shared storage.** A Kubernetes `PersistentVolumeClaim` (`ReadOnlyMany` or `ReadWriteMany`), NFS, or a bucket mounted with a FUSE driver. The [cloud cluster guide](cluster-management.md) mounts a GCS bucket at `/home/ray/shared` on every node, so a checkpoint trained there at `/home/ray/shared/experiments/.../serialized/` is servable from the same path.
- **Download at startup.** An init container copies the checkpoints from object storage into a volume the Ray containers mount.

Paths in `serve.yml` may be relative; `warprec.serve` makes them absolute before handing them to the replicas.

### The Image

Build on the official Ray image matching the Ray version WarpRec installs, so that every node runs the same Ray and Python:

```dockerfile
FROM rayproject/ray:2.58.0-py312-cpu

# CPU-only PyTorch: on Linux the default wheels bundle CUDA and add gigabytes.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        "warprec[serving,graph]==2.0.0"

# The checkpoints and the configuration, if they ship with the image.
COPY --chown=ray:users models/ /models/
COPY --chown=ray:users serve.yml /app/serve.yml

EXPOSE 8000
CMD ["warprec.serve", "-c", "/app/serve.yml"]
```

- The Ray images are published for both `amd64` and `arm64`.
- For GPUs, start from `rayproject/ray:2.58.0-py312-gpu` and drop the `--extra-index-url` line, so that PyTorch comes with CUDA.
- Use `warprec[mcp,graph]` for the MCP endpoint.

The same image serves as the `CMD` of a container and as the head and worker image of a KubeRay cluster, where KubeRay overrides the command.

### A `serve.yml` for Containers

```yaml
server:
  host: 0.0.0.0          # listen beyond the container
  port: 8000
endpoints:
  - name: sasrec
    checkpoint: /models/sasrec.pth
  - name: bpr
    checkpoint: /models/bpr.pth
    unknown_user: popular
```

Leave `api_key` out of the file and pass `WARPREC_API_KEY` as an environment variable, from a Kubernetes `Secret` where there is one.

### Resources

- **CPU.** Each model replica asks Ray for one CPU unless `deployment.ray_actor_options` says otherwise; the gateway asks for none. Ray counts the CPUs of its container, including a CPU limit, so a pod limited to 2 CPUs fits two replicas. With more models than CPUs, give each a fraction, for example `ray_actor_options: {num_cpus: 0.5}`, or the extra replicas wait to be scheduled.
- **GPU.** `device: cuda` asks Ray for one GPU per replica; `num_gpus: 0.25` shares one GPU between four replicas. On Kubernetes, give the container `nvidia.com/gpu` limits and Ray picks the GPUs up. Models that keep no tensors, such as the neighbourhood and EASE families, are served on the CPU and reserve no GPU.
- **Shared memory.** Ray keeps its object store in `/dev/shm`. Docker's default of 64 MB is too small: pass `--shm-size`, or mount a memory-backed `emptyDir` on Kubernetes.
- **Memory.** Each replica holds its model and the training data serving needs, which is the size of its checkpoint, plus PyTorch.

### Health, Shutdown and Ports

- **Health.** `GET /healthz` answers without the API key, which makes it suitable for load balancers and Kubernetes probes. Ray's own proxy also answers `/-/healthz` and `/-/routes`.
- **Shutdown.** `warprec.serve` shuts Serve and Ray down cleanly on `SIGTERM` and exits with status 0. This takes about 15 seconds, more than Docker's default stop timeout of 10, so stop containers with `docker stop -t 30` and give pods a `terminationGracePeriodSeconds` of about 60.
- **Port.** Serve listens on port 8000, the port KubeRay's Serve service expects. Keep it there on Kubernetes.

### Exporting the Application for a Cluster

On a Ray cluster, the application is described by a [Ray Serve config file](https://docs.ray.io/en/latest/serve/production-guide/config.html) rather than started by a process. `warprec.serve` writes one:

```bash
warprec.serve -c serve.yml --export serve_app.yaml
```

- The file runs `warprec.serving.app:app_builder` with the serving configuration inlined.
- Paths become absolute, and they must exist where the export runs.
- The proxy listens on `0.0.0.0` unless `server.host` is set.
- The API key is left out: set `WARPREC_API_KEY` in the cluster's environment.

When the checkpoints live in an image, export from inside that image, so the paths are the ones the cluster will see:

```bash
docker run --rm -v "$PWD":/out my-registry/warprec-serve:2.0.0 \
    warprec.serve -c /app/serve.yml --export /out/serve_app.yaml
```

## One Machine

```bash
pip install "warprec[serving]"
warprec.serve -c serve.yml
```

`warprec.serve` starts a local Ray instance and serves until it is stopped. Batching, several replicas per model and GPUs all work within the machine. To keep it running as a service, hand it to the system's process manager. With systemd, for example:

```ini
[Unit]
Description=WarpRec model serving
After=network-online.target

[Service]
WorkingDirectory=/srv/warprec
Environment=WARPREC_API_KEY=change-me
ExecStart=/srv/warprec/venv/bin/warprec.serve -c /srv/warprec/serve.yml
Restart=on-failure
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

## Docker

Build the [image](#the-image) and run it:

```bash
docker build -t my-registry/warprec-serve:2.0.0 .
docker run -d --name warprec -p 8000:8000 --shm-size=2g \
    -e WARPREC_API_KEY=change-me my-registry/warprec-serve:2.0.0

curl localhost:8000/healthz
docker stop -t 30 warprec
```

To serve new checkpoints without rebuilding, mount them over `/models` instead of copying them in: `-v /srv/models:/models:ro`.

## Kubernetes Deployment

Without KubeRay, the container above runs as an ordinary `Deployment`. Each pod is a complete, independent server with its own Ray inside, so scaling means more pods behind a `Service`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: warprec-api-key
stringData:
  WARPREC_API_KEY: change-me
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: warprec-serve
spec:
  replicas: 2
  selector:
    matchLabels: {app: warprec-serve}
  template:
    metadata:
      labels: {app: warprec-serve}
    spec:
      terminationGracePeriodSeconds: 60
      containers:
        - name: warprec
          image: my-registry/warprec-serve:2.0.0
          ports:
            - containerPort: 8000
          envFrom:
            - secretRef: {name: warprec-api-key}
          readinessProbe:
            httpGet: {path: /healthz, port: 8000}
            periodSeconds: 5
          livenessProbe:
            httpGet: {path: /healthz, port: 8000}
            initialDelaySeconds: 180
            periodSeconds: 20
          resources:
            requests: {cpu: "4", memory: 6Gi}
            limits: {cpu: "4", memory: 6Gi}
          volumeMounts:
            - {name: dshm, mountPath: /dev/shm}
      volumes:
        - name: dshm
          emptyDir: {medium: Memory, sizeLimit: 2Gi}
---
apiVersion: v1
kind: Service
metadata:
  name: warprec-serve
spec:
  selector: {app: warprec-serve}
  ports:
    - port: 8000
      targetPort: 8000
```

- **Strengths:** this is the simplest way onto Kubernetes. Any `HorizontalPodAutoscaler` works on it, and rollouts are ordinary `Deployment` rollouts.
- **Limitations:** every pod loads every model, and the replicas of one model cannot spread across pods. A large model, or very uneven traffic between models, is better served by [KubeRay](#kuberay-rayservice).
- **CPU limit:** the limit is what Ray sees, so size it to the number of model replicas (see [Resources](#resources)).

## Ray Cluster on Virtual Machines

A Ray cluster started with `ray up`, on GCP, AWS, Azure or your own machines (see [Cloud Clustering](cluster-management.md)), serves the models as replicas spread over its nodes. WarpRec and its extras must be installed on every node, and the checkpoints must be at the same path on every node, as on the shared mount the clustering guide sets up.

There are two ways to put the application on the cluster.

**From the head node, attached.** Set `server.ray_address: auto` and start `warprec.serve` on the head node:

```bash
ray attach cluster.yml
warprec.serve -c serve.yml          # server.ray_address: auto
```

The application lives as long as the command. Stopping it deletes only this application and leaves other Serve applications on the cluster running.

**Detached, from anywhere.** Export the application and deploy it through the dashboard:

```bash
warprec.serve -c serve.yml --export serve_app.yaml
ray dashboard cluster.yml                                  # keeps a tunnel to port 8265 open
serve deploy serve_app.yaml --address http://localhost:8265
serve status --address http://localhost:8265               # wait for RUNNING
```

- The application keeps running after the terminal closes; `serve shutdown --address http://localhost:8265` removes it.
- Set `WARPREC_API_KEY` in the environment the nodes start with, for example in the cluster configuration's `setup_commands` or the environment of `ray start`.
- Open port 8000 on the nodes behind your load balancer.

## KubeRay `RayService`

[KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) runs Ray clusters on Kubernetes, and its `RayService` resource runs a Ray Serve application on one. This is the recommended production setup on Kubernetes:

- KubeRay starts the cluster and deploys the exported application.
- It recovers failed pods and exposes the application through a `Service`.
- A change to the application is applied in place. A change to the cluster, such as a new image, brings up a new cluster and switches traffic to it once it is ready, without downtime.

**1. Install the KubeRay operator** with Helm:

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
kubectl create namespace ray-system
helm install kuberay-operator kuberay/kuberay-operator --version 1.7.0 -n ray-system
kubectl get pods -n ray-system
```

**2. Build and push the [image](#the-image)**, with the checkpoints in it or on a volume, and **export the application** from it ([Exporting the Application](#exporting-the-application-for-a-cluster)).

**3. Create the API key secret:**

```bash
kubectl create secret generic warprec-api-key --from-literal=WARPREC_API_KEY=change-me
```

**4. Describe the service.** `serveConfigV2` is the exported file, unchanged:

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: warprec
spec:
  # warprec.serve -c serve.yml --export serve_app.yaml
  serveConfigV2: |
    proxy_location: EveryNode
    http_options:
      host: 0.0.0.0
      port: 8000
    applications:
    - name: warprec
      route_prefix: /
      import_path: warprec.serving.app:app_builder
      args:
        config:
          server: {host: 0.0.0.0, port: 8000, route_prefix: /, api_key: null, mcp: false, ray_address: null}
          endpoints:
          - name: sasrec
            checkpoint: /models/sasrec.pth
            deployment:
              autoscaling_config: {min_replicas: 1, max_replicas: 6, target_ongoing_requests: 32}
          - name: bpr
            checkpoint: /models/bpr.pth
            unknown_user: popular
  rayClusterConfig:
    rayVersion: "2.58.0"
    enableInTreeAutoscaling: true
    headGroupSpec:
      rayStartParams:
        num-cpus: "0"          # keep model replicas off the head
      template:
        spec:
          containers:
            - name: ray-head
              image: my-registry/warprec-serve:2.0.0
              envFrom:
                - secretRef: {name: warprec-api-key}
              resources:
                requests: {cpu: "2", memory: 4Gi}
                limits: {memory: 4Gi}
    workerGroupSpecs:
      - groupName: cpu-workers
        replicas: 1
        minReplicas: 1
        maxReplicas: 4
        rayStartParams: {}
        template:
          spec:
            containers:
              - name: ray-worker
                image: my-registry/warprec-serve:2.0.0
                envFrom:
                  - secretRef: {name: warprec-api-key}
                resources:
                  requests: {cpu: "4", memory: 8Gi}
                  limits: {cpu: "4", memory: 8Gi}
```

The exported file lists every key of every endpoint. It is shortened here.

**5. Apply it and wait** for the service to report it is ready:

```bash
kubectl apply -f rayservice.yaml
kubectl get rayservice warprec
kubectl describe rayservice warprec
```

**6. Reach it.** KubeRay creates the Service `warprec-serve-svc` on port 8000:

```bash
kubectl port-forward svc/warprec-serve-svc 8000:8000
curl localhost:8000/healthz
```

For traffic from outside the cluster, put an `Ingress` (or a `LoadBalancer` Service) in front of it:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: warprec
spec:
  ingressClassName: nginx
  rules:
    - http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: warprec-serve-svc
                port: {number: 8000}
```

### Scaling on KubeRay

Two autoscalers work together:

- **Ray Serve** adds and removes model replicas. Each endpoint sets its policy in `deployment.autoscaling_config` of `serve.yml`, as `sasrec` does above.
- **Ray's node autoscaler**, turned on by `enableInTreeAutoscaling`, adds worker pods when replicas cannot be scheduled, and removes idle ones, between each group's `minReplicas` and `maxReplicas`.

On a cloud cluster, let the Kubernetes cluster autoscaler add nodes for the new pods.

### GPUs on KubeRay

Add a GPU worker group and place the GPU endpoints on it with `device: cuda`:

```yaml
    workerGroupSpecs:
      - groupName: gpu-workers
        replicas: 1
        minReplicas: 0
        maxReplicas: 2
        rayStartParams: {}
        template:
          spec:
            containers:
              - name: ray-worker
                image: my-registry/warprec-serve:2.0.0-gpu    # built FROM rayproject/ray:2.58.0-py312-gpu
                envFrom:
                  - secretRef: {name: warprec-api-key}
                resources:
                  limits: {cpu: "8", memory: 32Gi, nvidia.com/gpu: "1"}
```

### Checkpoints on a Volume

To update models without rebuilding the image, keep the checkpoints on a `PersistentVolumeClaim`. Mount it at the same path in the head and in every worker group:

```yaml
          spec:
            containers:
              - name: ray-worker
                image: my-registry/warprec-serve:2.0.0
                volumeMounts:
                  - {name: models, mountPath: /models, readOnly: true}
            volumes:
              - name: models
                persistentVolumeClaim: {claimName: warprec-models, readOnly: true}
```

Changing the files on the volume does not restart anything. To serve the new checkpoints, change `serveConfigV2`, for example by pointing an endpoint at the new file, and KubeRay redeploys the application in place.

## KubeRay `RayCluster` with `serve deploy`

While iterating, a plain KubeRay `RayCluster`, without the `RayService` around it, can receive the application by hand. The flow is the same as on [virtual machines](#ray-cluster-on-virtual-machines), through the head's dashboard:

```bash
kubectl port-forward svc/<raycluster-name>-head-svc 8265:8265
serve deploy serve_app.yaml --address http://localhost:8265
```

Unlike a `RayService`, a `RayCluster` does not redeploy the application after the cluster restarts, so use it for development and keep the `RayService` for production.

## Connecting Agents

Every option above serves the same API. With `server.mcp: true` and the `mcp` extra, it also serves MCP tools at `/mcp`. An agent connects to whatever address the deployment exposes: `localhost:8000`, the container's published port, the `Ingress` host, or `warprec-serve-svc` from inside the cluster. See [Serving Models](../serving/index.md#mcp).
