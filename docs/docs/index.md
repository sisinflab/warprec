<style>
  .feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
  }
  .feature-card {
    background: var(--md-code-bg-color);
    border: 1px solid var(--md-default-fg-color--lightest);
    border-radius: 0.5rem;
    padding: 1rem 1.2rem;
  }
  .feature-card strong {
    display: block;
    margin-bottom: 0.3rem;
  }
</style>

# Unifying Academic Rigor and Industrial Scale

**A high-performance, backend-agnostic framework for reproducible, scalable, and responsible Recommender Systems.**

Innovation in Recommender Systems is impeded by a fractured ecosystem. Researchers must choose between the ease of in-memory academic tools and the costly, complex rewriting required for distributed industrial engines. WarpRec eliminates this trade-off. Models defined in WarpRec transition seamlessly from local debugging to distributed training on Ray clusters, without changing a single line of code.

<div class="feature-grid">
  <div class="feature-card">
    <strong>65+ Algorithms</strong>
    From matrix factorization to graph-based and sequential architectures, spanning 6 model families.
  </div>
  <div class="feature-card">
    <strong>40 GPU-Accelerated Metrics</strong>
    Accuracy, rating, coverage, novelty, diversity, bias, fairness, and multi-objective evaluation.
  </div>
  <div class="feature-card">
    <strong>19 Data Strategies</strong>
    13 filtering and 6 splitting strategies for rigorous, leak-free experimental protocols.
  </div>
  <div class="feature-card">
    <strong>Backend-Agnostic</strong>
    Write once, run anywhere: Pandas, Polars, or Spark via the Narwhals abstraction layer.
  </div>
  <div class="feature-card">
    <strong>Distributed Training</strong>
    Scale from a laptop to a multi-node GPU cluster with Ray. Grid, Bayesian, and bandit-based HPO built in.
  </div>
  <div class="feature-card">
    <strong>Green AI</strong>
    First RS framework with native CodeCarbon integration for real-time energy and carbon tracking.
  </div>
  <div class="feature-card">
    <strong>Agentic AI Ready</strong>
    Native MCP server turns any trained model into a queryable tool for LLMs and autonomous agents.
  </div>
  <div class="feature-card">
    <strong>Statistical Rigor</strong>
    Automated hypothesis testing with Bonferroni, Holm-Bonferroni, and FDR corrections.
  </div>
</div>

## Quick Start

Define the complete experimental pipeline through a single YAML configuration file. To explore the full capabilities of the framework, refer to the reference test configuration located at `config/quick_start.yml`, which provides a minimal yet comprehensive example of the supported settings and components.

```yaml
reader:
  loading_strategy: dataset
  data_type: transaction
  reading_method: local
  local_path: path/to/my/dataset.csv
  rating_type: explicit
  sep: ','
  labels:
    user_id_label: user_id
    item_id_label: item_id
    rating_label: rating
    timestamp_label: timestamp
writer:
  dataset_name: MyExperiment
  writing_method: local
  local_experiment_path: experiments/test/
splitter:
  test_splitting:
    strategy: temporal_holdout
    ratio: 0.1
models:
  ItemKNN:
    k: 100
    similarity: cosine
evaluation:
  top_k: [10, 20, 50]
  metrics: [nDCG, Precision, Recall, HitRate]
  strategy: sampled
  num_negatives: 99
```

Start your local instance of Ray:

```bash
ray start --head
```

Then run the following command:

```bash
python -m warprec.run -c config/quick_start.yml -p train
```

!!! note

    WarpRec supports three execution pipelines: **Training** (full HPO), **Design** (rapid prototyping), and **Evaluation** (pre-trained checkpoints). See [Quick Start](get-started/quick-start.md) for detailed examples covering local, distributed, and agentic workflows.
