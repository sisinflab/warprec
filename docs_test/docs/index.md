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

# WarpRec: Unifying Academic Rigor and Industrial Scale

**A high-performance, backend-agnostic framework for reproducible, scalable, and responsible Recommender Systems.**

Innovation in Recommender Systems is impeded by a fractured ecosystem. Researchers must choose between the ease of in-memory academic tools and the costly, complex rewriting required for distributed industrial engines. WarpRec eliminates this trade-off. Models defined in WarpRec transition seamlessly from local debugging to distributed training on Ray clusters, without changing a single line of code.

<div class="feature-grid">
  <div class="feature-card">
    <strong>55 Algorithms</strong>
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

## Get Started in Seconds

Define your entire experiment in a single YAML file:

```yaml
reader:
    loading_strategy: dataset
    reading_method: local
    local_path: data/movielens.tsv
    rating_type: implicit
writer:
    dataset_name: QuickStart
    writing_method: local
splitter:
    test_splitting:
        strategy: temporal_holdout
        ratio: 0.1
models:
    LightGCN:
        embedding_size: 64
        n_layers: 3
        reg_weight: 0.001
        batch_size: 2048
        epochs: 200
        learning_rate: 0.001
evaluation:
    top_k: [10, 20]
    metrics: [nDCG, Precision, Recall, HitRate]
```

Then run:

```bash
python -m warprec.run -c config.yml -p train
```

!!! note

    WarpRec supports three execution pipelines: **Training** (full HPO), **Design** (rapid prototyping), and **Evaluation** (pre-trained checkpoints). See [Quick Start](getting-started/quick-start.md) for detailed examples covering local, distributed, and agentic workflows.
