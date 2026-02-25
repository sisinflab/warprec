# WarpRec: Unifying Academic Rigor and Industrial Scale for Responsible, Reproducible, and Efficient Recommendation

**Marco Avolio\***
Wideverse, Italy

**Potito Aghilar\***
Politecnico di Bari, Italy
Wideverse, Italy

**Sabino Roccotelli\***
Politecnico di Bari, Italy
Wideverse, Italy

**Vito Walter Anelli\***
Politecnico di Bari, Italy

**Chiara Mallamaci**
Politecnico di Bari, Italy

**Vincenzo Paparella**
ISTI-CNR, Pisa, Italy

**Marco Valentini**
Politecnico di Bari, Italy

**Alejandro Bellogín**
UAM, Madrid, Spain

**Michelantonio Trizio**
Wideverse, Italy

**Joseph Trotta**
OVS, Italy

**Antonio Ferrara**
Politecnico di Bari, Italy

**Tommaso Di Noia**
Politecnico di Bari, Italy

*\*These authors contributed equally to this research.*

## Abstract
Innovation in Recommender Systems is currently impeded by a fractured ecosystem, where researchers must choose between the ease of in-memory experimentation and the costly, complex rewriting required for distributed industrial engines. To bridge this gap, we present WarpRec, a high-performance framework that eliminates this trade-off through a novel, backend-agnostic architecture. It includes 50+ state-of-the-art algorithms, 40 metrics, and 19 filtering and splitting strategies that seamlessly transition from local execution to distributed training and optimization. The framework enforces ecological responsibility by integrating CodeCarbon for real-time energy tracking, showing that scalability need not come at the cost of scientific integrity or sustainability. Furthermore, WarpRec anticipates the shift toward Agentic AI, leading Recommender Systems to evolve from static ranking engines into interactive tools within the Generative AI ecosystem. In summary, WarpRec not only bridges the gap between academia and industry but also can serve as the architectural backbone for the next generation of sustainable, agent-ready Recommender Systems. Code is available at https://github.com/sisinflab/warprec/

## CCS Concepts
• **Information systems** → **Recommender systems.**

## Keywords
Evaluation, Agentic AI, Green AI, Reproducibility, Scalability

---

## 1 Introduction
The dual role of Recommender Systems (RS) as scalable industrial tools and rigorous scientific instruments underscores their significance across both commercial and academic sectors. Hence, the modern RS pipeline is expected to reconcile conflicting imperatives: it must satisfy the massive scalability requirements of industrial applications while adhering to the scientific rigor and ecological responsibility demanded by the research community [10]. Furthermore, with the rise of Agentic AI [85, 97, 98], recommenders are increasingly tasked with serving as interactive tools for Large Language Models (LLMs) via standardized interfaces such as the Model Context Protocol[^1]. However, the advancement of the field is currently stifled by a fundamental architectural dichotomy between academia and industry. Researchers typically rely on flexible, in-memory libraries (e.g., RecBole [102], Elliot [5]) designed for rapid prototyping. While excellent for algorithmic exploration, these tools rely heavily on eager execution engines (e.g., Pandas) that fail to scale beyond single-node memory limits. Conversely, industrial practitioners utilize distributed frameworks (e.g., NVIDIA Merlin [83]) optimized for throughput. These systems, while scalable, often lack the flexible evaluation protocols, statistical rigor, and ease of customization required for scientific inquiry. This divide creates a “Deployment Chasm”: novel algorithms developed in academia require a complete, costly rewrite to function in production environments [4, 9].

We present **WarpRec**, a high-performance framework that eliminates these trade-offs. It introduces a backend-agnostic architecture built upon Narwhals[^2], enabling a “write-once, run-anywhere” paradigm. Models defined in WarpRec transition seamlessly from local debugging to distributed training on Ray clusters [49]. The framework is engineered with extreme modularity; it is composed of five decoupled engines that function as independent, composable units. This allows researchers to inject custom logic, swap data backends, or utilize the framework’s evaluation suite (comprising 40 metrics and 19 filtering/splitting strategies) within external pipelines. WarpRec supports 55 state-of-the-art algorithms, ranging from matrix factorization to the latest graph-based and sequential architectures, all capable of running locally or at cluster scale. Beyond unifying scale, WarpRec addresses these functional gaps:

*   **Green AI & Sustainability:** It is the first framework to enforce ecological accountability by integrating CodeCarbon[^3] for real-time energy tracking and Green AI profiling.
*   **Agentic Readiness:** Anticipating the shift toward autonomous systems, WarpRec natively implements the Model Context Protocol server interface. This transforms the recommender from a static predictor into a queryable agent capable of complex reasoning tasks within an LLM loop.
*   **Scientific Rigor:** It automates reproducibility, statistical tests and corrections, e.g., Bonferroni [23] and FDR [12].

The remainder of this paper details WarpRec’s architecture, validates its performance through extensive benchmarks against state-of-the-art frameworks, and demonstrates its readiness for the next generation of AI-driven recommendation.

[^1]: https://modelcontextprotocol.io/
[^2]: https://narwhals-dev.github.io/narwhals/
[^3]: https://mlco2.github.io/codecarbon/

---

**Figure 1: The modular architecture of WarpRec.** Five decoupled modules manage the end-to-end recommendation lifecycle, from data ingestion and processing to model training and evaluation. An Application Layer exposes the recommender through a REST API and MCP agentic interface.

---

## 2 The Fragmented Landscape
The landscape of recommender frameworks is fractured into distinct silos, none capable of satisfying the demands of modern research.

**The Academic Silo.** Academic frameworks have evolved from early tools for classic (MyMediaLite [27], LensKit [24], Surprise [38]) and deep learning models (LibRec [30], Spotlight [41], OpenRec [89]) into comprehensive libraries (RecBole [102], Cornac [63]) offering hundreds of models and frameworks that prioritize reproducibility [20, 58] and scientific rigor through automated experimental lifecycles (Elliot [5], DaisyRec [75]). Despite this progress, they remain confined to a single-node paradigm, that prevents industrial scaling and necessitates manual infrastructure for distributed hyperparameter optimization (HPO)[90] on engines like Ray [49] or Spark [95] and for using strategies such as ASHA [43]. Finally, even rigorous frameworks like Elliot often omit multiple hypothesis testing corrections, leaving them vulnerable to p-hacking [78].

**The Industrial Silo.** Industrial-grade frameworks such as NVIDIA Merlin [54], Apache Spark MLlib [48, 82], or MS Recommenders [28] are architected to address the extreme computational demands of production-scale environments. These systems overcome the limitations of academic tools by employing distributed dataframes and GPU offloading to facilitate the training of complex architectures on billions of data points [18, 61]. However, these tools often prioritize *serving* over *science*, lacking fine-grained experimental control. Industrial frameworks usually provide limited data-splitting and evaluation metrics, typically excluding significance testing and beyond-accuracy dimensions such as bias, fairness, and diversity [39].

**Green AI.** The field has pivoted from accuracy-maximizing “Red AI” to “Green AI”, elevating energy efficiency and carbon transparency to first-class metrics alongside traditional performance [10, 67]. This is particularly critical in Recommender Systems, where the combination of billion-parameter embedding tables and exhaustive hyperparameter optimization grids leads to extreme energy consumption [68, 81]. Recent works have shown that marginal gains in performance often require exponential increases in carbon emissions [8, 68, 69]. Despite the critical importance of sustainability, current RS frameworks remain fundamentally energy-blind. Leading platforms [5, 63, 102] provide no native mechanisms, such as CodeCarbon[^3] or CarbonTracker [7], to quantify carbon emissions or power consumption within the experimental pipeline.

**Agentic Interoperability in Recommendation.** Artificial intelligence is shifting from monolithic models to agentic workflows [98], where agents call external tools [65] and interleave reasoning with actions [92]. In this paradigm, the recommender becomes a callable tool within an agent’s decision-making process [53]. This new role requires interactive dialogue to iteratively refine results [85, 97], yet current frameworks lack the standardized interfaces to enable this. WarpRec addresses this gap by natively implementing the Model Context Protocol, enabling standardized agent-ready interaction.

## 3 WarpRec
The architecture of WarpRec, illustrated in Figure 1, is designed with principles of modularity and separation of concerns, ensuring each phase of the recommendation lifecycle is independent and extensible. Table 1 compares its features against established frameworks.

### 3.1 Pipelines and Callbacks
To streamline the research lifecycle, WarpRec abstracts complex workflows into three standardized execution pipelines:
*   **Training Pipeline**, which orchestrates the full end-to-end process, automating hyperparameter optimization (HPO);
*   **Design Pipeline**, optimized for rapid prototyping and architectural validation without HPO overhead;
*   **Evaluation Pipeline**, dedicated to post-hoc analysis and inference using pre-trained checkpoints.

---

**Table 1: Comprehensive comparison of WarpRec against frameworks identified in the literature (with year of the last update) across six dimensions: Data, Optimization & Training, Extensions & Integration, Rigor & Reproducibility, Green AI, Metrics, and Statistics.**

| Framework | Backend (Eager/Fast/Distrib/Cloud) | Optim. & Train. (Multi-GPU/Cluster/Search/Pruning) | Ext. & Int. (Callbacks/Agentic/REST) | Rigor & Repro. (Tests/Corr/Seed/Checkpt) | Green AI (Carbon/Energy) | Metrics (Bias/Fair/Nov/Multi/GPU) | Stats (# Models/Split/Metrics) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Classic** | | | | | | | |
| LibRec (2022) [30] | ✓ / ✓ / - / - | - / - / ✓ / - | - / - / - | - / - / ✓ / - | - / - | 0 / 0 / 0 / 0 / - | 55 / 10 / 11 |
| MyMediaLite (2020) [27] | - / - / - / - | - / - / ✓ / - | - / - / - | - / - / ✓ / - | - / - | 0 / 0 / 0 / 0 / - | 18 / 6 / 8 |
| Surprise (2024) [38] | - / - / - / ✓ | - / - / ✓ / - | - / - / - | - / - / ✓ / - | - / - | 0 / 0 / 0 / 0 / - | 7 / 6 / 4 |
| **Research** | | | | | | | |
| LensKit (2026) [24] | ✓ / ✓ / - / - | ✓ / ✓ / ✓ / - | - / - / - | - / - / ✓ / ✓ | - / - | 2 / 0 / 3 / 0 / - | 12 / 5 / 11 |
| RecBole (2025) [102] | ✓ / - / - / - | ✓ / - / ✓ / ✓ | - / - / - | - / - / ✓ / ✓ | - / - | 2 / 0 / 3 / 0 / ✓ | 159 / 8 / 16 |
| DaisyRec (2024) [75] | ✓ / - / - / - | - / - / ✓ / - | - / - / - | - / - / ✓ / - | - / - | 1 / 0 / 2 / 0 / - | 13 / 7 / 11 |
| Spotlight (2020) [41] | - / - / - / - | - / - / ✓ / - | - / - / - | - / - / - / - | - / - | 0 / 0 / 0 / 0 / - | 2 / 2 / 4 |
| Elliot (2023) [5] | ✓ / - / - / - | - / - / ✓ / - | - / - / - | ✓ / - / ✓ / - | - / - | 5 / 9 / 6 / 0 / - | 77 / 13 / 38 |
| **Specialized** | | | | | | | |
| Cornac (2025) [63] | ✓ / - / - / - | - / - / ✓ / - | - / - / - | - / - / ✓ / - | - / - | 0 / 0 / 0 / 0 / - | 58 / 4 / 12 |
| FuxiCTR (2025) [104] | ✓ / ✓ / - / - | ✓ / - / ✓ / - | - / - / - | - / - / ✓ / ✓ | - / - | 0 / 0 / 0 / 0 / ✓ | 51 / 2 / 4 |
| ClayRS (2024) [47] | ✓ / - / - / - | - / - / ✓ / - | - / - / - | ✓ / ✓ / ✓ / - | - / - | 1 / 1 / 3 / 0 / - | 17 / 3 / 15 |
| Informfully (2026) [36] | ✓ / - / - / - | - / - / ✓ / - | - / - / ✓ | - / - / ✓ / - | - / - | 0 / 0 / 10 / 0 / - | 10 / 5 / 22 |
| RecList (2023) [19] | ✓ / - / - / ✓ | - / - / - / - | ✓ / - / - | - / - / ✓ / - | - / - | 1 / 3 / 1 / 0 / - | - / - / 15 |
| **Industry Scale** | | | | | | | |
| MS Recommenders (2026) [28] | ✓ / - / ✓ / ✓ | ✓ / ✓ / ✓ / ✓ | ✓ / - / ✓ | - / - / ✓ / ✓ | - / - | 0 / 1 / 4 / 0 / - | 15 / 3 / 15 |
| Merlin (2024) [83] | ✓ / ✓ / ✓ / ✓ | ✓ / ✓ / ✓ / ✓ | ✓ / - / ✓ | - / - / ✓ / ✓ | - / - | 0 / 0 / 0 / 0 / ✓ | 8 / 8 / 5 |
| **WarpRec** | **✓ / ✓ / ✓ / ✓** | **✓ / ✓ / ✓ / ✓** | **✓ / ✓ / ✓** | **✓ / ✓ / ✓ / ✓** | **✓ / ✓** | **5 / 10 / 6 / 2 / ✓** | **55 / 19 / 40** |

---

These workflows are controlled via declarative configuration files, avoiding boilerplate orchestration code. This high degree of modularity is specifically intended for industrial applications and advanced custom workflows. Rather than being restricted to the predefined pipelines, practitioners can extract and integrate individual modules into existing production ecosystems. The framework further integrates an event-driven **Callback** system allowing custom hooks at specific stages, facilitating the injection of custom logic and enabling complex experiments without modifying the core pipeline.

### 3.2 Reader Module
The *Reader Module* is designed to efficiently ingest user-item interactions and metadata, abstracting the complexity of data retrieval. The module leverages Narwhals[^2] as a backend-agnostic compatibility layer. Instead of enforcing a specific data structure, this design allows to abstract the data loading process, uniforming the data access interface without incurring costly conversion overheads. Furthermore, the module decouples logical data access from physical storage. It supports seamless ingestion from both local filesystems and cloud-based object storage. This architectural choice ensures that the framework remains environment-agnostic, enabling researchers to transition effortlessly from local prototyping to large-scale cloud experimentation without modifying their ingestion pipelines.

### 3.3 Data Engine
The *Data Engine* module is responsible for transforming raw transactions into refined, model-ready structures. To ensure experimental rigor and flexibility, this module is composed of three specialized components: (i) Filtering, (ii) Splitting, and (iii) Dataset Management.

**Filtering.** The *Filtering* component includes 13 distinct strategies, organized into three functional families [5]: (i) *filter-by-rating* strategies, which mitigate noise by pruning interactions based on global or distributional value thresholds; (ii) *k-core* decomposition, which iteratively removes users and items with fewer than $k$ connections; and (iii) *cold-user/item* heuristics, which target the cold-start problem by discarding entities that fail to meet minimum interaction constraints.

**Splitter.** To prevent data leakage and ensure rigorous evaluation [29], the *Splitter* component partitions the filtered dataset using 6 distinct strategies. WarpRec supports both *random* and *temporal* configurations for *Hold-Out* and *Leave-k-Out* protocols, *Fixed Timestamp* splitting and *k-fold Cross-Validation*. The module enforces system-wide determinism by anchoring all stochastic partitioning to global random seeds, guaranteeing experimental reproducibility.

**Dataset.** This component orchestrates the transformation of raw partitions into high-performance internal representations. It unifies *Interaction*, *Session*, and *Context Management* to standardize sparse user-item signals, variable-length behavioral sequences, and auxiliary metadata. The *Dataset* enforces strict *Data Alignment* by mapping external identifiers to contiguous internal indices, ensuring referential integrity before materializing optimized *Training and Evaluation Structures* designed for high-throughput ingestion.

### 3.4 Recommendation Engine
This module governs the lifecycle and training of diverse model architectures, enabling training, hyperparameter search, and resource management. It includes two components: (i) Models and (ii) Trainer.

**Table 2: Statistical summary of the experimental datasets.**

| Dataset | # Users | # Items | # Interactions | Sparsity |
| :--- | :--- | :--- | :--- | :--- |
| MovieLens-1M | 6,040 | 3,883 | 1,000,209 | 95.7353% |
| MovieLens-32M | 200,948 | 87,585 | 32,000,204 | 99.8182% |
| NetflixPrize-100M | 480,189 | 17,770 | 100,480,507 | 98.8281% |

**Models.** This component encapsulates recommendation algorithms as decoupled entities, independent of data ingestion and evaluation protocols. In its current release, the framework provides a robust repository of 55 built-in algorithms spanning 6 fundamental classes [60]: (i) *Unpersonalized*, (ii) *Content-Based*, (iii) *Collaborative Filtering* (CF), (iv) *Context-Aware*, (v) *Sequential*, and (vi) *Hybrid* models. Within the CF family, the framework covers several dominant paradigms, including: Autoencoders (e.g., EASE$^R$ [71], MultiVAE [45]); Graph-based architectures (e.g., LightGCN [34], DGCF [84]); K-Nearest Neighbors (e.g., UserKNN [59], ItemKNN [64]); Latent Factor models (e.g., BPRMF [57], SLIM [51]); and Neural recommenders (e.g., NeuMF [35], ConvNCF [22]). To handle heterogeneous metadata, WarpRec incorporates robust factorization-based baselines–such as FM [56], AFM [88], and NFM [33]–alongside deep architectures designed for high-dimensional sparse and dense features, including Wide&Deep [18], DeepFM [31], and xDeepFM [44]. The Sequential suite is equally comprehensive, encompassing CNN-based (Caser [76]), Markov-Chain (FOSSIL [32]), RNN (GRU4Rec [37]), and models based on Transformer architecture (BERT4Rec [73]). To enable modern research, WarpRec integrates a wide range of state-of-the-art models, including LightCCF [101], EGCF [100], ESIGCF [91], MixRec [99], LightGCN++[42], gSASRec[55], and LinRec [46].

**Trainer.** This component serves as the framework’s core execution engine, orchestrating model optimization and state persistence via automated checkpointing to enable seamless experiment resumption. Designed to bridge the gap between academic research and production MLOps, it supports distributed computing across multi-GPU clusters while incorporating comprehensive experiment tracking hooks for rigorous monitoring.

*Execution Environments and Distributed Training.* WarpRec ensures strict code portability from local prototyping to industrial deployment through seamless vertical and horizontal scaling. The framework supports execution from single-node to multi-GPU configurations, leveraging Ray for multi-node orchestration [49]. This integration enables elastic scaling across cloud infrastructures, optimizing resource allocation and reducing computational costs.

*Hyperparameter Tuning.* To automate hyperparameter discovery, the module integrates robust HPO strategies ranging from Grid and Random search to Bayesian and bandit-based optimization (e.g., *HyperOpt* [14], *Optuna* [3], *BoHB* [25]) over heterogeneous search spaces. Computational efficiency is maximized via a dual-layer stopping mechanism: system-level pruning using the *ASHA* scheduler [43] and model-level early stopping based on convergence metrics.

*Dashboarding.* To facilitate real-time observability, the framework integrates TensorBoard, Weights & Biases, and MLflow [94]. In line with Green AI principles [87], CodeCarbon[^3] was adopted to automatically quantify energy consumption and carbon emissions.

### 3.5 Evaluation
To guarantee scientific reproducibility, the *Evaluation* module provides a suite of multidimensional metrics while supporting statistical hypothesis testing and error correction.

**Metrics.** WarpRec integrates a comprehensive suite of 40 metrics organized into distinct functional families: *Accuracy* [66, 103], *Rating*, *Coverage*, *Novelty* [80], *Diversity* [96], *Bias* [1, 2, 77, 93, 106], and *Fairness* [21, 105]. As highlighted in Table 1, WarpRec distinguishes itself as the sole framework to natively support *Multi-objective* metrics [52, 107]. This capability advances the state-of-the-art by enabling model selection through the optimization of competing goals (e.g., balancing accuracy and popularity bias). To ensure scalability, the metric computation is fully GPU-accelerated, drastically reducing the latency of large-scale experimental loops.

**Statistical Hypothesis Testing.** To guarantee methodological rigor [26, 62], WarpRec integrates a comprehensive hypothesis testing suite into the evaluation pipeline. The framework automates significance testing, supporting both paired comparisons (e.g., Student’s t-test [72], Wilcoxon signed-rank [86]) and independent-group analyses (e.g., Mann-Whitney U [50]). WarpRec mitigates the Multiple Comparison Problem [17] through Bonferroni [16] and FDR [13] corrections for Type I errors, ensuring that performance gains are statistically robust.

### 3.6 Writer Module
The *Writer Module* ensures reproducibility and streamlined monitoring via a storage-agnostic interface that persists artifacts to local or cloud backends. Beyond standard performance tables and statistical significance tests, WarpRec automatically serializes granular per-user metrics, optimized hyperparameters, and trained model weights. Furthermore, the framework logs recommendation lists, execution metadata, execution times, and carbon-emission estimates, supporting both rigorous experimentation and sustainable AI research.

### 3.7 Application Layer
Finally, WarpRec bridges the gap between rapid experimentation and production deployment through a strictly modular architecture. Unlike monolithic frameworks, WarpRec decouples the modeling core from the training infrastructure, allowing trained models to be isolated as standalone artifacts with zero additional engineering effort. This design empowers a versatile model serving layer where specific recommenders can be instantly exposed via: (i) RESTful APIs for high-throughput, real-time inference in standard microservices; and (ii) an MCP server, enabling Large Language Models and autonomous agents to dynamically query the recommender as a tool.

## 4 Performance and Multi-faceted Analysis
We show that WarpRec bridges the gap between academic rigor and industrial scalability by answering the following research questions:
*   **RQ1: Scalability and Performance.** How do WarpRec’s architectural optimizations impact performance when scaling from academic prototyping to massive industrial datasets?
*   **RQ2: Green AI Analysis.** Can WarpRec effectively quantify the carbon footprint in large-scale recommendation workflows?
*   **RQ3: Agentic AI.** Does WarpRec provide the modularity and abstraction to serve as a backend for autonomous agentic systems?

---

**Table 3: Computational resource requirements (CPUs / GPUs / RAM in GB) by dataset and execution mode. P: Parallel; S: Serial.**

| Data | Mode | EASE$^R$ | ItemKNN | LightGCN | NeuMF | SASRec |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MovieLens-1M | S | 16/1/64 | – | 16/1/64 | 16/1/64 | 16/1/64 |
| | P | – | – | 32/1/64 | 32/1/64 | 32/2/64 |
| MovieLens-32M | S | 16/1/256 | – | 16/1/128 | 16/1/128 | 16/1/128 |
| | P | – | – | 32/4/128 | 32/4/128 | 32/4/128 |
| NetflixPrize-100M | S | 16/1/192 | 16/1/192 | 16/1/192 | 16/1/192 | 16/1/192 |
| | P | – | – | 32/4/192 | 32/4/192 | 32/4/192 |

---

### 4.1 Experimental Setup
To rigorously evaluate the scalability and efficiency of WarpRec, we designed a comprehensive experimental protocol spanning a diverse suite of recommendation frameworks, datasets of varying magnitude, and strict reproducibility standards.

**Baselines.** We consider frameworks that provide experimentation pipelines out-of-the-box, eliminating error-prone custom implementations (e.g., NVIDIA Merlin). Accordingly, we compare against 5 leading frameworks, such as Cornac [63], DaisyRec [74], Elliot [5], and RecBole [102] for academic benchmarking, and Microsoft Recommenders [28] for enterprise-scale deployment.

**Models.** We benchmark five representative algorithms that span different learning paradigms, including EASE$^R$ [71], NeuMF [35], LightGCN [34], SASRec [40] and ItemKNN [64] (included as a reference baseline for environmental impact comparisons within our analysis).

**Datasets.** Our benchmarking spans three scaling datasets, from academic prototyping to industrial deployment size: MovieLens-1M, MovieLens-32M[^4], and NetflixPrize-100M [11]. The dataset statistics are summarized in Table 2. We limited the dataset scale to permit competitor execution success, as WarpRec’s capacity extends further.

**Splitting Strategy.** We employ 90-10 random holdout for EASE$^R$, LightGCN, NeuMF, ItemKNN and temporal holdout for SASRec.

**Optimization Protocol.** Models are tuned via exhaustive grid search of 6 configurations per model from established literature [6, 15] (details on GitHub[^5]). We train for 10 epochs on MovieLens and 2 on Netflix-100M. Best models are selected via validation nDCG@10, using a 8,192 batch size and a 24h timeout per trial. Given the memory requirement of EASE$^R$ it is only considered for serial execution.

**Hardware and Resources.** Experiments were conducted on 16-core CPUs and 64GB NVIDIA A100 GPUs. We compare *serial* (sequential) and *parallel* (concurrent via Ray) execution modes to evaluate scalability. Resource allocations are detailed in Table 3.

**Evaluation Metrics.** We evaluate WarpRec’s efficiency, considering $C$ concurrent trials, by measuring: (i) Preprocessing time, including ingestion and splitting; (ii) Training and (iii) Evaluation times, measured as the average per-trial optimization and validation; (iv) HPO duration, the total wall-clock time for hyperparameter exploration[^6]; and the total time aggregates preprocessing and HPO. Finally, we report nDCG@10 with full ranking evaluation to ensure that speed-ups do not compromise algorithmic fidelity.

**Reproducibility and Carbon Footprint.** Experiments are tracked via Weights & Biases using fixed seed, while CodeCarbon monitors energy and emissions. All code and configurations are on GitHub[^5].

[^4]: Both available at https://grouplens.org/datasets/movielens/
[^5]: https://github.com/sisinflab/warprec-benchmark-2026/
[^6]: WarpRec can perform per-epoch evaluation, but we excluded it since others do not.

---

**Table 4: End-to-end performance and scalability benchmark. WarpRec is compared against leading frameworks on datasets of increasing scale; † denotes its serial execution without parallelization. Best results are reported in bold, second best are underlined.**

**NetflixPrize-100M (Medium Scale)**

| Model | Fram. | C | Prep. | Train | Eval | HPO | Total | nDCG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EASE$^R$** | Cornac | 1 | 7m 58s | **2m 35s** | 29m 45s | <u>3h 14m</u> | <u>3h 22m</u> | 0.3143 |
| | Elliot | 1 | 2m 39s | <u>2m 54s</u> | 5h 17m | *Time Limit Exceeded (4/6)* | | |
| | RecBole | 1 | 11m 17s | 26m 29s | **20m 21s** | 4h 41m | 5h 49m | 0.3283 |
| | DaisyRec | 1 | <u>1m 35s</u> | 1h 17m | *Out of Memory - RAM* | | | |
| | **WarpRec**$^\dagger$ | 1 | **30.25s** | 3m 9s | <u>21m 13s</u> | **2h 26m** | **2h 26m** | 0.3035 |
| **NeuMF** | Cornac | 1 | 7m 33s | 19m 55s | 16m 24s | 3h 38m | 3h 45m | 0.1267 |
| | Elliot | 1 | *Time Limit Exceeded (0/6)* | | | | | |
| | RecBole | 1 | 11m 13s | 1h 23m | 36m 57s | 11h 59m | 13h 6m | 0.1080 |
| | DaisyRec | 1 | 1m 36s | 2h 39m | *Out of Memory - VRAM* | | | |
| | MS Rec. | 1 | 46m 13s | *Time Limit Exceeded (0/6)* | | | | |
| | **WarpRec**$^\dagger$ | 1 | <u>29.95s</u> | **9m 13s** | <u>2m 30s</u> | <u>1h 10m</u> | <u>1h 10m</u> | 0.1323 |
| | **WarpRec** | 6 | **21.87s** | <u>19m 34s</u> | **2m 29s** | **28m 0s** | **28m 22s** | 0.1323 |
| **LightGCN** | Cornac | 1 | 7m 46s | 10h 4m | 12m 26s | *Time Limit Exceeded (2/6)* | | |
| | Elliot | 1 | 2m 33s | 15h 53m | 42.13s | *Time Limit Exceeded (1/6)* | | |
| | RecBole | 1 | 11m 33s | **1h 27m** | 4m 19s | **9h 7m** | **10h 17m** | 0.0247 |
| | DaisyRec | 1 | 1m 55s | 14h 21m | 1m 7s | *Time Limit Exceeded (1/6)* | | |
| | MS Rec. | 1 | 4m 36s | *Out of Memory - VRAM* | | | | |
| | **WarpRec**$^\dagger$ | 1 | <u>29.53s</u> | <u>4h 53m</u> | **39.15s** | *Time Limit Exceeded (2/6)* | | |
| | **WarpRec** | 6 | **26.38s** | 14h 35m | <u>40.81s</u> | <u>22h 4m</u> | <u>22h 5m</u> | 0.1779 |
| **SASRec** | RecBole | 1 | 42m 37s | 2h 37m | 2m 59s | 15h 59m | 20h 15m | 0.0515 |
| | **WarpRec**$^\dagger$ | 1 | **21.19s** | **36m 32s** | <u>51.85s</u> | <u>3h 44m</u> | <u>3h 44m</u> | 0.1204 |
| | **WarpRec** | 6 | <u>24.86s</u> | <u>1h 12m</u> | **51.68s** | **1h 56m** | **1h 56m** | 0.1204 |

**MovieLens-32M (Small Scale)**

| Model | Fram. | C | Prep. | Train | Eval | HPO | Total | nDCG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EASE$^R$** | Cornac | 1 | 1m 58s | **26m 46s** | *Out of Memory - RAM* | | | |
| | Elliot | 1 | 50.71s | 34m 8s | 10h 12m | *Time Limit Exceeded (2/6)* | | |
| | RecBole | 1 | 3m 19s | 51m 0s | **17m 28s** | <u>6h 51m</u> | <u>7h 10m</u> | 0.3609 |
| | DaisyRec | 1 | <u>24.93s</u> | 3h 9m | *Out of Memory - RAM* | | | |
| | **WarpRec**$^\dagger$ | 1 | **9.343s** | <u>29m 2s</u> | <u>32m 14s</u> | **6h 7m** | **6h 7m** | 0.3399 |
| **NeuMF** | Cornac | 1 | 2m 9s | <u>27m 13s</u> | 21m 59s | 4h 55m | 4h 57m | 0.1264 |
| | Elliot | 1 | 16h 9m | 2h 26m | 26m 19s | *Time Limit Exceeded (3/6)* | | |
| | RecBole | 1 | 3m 29s | 10h 51m | 1h 3m | *Time Limit Exceeded (2/6)* | | |
| | DaisyRec | 1 | 25.20s | 1h 44m | *Out of Memory - VRAM* | | | |
| | MS Rec. | 1 | 2h 6m | *Time Limit Exceeded (0/6)* | | | | |
| | **WarpRec**$^\dagger$ | 1 | <u>9.70s</u> | **13m 0s** | **4m 55s** | <u>1h 47m</u> | <u>1h 47m</u> | 0.1456 |
| | **WarpRec** | 6 | **9.03s** | 58m 21s | <u>4m 57s</u> | **1h 34m** | **1h 34m** | 0.1456 |
| **LightGCN** | Cornac | 1 | 1m 58s | 5h 59m | 20m 32s | *Time Limit Exceeded (3/6)* | | |
| | Elliot | 1 | 49.73s | 7h 37m | **15.45s** | *Time Limit Exceeded (2/6)* | | |
| | RecBole | 1 | 3m 11s | **29m 53s** | 1m 51s | **3h 10m** | **3h 29m** | 0.0237 |
| | DaisyRec | 1 | 26.07s | 6h 21m | 1m 17s | *Time Limit Exceeded (3/6)* | | |
| | MS Rec. | 1 | 1m 16s | *Out of Memory - VRAM* | | | | |
| | **WarpRec**$^\dagger$ | 1 | <u>10.34s</u> | <u>3h 47m</u> | 39.54s | <u>22h 46m</u> | <u>22h 46m</u> | 0.2061 |
| | **WarpRec** | 6 | **9.14s** | 7h 16m | <u>38.94s</u> | <u>10h 7m</u> | <u>10h 7m</u> | 0.2061 |
| **SASRec** | RecBole | 1 | 12m 48s | 4h 11m | 1m 6s | *Time Limit Exceeded (5/6)* | | |
| | **WarpRec**$^\dagger$ | 1 | <u>8.18s</u> | **1h 0m** | <u>40.24s</u> | <u>6h 4m</u> | <u>6h 4m</u> | 0.0889 |
| | **WarpRec** | 6 | **7.91s** | <u>1h 45m</u> | **39.94s** | **2h 36m** | **2h 36m** | 0.0889 |

**MovieLens-1M (Tiny Scale)**

| Model | Fram. | C | Prep. | Train | Eval | HPO | Total | nDCG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EASE$^R$** | Cornac | 1 | 2.97s | **1.31s** | 7.21s | <u>51.15s</u> | <u>54.12s</u> | 0.2736 |
| | Elliot | 1 | 2.14s | <u>1.44s</u> | 13.60s | 1m 30s | 1m 32s | 0.2920 |
| | RecBole | 1 | 5.39s | 7.38s | <u>3.76s</u> | 1m 7s | 1m 39s | 0.3221 |
| | DaisyRec | 1 | <u>0.59s</u> | 46.85s | 5.25s | 5m 12s | 5m 13s | 0.0888 |
| | **WarpRec**$^\dagger$ | 1 | **0.33s** | 2.32s | **2.77s** | **30.52s** | **30.85s** | 0.2850 |
| **NeuMF** | Cornac | 1 | 2.90s | 47.40s | 7.51s | 5m 29s | 5m 32s | 0.1927 |
| | Elliot | 1 | 14.70s | 1m 0s | 2.13s | 6m 14s | 6m 29s | 0.2209 |
| | RecBole | 1 | 6.41s | 2m 17s | 10.27s | 14m 47s | 15m 25s | 0.1664 |
| | DaisyRec | 1 | **0.55s** | 1m 10s | **0.66s** | 7m 6s | 7m 7s | 0.0030 |
| | MS Rec. | 1 | 25.97s | 3m 11s | 2m 43s | 35m 25s | 35m 51s | 0.2630 |
| | **WarpRec**$^\dagger$ | 1 | <u>0.94s</u> | **18.93s** | <u>1.26s</u> | <u>2m 1s</u> | <u>2m 2s</u> | 0.1943 |
| | **WarpRec** | 6 | 0.97s | 32.70s | 1.25s | **52.48s** | **53.45s** | 0.1943 |
| **LightGCN** | Cornac | 1 | 2.93s | 2m 36s | 4.67s | 16m 6s | 16m 9s | 0.2482 |
| | Elliot | 1 | 1.69s | 1m 43s | **0.36s** | 10m 24s | 10m 26s | 0.2269 |
| | RecBole | 1 | 5.29s | 44.21s | 2.81s | 4m 42s | 5m 13s | 0.0216 |
| | DaisyRec | 1 | **0.51s** | 1m 16s | <u>0.46s</u> | 7m 40s | 7m 41s | 0.0512 |
| | MS Rec. | 1 | 1.40s | 1m 0s | 1.74s | 7m 53s | 7m 54s | 0.1860 |
| | **WarpRec**$^\dagger$ | 1 | <u>0.95s</u> | **22.26s** | 0.87s | <u>2m 19s</u> | <u>2m 20s</u> | 0.2122 |
| | **WarpRec** | 6 | 1.02s | 1m 26s | 0.82s | **1m 41s** | **1m 42s** | 0.2122 |
| **SASRec** | RecBole | 1 | 23.16s | 7m 3s | 1.72s | 42m 34s | 44m 53s | 0.0689 |
| | **WarpRec**$^\dagger$ | 1 | 0.99s | **1m 55s** | **1.01s** | <u>11m 36s</u> | <u>11m 37s</u> | 0.0649 |
| | **WarpRec** | 6 | **0.95s** | <u>4m 49s</u> | <u>1.12s</u> | **6m 26s** | **6m 27s** | 0.0649 |

---

**Table 5: Green AI Profiling of WarpRec on NetflixPrize-100M, captured via CodeCarbon. Best results are reported in bold.**

| Metric | ItemKNN | EASE$^R$ | NeuMF | LightGCN | SASRec |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Emissions | **0.0002** | 0.0005 | 0.0004 | 0.0095 | 0.0012 |
| Emissions Rate | **2.96e$^{-7}$** | 3.09e$^{-7}$ | 4.97e$^{-7}$ | 4.00e$^{-7}$ | 5.33e$^{-7}$ |
| CPU Power | **137.8144** | 155.3458 | 219.4410 | 154.1137 | 152.7022 |
| GPU Power | 75.9096 | **71.7731** | 177.2569 | 157.3857 | 278.5554 |
| CPU Energy | **0.0248** | 0.0638 | 0.0530 | 1.0235 | 0.0967 |
| GPU Energy | **0.0133** | 0.0292 | 0.0424 | 1.0373 | 0.1847 |
| RAM Energy | **0.0095** | 0.0220 | 0.0129 | 0.3584 | 0.0345 |
| Energy Consumed | **0.0476** | 0.1150 | 0.1083 | 2.4192 | 0.3159 |
| Peak RAM Usage | 74.8316 | 67.5962 | 99.3873 | **53.6099** | 86.1017 |

*Emissions: kg CO2eq; Emissions Rate: kg CO2eq/h; Power: W; Energy: kWh; RAM: GB.*

---

### 4.2 RQ1: Scalability and Performance
To address RQ1, Table 4 details the efficiency results for increasing scale datasets of WarpRec against academic and industrial competitor frameworks. The failures to complete the pipeline can be due to either memory exhaustion (Out-of-Memory errors in both RAM and VRAM) or exceeding the execution time limits. They are highlighted in the table using a gray box starting at the specific stage where execution was interrupted. In the case of *Time Limit Exceeded* errors, the notation ($x/6$) indicates the $x$ number of hyperparameter configurations successfully completed before the timeout. The experimental results reveal significant disparities in how various frameworks handle increasing data scales. On the smallest dataset (MovieLens-1M), all evaluated frameworks completed the full recommendation pipeline, confirming their suitability for small-scale academic prototyping. However, as the data scale increases to MovieLens-32M and NetflixPrize-100M, critical nuances regarding architectural robustness emerge. Several established frameworks exhibit systemic limitations when tasked with medium-to-large-scale experiments. These frameworks frequently fail to complete the pipeline. In contrast, WarpRec consistently executes the end-to-end workflow across all benchmarks, positioning it as a robust infrastructure for both high-stakes research and industrial deployment. Notably, LightGCN is the most computationally intensive model across all frameworks. Except for RecBole and WarpRec, all other frameworks failed to process the NetflixPrize-100M dataset for this model. Due to the tight constraints of our experimental setup, RecBole’s LightGCN implementation struggled to reach convergence within the time limit. Notably, algorithms such as EASE$^R$ (frequently favored in industrial applications for their balance of simplicity and performance [70, 79]) require the inversion of a dense Gram matrix, which poses a significant challenge to data engines. WarpRec outperforms all competitors across all dataset scales for this model.

### 4.3 RQ2: Green AI Analysis
Leveraging WarpRec’s native integration with CodeCarbon, we profiled the environmental impact of training recommendation models on the NetflixPrize-100M dataset (Table 5). A cross-analysis with end-to-end performance (Table 4) reveals that total energy consumption is driven more by training duration than by instantaneous power draw. For instance, while SASRec exhibits the highest peak GPU power (278.6 W), its relatively rapid convergence results in a moderate total energy footprint (0.32 kWh). In contrast, LightGCN, despite a lower average power draw (157.4 W), requires a significantly more extensive training phase to converge, resulting in a massive aggregate consumption of 2.42 kWh and the highest carbon emissions (0.0095 kg CO2eq). Conversely, shallow architectures demonstrate a superior balance between efficiency and effectiveness; specifically, EASE$^R$ consumes approximately 95% less energy (0.115 kWh) than the deep graph-based baselines, demonstrating that state-of-the-art recommendation performance can be achieved with minimal environmental cost.

---

> **User:** “ I’ve recently watched Pulp Fiction, Forrest Gump, and Full Metal Jacket. Do you have any suggestions for what I should watch next? ”
>
> **Agent (Internal Thought):** *User requests recommendations based on viewing history. Calling → WarpRec SASRec Tool...*
>
> **Tool Call:** `WarpRec_SASRec.recommend (item_sequence=[‘Pulp Fiction’, ‘Forrest Gump’, ‘Full Metal Jacket’], top_k=3)`
>
> **WarpRec Tool Output:** *[1. ‘Star Wars: Episode IV - A New Hope’, 2. ‘The Godfather’, 3. ‘Schindler’s List’]*
>
> **Agent:** “ Here are some movies you could like: **The Godfather** would be an excellent choice given your appreciation for character-driven storytelling and grittier narratives. For something epic and immersive, the original **Star Wars IV** captures the iconic ’70s era you seem to enjoy. **Schindler’s List** shares the historical depth and emotional weight of your recent watches. Do you prefer something more action-adventure oriented, or another powerful drama? ”

**Figure 2: Sequential Recommendation via WarpRec MCP Interface.** The AI Agent leverages a SASRec model trained on MovieLens-32M.

---

### 4.4 RQ3: Interoperability with Agentic Pipelines
To address RQ3, we assess the capability of WarpRec’s *Application Layer*, which abstracts away the complexities of serving, ensuring that trained models are instantly available for inference without modification. As illustrated in Figure 2, the system performs real-time inference using a SASRec model trained on the MovieLens-32M dataset (achieving an nDCG@10 of 0.9257 with 100 negative samples). Thanks to its sequential nature, the model does not require a pre-trained user embedding to process the input history. After receiving the response from the WarpRec tool, the agent generates a natural language explanation identifying latent semantic connections to enrich the raw recommendation list. The recommendations are presented highlighting “character-centered storytelling” and “more raw storytelling,” thus combining the algorithmic result and agentic reasoning. This shows how WarpRec’s MCP interface bridges the gap between recommendation mechanisms and agentic workflows.

## 5 Conclusion
WarpRec represents a paradigm shift in Recommender Systems engineering, resolving the long-standing dichotomy between academic prototyping and industrial recommendation through a backend-agnostic architecture for reproducible, scalable, and responsible research. Our analysis demonstrates that WarpRec achieves industrial-grade throughput while retaining the flexibility of lightweight academic libraries. It shows that a single codebase can easily transition from local exploration to distributed training on Ray clusters. By operationalizing Green AI principles and enforcing statistical rigor by default, WarpRec ensures that future innovations are scientifically valid and environmentally responsible. Finally, WarpRec transforms the recommender from a static ranking engine into a dynamic, queryable partner for Agentic AI.