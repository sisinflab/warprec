# SYSTEM ROLE & PERSONA
Act as a **Senior Technical Writer** specializing in Machine Learning and MLOps documentation, combined with the deep domain expertise of a **Recommender Systems (RS) Researcher** and the marketing acumen of a **Developer Advocate**. 

# CONTEXT & OBJECTIVE
I am providing you with the codebase and the foundational research paper for **WarpRec**, a high-performance, backend-agnostic recommendation framework designed to bridge the "Deployment Chasm" between academic rigor and industrial scale. 

Your objective is to generate a comprehensive, exhaustive, and highly polished documentation suite ready to be deployed on **ReadTheDocs** (using Markdown/MyST). The current documentation in the repository is an incomplete draft. You must rewrite and expand it to serve two primary purposes:
1. **Scientific & Technical Reference:** It must be mathematically rigorous, providing exhaustive taxonomies, class/method signatures, and LaTeX formulas for models and metrics.
2. **Framework Advertisement:** It must clearly articulate the Unique Selling Points (USPs) of WarpRec (e.g., Green AI tracking, Agentic AI/MCP readiness, Ray distributed training, Narwhals backend-agnosticism) in a professional, engaging manner.

# TONE & STYLE GUIDELINES
* **Professional & Scientific:** Use precise academic terminology (e.g., "bipartite graph," "collaborative filtering," "hyperparameter optimization," "Type I error corrections").
* **Clear & Actionable:** Provide copy-pasteable code snippets for both local prototyping and distributed cluster execution.
* **Visually Structured:** Make heavy use of Markdown tables, admonitions (`> **Note:**`, `> **Warning:**`, `> **Tip:**`), and hierarchical headers.
* **Mathematical Rigor:** Use LaTeX blocks (`$$ ... $$` or `$ ... $`) for all metric definitions (e.g., nDCG, HR, Fairness metrics) and model objective functions (e.g., BPR loss, Matrix Factorization).

---

# EXECUTION PHASES & ACTIONS

Please generate the documentation by strictly following these sequential phases. Treat each phase as a separate documentation page or section. 

### Phase 1: The "Pitch" & Architecture Overview (`index.md` & `architecture.md`)
* **Action 1:** Write a compelling landing page that advertises WarpRec. Highlight the core problem (the fragmented landscape between eager-execution academic tools and rigid industrial frameworks) and how WarpRec solves it.
* **Action 2:** Detail the 5 decoupled engines (Reader, Data Engine, Recommendation Engine, Evaluation, Writer) and the Application Layer (REST API & MCP).
* **Action 3:** Highlight the 4 Pillars of WarpRec: Scalability (Ray), Green AI (CodeCarbon), Agentic Readiness (MCP), and Scientific Rigor (Bonferroni/FDR corrections).

### Phase 2: Getting Started & Quickstarts (`quickstart.md`)
* **Action 1:** Provide a step-by-step installation guide.
* **Action 2:** Create three distinct, fully coded examples:
  1. *The Academic:* A local, in-memory script using the Design Pipeline for rapid prototyping.
  2. *The Industrial:* A distributed training script using Ray and the Training Pipeline with HPO (Hyperparameter Optimization).
  3. *The Agentic:* A quickstart showing how to spin up the MCP server and query the recommender via an LLM.

### Phase 3: Exhaustive Taxonomies (`taxonomies.md`)
* **Action 1:** Create exhaustive, well-formatted Markdown tables categorizing the **55 state-of-the-art algorithms** supported by WarpRec. Group them by family: Unpersonalized, Content-Based, Collaborative Filtering (Autoencoders, Graph-based, KNN, Latent Factor, Neural), Context-Aware, Sequential, and Hybrid.
* **Action 2:** Create a taxonomy table for the **40 metrics**, grouped by family: Accuracy, Rating, Coverage, Novelty, Diversity, Bias, Fairness, and Multi-objective.
* **Action 3:** Create a taxonomy table for the **19 filtering and splitting strategies** (e.g., $k$-core, cold-user, temporal holdout, cross-validation).

### Phase 4: Deep Dive API Reference & Signatures (`api_reference/`)
* **Action 1:** For the **Metrics**, generate detailed class/method signatures based on the provided code. For each metric, include:
  * The Python method signature with type hints.
  * A brief scientific description.
  * The exact mathematical formula using **LaTeX**.
  * *Example format:*
    ```python
    def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    ```
    *Description:* Normalized Discounted Cumulative Gain at rank $k$.
    *Formula:* $$ \text{nDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k} $$
* **Action 2:** For the **Models**, generate detailed class signatures. For each model, include:
  * The class signature with key hyperparameters.
  * The theoretical background and the loss function in **LaTeX**.
  * A code snippet showing how to instantiate and train the model.
* **Action 3:** For the **Data Engine**, document the `Reader` module, explaining how Narwhals is used to abstract Pandas/Polars/Spark dataframes.

### Phase 5: Advanced Features & USPs (`advanced/`)
* **Action 1: Green AI Profiling:** Document how to enable CodeCarbon tracking. Show an example of the output metrics (Emissions, CPU/GPU Power, RAM Energy) and explain how researchers can use this for sustainable AI reporting.
* **Action 2: Agentic AI & MCP:** Write a detailed guide on the Model Context Protocol integration. Provide a mock dialogue/code example showing an LLM (like Claude) calling the `WarpRec_SASRec.recommend` tool and reasoning over the output.
* **Action 3: Statistical Rigor:** Document the hypothesis testing suite. Explain how to apply paired tests (Student's t-test, Wilcoxon) and independent tests (Mann-Whitney U), and how to enable Multiple Comparison Problem corrections (Bonferroni, FDR).

---

# WORKFLOW INSTRUCTIONS
1. **Do not generate the entire documentation at once.** 
2. First, output the proposed file structure (`toctree`) for the ReadTheDocs sidebar.
3. Next, generate the content for **Phase 1** and **Phase 2** only. 
4. Stop and ask for my feedback and approval. Once approved, I will prompt you to continue with Phases 3, 4, and 5. 
5. **Do not use placeholders** like "Insert code here" or "Add formula here". Extract the actual logic, parameters, and math from the provided paper and codebase.

---

# INPUTS FOR PROCESSING

**1. The Research Paper:**
@Paper.md

**2. The Codebase:**
@Codebase
