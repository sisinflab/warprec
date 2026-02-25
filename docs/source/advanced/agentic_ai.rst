.. _agentic_ai:

####################################
Agentic AI & Model Context Protocol
####################################

WarpRec anticipates the shift toward autonomous AI systems by natively implementing the **Model Context Protocol (MCP)** server interface. This transforms the recommender from a static predictor into a queryable tool that LLMs and autonomous agents can dynamically invoke within their decision-making loops.

.. contents:: On this page
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

-----

The Agentic Paradigm
=====================

Artificial intelligence is shifting from monolithic models to **agentic workflows**, where autonomous agents call external tools and interleave reasoning with actions. In this paradigm, the recommender becomes a callable tool within an agent's decision-making process. This new role requires interactive dialogue to iteratively refine results.

Current RS frameworks lack the standardized interfaces to enable this. WarpRec addresses the gap by exposing trained models through:

1. **MCP Server** (via `FastMCP <https://github.com/jlowin/fastmcp>`_) — Standardized tool discovery and invocation for MCP-compatible clients (Claude Desktop, custom agents, etc.).
2. **REST API** (via `FastAPI <https://fastapi.tiangolo.com/>`_) — Versioned HTTP endpoints for standard microservices integration.

-----

MCP Server Setup
================

**Prerequisites:**

- A pre-trained model checkpoint (from the :ref:`pipeline-train` or :ref:`pipeline-design`).
- The ``infer-api/`` directory containing ``mcp_server.py``.

**Step 1: Configure the environment.**

.. code-block:: bash

    cd infer-api
    cp .env.example .env

Edit ``.env`` to point to your checkpoints and datasets:

.. code-block:: bash

    # Device for PyTorch inference
    DEVICE=cpu

    # Paths to model checkpoints and dataset files
    CHECKPOINTS_DIR=checkpoints
    DATASETS_DIR=datasets

    # Models and datasets to load
    MODELS=SASRec
    DATASETS=movielens

    # MCP Server
    MCP_HOST=0.0.0.0
    MCP_PORT=8082

**Step 2: Start the MCP server.**

.. code-block:: bash

    python mcp_server.py

The server automatically discovers available model-dataset combinations and exposes them as MCP tools. For a SASRec model trained on MovieLens, the tool ``recommend_movielens_sequential`` becomes available.

-----

MCP Tool Interface
==================

Each exposed tool follows a consistent interface:

.. code-block:: text

    Tool: recommend_{dataset}_{model_type}

    Parameters:
      item_sequence (List[str])  — Ordered list of item names (titles, IDs)
      top_k (int)                — Number of recommendations to return

    Returns:
      List[str]                  — Ranked list of recommended item names

The MCP protocol allows any compatible client to **discover** available tools at runtime, query their schemas, and invoke them with structured arguments.

-----

Example: LLM Agent Interaction
===============================

The following illustrates how an AI agent uses the WarpRec MCP tool to generate personalized recommendations. This example uses a SASRec model trained on MovieLens-32M:

.. code-block:: text

    User:
      "I've recently watched Pulp Fiction, Forrest Gump, and Full Metal Jacket.
       Do you have any suggestions for what I should watch next?"

    Agent (Internal Thought):
      User requests recommendations based on viewing history.
      Calling WarpRec SASRec Tool...

    Tool Call:
      recommend_movielens_sequential(
          item_sequence=["Pulp Fiction", "Forrest Gump", "Full Metal Jacket"],
          top_k=3,
      )

    WarpRec Tool Output:
      ["Star Wars: Episode IV - A New Hope", "The Godfather", "Schindler's List"]

    Agent Response:
      "Here are some movies you could like: The Godfather would be an excellent
       choice given your appreciation for character-driven storytelling and
       grittier narratives. For something epic and immersive, the original
       Star Wars IV captures the iconic '70s era you seem to enjoy.
       Schindler's List shares the historical depth and emotional weight of
       your recent watches. Do you prefer something more action-adventure
       oriented, or another powerful drama?"

The agent combines the raw recommendation list from WarpRec with its own reasoning to provide a rich, conversational experience — identifying latent semantic connections such as "character-driven storytelling" and "historical depth" to enrich the output.

-----

REST API Alternative
====================

For standard microservices integration, WarpRec also provides a FastAPI-based REST server:

.. code-block:: bash

    cd infer-api
    python server.py

This exposes versioned endpoints at ``/api/warprec/v1/`` for three inference modes:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Endpoint
     - Description
   * - ``/api/warprec/v1/sequential/``
     - Sequential recommendation from an ordered item history.
   * - ``/api/warprec/v1/collaborative/``
     - Collaborative filtering recommendation from user-item interactions.
   * - ``/api/warprec/v1/contextual/``
     - Context-aware recommendation incorporating metadata features.

An interactive Swagger UI is available at ``http://localhost:8081/docs`` for testing and exploration.

-----

Environment Variables
=====================

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Default
     - Description
   * - ``SERVER_HOST``
     - ``0.0.0.0``
     - Host for the REST API server.
   * - ``SERVER_PORT``
     - ``8081``
     - Port for the REST API server.
   * - ``DEVICE``
     - ``cpu``
     - PyTorch device for inference (``cpu`` or ``cuda``).
   * - ``CHECKPOINTS_DIR``
     - ``checkpoints``
     - Directory containing trained model checkpoints.
   * - ``DATASETS_DIR``
     - ``datasets``
     - Directory containing dataset files (mappings, metadata).
   * - ``MODELS``
     - —
     - Comma-separated list of model names to load.
   * - ``DATASETS``
     - —
     - Comma-separated list of dataset names to load.
   * - ``MCP_HOST``
     - ``0.0.0.0``
     - Host for the MCP server.
   * - ``MCP_PORT``
     - ``8082``
     - Port for the MCP server.

-----

Architecture
============

The Application Layer is strictly decoupled from the training infrastructure:

1. **Training** produces checkpoint artifacts (model weights, item mappings, dataset metadata).
2. **Serving** loads these artifacts into a ``ModelManager`` that instantiates the appropriate model and dataset.
3. **Inference services** (``SequentialService``, ``CollaborativeService``, ``ContextualService``) wrap model-specific prediction logic.
4. Both the **MCP server** and **REST API** delegate to these services, ensuring consistent behavior across protocols.

This design means that any model trained in WarpRec can be served with **zero additional engineering effort** — simply point the server at the checkpoint directory.

.. seealso::

   :doc:`/pipelines/index` for how to produce model checkpoints via the Training Pipeline.
