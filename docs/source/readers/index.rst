#######
Readers
#######

.. toctree::
   :hidden:
   :maxdepth: 2

   local

The WarpRec data reading module provides a unified interface to ingest datasets for recommendation tasks. It is designed to be flexible and extensible, allowing users to load interaction data from different sources, including:

- Local files

The module abstracts the underlying data source, returning a consistent ``DataFrame`` object that contains user-item interactions and optionally also side information, and clustering information. This ensures that downstream components, such as dataset splitters, models, and callbacks, can operate without concern for the original data format or storage location.

Key features include:

- Automatic reading of data from source
- Support of different data types

This design allows WarpRec to maintain **flexibility, reproducibility, and scalability**, supporting a wide range of experimental pipelines and real-world recommendation scenarios.
