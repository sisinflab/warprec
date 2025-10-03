#######
Writers
#######

.. toctree::
   :hidden:
   :maxdepth: 2

   local

The WarpRec data writing module provides a unified interface to **persist** the results and artifacts generated during the recommendation pipeline. It is designed to be flexible and extensible, allowing users to save data to different destinations, including:

- Local files

Key features include:

- Automatic saving of results and artifacts to the configured destination.
- Support for different output formats (e.g., CSV, TSV).

This design allows WarpRec to maintain **consistency, traceability, and interoperability**, facilitating the sharing of experimental outcomes and the integration with external systems.
