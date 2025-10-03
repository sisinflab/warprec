#################
Proxy Recommender
#################

The **Proxy Recommender** is a flexible evaluation component designed to assess recommendation outputs produced by external frameworks.
It operates exclusively in a **local context** and is **incompatible with the Ray-based training module**, requiring execution within a **custom evaluation pipeline**.

Users can either run the provided sample script or implement their own evaluation logic by directly interfacing with the Proxy Recommender.
This design enables seamless integration of third-party recommendation results into the framework’s evaluation workflow.

To implement your custom script you can follow the documentation regarding :ref:`Scripting <scripting>`.
WarpRec also provides a sample script located at *sample_pipelines/proxy_evaluation.py* that the user can use as-is or modify to fit their needs.

Work in progress.
