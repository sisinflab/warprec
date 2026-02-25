.. _api_pipelines:

################
Pipelines API
################

The pipeline functions orchestrate WarpRec's end-to-end workflows. Each function takes a path to a YAML configuration file and executes the corresponding pipeline.

.. autofunction:: warprec.pipelines.design.design_pipeline

.. autofunction:: warprec.pipelines.train.train_pipeline

.. autofunction:: warprec.pipelines.eval.eval_pipeline
