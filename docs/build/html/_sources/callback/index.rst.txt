.. _callback:

#################
Callbacks
#################

.. toctree::
    :hidden:
    :maxdepth: 1

    implement

WarpRec offers a comprehensive set of tools to customize the model training process. This flexibility is essential, as recommendation models rely on diverse types of information, making it impractical to provide a one-size-fits-all solution.

To address this need for customization, WarpRec introduces the **custom Callback system**.

WarpRec provides customizable ``Callback`` functionality. When using the framework, you can either launch an experiment via a configuration file—accessing the main training and inference pipelines—or use a custom script to directly interact with WarpRec's internal components.

In certain scenarios, you may want to modify the workflow slightly or perform additional computations during execution. This can be achieved seamlessly using WarpRec’s ``Callback`` system.

Using a Custom Callback
------------------------

To integrate a custom callback into the main pipeline, follow these two steps:

1. **Implement the Callback:** Create a script containing a class that extends the base ``WarpRecCallback``.
2. **Register the Callback:** Add the callback definition to your configuration file. For more details on configuration, see the :ref:`configuration guide <general>`.

That’s it! In :ref:`this <callback_implement>` section you can find an easy to follow tutorial on how to implement your first callback.
