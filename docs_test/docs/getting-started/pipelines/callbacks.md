# Callbacks

WarpRec offers a comprehensive set of tools to customize the model training process. This flexibility is essential, as recommendation models rely on diverse types of information, making it impractical to provide a one-size-fits-all solution.

To address this need for customization, WarpRec introduces the **custom Callback system**.

WarpRec provides customizable `Callback` functionality. When using the framework, you can either launch an experiment via a configuration file — accessing the main training and inference pipelines — or use a custom script to directly interact with WarpRec's internal components.

In certain scenarios, you may want to modify the workflow slightly or perform additional computations during execution. This can be achieved seamlessly using WarpRec's `Callback` system.

## Using a Custom Callback

To integrate a custom callback into the main pipeline, follow these two steps:

1. **Implement the Callback:** Create a script containing a class that extends the base `WarpRecCallback`.
2. **Register the Callback:** Add the callback definition to your configuration file. For more details on configuration, see the [General Configuration](../../core/configuration/general.md) guide.

For a detailed implementation tutorial, see the [Callbacks Guide](../../guides/callbacks.md).

---

## Available Callbacks

WarpRec provides a set of built-in callbacks that are triggered at specific stages of the pipeline. **WarpRecCallback** is the base class for all WarpRec callbacks.

| Callback name | Origin | Description |
|---|---|---|
| `on_data_reading` | WarpRec | Invoked after data reading. |
| `on_dataset_creation` | WarpRec | Invoked after dataset initialization. |
| `on_training_complete` | WarpRec | Invoked after model training completion. |
| `on_evaluation_complete` | WarpRec | Invoked after model evaluation completion. |

WarpRec callbacks also inherit all the lifecycle hooks defined by the Ray Tune **Callback** class. For more details on those, refer to the [Ray Tune documentation](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Callback.html).
