# WarpRec Configurations

WarpRec offers a wide range of configuration options to help tailor your training process to your data and goals. Configurations are a key component of WarpRec's [utils](../README.md) package. Here's a quick overview of what you can customize through configuration files:

- Data reading
- Splitting strategies
- Models to train
- Hardware resources allocation
- Number of experiments to run
- Evaluation process
- Output results and recommendations
- And much more!

To make everything easier to digest, this documentation includes a Table of Contents for easy navigation. Each section includes examples to help you understand how to configure WarpRec more effectively.

## 📚 Table of Contents
- 📖 [Reader Configuration](#📖-reader-configuration)
- ✍️ [Writer Configuration](#️✍️-writer-configuration)
- 🧹 [Filtering Configuration](#🧹-filtering-configuration)
- 🔀 [Splitter Configuration](#🔀-splitter-configuration)
- 🖥️ [Dashboard Configuration](#🖥️-dashboard-configuration)
- 🧠 [Models Configuration](#🧠-models-configuration)
- 📊 [Evaluation Configuration](#📊-evaluation-configuration)
- ⚙️ [General Configuration](#️⚙️-general-configuration)

## 📖 Reader Configuration

The `Reader Configuration` section handles data loading and preprocessing. The data loaded here is transformed into the internal structures used by WarpRec’s recommendation models.

### 🔧 Available Keywords

- **loading_strategy**: The strategy to use to load the data. Can be 'dataset' or 'split'.
- **data_type**: The type of data accepted by the WarpRec. As of right now, only 'transaction' data is supported.
- **reading_method**: The source from which the data will be read. As of right now, only 'local' is supported.
- **local_path**: The local path from which the data is read. This keyword is used along with the `reading_method` when is set to `local`.
- **sep**: The separator of the file to be read. Defaults to `'\t'`.
- **header**: Whether to treat the first row of the file as a header or not. Defaults to True.
- **rating_type**: The type of rating to be used. Can be set to 'implicit' or 'explicit'. If set to explicit, the framework will look for a score column inside the transaction data, this column must contain a score given by the user to the item for that specific transaction. If set to `implicit`, the column can be omitted and each transaction will be assigned a score of 1.
- **split**: This is a nested section of the reader configuration, it is used in case of data that has already been split.
- **side**: This is a nested section of the reader configuration, it is used if the user wants to load side information of the items alongside the transaction data.
- **clustering**: This is a nested section of the reader configuration, it is used if the user wants to load clustering information of the users or the items alongside the transaction data.
- **labels**: This is a nested section of the reader configuration, it is used if the user wants to read columns with different naming from the standard.
- **dtypes**: This is a sub category of the reader configuration, it is used if the user wants to read columns with different typing from the standard.

#### 🔀 Split Reading

WarpRec also supports the reading process of already split data. This can be done using the `split` nested section. To use this submodule, you first need to set the `reading strategy` to **split**, then you provide the following information:

- **local_path**: The path to the directory where the split data is stored. All the splits must be in the same directory.
- **ext**: The extension used by the split files. Defaults to `.tsv`.
- **sep**: The separator used by the split files. Defaults to `\t`.
- **header**: Whether to treat the first row of the file as a header or not. Defaults to True.

#### 🧩 Side Information Reading

WarpRec also supports reading side information. This can be done using the `side` nested section. To use this sub module you can simply add it to your configuration file, then you can provide the following information:

- **local_path**: The path to the side information.
- **sep**: The separator used for the side information file.
- **header**: Whether to treat the first row of the file as a header or not. Defaults to True.

#### 👥 Clustering Information Reading

WarpRec also supports reading clustering information. This can be done using the `clustering` nested section. To use this sub module you can simply add it to your configuration file, then you can provide the following information:

- **user_local_path**: The path to the user clustering information.
- **item_local_path**: The path to the item clustering information.
- **user_sep**: The separator used for the user clustering file.
- **item_sep**: The separator used for the item clustering file.
- **user_header**: Whether to treat the first row of the user file as a header or not. Defaults to True.
- **item_header**: Whether to treat the first row of the item file as a header or not. Defaults to True.

#### 🏷️ Labels

WarpRec expects data to be formatted as follows:

```plaintext
user_id,item_id,rating,timestamp
1,42,2,1651939200
2,17,5,1651942800
3,42,1,1651946400
...
```

The labels are expected to be exactly the same as provided in this example. This is done in case you have a file with multiple columns but only a few of those are used inside the framework. If your labels differ, you can use the `label` nested section, then you can provide the following information:

- **user_id_label**: The custom label for the user id.
- **item_id_label**: The custom label for the item id.
- **rating_label**: The custom label for the rating.
- **timestamp_label**: The custom label for the timestamp.

Some information to keep in mind regarding labels:

- Only `user_id` and `item_id` are strictly required.
- If you're using implicit feedback, `rating_label` is optional.
- `timestamp_label` is only needed for time-based split strategies.
- In case `header=False`, columns are expected to be ordered correctly.

#### 🧬 Dtypes

By default, WarpRec assumes:

- IDs are integers
- Ratings are floats
- Timestamps are integers

If your data uses different types (e.g. string-based IDs), you can override them:

- **user_id_type**: The custom dtype for the user id.
- **item_id_type**: The custom dtype for the item id.
- **rating_type**: The custom dtype for the rating.
- **timestamp_type**: The custom dtype for the timestamp.

As a side note, when `header=False` dtypes will be ignored.

Supported dtypes:

- `int8`, `int16`, `int32`, `int64`
- `float32`, `float64`
- `str`

### 📌 Example of Configuration

Here is a complete example that uses all available customization options for the reader module. This setup loads a formatted version of the MovieLens dataset, along with side and clustering information.

 ```yaml
    reader:
        loading_strategy: dataset
        data_type: transaction
        reading_method: local
        local_path: tests/test_dataset/movielens.csv
        rating_type: explicit
        sep: ','
        labels:
            user_id_label: uid
            item_id_label: iid
            rating_label: rating
            timestamp_label: time_ms
        dtypes:
            user_id_type: str
        side:
            local_path: tests/test_dataset/movielens_side.csv
            sep: ','
        clustering:
            user_local_path: tests/test_dataset/movielens_user_cluster.csv
            item_local_path: tests/test_dataset/movielens_item_cluster.csv
            user_sep: ','
            item_sep: ','
...
```

### ⚠️ Notes and Validation

- The dataset provided through the `local_path` must contain the labels provided inside `labels`. An invalid dataset will raise an exception.
- Reading a dataset that was previously split is only possible if all the split files are in the same directory and they are in the WarpRec format.

## ✍️ Writer Configuration

The `Writer Configuration` section defines how and where WarpRec saves the output of your experiments, such as generated recommendations and processed data splits.

WarpRec allows customization of file format, paths, naming conventions, and more. This is especially useful when running multiple experiments or managing different datasets.

### 🔧 Available Keywords

- **dataset_name**: The name of the dataset currently being used. This will be used to name directories and files for easier traceability.
- **writing_method**: The strategy used to write output files. Currently, only 'local' is supported.
- **local_experiment_path**: The path where all files related to the experiment (e.g., recommendations, metrics, splits) will be stored. Required when using local as the writing method.
- **setup_experiment**: Whether to set up a new experiment folder with proper structure. Default is true.
- **save_split**: Whether to save the train/validation/test splits created during preprocessing. Default is false.
- **results**: A nested section to control result file customization.
- **split**: A nested section to control split file customization.
- **recommendation**: A nested section to control recommendation file customization.

#### 📝 Results
The `results` section lets you control the structure of the result files.

- **sep**: The separator used for the output files. Default is `\t`.
- **ext**: The extension of the output files. Default is `.tsv`.

#### ✂️ Split
The `split` section lets you control the structure of the split files.

- **sep**: The separator used for the output files. Default is `\t`.
- **ext**: The extension of the output files. Default is `.tsv`.
- **header**: Whether to treat the first row of the file as a header or not. Defaults to True.

#### 🛍️ Recommendation
The `recommendation` section lets you control the structure of the recommendation files.

- **sep**: The separator used for the output files. Default is `\t`.
- **ext**: The extension of the output files. Default is `.tsv`.
- **header**: Whether to treat the first row of the file as a header or not. Defaults to `True`.
- **k**: The number of recommendation per user to produce. Defaults to `50`.
- **user_label**: The user label used in the file. Defaults to `user_id`.
- **item_label**: The item label used in the file. Defaults to `item_id`.
- **rating_label**: The rating label used in the file. Defaults to `rating`.

### 📌 Example of Writer Configuration

Below is a full example of a `writer configuration` that saves output locally with a custom file format:

```yaml
writer:
    dataset_name: movielens
    writing_method: local
    local_experiment_path: results/movielens_experiment
    setup_experiment: true
    save_split: true
    results:
        sep: ','
    split:
        sep: ','
        ext: .csv
    recommendation:
        sep: ','
        ext: .csv
        header: false
        k: 100
...
```

### ⚠️ Notes and Validation

- The `local_experiment_path` must exist if you choose `local` as the `writing_method`. Otherwise, WarpRec will raise an error.
- Custom separators (e.g., `|`, `,`, etc.) must be valid characters. Invalid separators will raise an exception.
- Only local writing is currently supported, but this structure allows easy extension for remote/cloud options in future versions.

## 🧹 Filtering Configuration

The `Filtering Configuration` section defines how your dataset is filtered before the splitting process. Filtering is crucial in scenarios where you dataset contains redundant information or it's dimensions are too much to handle with the resources at hand. Filtering strategies will be executed in order, so be careful how information is provided in the configuration. Filtering strategies are expected to be in this format:

```yaml
filtering:
    strategy_name_1:
        arg_name_1: value_1
    strategy_name_2:
        arg_name_1: value_1
        arg_name_2: value_2
...
```

and the strategies will be executed from top to bottom. These are all the strategies available:

`MinRating`: All the transactions with a rating value lower then the min_rating will be discarded. Cannot be used in implicit rating scenarios.

```yaml
filtering:
    MinRating:
        min_rating: 3.0
...
```

`UserAverage`: All the transactions below the average of each user will be discarded. Cannot be used in implicit rating scenarios.

```yaml
filtering:
    UserAverage: {} # No parameters needed
...
```

`UserMin`: All the transactions containing an user with less then the specified number of interactions, will be discarded.

```yaml
filtering:
    UserMin:
        min_interactions: 5
...
```

`UserMax`: All the transactions containing an user with more then the specified number of interactions, will be discarded. Useful to analyze cold scenarios.

```yaml
filtering:
    UserMax:
        max_interactions: 2
...
```

`ItemMin`: All the transactions containing an item with less then the specified number of interactions, will be discarded.

```yaml
filtering:
    ItemMin:
        min_interactions: 5
...
```

`ItemMax`: All the transactions containing an item with more then the specified number of interactions, will be discarded. Useful to analyze cold scenarios.

```yaml
filtering:
    ItemMax:
        max_interactions: 2
...
```

`IterativeKCore`: Iteratively apply UserMin and ItemMin until convergence is reached.

```yaml
filtering:
    IterativeKCore:
        min_interactions: 5
...
```

`NRoundsKCore`: Apply UserMin and ItemMin a fixed number of times. This is a simpler version of `IterativeKCore`, can be used in cases where convergence is not needed.

```yaml
filtering:
    NRoundsKCore:
        rounds: 3
        min_interactions: 5
...
```

### 📌 Example of Filtering Configuration

Below is a full example of a `filtering configuration` that first removes all the ratings below 3.0, then keeps only the users with at least 10 interactions:

```yaml
filtering:
    MinRating:
        min_rating: 3.0
    UserMin:
        min_interactions: 10
...
```

### ⚠️ Notes and Validation

- The naming of the strategy and the parameters must be correct for them to properly work.
- Changing the order of the filtering will most likely change the end results of the filtering process.

## 🔀 Splitter Configuration

The `Splitter Configuration` section defines how your dataset is split into training, validation, and test sets before training begins. This step is crucial for building a reliable evaluation pipeline and ensuring fair comparisons between models.

WarpRec provides several splitting strategies that can be configured to match your experimental needs, including:

- Temporal strategies
- Random strategies
- Timestamp slicing

### 🔧 Available Keywords

- **strategy**: The splitting strategy to apply. Options currently include:
    - *temporal_holdout*: Splits data using the temporal value of the transaction. Test/Validation set will be the latest transaction for each user. Ratio value must be provided in the configuration file. In edge cases, the transaction will be considered only in the training set.
    - *temporal_leave_k_out*: Splits data using the temporal value of the transaction. Test/Validation set will be the latest transaction for each user. K value must be provided in the configuration file. In edge cases, the transaction will be considered only in the training set.
    - *random_holdout*: Splits data randomly. Ratio value must be provided in the configuration file. In edge cases, the transaction will be considered only in the training set.
    - *random_leave_k_out*: Splits data randomly. K value must be provided in the configuration file. In edge cases, the transaction will be considered only in the training set.
    - *timestamp_slicing*: Splits data given a fixed timestamp. Timestamp must be provided in the configuration file. If 'best' is chosen, the algorithm will provide to find the best split.
- **test_ratio**: The ratio of the test set. Must be provided if the strategy expects a ratio.
- **val_ratio**: The ratio of the validation set. Must be provided if the strategy expects a ratio.
- **test_k**: The k value of the test set. Must be provided if the strategy expects a k value.
- **val_k**: The k value of the validation set. Must be provided if the strategy expects a k value.
- **timestamp**: The timestamp used in the `timestamp_slicing` strategy. Can either be a timestamp value or 'best'.
- **seed**: The seed used for random strategies. Defaults to 42.

### 📌 Example of Splitter Configuration

Below is a full example of a `splitter configuration` that splits data based on time of transaction:

```yaml
splitter:
    strategy: temporal_holdout
    test_ratio: 0.2
    val_ratio: 0.1
...
```

### ⚠️ Notes and Validation

- Not every field must be provided. Each strategy needs different values.
- The test set is required, the validation set is optional and can be omitted by simply not passing any value.
- Temporal strategies require the timestamp to be passed during the reading process.

## 🖥️ Dashboard Configuration

The `Dashboard Configuration` section defines which dashboard to activate during the training process. TensorBoard dashboard is activated by default and cannot be disabled.

### 🔧 Available Keywords

The `Dashboard Configuration` contains different nested sections dedicated to each dashboard that can be activated.

- **wandb**: The nested section dedicated to Weights & Biases dashboard.
- **mlflow**: The nested section dedicated to MLFlow dashboard.
- **codecarbon**: The nested section dedicated to Codecarbon dashboard.

### 🟡 Wandb

WarpRec, through Ray, supports the Weights & Biases dashboard.

- **enabled**: A flag indicating whether to activate the dashboard. Defaults to False.
- **project**: Name of the project.
- **group**: Name of the group.
- **api_key_file**: Path to the file with the API key.
- **api_key**: API key.
- **excludes**: List of parameters to exclude from logging.
- **log_config**: Whether or not to log the configuration.
- **upload_checkpoints**: Whether or not to upload the checkpoints.

### 🔵 MLFlow

WarpRec, through Ray, supports the MLFlow dashboard.

- **enabled**: A flag indicating whether to activate the dashboard. Defaults to False.
- **tracking_uri**: URI of the MLflow tracking server.
- **registry_uri**: URI of the MLflow model registry.
- **experiment_name**: Name of the MLflow experiment.
- **tags**: Tags to be added to the MLflow run.
- **tracking_token**: Token for MLflow tracking.
- **save_artifacts**: Whether or not to save the artifacts. Defaults to False.

### 🌱 Codecarbon

WarpRec, through Ray, supports the Codecarbon dashboard.

- **enabled**: A flag indicating whether to activate the dashboard. Defaults to False.
- **save_to_api**: Whether or not to save the results to CodeCarbon API.
- **save_to_file**: Whether or not to save the results to a file.
- **output_dir**: Directory where to save the results.
- **tracking_mode**: Tracking mode for CodeCarbon. Options are "machine" or "process".

### 📌 Example of Dashboard Configuration

Below is a full example of a `dashboard configuration` to track your experiments:

```yaml
dashboard:
    mlflow:
        enabled: true
        experiment_name: MyExperiment
...
```

### ⚠️ Notes and Validation

- If you are using a remote dashboard, like Weights & Biases and you are trying to save artifacts, Ray checkpoints might need to not be deleted.
- TensorBoard dashboard is always active and can be accessed locally.

## 🧠 Models Configuration

The `Model Configuration` section defines how each model inside your experiment should be trained.

WarpRec provides several options when it comes to setting up the training of your models that can be configured to match your experimental needs, including:

- Search Space
- Scheduling
- Resource Usage

### 🔧 Available Keywords

The `Model Configuration` is different from other configurations as it presents two main nested sections:

- **meta**: The meta parameters of the model. Meta parameters affect initialization of the model, implementation and checkpoints.
- **optimization**: A nested section containing all the information about the hyperparameter optimization done through Ray Tune.
- **early_stopping**: An optional strategy which will stop the trial if the model has reached a plateau.
- **parameters**: The parameters of the model.

#### 📝 Meta

The `meta` section let you decide some information about the model that do not interfere with the training.

- **save_model**: Flag that decides whether or not to save the model in the experiment directory. Defaults to false.
- **save_recs**: Flag that decides whether or not to save the recommendations. Defaults to false.
- **keep_all_ray_checkpoints**: Flag that decides whether or not to keep all the checkpoints that Ray Tune will create. On a large scale training this option is advised to be set on false. Defaults to false.
- **load_from**: Local path to a model weights to be loaded. Defaults to None.
- **implementation**: The implementation used during the training, if more than one is present. Defaults to latest.

#### ⚙️ Optimization

The `optimization` section let you decide how to train your model.

- **strategy**: The strategy used for hyperparameter optimization. Defaults to grid. Supported strategies are:
    - *grid*: Performs an exhaustive search of the whole search space.
    - *random*: Performs a random search of the search space.
    - *hopt*: Using the HyperOpt algorithm to explore the search space efficiently.
    - *optuna*: Using the Optuna algorithm to explore the search space efficiently.
    - *bohb*: Using the BOHB algorithm to explore the search space efficiently.
- **scheduler**: The scheduler used for hyperparameter optimization. Defaults to fifo. Supported strategies are:
    - *fifo*: Classic First In First Out.
    - *asha*: The ASHA scheduler for a more optimized scheduling approach.
- **properties**: A nested section dedicated for strategy and scheduler parameters.
- **validation_metric**: The validation metric used during training. Defaults to nDCG@5.
- **device**: The device used during training and evaluation. Defaults to cpu. Supports cuda devices and also cuda devices with indexing, like cuda:1.
- **num_samples**: The number of samples to generate for the different strategies. If the strategy is set to grid, then this field must be set to 1. Defaults to 1.
- **cpu_per_trial**: The number of cpu cores per trial. Must be greater than 0. Supports floating numbers. Defaults to the maximum number of cpu cores available locally.
- **gpu_per_trial**: The number of gpus per trial. Supports floating numbers. Defaults to 0.

#### 🧩 Properties

The `properties` section is used to provide further information to the strategy or the scheduler, if needed.

- **mode**: This values is used to determine whether to maximize the value of the validation metric or to minimize it. Accepted values are min and max. Defaults to max.
- **seed**: The seed of the experimentation, used to set up the model initialization. Defaults to 42.
- **time_attr**: The measure of time used by the scheduler.
- **max_t**: Max time unit given to each trial.
- **grace_period**: Min time unit given to each trial.
- **reduction_factor**: Reduction rate of trials. Used inside ASHA scheduler to reduce the number of trials.

#### 🛑 Early Stopping

The `early_stopping` section is used to optionally add a stopping criteria to each trial.

- **monitor**: The value to monitor during training. Can either be "score", to monitor the validation metric, or "loss".
- **patience**: Number of consecutive evaluations without improvement after which training will be stopped. This value is *required* if the early stopping is included in the training.
- **grace_period**: Minimum number of evaluations to run before applying early stopping, regardless of performance.
- **min_delta**: Minimum change in the monitored value to qualify as an improvement. Changes smaller than this threshold are considered as no improvement.

### 📌 Example of Model Configuration

Below is a full example of a `model configuration` that trains a model on given data:

```yaml
models:
    EASE:
        optimization:
            strategy: grid
            scheduler: fifo
            properties:
                mode: max
                seed: 42
            device: cpu
            validation_metric: nDCG@10
            cpu_per_trial: 12
        l2: 10
...
```

Also this is an example configuration including a more complex parameter exploration, with early stopping:

```yaml
models:
    NeuMF:
        optimization:
          strategy: hopt
          num_samples: 50
        early_stopping:
          patience: 5
          grace_period: 5
        mf_embedding_size: [16, 32, 64]
        mlp_embedding_size: [16, 32, 64]
        mlp_hidden_size: [[32, 16], [64, 32], [64, 32, 16]]
        mf_train: [True, False]
        mlp_train: [True, False]
        dropout: [0, 0.01]
        weight_decay: 1e-5
        epochs: 100
        learning_rate: [1e-3, 1e-4, 1e-5]
        neg_samples: [0, 1, 2]
...
```

### ⚠️ Notes and Validation

- Each model trained with WarpRec requires it's own configuration.
- Trials of a single model can be computed in parallel. If more than one model is provided, the next training session will wait the pervious one to be over before starting.
- Parameters of the model, like on the example, depend on the model. Check the model documentation for more information about the parameters.
- Saving recommendation will lead to disk space occupation. Save recommendations only if you need them.

### 🧪 Advanced Model Configuration

If you are an expert user and want to use WarpRec to its fullest potential, this is the section where we will explain how you can personalize the your model hyperparameter tuning. Let's start with an example, we want to train the ItemKNN model on our data. This is a possible configuration:

```yaml
models:
    ItemKNN:
        k: 10
        similarity: cosine
        normalize: True
...
```

This configuration will run a grid search on a single combination of parameters. Let's do a little step forward an try to make a better grid search. This would be a more refined configuration:

```yaml
models:
    ItemKNN:
        k: [10, 20, 30, 40, 50]
        similarity: [cosine, euclidean]
        normalize: [True, False]
...
```

With this configuration, you can explore better the possible hyperparameter. In this case, you are actually running a total of *5 * 2 * 2 = 20* trials. If you have enough resources, you can adapt the number of cpu cores given to each trial, to execute more trials simultaneously. Here's how you can do it:

```yaml
models:
    ItemKNN:
        optimization:
            cpu_per_trial: 2
        k: [10, 20, 30, 40, 50]
        similarity: [cosine, euclidean]
        normalize: [True, False]
...
```

If your system has for example 12 cores, then this configuration will execute at most 6 trials in parallel. **Be sure to check if your system can handle this many trials in parallel**. This method of approach is perfect if you want a quick and easy hyperparameter exploration. In case you need to do a really extensive search, then it is advised to use a search algorithm like *HyperOpt* or *Optuna*. To do so, in WarpRec, it's really easy; in the optimization section you can add the strategy keyword and select the algorithm.

Selecting a more refined search algorithm is usually not sufficient to exhaustively explore the search space. In order to do so you can specify, for each parameter, the *search space* to use. Let's give a quick run down on search spaces:

- *uniform*: Uniform search space.
- *quniform*: Quantized uniform search space.
- *loguniform*: Logarithmic uniform search space.
- *qloguniform*: Quantized logarithmic uniform search space.
- *randn*: Random number search space.
- *qrandn*: Quantized random number search space.
- *randint*: Random integer search space.
- *qrandint*: Quantized random integer search space.
- *lograndint*: Logarithmic random integer search space.
- *qlograndint*: Quantized logarithmic random integer search space.
- *choice*: Default option for most strategies.
- *grid*: Default option used for grid search.

These search spaces can be selected directly inside the configuration file. Normal search spaces need min value and a max value. Quantized search spaces also need a quantization constant. Logarithmic search spaces use base 10 as default but can also take as input the logarithm base. Here is a quick configuration as an example:

```yaml
param_1: ['uniform', 0.0, 1.0]
param_2: ['qrandint', 10, 500, 5]
param_3: ['qloguniform', 0.0, 1.0, 0.005, 2]
```

With all of this information now we can create a really extensive experiment to really explore the search space of our algorithm. Here's an example configuration:

```yaml
models:
    ItemKNN:
        optimization:
            strategy: hopt
            validation_metric: Recall@5
            num_samples: 200
            cpu_per_trial: 2
        k: ['qrandint', 10, 1000, 5]    # This will randomly sample multiples of 5 between 10 and 1000
        similarity: [cosine, euclidean] # This will randomly choose between all the possible values
        normalize: [True, False]        # This will randomly choose between all the possible values
...
```

And with this, you have a configuration that will search through 200 possible different values for the ItemKNN model, using your data.

## 📊 Evaluation Configuration

The `Evaluation Configuration` section defines what metrics should be evaluated on each model trained.

### 🔧 Available Keywords

The `Evaluation Configuration` can be configured using the following keywords:

- **top_k**: The cutoff used to compute ranking metrics.
- **metrics**: The metrics to be evaluated.
- **max_metric_per_row**: The metric to be logged in each row. Defaults to 4.
- **beta**: The beta value used by the F1-score metric. Defaults to 1.0.
- **pop_ratio**: The ratio of transactions that will be considered popular. Defaults to 0.8.
- **save_evaluation**: Flag that decides whether or not to save the evaluation. Defaults to true.

### 📌 Example of Evaluation Configuration

Below is a full example of a `evaluation configuration` that evaluates the best model trained on the current iteration:

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [nDCG, Precision, Recall, HitRate]
...
```

### ⚠️ Notes and Validation

- Beta and Pop ratio values are set to the most commonly used values. Changing these parameters will affect the results.
- During evaluation, only user with relevant items will be considered.

## ⚙️ General Configuration

The `General Configuration` section defines some parameters that will affect the overall behavior of WarpRec.

### 🔧 Available Keywords

The `General Configuration` can be configured using the following keywords:

- **precision**: The precision to be used inside the experiment. Defaults to float32.
- **batch_size**: The batch size to be used inside the experiment. Defaults to 1024.
- **ray_verbose**: . The Ray Tune verbosity value. Ray Tune accepts verbosity levels in a range from 0 to 3. Defaults to 1.
- **callback**: A nested section dedicated to the optional callback.

#### 📞 Callback

The `callback` section let point at some custom callback implementation and optionally pass it some information directly from configuration.

- **callback_path**: The path to the python script containing the callback implementation.
- **callback_name**: The name of the implementation of the callback. The class must inherit from [WarpRecCallback](../callback.py)
- **args**: A list of arguments to pass the callback constructor.
- **kwargs**: A dictionary of arguments to pass the callback constructor.

### 📌 Example of Recommendation Configuration

Below is a full example of a `recommendation configuration`:

```yaml
general:
    precision: float64
    batch_size: 2048
    ray_verbose: 0
    callback:
        callback_path: path/to/the/script.py
        callback_name: class_name
        args: [arg_1, arg_2, ...]
        kwargs:
            kwargs_1: kwargs_value_1
            kwargs_2: kwargs_value_2
            ...
...
```

### ⚠️ Notes and Validation

- Increasing the precision of the experiment will also require more memory to execute most models. Usually float32 is more than sufficient.
