.. _reader:

#####################
Reader Configuration
#####################

The **Reader Configuration** module is responsible for handling data loading and preprocessing.
Data ingested through this configuration is automatically transformed into the internal structures used by WarpRec’s recommendation models.
This ensures consistency and reproducibility across different datasets and use cases.

---------------------
Available Keywords
---------------------

The following keywords are available to configure the reader:

- **loading_strategy**: Defines the loading mechanism for the data. Supported values:

  * ``dataset`` – Loads the dataset directly from the specified path.
  * ``split`` – Loads datasets that have already been partitioned into train, validation, and test sets.

- **data_type**: The data format accepted by WarpRec. Currently, only ``transaction`` data is supported.

- **reading_method**: The source from which the data is read:

  * ``local`` – Reads data from the local filesystem.
  * ``azure_blob`` – Reads data from an Azure Blob Storage. Needs *remote-io* extra to be installed.

- **local_path**: The path to the dataset on the local filesystem.
  This option is mandatory when ``reading_method`` is set to ``local``.

- **azure_blob_name**: The name of the Azure Blob to read from.
  This option is mandatory when ``reading_method`` is set to ``azure_blob``.

- **sep**: Column separator used in the dataset file. Defaults to ``'\t'``.

- **header**: Boolean flag to indicate whether the first row of the file is a header. Defaults to ``True``.

- **rating_type**: Specifies the feedback type:

  * ``implicit`` – Each transaction is automatically assigned a score of 1.
  * ``explicit`` – The dataset must contain a numerical score column representing user feedback.

- **file_format**: The format of the input file. Supported values:

  * ``tabular`` – Standard delimited text files (e.g., CSV, TSV).
  * ``parquet`` – Columnar storage format optimized for performance.

- **split**: A nested configuration block for datasets that are already split.

- **side**: A nested configuration block for loading **side information** associated with items.

- **clustering**: A nested configuration block for loading **user or item clustering information**.

- **labels**: A nested configuration block that maps custom dataset column names to WarpRec’s internal schema.

- **dtypes**: A nested configuration block that overrides default datatypes for each column.

.. important::
   The ``labels`` and ``dtypes`` sections allow **flexibility** in handling heterogeneous datasets.
   This is especially useful when adapting real-world data sources that do not follow WarpRec’s default schema.

----------------------
Split Data Reading
----------------------

WarpRec supports ingestion of datasets that are **already split** into train, validation, and test files.
This is achieved through the ``split`` nested configuration block. To enable this feature:

1. Set ``loading_strategy`` to ``split``.
2. Provide the following options:

- **local_path**: Directory containing all split files. All splits must be located in the same directory.
- **azure_blob_prefix**: Prefix path in the Azure Blob Storage where split files are located.
- **ext**: Extension of the split files. Defaults to ``.tsv``.
- **sep**: Column separator used by the split files. Defaults to ``'\t'``.
- **header**: Whether to treat the first row of the split files as headers. Defaults to ``True``.
- **file_format**: The format of the split files. Supported values are ``tabular`` and ``parquet``.

.. note::
   Split datasets must be in a **WarpRec-compatible format**.
   Mismatched labels or inconsistent file structures will result in errors.

-----------------------------
Side Information Reading
-----------------------------

WarpRec allows integration of **side information** (e.g., item attributes or metadata).
This can be configured using the ``side`` nested section:

- **local_path**: Path to the file containing side information.
- **azure_blob_name**: Name of the Azure Blob containing side information.
- **sep**: Column separator for the side information file.
- **header**: Whether the first row of the file is a header. Defaults to ``True``.
- **file_format**: The format of the side information file. Supported values are ``tabular`` and ``parquet``.

.. tip::
   Side information can improve model performance, especially in **cold-start scenarios**. Not every model uses side information, check the :ref:`Recommenders Documentation <recommender>` for further details on each model.

---------------------------------
Clustering Information Reading
---------------------------------

WarpRec can ingest **user or item clustering information** for advanced evaluation metrics.
This is configured under the ``clustering`` nested section:

- **user_local_path**: Path to the user clustering file.
- **item_local_path**: Path to the item clustering file.
- **user_azure_blob_name**: Name of the Azure Blob containing user clustering information.
- **item_azure_blob_name**: Name of the Azure Blob containing item clustering information.
- **user_sep**: Column separator for the user clustering file.
- **item_sep**: Column separator for the item clustering file.
- **user_header**: Whether the user clustering file contains a header. Defaults to ``True``.
- **item_header**: Whether the item clustering file contains a header. Defaults to ``True``.
- **user_file_format**: The format of the user clustering file. Supported values are ``tabular`` and ``parquet``.
- **item_file_format**: The format of the item clustering file. Supported values are ``tabular`` and ``parquet``.

----------------
Labels
----------------

WarpRec expects datasets to follow a strict column schema:

.. csv-table:: Example Dataset (with rating and timestamp)
   :header: "user_id", "item_id", "rating", "timestamp"
   :widths: 15, 15, 10, 30
   :align: left

   1, 1193, 5, 978300760
   1, 661, 3, 978302109
   2, 1357, 5, 978298709
   3, 2393, 4, 978297054

If your dataset uses different column names, you can override them using the ``labels`` section:

- **user_id_label**: Custom label for user ID.
- **item_id_label**: Custom label for item ID.
- **rating_label**: Custom label for rating column.
- **timestamp_label**: Custom label for timestamp column.

.. important::
   - ``user_id`` and ``item_id`` are **mandatory**.
   - ``rating_label`` is optional if using implicit feedback.
   - ``timestamp_label`` is only required for time-based split strategies or sequential models.
   - If ``header=False``, column order must match WarpRec’s schema.

----------------
Dtypes
----------------

By default, WarpRec assumes the following datatypes:

- IDs → integers
- Ratings → floats
- Timestamps → integers

To override these defaults, use the ``dtypes`` section:

- **user_id_type**: Datatype for user IDs.
- **item_id_type**: Datatype for item IDs.
- **rating_type**: Datatype for ratings.
- **timestamp_type**: Datatype for timestamps.

.. note::
   When ``header=False``, dtype specifications will be **ignored**. Supported datatypes include:

   - ``int8``, ``int16``, ``int32``, ``int64``
   - ``float32``, ``float64``
   - ``str``

---------------------------
Example Configuration
---------------------------

Below is an example configuration for loading a **MovieLens dataset** with explicit feedback, side information, and clustering data.

.. code-block:: yaml

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
