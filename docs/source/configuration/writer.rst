#####################
Writer Configuration
#####################

The **Writer Configuration** module defines how and where WarpRec saves the output of experiments.
This includes generated recommendations, evaluation metrics, and data splits created during preprocessing.

By customizing file formats, paths, and naming conventions, WarpRec allows precise control of output organization.
This feature is especially useful when running multiple experiments simultaneously or when managing several datasets.

.. note::
   The writer is essential for **reproducibility** and **traceability**.
   A properly configured writer ensures that experiment outputs are consistently stored and can be easily compared across runs.

.. warning::
   Misconfigured writer settings can result in **loss of results** or overwriting previous experiment outputs.
   Always verify the experiment path and labels before running large-scale experiments.

---------------------
Available Keywords
---------------------

The following keywords are available for configuring the writer:

- **dataset_name**: The name of the dataset currently being used.
  This value will be used in naming output directories and files to ensure traceability.

- **writing_method**: The strategy used to write output files:

  * ``local`` – Writes data to local filesystem.
  * ``azure_blob`` – Writes data to an Azure Blob Storage. Needs *remote-io* extra to be installed.

- **local_experiment_path**: The directory path where experiment files will be stored.
  This parameter is **required** when using the ``local`` writing method.

- **azure_blob_experiment_container**: The name of the Azure Blob container where experiment files will be stored.
  This parameter is **required** when using the ``azure_blob`` writing method.

- **save_split**: Whether to save the train/validation/test splits generated during preprocessing. Defaults to ``False``.

- **results**: A nested section controlling the formatting of result files.

- **split**: A nested section controlling the formatting of split files.

- **recommendation**: A nested section controlling the formatting of recommendation files.

.. important::
   Consistency between **Reader** and **Writer** configurations is crucial.
   For example, if custom labels are defined in the reader, they must be mirrored in the writer to ensure correct file generation.

-----------------
Results
-----------------

The ``results`` nested section defines how **result files** (e.g., evaluation metrics) are formatted:

- **sep**: Column separator for output files. Default: ``\t``.
- **ext**: Extension for output files. Default: ``.tsv``.

.. tip::
   Using a comma (``','``) as separator and ``.csv`` as extension makes outputs compatible with spreadsheet tools such as Excel.

-----------------
Split
-----------------

The ``split`` nested section configures how **data split files** are saved:

- **sep**: Column separator for split files. Default: ``\t``.
- **ext**: Extension for split files. Default: ``.tsv``.
- **header**: Boolean flag indicating whether the first row should contain column names. Defaults to ``True``.


-----------------
Recommendation
-----------------

The ``recommendation`` nested section defines the structure of **recommendation output files**:

- **sep**: Column separator for output files. Default: ``\t``.
- **ext**: Extension for output files. Default: ``.tsv``.
- **header**: Whether to include column headers in the output. Defaults to ``True``.
- **k**: Number of recommendations generated per user. Defaults to ``50``.
- **user_label**: Label used for the user column. Defaults to ``user_id``.
- **item_label**: Label used for the item column. Defaults to ``item_id``.
- **rating_label**: Label used for the recommendation score. Defaults to ``rating``.

.. note::
   The ``k`` parameter controls the **top-k recommendations** stored per user.
   Increasing this value may lead to larger files and slower post-processing.

---------------------------
Example Configuration
---------------------------

The following example shows a complete writer configuration that saves experiment results locally with customized formatting:

.. code-block:: yaml

   writer:
       dataset_name: movielens
       writing_method: local
       local_experiment_path: results/movielens_experiment
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
