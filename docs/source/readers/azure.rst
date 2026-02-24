Azure Reader
=============

The ``Azure Reader`` of **WarpRec** handles the reading of raw data, pre-split data, and side information directly from **Azure Blob Storage**.
It provides seamless integration with Azure’s cloud storage while maintaining the same flexibility and configurability of the ``Local Reader``.

Reading from a Single Azure Source
----------------------------------

When reading from a single Azure source, WarpRec expects the data to be stored as a **blob** within a specified container.
The file format and structure are identical to those used by the ``Local Reader``::

    user_id,item_id,rating,timestamp
    1,42,2,1651939200
    2,17,5,1651942800
    3,42,1,1651946400
    ...

WarpRec will automatically handle the download or in-memory reading of the blob content. The file can be in **CSV**, **TSV**, or any other tabular text format, as long as the separator matches the configuration.

- **Header and Columns:**
    - The same column labels are required: ``user_id``, ``item_id``, ``rating``, and ``timestamp`` (order-independent).
    - Column names can be customized through configuration.
    - Extra columns are ignored unless specified in the configuration.

- **Separators:** The separator must be fixed and match the one defined in the configuration (e.g., comma, tab, semicolon).

- **Required Columns:**
    - ``rating`` is mandatory only for the **explicit** rating type.
    - ``timestamp`` is mandatory only when a **temporal splitting strategy** is used.
      Numeric timestamps are recommended for best performance.

Reading Pre-split Data from Azure
---------------------------------

WarpRec supports reading pre-split datasets stored in Azure Blob Storage.
The structure should mirror the one used for local data but in a **cloud-based directory-like organization**::

    azure-container/
    ├── split_dir/
    │   ├── train.tsv
    │   ├── validation.tsv
    │   ├── test.tsv
    │   ├── 1/
    │   │   ├── train.tsv
    │   │   ├── validation.tsv
    │   └── 2/
    │       ├── train.tsv
    │       ├── validation.tsv

- Each split file must conform to the same schema as single-source datasets.
- Both **training** (e.g., ``train.tsv``) and **test** (e.g., ``test.tsv``) sets must be provided.
- Optional fold subdirectories (e.g., ``1/``, ``2/``) are supported for cross-validation setups.
- WarpRec automatically lists and downloads blobs from the specified path.

Reading Side Information
------------------------

Side information files stored in Azure Blob Storage follow the same structure and requirements as local files::

    item_id,feature_1,feature_2,...
    1,2,1,...
    2,3,1,...
    3,1,5,...
    ...

- **Column Ordering:**
    - The **first column** must contain the **item ID**.
    - All subsequent columns are treated as **numerical features**.

- **Data Type:** All values must be **numeric** and **preprocessed** before upload.

- **Error Handling:**
  WarpRec checks model requirements during configuration.
  If a model needs side information and none is provided, the experiment will automatically stop with a clear message.

Reading Clustering Information
------------------------------

Clustering files can also be read directly from Azure Blob Storage, using the same format::

    user_id,cluster
    1,2
    2,3
    3,1
    ...

- **Header:** The header must be consistent with the other files.
- **Cluster Numeration:**
  Cluster IDs must start from **1**, as ``cluster 0`` is reserved as a fallback.
  If cluster numbering is inconsistent, WarpRec will automatically reindex them as needed.
