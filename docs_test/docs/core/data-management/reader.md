# Readers

The WarpRec data reading module provides a unified interface to ingest datasets for recommendation tasks. It is designed to be flexible and extensible, allowing users to load interaction data from different sources, including:

- Local files
- Azure Blob Storage

The module abstracts the underlying data source, returning a consistent `DataFrame` object that contains user-item interactions and optionally also side information, and clustering information. This ensures that downstream components, such as dataset splitters, models, and callbacks, can operate without concern for the original data format or storage location.

Key features include:

- Automatic reading of data from source
- Support of different data types

This design allows WarpRec to maintain **flexibility, reproducibility, and scalability**, supporting a wide range of experimental pipelines and real-world recommendation scenarios.

---

## Local Reader

The `Local Reader` of **WarpRec** handles the reading of raw data from a local source.

### Reading from a Single Local Source

When reading from a single local source, WarpRec expects the data to be in one file, typically organized in a tabular format:

```
user_id,item_id,rating,timestamp
1,42,2,1651939200
2,17,5,1651942800
3,42,1,1651946400
...
```

WarpRec is a highly customizable framework; here are the requirements and customization options for the raw data file:

- **Header and Columns:**
    - A header with the following labels is expected (order is not important): `user_id`, `item_id`, `rating`, `timestamp`.
    - Column labels can be customized through configuration.
    - The file can contain more columns; only those with the configured names will be considered.
- **Separators:** Values must be split by a fixed separator, which can be customized.
- **Required Columns:**
    - The `rating` column is required **only** for the `explicit` rating type.
    - The `timestamp` column is required **only** if a temporal strategy is used. Timestamps should ideally be provided in **numeric format** for full support, although string formats are accepted but may result in unexpected errors.

### Reading Pre-split Local Data

When reading pre-split data from a local source, WarpRec expects the split files to reside within the same directory. The required directory structure is as follows:

```
split_dir/
├── train.tsv
├── validation.tsv
├── test.tsv
├── 1/
|   ├── train.tsv
|   ├── validation.tsv
└── 2/
    ├── train.tsv
    ├── validation.tsv
```

- Each individual file is expected to follow the **same format** as unsplit dataset files.
- In this setup, both the **training** (e.g., `train.tsv`) and **test** (e.g., `test.tsv`) sets must be provided.
- The train/validation folds (e.g., directories `1/`, `2/`) are optional.

### Reading Side Information

Side information is used to train certain models and evaluate specific metrics. WarpRec expects the side information file to be formatted as:

```
item_id,feature_1,feature_2,...
1,2,1,...
2,3,1,...
3,1,5,...
...
```

- **Column Ordering is Crucial:**
    - The **first column** must contain the **item ID**.
    - All other columns will be interpreted as features.
- **Data Type:** WarpRec expects all feature data in this file to be **numerical**. The user must provide preprocessed input.
- **Error Handling:** During the configuration evaluation process, you will be notified if you attempt to use a model that requires side information but none has been provided. In that case, the experiment will be terminated.

### Reading Clustering Information

When reading clustering information, WarpRec expects the file to be formatted as follows:

```
user_id,cluster
1,2
2,3
3,1
...
```

- **Header:** The header is important and needs to be consistent with the other files.
- **Cluster Numeration:** The clusters must be numbered starting from **1**, as `cluster 0` is reserved as a fallback.
    - In case of incorrect numeration, the framework will automatically handle this step.

---

## Azure Reader

The `Azure Reader` of **WarpRec** handles the reading of raw data, pre-split data, and side information directly from **Azure Blob Storage**. It provides seamless integration with Azure's cloud storage while maintaining the same flexibility and configurability of the `Local Reader`.

### Reading from a Single Azure Source

When reading from a single Azure source, WarpRec expects the data to be stored as a **blob** within a specified container. The file format and structure are identical to those used by the `Local Reader`:

```
user_id,item_id,rating,timestamp
1,42,2,1651939200
2,17,5,1651942800
3,42,1,1651946400
...
```

WarpRec will automatically handle the download or in-memory reading of the blob content. The file can be in **CSV**, **TSV**, or any other tabular text format, as long as the separator matches the configuration.

- **Header and Columns:**
    - The same column labels are required: `user_id`, `item_id`, `rating`, and `timestamp` (order-independent).
    - Column names can be customized through configuration.
    - Extra columns are ignored unless specified in the configuration.

- **Separators:** The separator must be fixed and match the one defined in the configuration (e.g., comma, tab, semicolon).

- **Required Columns:**
    - `rating` is mandatory only for the **explicit** rating type.
    - `timestamp` is mandatory only when a **temporal splitting strategy** is used. Numeric timestamps are recommended for best performance.

### Reading Pre-split Data from Azure

WarpRec supports reading pre-split datasets stored in Azure Blob Storage. The structure should mirror the one used for local data but in a **cloud-based directory-like organization**:

```
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
```

- Each split file must conform to the same schema as single-source datasets.
- Both **training** (e.g., `train.tsv`) and **test** (e.g., `test.tsv`) sets must be provided.
- Optional fold subdirectories (e.g., `1/`, `2/`) are supported for cross-validation setups.
- WarpRec automatically lists and downloads blobs from the specified path.

### Reading Side Information

Side information files stored in Azure Blob Storage follow the same structure and requirements as local files:

```
item_id,feature_1,feature_2,...
1,2,1,...
2,3,1,...
3,1,5,...
...
```

- **Column Ordering:**
    - The **first column** must contain the **item ID**.
    - All subsequent columns are treated as **numerical features**.

- **Data Type:** All values must be **numeric** and **preprocessed** before upload.

- **Error Handling:**
  WarpRec checks model requirements during configuration. If a model needs side information and none is provided, the experiment will automatically stop with a clear message.

### Reading Clustering Information

Clustering files can also be read directly from Azure Blob Storage, using the same format:

```
user_id,cluster
1,2
2,3
3,1
...
```

- **Header:** The header must be consistent with the other files.
- **Cluster Numeration:**
  Cluster IDs must start from **1**, as `cluster 0` is reserved as a fallback. If cluster numbering is inconsistent, WarpRec will automatically reindex them as needed.
