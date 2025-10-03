Local Reader
============

The ``Local Reader`` of **WarpRec** handles the reading of raw data from a local source.

Reading from a Single Local Source
----------------------------------

When reading from a single local source, WarpRec expects the data to be in one file, typically organized in a tabular format::

    user_id,item_id,rating,timestamp
    1,42,2,1651939200
    2,17,5,1651942800
    3,42,1,1651946400
    ...

WarpRec is a highly customizable framework; here are the requirements and customization options for the raw data file:

- **Header and Columns:**
    - A header with the following labels is expected (order is not important): ``user_id``, ``item_id``, ``rating``, ``timestamp``.
    - Column labels can be customized through configuration.
    - The file can contain more columns; only those with the configured names will be considered.
- **Separators:** Values must be split by a fixed separator, which can be customized.
- **Required Columns:**
    - The ``rating`` column is required **only** for the ``explicit`` rating type.
    - The ``timestamp`` column is required **only** if a temporal strategy is used. Timestamps should ideally be provided in **numeric format** for full support, although string formats are accepted but may result in unexpected errors.

Reading Pre-split Local Data
----------------------------

When reading pre-split data from a local source, WarpRec expects the split files to reside within the same directory. The required directory structure is as follows::

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

- Each individual file is expected to follow the **same format** as unsplit dataset files.
- In this setup, both the **training** (e.g., ``train.tsv``) and **test** (e.g., ``test.tsv``) sets must be provided.
- The train/validation folds (e.g., directories ``1/``, ``2/``) are optional.

Reading Side Information
------------------------

Side information is used to train certain models and evaluate specific metrics. WarpRec expects the side information file to be formatted as::

    item_id,feature_1,feature_2,...
    1,2,1,...
    2,3,1,...
    3,1,5,...
    ...

- **Column Ordering is Crucial:**
    - The **first column** must contain the **item ID**.
    - All other columns will be interpreted as features.
- **Data Type:** WarpRec expects all feature data in this file to be **numerical**. The user must provide preprocessed input.
- **Error Handling:** During the configuration evaluation process, you will be notified if you attempt to use a model that requires side information but none has been provided. In that case, the experiment will be terminated.

Reading Clustering Information
------------------------------

When reading clustering information, WarpRec expects the file to be formatted as follows::

    user_id,cluster
    1,2
    2,3
    3,1
    ...

- **Header:** The header is important and needs to be consistent with the other files.
- **Cluster Numeration:** The clusters must be numbered starting from **1**, as ``cluster 0`` is reserved as a fallback.
    - In case of incorrect numeration, the framework will automatically handle this step.
