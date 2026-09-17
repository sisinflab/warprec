# Readers

The WarpRec data reading module provides a unified interface to ingest datasets for recommendation tasks. It is designed to be flexible and extensible, allowing users to load interaction data from different sources, including:

- Local files
- Azure Blob Storage

The module abstracts the underlying data source, returning a consistent `DataFrame` object that contains user-item interactions and optionally also side information, and clustering information. This ensures that downstream components, such as dataset splitters, models, and callbacks, can operate without concern for the original data format or storage location.

WarpRec supports reading from **local files** and **Azure Blob Storage**. The backend is selected through configuration; the data format and requirements are identical regardless of the source. When using Azure Blob Storage, WarpRec automatically handles blob download or in-memory reading.

!!! info "API Reference"

    For class signatures and parameters, see the [Data Management API Reference](../api-reference/data/reader.md).

---

## Reading from a Single Source

WarpRec expects the data to be in one file, typically organized in a tabular format:

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
- **Separators:** Values must be split by a fixed separator, which can be customized (e.g., comma, tab, semicolon).
- **Required Columns:**
    - The `rating` column is required **only** for the `explicit` rating type.
    - The `timestamp` column is required **only** if a temporal strategy is used. Timestamps should ideally be provided in **numeric format** for full support, although string formats are accepted but may result in unexpected errors.

---

## Reading Pre-split Data

When reading pre-split data, WarpRec expects the split files to reside within the same directory (or virtual folder, in the case of Azure). The required directory structure is as follows:

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

---

## Reading Side Information

Side information describes the **items** of the dataset. It feeds content-based and hybrid models, provides item features to context-aware models, and is used to evaluate specific metrics. WarpRec expects the file to be formatted as:

```
item_id,Action,Comedy,Drama
1,0,1,0
2,1,0,1
3,0,0,1
...
```

- **Column Ordering is Crucial:**
    - The **first column** must contain the **item ID**.
    - All other columns will be interpreted as features.
- **Data Type:** both numeric and textual columns are accepted, and the two are interpreted differently. See [How Columns Are Interpreted](#how-columns-are-interpreted) below.
- **Error Handling:** During the configuration evaluation process, you will be notified if you attempt to use a model that requires side information but none has been provided. In that case, the experiment will be terminated.

### How Columns Are Interpreted

WarpRec derives two representations from the same file, because model families consume attributes in different ways. Content-based and hybrid models (`VSM`, `AttributeItemKNN`, `AttributeUserKNN`, `CEASE`, `AddEASE`) need an item-by-feature matrix to take similarities over, while context-aware models (`DeepFM`, `NFM`, `xDeepFM`, ...) need one embedding per attribute value. Both are built automatically, from the same columns:

| Column kind | Interpretation | Features produced |
|---|---|---|
| Numeric | kept as feature **values** | one per column |
| Non-numeric | treated as **categorical** and expanded into indicators | one per distinct value |

This means the wide layout shown above, where each attribute is already a binary column, is preserved exactly as written. A compact categorical layout works just as well, and is expanded for you:

```
item_id,genre,director
1,comedy,Lubitsch
2,drama,Wilder
...
```

!!! warning "Categorical attributes encoded as integers"

    A numeric column is taken at face value, which is what you want for a genuine measurement such as a release year or a duration. A column holding category **codes** (`genre = 1, 2, 3`) therefore becomes a single ordinal feature, in which genre `1` and genre `2` look far more alike than genre `1` and genre `5`. That is rarely the intent for a similarity-based model. Either leave such attributes as text and let WarpRec expand them, or one-hot encode them yourself.

!!! tip "Several rows per item"

    A file may carry more than one row for the same item, as is natural for a tag list. Every row contributes its features to that item, so a long layout works as well as a wide one.

!!! note "Items without attributes"

    Items that do not appear in the side information file are removed from the experiment together with their interactions, so that every model in the run is compared on the same catalogue.

---

## Reading Clustering Information

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
