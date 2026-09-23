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

!!! tip "Items without interactions: cold start"

    The opposite case, an item that carries attributes but was never interacted with, is dropped by default. Setting `reader.side.keep_unseen_items: True` keeps it in the catalogue, which is what a cold-start experiment needs: the item becomes an all-zero column of the interaction matrix, so a content-based or hybrid model can recommend it from its attributes while the collaborative models score it last.

---

## Reading Contextual Information

Contextual columns describe the **circumstances of an interaction** rather than the user or the item: the time of day, the weather, the device, whoever the user was with. They are named in `reader.labels.context_labels` and live in the interaction file itself, one row per interaction:

```
user_id,item_id,rating,daytime,weather
1,42,4,morning,sunny
1,42,5,evening,rainy
2,17,3,evening,sunny
...
```

### How a Contextual Column Is Interpreted

A context column is read as one of three kinds of field, and each contributes exactly one vector to the models, so the number of fields never changes:

| Kind | When | How it is used |
|---|---|---|
| **Categorical** | the default | The value is mapped to an index, with `0` reserved for values unseen during training |
| **Numeric** | the column holds floats, or is declared as a float type in `dtypes.context_types` | The value is kept as it is and scales a single embedding, so the ordering the numbers carry survives |
| **Multi-valued** | the column is named in `dtypes.context_separators` | The cell is split on the separator, the values share one vocabulary, and their embeddings are pooled into one vector |

```yaml
reader:
    labels:
        context_labels: [weather, temperature, genres]
    dtypes:
        context_types:
            temperature: float32      # a measurement, not a category
        context_separators:
            genres: '|'               # "action|comedy" is two values, not one category
training:
    sequence_pooling: mean            # how multi-valued fields are combined
```

!!! warning "A measurement encoded as a category"

    A numeric column is only treated as a measurement when it reads as a float. An integer column such as a year or a rating count is taken as categorical, which invents one token per distinct value and throws the ordering away. Declare it as `float32` under `dtypes.context_types` when you want it treated as a quantity.

!!! note "Why the default pooling is the mean"

    `mean` is the embedding equivalent of the normalised multi-hot block that the factorization-machine literature defines these models over, so a field's contribution does not grow with the number of values a row happens to hold. `sum` and `max` are available through `training.sequence_pooling`.

- **Repeated pairs are expected and preserved.** The first two rows above describe the same user and the same item in two different situations. Both are used for training: that is the signal a context-aware model exists to learn.
- **Every context column is categorical.** Values are mapped to integer indices, with `0` reserved for values that were not seen during training, so a value appearing only in the test set is treated as unknown rather than as a new category.
- **The interaction matrix still holds one cell per pair.** Models that work on the matrix rather than on the rows — collaborative filtering, content-based — aggregate the repeated rows according to `reader.duplicates`, which defaults to `max`.

!!! note "Contexts and the models that ignore them"

    Only context-aware models read these columns. Providing them does not change what a collaborative model sees, beyond the duplicate aggregation described above.

---

## Reading a Knowledge Graph

A knowledge graph states facts about the **entities** an item stands for: who directed a film, which genre it belongs to, where its director was born. It feeds the knowledge-aware models. WarpRec expects it as **two files**, which is the shape the published graphs are distributed in.

The first holds the facts, as `(head, relation, tail)` triples:

```
m1	film.directed_by	person.kubrick
m1	film.genre	genre.drama
m2	film.genre	genre.drama
person.kubrick	person.born_in	country.uk
...
```

The second says which entity each catalogue item stands for, because the graph is written about entities and the interactions are written about items:

```
1	m1
2	m2
3	m3
...
```

- **Column Ordering is Crucial:** both files are read positionally. The triples file is `head`, `relation`, `tail`; the alignment file is `item_id`, `entity_id`. The names are set through `column_names` and `link_column_names`.
- **Header:** both files are assumed to have **no header row** by default, which is how such graphs are usually published. Set `header: True` when yours does.
- **Both files are required.** The triples alone do not connect the graph to the catalogue.
- **Error Handling:** during the configuration evaluation process, you will be notified if you attempt to use a model that requires a knowledge graph but none has been provided. In that case, the experiment will be terminated.

### How the Graph Is Interpreted

Entities and relations are mapped into an index space of their own, in the same way users and items are:

| File content | Interpretation |
|---|---|
| A head or a tail | One **entity**; both ends share a single vocabulary, so the same identifier is the same node wherever it appears |
| A relation | One **relation**, embedded separately from the entities |
| An alignment row | One item of the catalogue is declared to *be* a given entity |

!!! note "Entities beyond the items"

    An entity that stands for no item is kept. A fact two hops away from an item, such as the country a film's director was born in, is exactly what the propagating models exist to reach, so the graph is never trimmed to the catalogue.

!!! note "Items the graph is silent about"

    An item that is not aligned to any entity, or is aligned to an entity that appears in no triple, keeps all of its interactions and simply carries no facts. Such an item is scored from the collaborative half of the model alone; it is never dropped from the catalogue.

!!! tip "Alignments beyond the catalogue"

    The alignment file is written for the whole graph, not for one split, so it may name items that filtering or splitting removed. Those rows are ignored.

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
