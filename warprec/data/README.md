# 📦 [WarpRec](../../README.md) Data

The `data` module of WarpRec offers a wide variety of data structures used by the framework to handle the raw data and transform it into the correct form desired by the different recommendation models.

## 📚 Table of Contents
-  🔍 [Reader](#🔍-reader)
    - 📁 [Reading from a local source](#📁-reading-from-a-local-source)
    - 📁 [Reading splits from a local source](#📁-reading-splits-from-a-local-source)
    - 📁 [Reading side information from a local source](#📁-reading-side-information-from-a-local-source)
    - 📁 [Reading clustering information from a local source](#📁-reading-clustering-information-from-a-local-source)
-  ✍️ [Writer](#️✍️-writer)
    - 🗂️ [Writing in a local source](#️🗂️-writing-in-a-local-source)
-  🔀 [Splitter](#🔀-splitter)
    - 🎲 [Random splitting](#🎲-random-splitting)
    - ⌛ [Temporal splitting](#⌛-temporal-splitting)
    - 📅 [Timestamp slicing](#📅-timestamp-slicing)
-  📊 [Dataset](#📊-dataset)
    - 🤝 [Interactions](#🤝-interactions)
    - 🧠 [Sessions](#🧠-sessions)

## 🔍 Reader

The `Reader` module of WarpRec handles the reading of the raw data from a given source. The module supports different types of data and takes care of the required preprocessing for you.

### 📁 Reading from a local source

When reading from a local source, WarpRec expects the data to be in a single file organized as such:

```plaintext
user_id,item_id,rating,timestamp
1,42,2,1651939200
2,17,5,1651942800
3,42,1,1651946400
...
```

WarpRec is a highly customizable framework, and here are some of the requirements for the raw data file and what can be customized:

- A header [user_id,item_id,rating,timestamp] not in a particular order.
    - Column labels can be customized through configuration.
    - The file can contain more columns, only the ones with the correct names will be considered.
- Values split by a fixed separator, which can be customized.
- The rating column is required only for the `explicit` rating type.
- The timestamp column is required only if a temporal strategy is used. Timestamps should be provided in numeric format for full support — although string formats are accepted, they may result in unexpected errors.

### 📁 Reading splits from a local source

When reading splits from a local source, WarpRec expects the files to be in the same directory. The structure should be like this:

```plaintext
split_dir/
├── train.tsv
├── test.tsv
└── val.tsv
```

Each individual file is expected to follow the same format as unsplit dataset files. In this setup, both the training and test sets must be provided, while the validation set is optional.

### 📁 Reading side information from a local source

When reading side information from a local source, WarpRec expects the file to be formatted as such:

```plaintext
item_id,feature_1,feature_2,...
1,2,1,...
2,3,1,...
3,1,5,...
...
```

In this case, the ordering of the columns is crucial: the first column must contain the item ID, while all the other columns will be interpreted as features. WarpRec expects all the data in this file to be numerical, so the user must provide preprocessed input. Side information is used to train certain models and to evaluate specific metrics. During the configuration evaluation process, you will be notified if you attempt to use a model that requires side information but none has been provided. In that case, the experiment will be terminated.

### 📁 Reading clustering information from a local source

When reading clustering information from a local source, WarpRec expects the file to be formatted as follows:

```plaintext
user_id,cluster
1,2
2,3
3,1
...
```

Some general information about the clustering files accepted:

- The header is important and needs to be consistent with the other files.
- The clusters must be numbered starting from 1, since `cluster 0` is reserved as a fallback.
    - In case of incorrect numeration, the framework will take care of this step for you.

## ✍️ Writer

The `Writer` module of WarpRec is a lightweight solution to track all the experiment results, models and configuration.

### 🗂️ Writing in a local source

When starting an experiment, WarpRec will set up a directory for your experiment by default, unless this behavior is disabled via configuration. This is the general structure of an experimentation folder:

```plaintext
experiment_dir/
├── evaluation/
├── params/
├── recs/
├── serialized/
├── split/
└── config.json
```

Results produced with WarpRec will be saved inside this folder with a timestamp. Let's describe now each element that you will find inside the experimentation folder:

- **evaluation folder**: This folder will contain the results of the evaluation of each model, unless prompted not to do so.
- **recs folder**: This folder will contain the final recommendation of each model, if the option has been set to true.
- **serialized folder**: This folder will contain the serialized version of each model, if the option has been set to true.
- **split folder**: This folder will contain the splits of the dataset, if the option has been set to true.
- **config.json**: This is an extended version of the configuration used for the experiment, if some values have been set to default, inside this configuration you can check precisely what parameters WarpRec used to perform each operation.

## 🔀 Splitter

The `Splitter` module in WarpRec is responsible for dividing the data into training and test sets, with an optional validation set. While the test set is mandatory, including a validation set is optional. The module supports various data splitting strategies, each of which is described in the following section along with how it partitions the data.

The WarpRec splitting module is built to be efficient on large datasets and robust in edge cases where certain users don't respect splitting criteria: in that case WarpRec handles the user by ensuring that they appear in the training set at least once.

### 🎲 Random splitting

The `Random splitting` will split the data using a random number generator. In order to be reproducible, every experiment will be associated with a seed. Random strategies available in WarpRec are:

- **random holdout**: The random holdout will extract from the original data a given ratio of transactions.
- **random leave k out**: The random leave k out will extract from the original data a given number of transactions for each user.

### ⌛ Temporal splitting

The `Temporal splitting` will split the data using the timestamp. If a temporal splitting strategy is chosen inside the configuration, the timestamp column will be mandatory inside the raw data. Temporal strategies available in WarpRec are:

- **temporal holdout**: The temporal holdout will extract from the original data a given ratio of the latest transactions.
- **temporal leave k out**: The temporal leave k out will extract from the original data a given number of the latest transactions for each user.

### 📅 Timestamp slicing

The `Timestamp slicing` will split the data based on a timestamp given as input. Every transaction before that timestamp will be considered training, every transaction after will be considered test. Validation is not available for this strategy. There is also a variant of this strategy called `best`. In that case WarpRec will find the timestamp that will better split the data.

## 📊 Dataset

The `Dataset` module in WarpRec is at the core of data management for recommendation models. It's designed as a heterogeneous ecosystem that facilitates preprocessing, alignment of auxiliary information (side information), and efficient data organization for training and evaluation.

Its main functionalities include:

- **Sub-structure data handling**: Organizes and structures raw data into formats usable by the framework.
- **Side information alignment**: Automatically integrates and aligns additional information (like item features) with transaction data.
- **ID to index mapping**: Converts original user and item IDs into sequential numerical indices, optimizing memory and performance.
- **Batch iteration**: Provides an efficient mechanism for iterating over data in batches, essential for training deep learning models.

### 🤝 Interactions

`Interactions` is a fundamental class within the `Dataset` module, responsible for **storing and efficiently accessing interaction data** (users, items, ratings, timestamps) in various formats.

**Supported data formats**:
- **Dictionary**: Useful for quick access to a single user's interactions (e.g., `{user_id: {item_id: rating}}`).
- **DataFrame**: Maintains a copy of the raw data in tabular format for easy manipulation.
- **CSR Sparse Matrix**: The primary representation for interaction data, optimized for efficient sparse matrix operations, common in recommendation models.
- **CSR Sparse Matrix for Side Information**: If side information is present, it's converted into a sparse matrix for efficient access.

### 🧠 Sessions
The `Sessions` class is part of the `Dataset` module and is designed to **generate session-aware training data for sequential recommendation models**.

This component processes raw interaction data (user, item, timestamp) to produce user-item sequences and corresponding targets, optionally with negative sampling, enabling models to learn temporal user behavior.

Key responsibilities:

- Convert timestamped interaction logs into user-item sequences.
- Automatically truncate sequences to a fixed length (`max_seq_len`).
- Support user-specific batching with optional user IDs.
- Efficient negative sampling avoiding items in the history or current target.
- Internal caching mechanism to speed up repeated dataloader creation.
