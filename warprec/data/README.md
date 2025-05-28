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

WarpRec is a highly customizable framework, here are some of the requirements of the raw data file and also what is customizable:

- A header [user_id,item_id,rating,timestamp] not in a particular order.
    - Column labels can be customized through configuration.
    - The file can contain more columns, only the ones with the correct names will be considered.
- Values separated by a fixed separator that can be customized.
- Rating column is required only for the `explicit` rating type.
- Timestamp column is required only if a temporal strategy is used.

### 📁 Reading splits from a local source

When reading splits from a local source, WarpRec expects the files to be in the same directory. The structure should be similar to this:

```plaintext
split_dir/
├── train.tsv
├── test.tsv
└── val.tsv
```

The single files are expected to be in the same format as regular data files. In this case, for pre-split data, training set and test set are required, validation split is optional.

### 📁 Reading side information from a local source

When reading side information from a local source, WarpRec expects the file to be formatted as such:

```plaintext
item_id,feature_1,feature_2,...
1,2,1,...
2,3,1,...
3,1,5,...
...
```

In this case the ordering of the columns if fundamental: the first column must the the item ID, while the all the other columns will be considered features. WarpRec expects the information inside this file to be all numerical, so the user must provide already processed data. Side information are used to train some models and to evaluate certain metrics. During the process of configuration evaluation, you will be notified if you are trying to use a model that requires side information, but none have been provided. In that case, the experiment will be terminated.

### 📁 Reading clustering information from a local source

When reading clustering information from a local source, WarpRec expects the file to be formatted as such:

```plaintext
user_id,cluster
1,2
2,3
3,1
...
```

Some general information about the clustering files accepted:

- Labels are important, they need to be consisted with the other files.
- The cluster must be numerated starting from 1, as the `cluster 0` is used as fallback.
    - In case of incorrect numeration, the framework will take care of this step for you.

## ✍️ Writer

The `Writer` module of WarpRec is a lightweight solution to track all the experiment results, models and configuration.

### 🗂️ Writing in a local source

When starting an experiment, WarpRec will setup a directory for you experiment (unless prompted to not do so). This is the general structure of an experimentation folder:

```plaintext
experiment_dir/
├── timestamp_1/
|   ├── evaluation/
|   ├── recs/
|   ├── serialized/
|   ├── split/
|   └── config.json
├── timestamp_2/
|   ├── evaluation/
...
```

Every timestamp represents an experiment for that specific experimentation folder. Let's describe now each element that you will find inside the experimentation folder:

- **evaluation folder**: This folder will contain the results of the evaluation of each model, unless prompted not to do so.
- **recs folder**: This folder will contain the final recommendation of each model, if the option has been set to true.
- **serialized folder**: This folder will contain the serialized version of each model, if the option has been set to true.
- **split folder**: This folder will contain the splits of the dataset, if the option has been set to true.
- **config.json**: This is an extended version of the configuration used for the experiment, if some values have been set to default, inside this configuration you can check precisely what parameters WarpRec used to perform each operation.

## 🔀 Splitter

The `Splitter` module of WarpRec is dedicated to the split of the data into train, test and validation set. The test set is mandatory, while the validation set is optional. The module supports different strategies to split the data, in the following section you can find a description of every strategy and how it splits the data.

The WarpRec splitting module is built to be efficient on large datasets and also is robust in edge cases where certain users don't respect splitting criteria: in that case WarpRec handles the user by ensuring that he appears inside the training set at least one time.

### 🎲 Random splitting

The `Random splitting` will split the data using a random number generator. In order to be reproducible, every experiment will be associated with a seed. Random strategies available in WarpRec are:

- **random holdout**: The random holdout will extract from the original data a given ratio of transactions.
- **random leave k out**: The random leave k out will extract from the original data a given number of transactions for each user.

### ⌛ Temporal splitting

The `Temporal splitting` will split the data using the timestamp. If a temporal splitting strategy is chosen inside the configuration, the timestamp column will be mandatory inside the raw data. Temporal strategies available in WarpRec are:

- **temporal holdout**: The temporal holdout will extract from the original data a given ration of the latest transactions.
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
