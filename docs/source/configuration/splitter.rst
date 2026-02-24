#########################
Splitter Configuration
#########################

The **Splitter Configuration** module defines how a dataset is partitioned into **training**, **validation**,
and **test sets** prior to model training.
Proper splitting is crucial to build a reliable evaluation pipeline and ensure fair comparison between models.

WarpRec provides multiple splitting strategies that can be tailored to your experimental needs, including:

- Temporal strategies
- Random strategies
- Timestamp slicing
- K-Fold Cross Validation

.. important::
  - Temporal strategies require that **timestamps** are present in the dataset loaded by the reader.
  - **Test set** is required for train and design pipelines; **validation set** is optional.

-----------------------------
Supported Splitting Strategies
-----------------------------

**1. Temporal Holdout**

Orders interactions by **timestamp** and reserves a portion of the most recent interactions as the test set.

.. code-block:: yaml

   splitter:
     test_splitting:
       strategy: temporal_holdout
       ratio: 0.1

**2. Temporal Leave-K-Out**

Orders interactions by **timestamp** and leaves exactly **K** interactions per user for the test set.
Users with fewer than K interactions remain entirely in the training set.

.. code-block:: yaml

   splitter:
     test_splitting:
       strategy: temporal_leave_k_out
       k: 1

**3. Random Holdout**

Randomly selects a portion of interactions to include in the test set.

.. code-block:: yaml

   splitter:
     test_splitting:
       strategy: random_holdout
       ratio: 0.1

**4. Random Leave-K-Out**

Randomly selects **K** interactions per user to include in the test set.
Users with fewer than K interactions remain entirely in the training set.

.. code-block:: yaml

   splitter:
     test_splitting:
       strategy: random_leave_k_out
       k: 1

**5. Timestamp Slicing**

Splits the dataset based on a specific **timestamp**:

- Interactions before the timestamp → training set
- Interactions after the timestamp → test set

WarpRec also supports the special keyword ``best`` to automatically select an optimal timestamp.

.. code-block:: yaml

   splitter:
     test_splitting:
       strategy: timestamp_slicing
       timestamp: 10009287 | best

**6. K-Fold Cross Validation**

Partitions the dataset into **K folds**, using K-1 folds for training and the remaining fold for validation.
The process is repeated K times to exhaust all possible fold combinations.

.. code-block:: yaml

   splitter:
     validation_splitting:
       strategy: k_fold_cross_validation
       folds: 10

.. note::
   - This strategy is applicable **only for validation sets**, not test sets.
   - Provides less biased and more accurate evaluation metrics, but requires additional training time.

-----------------------------
Example Splitter Configuration
-----------------------------

This example demonstrates a full configuration:
- Test set split using **temporal holdout** (10% of the most recent interactions)
- Validation set using **10-fold cross validation**

.. code-block:: yaml

   splitter:
     test_splitting:
       strategy: temporal_holdout
       ratio: 0.1
     validation_splitting:
       strategy: k_fold_cross_validation
       folds: 10
