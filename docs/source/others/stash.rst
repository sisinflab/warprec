#####
Stash
#####

The ``Stash`` acts as a flexible container for user-defined data structures.
In the field of Recommender Systems, which is constantly evolving, providing a
one-size-fits-all solution for every scenario is inherently difficult.
The ``Stash`` mechanism allows developers to persist arbitrary data that can be
retrieved later during training or evaluation.

The main way to interact with the ``Stash`` is through the **WarpRec Callback
System**, which provides hooks at key stages of the data processing and training
pipeline. You can find the full documentation of the callback system
:ref:`here <callback>`. Another way to access the ``Stash`` is through
a custom script that directly interacts with WarpRec's internal components.
You can find the full documentation on how to use WarpRec scripting :ref:`here <scripting>`.

Add Data to the Stash
---------------------

The WarpRec Dataset is designed to be easly serializable in order to be efficiently
stored in the Ray object store. During the training pipeline, loading custom data
inside the model can be done, but might results difficult. Using the ``Stash`` is a more
flexible way to store custom data that can be later retrieve during training.

WarpRec exposes two main methods to interact with the ``Stash``:

- `add_to_stash(key: str, value: Any) -> None`:
  This method allows you to add a new entry to the ``Stash``. The `key` parameter is a
  string that identifies the entry, while the `value` parameter can be any Python object.
- `get_stash() -> Dict[str, Any]`:
  This method retrieves the entire contents of the ``Stash`` as a dictionary.

Let's see an example of how to use these methods within a custom script to load images inside the dataset:

.. code-block:: python

    import os
    import torch
    from PIL import Image
    from torchvision import transforms

    # Fake dataset instance (already loaded somewhere in your script)
    dataset = main_dataset  # replace with your dataset object

    # Define a preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Fake image directory path
    image_dir = "/path/to/local/images/"

    # Example: assume items are identified by integer IDs starting from 0
    item_ids = range(dataset.num_items)

    image_tensors = []
    for item_id in item_ids:
        image_path = os.path.join(image_dir, f"{item_id}.jpg")

        if os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
            tensor_img = preprocess(img)
        else:
            # If no image is found, fallback to a zero tensor
            tensor_img = torch.zeros((3, 64, 64))

        image_tensors.append(tensor_img)

    # Stack into a single tensor: shape [num_items, 3, 64, 64]
    image_tensor_dataset = torch.stack(image_tensors, dim=0)

    # Store inside the stash for later retrieval
    dataset.add_to_stash("item_images", image_tensor_dataset)

At this point, the dataset’s ``Stash`` contains a tensor of preprocessed
images that can be retrieved and used in the model training phase. WarpRec always
passes the ``Stash`` to the model during its initialization, making it easy to
access custom data. This is an example on how to retrieve the images inside a custom model:

.. code-block:: python

    class CustomModel(Recommender):
        def __init__(self, params, interactions, *args, **kwargs):
            super().__init__(params, interactions, *args, **kwargs)
            self.item_images = kwargs.get("item_images")
