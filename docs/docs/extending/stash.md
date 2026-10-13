# Add Data to the Stash

The WarpRec Dataset is designed to be easily serializable in order to be efficiently stored in the Ray object store. During the training pipeline, loading custom data inside the model can be done, but might result difficult. Using the `Stash` is a more flexible way to store custom data that can be later retrieved during training.

WarpRec exposes two main methods to interact with the `Stash`:

- `add_to_stash(key: str, value: Any) -> None`:
  This method allows you to add a new entry to the `Stash`. The `key` parameter is a string that identifies the entry, while the `value` parameter can be any Python object.
- `get_stash() -> Dict[str, Any]`:
  This method retrieves the entire contents of the `Stash` as a dictionary.

Let's see an example of how to use these methods within a custom script to load images inside the dataset:

```python
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
```

At this point, the dataset's `Stash` contains a tensor of preprocessed images that can be retrieved and used in the model training phase. WarpRec always passes the `Stash` to the model during its initialization, making it easy to access custom data. This is an example on how to retrieve the images inside a custom model:

```python
class CustomModel(Recommender):
    def __init__(self, params, info, *args, seed=42, **kwargs):
        super().__init__(params, info, *args, seed=seed, **kwargs)
        self.item_images = kwargs.get("item_images")
```

## Reserved Names

The stash is passed as keyword arguments to every model and, through the evaluator, to every metric, next to the arguments WarpRec gives them itself. An entry named like one of those would replace it (a `num_items` entry used to replace the catalogue size of every metric) or make every model fail, so these names are refused:

- given to models: `name`, `params`, `info`, `interactions`, `transactions`, `knowledge`, `multimodal`, `sessions`, `seed`, `block_size`, `chunk_size`, `checkpoint`, `strict`;
- given to metrics: `k`, `num_users`, `num_items`, `item_interactions`, `item_indices`, `train_matrix`, `feature_lookup`, `user_cluster`, `item_cluster`, `propensity`.

`add_to_stash` raises an [`ExtensionError`](../api-reference/errors.md) (a `ValueError`) naming the key as soon as one is added. An entry written into the dictionary `get_stash()` returns is checked where the stash is passed on, when a model or the evaluator is built.

!!! note

    Every metric receives the whole stash, so an entry named like a metric's own parameter (for example `relevance` for EPC and EFD) sets that parameter for every metric that has it. Give stash entries names specific to your data, such as `item_images`.
