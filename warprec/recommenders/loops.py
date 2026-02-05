import os
from typing import Optional
from tqdm.auto import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import LRScheduler as LRSchedulerBaseClass

from warprec.data import Dataset
from warprec.common import standard_optimizer
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.utils.config import LRScheduler
from warprec.utils.registry import lr_scheduler_registry
from warprec.utils.logger import logger


def train_loop(
    model: IterativeRecommender,
    dataset: Dataset,
    epochs: int,
    lr_scheduler: Optional[LRScheduler] = None,
    device: str = "cpu",
):
    """Simple training loop decorated with tqdm.

    Args:
        model (IterativeRecommender): The model to train.
        dataset (Dataset): The dataset used to train the model.
        epochs (int): The number of epochs of the training.
        lr_scheduler (Optional[LRScheduler]): The learning rate scheduler configuration.
        device (str): The device used for training. Defaults to "cpu".
    """
    logger.msg(f"Starting the training of model {model.name}")

    model.to(device)

    # Compute optimization parameters
    num_workers = 0
    pin_memory = False
    persistent_workers = False
    if device == "cuda":
        allocated_cpus = os.cpu_count() or 1
        num_workers = max(allocated_cpus - 1, 1)
        pin_memory = True
        persistent_workers = True

    train_dataloader = model.get_dataloader(
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    optimizer = standard_optimizer(model)

    # Check for learning rate scheduler
    scheduler = None
    if lr_scheduler is not None:
        # Initialize the lr scheduler
        scheduler = lr_scheduler_registry.get(
            lr_scheduler.name, optimizer=optimizer, **lr_scheduler.params
        )

    model.train()
    for epoch in tqdm(range(epochs), desc="Training Model"):
        epoch_loss = 0.0
        for _, batch in tqdm(
            enumerate(train_dataloader),
            desc=f"Epoch {epoch + 1} Batch",
            leave=False,
            total=len(train_dataloader),
        ):
            batch = [x.to(device) for x in batch]
            optimizer.zero_grad()

            loss = model.train_step(batch, epoch)
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()

        if scheduler is not None and isinstance(scheduler, LRSchedulerBaseClass):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(epoch_loss)
            else:
                scheduler.step()

        tqdm.write(
            f"Epoch {epoch + 1}, Loss: {(epoch_loss / len(train_dataloader)):.4f}"
        )

    logger.positive(f"Training of {model.name} completed successfully.")
