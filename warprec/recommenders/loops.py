from typing import Optional
from tqdm.auto import tqdm

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import LRScheduler as LRSchedulerBaseClass

from warprec.data import Dataset
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.utils.config import LRScheduler
from warprec.utils.registry import lr_scheduler_registry
from warprec.utils.logger import logger


def train_loop(
    model: IterativeRecommender,
    dataset: Dataset,
    epochs: int,
    lr_scheduler: Optional[LRScheduler] = None,
    low_memory: bool = False,
    device: str = "cpu",
):
    """Simple training loop decorated with tqdm.

    Args:
        model (IterativeRecommender): The model to train.
        dataset (Dataset): The dataset used to train the model.
        epochs (int): The number of epochs of the training.
        lr_scheduler (Optional[LRScheduler]): The learning rate scheduler configuration.
        low_memory (bool): Wether or not to compute dataloader in
            lazy mode.
        device (str): The device used for training. Defaults to "cpu".
    """
    logger.msg(f"Starting the training of model {model.name}")

    model.to(device)

    train_dataloader = model.get_dataloader(
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        low_memory=low_memory,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay
    )

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
