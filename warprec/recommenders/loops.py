from tqdm.auto import tqdm

import torch

from warprec.data import Dataset
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.utils.logger import logger


def train_loop(
    model: IterativeRecommender, dataset: Dataset, epochs: int, low_memory: bool = False
):
    """Simple training loop decorated with tqdm.

    Args:
        model (IterativeRecommender): The model to train.
        dataset (Dataset): The dataset used to train the model.
        epochs (int): The number of epochs of the training.
        low_memory (bool): Wether or not to compute dataloader in
            lazy mode.
    """
    logger.msg(f"Starting the training of model {model.name}")

    train_dataloader = model.get_dataloader(
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        low_memory=low_memory,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay
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
            optimizer.zero_grad()

            loss = model.train_step(batch, epoch)
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()
        tqdm.write(
            f"Epoch {epoch + 1}, Loss: {(epoch_loss / len(train_dataloader)):.4f}"
        )

    logger.positive(f"Training of {model.name} completed successfully.")
