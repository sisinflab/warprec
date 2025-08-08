from tqdm.auto import tqdm

import torch

from warprec.data.dataset import Dataset
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.utils.logger import logger


def train_loop(model: IterativeRecommender, dataset: Dataset, epochs: int):
    """Simple training loop decorated with tqdm.

    Args:
        model (IterativeRecommender): The model to train.
        dataset (Dataset): The dataset used to train the model.
        epochs (int): The number of epochs of the training.
    """
    logger.msg(f"Starting the training of model {model.name}")

    train_dataloader = model.get_dataloader(
        interactions=dataset.train_set, sessions=dataset.train_session
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
        tqdm.write(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

    logger.positive(f"Training of {model.name} completed successfully.")
