from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor
import pandas as pd
import numpy as np
from pandas import DataFrame
from elliotwo.utils.config import Configuration
from elliotwo.data.writer import AbstractWriter
from elliotwo.data.dataset import AbstractDataset
from elliotwo.utils.logger import logger


class AbstractRecommender(nn.Module, ABC):
    """Abstract class that defines the basic functionalities of a recommendation model.

    Args:
        config (Configuration): The configuration file.
        dataset (AbstractDataset): The dataset on wich the train will be executed.
        params (dict): The parameters to set up the model.
        *args: Argument for PyTorch nn.Module.
        **kwargs: Keyword argument for PyTorch nn.Module.
    """

    def __init__(
        self,
        config: Configuration,
        dataset: AbstractDataset,
        params: dict,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._config = config
        self._dataset = dataset
        self._params = params
        self._name: str = "Name not set"
        # The train set sparse matrix
        self.interaction_matrix = self._dataset.train_set.get_sparse()

    @abstractmethod
    def fit(self):
        """This method will train the model on the dataset."""

    @abstractmethod
    def _serialize(self) -> dict:
        """This method will return the part of the recommender that was learned.

        Returns:
            dict: The dictionary containign all the important informations about the model.
        """

    @abstractmethod
    def _deserialize(self, deserialized_data: dict):
        """This method load the part of the recommender from a checkpoint."""

    @abstractmethod
    def forward(self) -> Tensor:
        """This method will return the prediction of the model.

        Returns:
            Tensor: The score matrix {user x item}.
        """

    def save_model(self, writer: AbstractWriter) -> None:
        """This method will save the model state using a writer.

        Args:
            writer (AbstractWriter): The writer to use to write the model state.
        """
        logger.msg(f"Starting serialization of the model {self.name}.")
        data_to_serialize = self._serialize()
        writer.write_model(data_to_serialize, self.name)
        logger.positive(
            f"Serialization process of the model {self.name} completed succefully."
        )

    def load_model(self, deserialized_data: dict) -> None:
        """This method will load a model from a given checkpoint.

        Args:
            deserialized_data (dict): The deserialized information that \
                will be used to restore the state of the model.
        """
        logger.msg(f"Loading previous state of the model {self.name}.")
        self._deserialize(deserialized_data)
        logger.positive(f"Loading of the model {self.name} completed succefully.")

    def get_recs(self, umap_i: dict, imap_i: dict, k: int) -> DataFrame:
        """This method turns the learned parameters into new \
            recommendations in DataFrame format.

        Args:
            umap_i (dict): The inverse mapping from index -> user_id.
            imap_i (dict): The inverse mapping from index -> item_id.
            k (int): The top k recommendation to be produced.

        Returns:
            DataFrame: A DataFrame containing the top k recommendations for each user.
        """
        # Extract informations from model
        scores = self.forward()
        top_k_items = torch.topk(scores, k, dim=1).indices
        user_ids = torch.arange(scores.shape[0]).unsqueeze(1).expand(-1, k)
        recommendations = torch.stack((user_ids, top_k_items), dim=2).reshape(-1, 2)

        # Extract user and items idxs
        user_idxs = recommendations[:, 0].tolist()
        item_idxs = recommendations[:, 1].tolist()

        # Map them back to original labels
        user_label = [umap_i[idx] for idx in user_idxs]
        item_label = [imap_i[idx] for idx in item_idxs]

        # Zip array and turn it into DataFrame
        real_recs = np.array(list(zip(user_label, item_label)))
        recommendations = pd.DataFrame(real_recs)

        return recommendations

    @property
    def name(self):
        return self._name


class ItemSimilarityRecommender(AbstractRecommender):
    """ItemSimilarityRecommender implementation.

    A ItemSimilarityRecommender is a Collaborative Filtering recommendation model \
        wich learns a similarity matrix B and produces recommendations using the computation: X@B.

    Args:
        config (Configuration): The configuration file.
        dataset (AbstractDataset): The dataset to train the model on.
        params (dict): The parameters of the model.
        *args: Argument for PyTorch nn.Module.
        **kwargs: Keyword argument for PyTorch nn.Module.
    """

    def __init__(
        self,
        config: Configuration,
        dataset: AbstractDataset,
        params: dict,
        *args,
        **kwargs,
    ):
        super().__init__(config, dataset, params, *args, **kwargs)
        self.item_similarity = None

    def _serialize(self) -> dict:
        """This method return the part of the recommender that was learned, \
            in this case the similarity matrix {item x item}.

        Returns:
            dict: The dictionary containign all the important informations about the model.
        """
        umap, imap = self._dataset.get_mappings()
        serialization_dict = {
            "model_name": self.__class__.__name__,
            "item_similarity": self.item_similarity,
            "user_mapping": umap,
            "item_mapping": imap,
        }
        return serialization_dict

    def _deserialize(self, deserialized_data: dict):
        """This method load the part of the recommender from a checkpoint, \
            in this case the similarity matrix {item x item}.

        Args:
            deserialized_data (dict): The data loaded with a reader, \
                must be compatible with the model itself.
        """
        if self.__class__.__name__ != deserialized_data["model_name"]:
            logger.negative(
                "You are trying to load a model informations from a different model."
            )
        self.item_similarity = deserialized_data["item_similarity"]
        self._dataset.update_mappings(
            deserialized_data["user_mapping"], deserialized_data["item_mapping"]
        )

    def forward(self) -> Tensor:
        """Prediction in the form of X@B where B is a {item x item} similarity matrix.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        r = self.interaction_matrix @ self.item_similarity

        # Masking interaction already seen in train
        r[self.interaction_matrix.nonzero()] = -torch.inf
        return torch.from_numpy(r)
