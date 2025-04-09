from .autoencoder import EASE
from .knn import ItemKNN, UserKNN
from .latent_factor import Slim
from .neural import NeuMF

__all__ = ["EASE", "ItemKNN", "UserKNN", "Slim", "NeuMF"]
