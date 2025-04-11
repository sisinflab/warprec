from .autoencoder import EASE
from .graph_based import RP3Beta
from .knn import ItemKNN, UserKNN
from .latent_factor import BPR, Slim
from .neural import NeuMF

__all__ = ["EASE", "RP3Beta", "ItemKNN", "UserKNN", "BPR", "Slim", "NeuMF"]
