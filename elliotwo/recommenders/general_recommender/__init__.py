from .autoencoder import EASE, MultiDAE
from .graph_based import RP3Beta
from .knn import ItemKNN, UserKNN
from .latent_factor import BPR, Slim
from .neural import NeuMF

__all__ = ["EASE", "MultiDAE", "RP3Beta", "ItemKNN", "UserKNN", "BPR", "Slim", "NeuMF"]
