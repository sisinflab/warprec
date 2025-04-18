from .autoencoder import CEASE, EASE, MultiDAE, MultiVAE
from .content_based import VSM
from .graph_based import RP3Beta
from .knn import ItemKNN, UserKNN
from .latent_factor import ADMMSlim, BPR, Slim
from .neural import NeuMF

__all__ = [
    "CEASE",
    "EASE",
    "MultiDAE",
    "MultiVAE",
    "VSM",
    "RP3Beta",
    "ItemKNN",
    "UserKNN",
    "ADMMSlim",
    "BPR",
    "Slim",
    "NeuMF",
]
