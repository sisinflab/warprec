from . import autoencoder
from . import content_based
from . import graph_based
from . import knn
from . import latent_factor
from . import neural
from . import unpersonalized
from .proxy import ProxyRecommender

__all__ = [
    "autoencoder",
    "content_based",
    "graph_based",
    "knn",
    "latent_factor",
    "neural",
    "unpersonalized",
    "ProxyRecommender",
]
