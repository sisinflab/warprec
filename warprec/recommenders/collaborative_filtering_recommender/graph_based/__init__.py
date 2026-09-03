# pylint: disable = R0903, R0401
from .rp3beta import RP3Beta

__all__ = ["RP3Beta"]

try:
    from .graph_utils import GraphRecommenderUtils, SparseDropout  # noqa: F401
    from .dgcf import DGCF  # noqa: F401
    from .egcf import EGCF  # noqa: F401
    from .esigcf import ESIGCF  # noqa: F401
    from .gcmc import GCMC  # noqa: F401
    from .lightccf import LightCCF  # noqa: F401
    from .lightgcl import LightGCL  # noqa: F401
    from .lightgcn import LightGCN  # noqa: F401
    from .lightgcnpp import LightGCNpp  # noqa: F401
    from .lightgode import LightGODE  # noqa: F401
    from .macrgcn import MACRGCN  # noqa: F401
    from .mixrec import MixRec  # noqa: F401
    from .ngcf import NGCF  # noqa: F401
    from .paac import PAAC  # noqa: F401
    from .popdcl import PopDCL  # noqa: F401
    from .recdcl import RecDCL  # noqa: F401
    from .sgcl import SGCL  # noqa: F401
    from .sgl import SGL  # noqa: F401
    from .simgcl import SimGCL  # noqa: F401
    from .simrec import SimRec  # noqa: F401
    from .ultragcn import UltraGCN  # noqa: F401
    from .xsimgcl import XSimGCL  # noqa: F401

    __all__.extend(
        [
            "DGCF",
            "GCMC",
            "EGCF",
            "ESIGCF",
            "GraphRecommenderUtils",
            "SparseDropout",
            "NGCFLayer",
            "LightCCF",
            "LightGCL",
            "LightGCN",
            "LightGCNpp",
            "LightGODE",
            "MACRGCN",
            "MixRec",
            "NGCF",
            "PAAC",
            "PopDCL",
            "RecDCL",
            "SGCL",
            "SGL",
            "SimGCL",
            "SimRec",
            "UltraGCN",
            "XSimGCL",
        ]
    )

except ImportError:
    from typing import Any

    from warprec.utils.registry import model_registry

    # Every model above needs PyTorch Geometric. Registering a stand-in for each
    # keeps the registry complete, so a configuration naming one of them fails
    # with a useful message instead of an unknown-model error.
    _PYG_MODELS = [
        "DGCF",
        "EGCF",
        "ESIGCF",
        "GCMC",
        "GraphRecommenderUtils",
        "LightCCF",
        "LightGCL",
        "LightGCN",
        "LightGCNpp",
        "LightGODE",
        "MACRGCN",
        "MixRec",
        "NGCF",
        "PAAC",
        "PopDCL",
        "RecDCL",
        "SGCL",
        "SGL",
        "SimGCL",
        "SimRec",
        "SparseDropout",
        "UltraGCN",
        "XSimGCL",
    ]

    def _unavailable(model_name: str) -> type:
        """Builds a stand-in that explains the missing dependency when used.

        Args:
            model_name (str): The name of the unavailable model.

        Returns:
            type: A class raising ImportError when instantiated.
        """

        class _Unavailable:
            """Placeholder used when PyTorch Geometric is not installed."""

            def __init__(self, *args: Any, **kwargs: Any):
                raise ImportError(
                    f"The {model_name} model requires PyTorch Geometric. "
                    'Install it with: pip install "warprec[graph]"'
                )

        _Unavailable.__name__ = model_name
        _Unavailable.__qualname__ = model_name
        return _Unavailable

    for _name in _PYG_MODELS:
        globals()[_name] = _unavailable(_name)
        model_registry.register(_name)(globals()[_name])

    __all__.extend(_PYG_MODELS)
