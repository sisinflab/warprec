from .rp3beta import RP3Beta

__all__ = ["RP3Beta"]

try:
    from .graph_utils import GraphRecommenderUtils, SparseDropout  # noqa: F401
    from .gcmc import GCMC  # noqa: F401
    from .lightgcn import LightGCN  # noqa: F401
    from .lightgcnpp import LightGCNpp  # noqa: F401
    from .ngcf import NGCF  # noqa: F401
    from .xsimgcl import XSimGCL

    __all__.extend(
        [
            "GCMC",
            "GraphRecommenderUtils",
            "SparseDropout",
            "NGCFLayer",
            "LightGCN",
            "LightGCNpp",
            "NGCF",
            "XSimGCL",
        ]
    )

except ImportError:
    from warprec.utils.registry import model_registry

    @model_registry.register("GCMC")
    class GCMC:  # type: ignore[no-redef]
        """Placeholder for GCMC model when PyG dependencies are not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "GCMC model requires PyG dependencies. "
                "Please install following the documentation you can find here: "
                "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
            )

    @model_registry.register("LightGCN")
    class LightGCN:  # type: ignore[no-redef]
        """Placeholder for LightGCN model when PyG dependencies are not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LightGCN model requires PyG dependencies. "
                "Please install following the documentation you can find here: "
                "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
            )

    @model_registry.register("LightGCNpp")
    class LightGCNpp:  # type: ignore[no-redef]
        """Placeholder for LightGCNpp model when PyG dependencies are not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LightGCNpp model requires PyG dependencies. "
                "Please install following the documentation you can find here: "
                "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
            )

    @model_registry.register("NGCF")
    class NGCF:  # type: ignore[no-redef]
        """Placeholder for NGCF model when PyG dependencies are not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "NGCF model requires PyG dependencies. "
                "Please install following the documentation you can find here: "
                "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
            )

    @model_registry.register("XSimGCL")
    class XSimGCL:  # type: ignore[no-redef]
        """Placeholder for XSimGCL model when PyG dependencies are not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "XSimGCL model requires PyG dependencies. "
                "Please install following the documentation you can find here: "
                "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
            )
