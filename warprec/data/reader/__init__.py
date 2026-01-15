# pylint: disable=too-few-public-methods
from .base_reader import Reader, ReaderFactory
from .local_reader import LocalReader

__all__ = ["Reader", "ReaderFactory", "LocalReader"]

try:
    from .azureblob_reader import AzureBlobReader

    __all__.append("AzureBlobReader")

except ImportError:

    class AzureBlobReader:  # type: ignore[no-redef]
        """Placeholder for AzureBlobReader when 'remote-io' extra is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "AzureBlobReader requires the 'remote-io' extra. "
                "Please install it with 'pip install warprec[remote-io]'"
            )
