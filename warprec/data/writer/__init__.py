# pylint: disable=too-few-public-methods
from .base_writer import Writer, WriterFactory
from .local_writer import LocalWriter

__all__ = ["Writer", "WriterFactory", "LocalWriter"]

try:
    from .azureblob_writer import AzureBlobWriter

    __all__.append("AzureBlobWriter")

except ImportError:

    class AzureBlobWriter:  # type: ignore[no-redef]
        """Placeholder for AzureBlobWriter when 'remote-io' extra is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "AzureBlobWriter requires the 'remote-io' extra. "
                "Please install it with 'pip install warprec[remote-io]'"
            )
