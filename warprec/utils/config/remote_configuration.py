from pydantic import BaseModel


class AzureConfig(BaseModel):
    """Configuration for Azure services.

    Attributes:
        storage_account_name (str): The name of the Azure Storage Account.
        container_name (str): The name of the Azure Container.
    """

    storage_account_name: str
    container_name: str
