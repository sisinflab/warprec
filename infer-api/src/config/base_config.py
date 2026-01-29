import os

from pydantic import BaseModel
from dotenv import load_dotenv


class BaseConfig(BaseModel):
    """Configuration data model for the application.
    
    Attributes:
        SERVER_HOST (str): The host address for the server.
        SERVER_PORT (int): The port number for the server.
    """
    
    SERVER_HOST: str
    SERVER_PORT: int
    SERVER_API_KEY: str
    
    def __init__(self, **data):
        
        # Load environment variables from .env file
        load_dotenv()
        
        super().__init__(
            SERVER_HOST     = os.getenv('SERVER_HOST', "0.0.0.0"),
            SERVER_PORT     = int(os.getenv('SERVER_PORT', "8000")),
            SERVER_API_KEY  = os.getenv('SERVER_API_KEY', ""),
            **data
        )