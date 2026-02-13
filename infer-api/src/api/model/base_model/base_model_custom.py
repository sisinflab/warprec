from pydantic import BaseModel, ConfigDict


class BaseModelCustom(BaseModel):
    '''Base model with custom configuration to exclude description from JSON schema, but not for id and date.
    
    This base model can be extended by other models to inherit the custom configuration.
    '''
    model_config = ConfigDict(json_schema_extra=lambda schema, _: schema.pop("description", None))