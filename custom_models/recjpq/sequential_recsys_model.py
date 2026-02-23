from __future__ import annotations
from typing import List, Type, Dict, Any
import torch
import torch.nn as nn

class SequentialModelConfig(object):
    def __init__(self, **kwargs):
        # Added **kwargs to handle the unpacking in from_config
        self.config = kwargs
    
    def as_dict(self) -> dict:
        return self.config 
    
    def get_model_architecture(self) -> Type['SequentialRecsysModel']:
        raise NotImplementedError()

class SequentialDataParameters(object):
    def __init__(self, num_users: int, num_items: int, sequence_length: int, batch_size: int) -> None:
        self.num_users = num_users
        self.num_items = num_items
        self.sequence_length = sequence_length
        self.batch_size = batch_size
    
    def as_dict(self):
        return self.__dict__

class SequentialRecsysModel(nn.Module):
    @classmethod
    def get_model_config_class(cls) -> Type[SequentialModelConfig]:
        raise NotImplementedError()

    def __init__(self, model_parameters: SequentialModelConfig, data_parameters: SequentialDataParameters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_parameters = model_parameters
        self.data_parameters = data_parameters
        
        # In PyTorch, layers are typically defined here (e.g., self.embedding = nn.Embedding(...))
    
    def forward(self, x):
        # Equivalent to call() in Keras
        raise NotImplementedError()

    def get_dummy_inputs(self) -> List[torch.Tensor]:
        raise NotImplementedError()

    def fit_biases(self, train_users):
        pass

    def pass_mappings(self, user_mappings, item_mappings): 
        pass

    # Write tensorboard stuff/metrics here
    def log(self):
        pass

    # --- ADDED: PyTorch Compatibility Methods for Keras API ---
    def save_weights(self, filepath):
      
        torch.save(self.state_dict(), filepath)

    @classmethod
    def from_config(cls, config: dict):
        data_parameters = SequentialDataParameters(**config['data_parameters'])
        
        # Instantiate the config class
        config_cls = cls.get_model_config_class()
        model_parameters = config_cls(**config['model_parameters'])
        
        model = cls(model_parameters, data_parameters)
        
        # In PyTorch, models are usually eagerly initialized. 
        # However, to replicate the 'build' logic (if using LazyModules) or simply to verify:
        dummy_data = model.get_dummy_inputs()
        
        # We switch to eval mode and no_grad for the dummy pass
        model.eval()
        with torch.no_grad():
            # If dummy_data is a list, we unpack it, otherwise pass as is
            if isinstance(dummy_data, list):
                model(*dummy_data)
            else:
                model(dummy_data)
        
        # Switch back to train mode by default
        model.train()
        
        return model
    
    def get_config(self):
        return get_config_dict(self.model_parameters, self.data_parameters)

def get_sequential_model(model_config: SequentialModelConfig, data_parameters: SequentialDataParameters):
    config = get_config_dict(model_config, data_parameters)
    model_arch = model_config.get_model_architecture()
    return model_arch.from_config(config)

def get_config_dict(model_config, data_parameters):
    model_config_dict = model_config.as_dict()
    data_parameters_dict = data_parameters.as_dict()
    config = {'model_parameters': model_config_dict, 'data_parameters': data_parameters_dict}
    return config