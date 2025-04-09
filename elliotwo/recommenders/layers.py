from typing import List

from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import normal_, xavier_normal_, xavier_uniform_


class MLP(nn.Module):
    """Simple implementation of MultiLayer Perceptron.

    Args:
        layers (List[int]): The hidden layers size list.
        dropout (float): The dropout probability.
        activation (str): The activation function to apply.
        batch_normalization (bool): Wether or not to apply batch normalization.
        initialization_method (str): The method of initialization to use.
        last_activation (bool): Wether or not to keep last non-linearity function.
    """

    def __init__(
        self,
        layers: List[int],
        dropout: float = 0.0,
        activation: str = "relu",
        batch_normalization: bool = False,
        initialization_method: str = None,
        last_activation: bool = True,
    ):
        super().__init__()
        self.init = initialization_method
        mlp_modules: List[Module] = []
        for input_size, output_size in zip(layers[:-1], layers[1:]):
            mlp_modules.append(nn.Dropout(p=dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if batch_normalization:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            if activation:
                mlp_modules.append(self._get_activation(activation))
        if activation is not None and not last_activation:
            mlp_modules.pop()
        self.mlp_layers = nn.Sequential(*mlp_modules)
        if initialization_method:
            self.apply(self._init_weights)

    def forward(self, input_feature: Tensor):
        """Simple forwarding, input tensor will pass
        through all the MLP layers.
        """
        return self.mlp_layers(input_feature)

    def _get_activation(self, activation: str = "relu"):
        """Retrieve the activation to use at the end
        of a MLP layer.
        """
        if activation == "relu":
            return nn.Sigmoid()
        if activation == "tanh":
            return nn.Tanh()
        if activation == "relu":
            return nn.ReLU()
        if activation == "leakyrelu":
            return nn.LeakyReLU()
        raise ValueError("Activation function not supported.")

    def _init_weights(self, module: Module):
        """Initialize the weights of a module."""
        if isinstance(module, nn.Linear):
            if self.init == "norm":
                normal_(module.weight.data, mean=0, std=0.01)
            elif self.init == "xavier_normal":
                xavier_normal_(module.weight.data)
            elif self.init == "xavier_uniform":
                xavier_uniform_(module.weight.data)
            else:
                raise ValueError("Initialization mode not supported.")
            if module.bias is not None:
                module.bias.data.fill_(0.0)
