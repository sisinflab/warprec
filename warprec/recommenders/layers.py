from typing import List

from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import normal_


def get_activation(activation: str = "relu") -> Module:
    """Get the activation function using enum.

    Args:
        activation (str): The activation layer to retrieve.

    Returns:
        Module: The activation layer requested.

    Raises:
        ValueError: If the activation is not known or supported.
    """
    match activation:
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()
        case "relu":
            return nn.ReLU()
        case "leakyrelu":
            return nn.LeakyReLU()
        case _:
            raise ValueError("Activation function not supported.")


def init_weights(module: Module):
    """Initialize the weights of a module."""
    if isinstance(module, nn.Linear):
        normal_(module.weight.data, mean=0, std=0.01)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    if isinstance(module, nn.Conv2d):
        normal_(module.weight.data, mean=0, std=0.01)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


class MLP(nn.Module):
    """Simple implementation of MultiLayer Perceptron.

    Args:
        layers (List[int]): The hidden layers size list.
        dropout (float): The dropout probability.
        activation (str): The activation function to apply.
        batch_normalization (bool): Wether or not to apply batch normalization.
        initialize (bool): Wether or not to initialize the weights.
        last_activation (bool): Wether or not to keep last non-linearity function.
    """

    def __init__(
        self,
        layers: List[int],
        dropout: float = 0.0,
        activation: str = "relu",
        batch_normalization: bool = False,
        initialize: bool = False,
        last_activation: bool = True,
    ):
        super().__init__()
        mlp_modules: List[Module] = []
        for input_size, output_size in zip(layers[:-1], layers[1:]):
            mlp_modules.append(nn.Dropout(p=dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if batch_normalization:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            if activation:
                mlp_modules.append(get_activation(activation))
        if activation is not None and not last_activation:
            mlp_modules.pop()
        self.mlp_layers = nn.Sequential(*mlp_modules)
        if initialize:
            self.apply(init_weights)

    def forward(self, input_feature: Tensor):
        """Simple forwarding, input tensor will pass
        through all the MLP layers.
        """
        return self.mlp_layers(input_feature)


class CNN(nn.Module):
    """Simple implementation of Convolutional Neural Network.

    Args:
        cnn_channels (List[int]): The output channels of each layer of the CNN.
        cnn_kernels (List[int]): The kernels of each layer.
        cnn_strides (List[int]): The strides of each layer.
        activation (str): The activation function to apply.
        initialize (bool): Wether or not to initialize the weights.

    Raises:
        ValueError: If the cnn_channels, cnn_kernels and cnn_strides lists
            do not have the same length.
    """

    def __init__(
        self,
        cnn_channels: List[int],
        cnn_kernels: List[int],
        cnn_strides: List[int],
        activation: str = "relu",
        initialize: bool = False,
    ):
        super().__init__()
        if not (len(cnn_channels) == len(cnn_kernels) == len(cnn_strides)):
            raise ValueError(
                "cnn_channels, cnn_kernels, and cnn_strides must have the same length."
            )

        cnn_modules: List[Module] = []
        in_channel = 1  # The first input channel will always be 1
        for i in range(len(cnn_channels)):
            out_channel = cnn_channels[i]
            kernel_size = cnn_kernels[i]
            stride = cnn_strides[i]

            # Append conv layer
            cnn_modules.append(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
            cnn_modules.append(get_activation(activation))
            in_channel = out_channel

        self.cnn_layers = nn.Sequential(*cnn_modules)

        if initialize:
            self.apply(init_weights)

    def forward(self, input_feature: Tensor):
        """Simple forwarding, input tensor will pass
        through all the CNN layers.
        """
        return self.cnn_layers(input_feature)
