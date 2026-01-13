from typing import List

import torch
from torch import nn, Tensor
from torch.nn import Module


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


class MLP(nn.Module):
    """Simple implementation of MultiLayer Perceptron.

    Args:
        layers (List[int]): The hidden layers size list.
        dropout (float): The dropout probability.
        activation (str): The activation function to apply.
        batch_normalization (bool): Wether or not to apply batch normalization.
        last_activation (bool): Wether or not to keep last non-linearity function.
    """

    def __init__(
        self,
        layers: List[int],
        dropout: float = 0.0,
        activation: str = "relu",
        batch_normalization: bool = False,
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
    ):
        super().__init__()
        if not len(cnn_channels) == len(cnn_kernels) == len(cnn_strides):
            raise ValueError(
                "cnn_channels, cnn_kernels, and cnn_strides must have the same length."
            )

        cnn_modules: List[Module] = []
        in_channel = 1  # The first input channel will always be 1

        # Iterate over th channels
        for out_channel, kernel_size, stride in zip(
            cnn_channels, cnn_kernels, cnn_strides
        ):
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

    def forward(self, input_feature: Tensor):
        """Simple forwarding, input tensor will pass
        through all the CNN layers.
        """
        return self.cnn_layers(input_feature)


class FactorizationMachine(nn.Module):
    """Calculates the Second-Order Interaction (FM part) over embeddings.

    Equation:
        0.5 * sum( (sum(v))^2 - sum(v^2) )

    Args:
        reduce_sum (bool): Whether to sum the result along the embedding dimension.
            - True: Output shape (Batch, 1). Used for standard FM.
            - False: Output shape (Batch, Embed_Dim). Used for NFM or DeepFM where
              you might want to feed the result into a DNN.
            Defaults to True.
    """

    def __init__(self, reduce_sum: bool = True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs (Tensor): A 3D tensor with shape (batch_size, context_size, embedding_size).

        Returns:
            Tensor: The second-order interaction result.
        """
        square_of_sum = torch.sum(inputs, dim=1) ** 2  # [batch_size, embedding_size]
        sum_of_square = torch.sum(inputs**2, dim=1)  # [batch_size, embedding_size]
        output = 0.5 * (square_of_sum - sum_of_square)

        if self.reduce_sum:
            output = torch.sum(output, dim=1, keepdim=True)  # [batch_size, 1]

        return output
