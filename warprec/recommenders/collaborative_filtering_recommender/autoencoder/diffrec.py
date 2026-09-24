# pylint: disable = R0801, E1102
import math
from typing import Any, Iterable, List, Optional, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


def timestep_embedding(steps: Tensor, width: int, max_period: int = 10000) -> Tensor:
    """Describe a diffusion step as a vector of sinusoids.

    The denoiser has to know how much noise it is looking at, and a single
    integer is a poor thing to feed a network. The usual answer, borrowed from
    the transformer literature, is a bank of sinusoids at geometrically spaced
    frequencies, which gives nearby steps similar descriptions without ever
    repeating.

    Args:
        steps (Tensor): The diffusion step of each row.
        width (int): How wide the description should be.
        max_period (int): The longest period in the bank.

    Returns:
        Tensor: The {row x width} description.
    """
    half = width // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=steps.device)
        / half
    )
    angles = steps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    if width % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


class Denoiser(nn.Module):
    """The network that reads a corrupted history and says what it should be.

    Args:
        n_items (int): The width of an interaction vector.
        hidden_dims (list): The widths of the hidden layers.
        time_size (int): The width of the step description.
        dropout (float): The dropout applied to the input.
        normalize (bool): Whether to scale each history to unit length first.

    Attributes:
        time_size (int): The width of the step description.
    """

    time_size: int

    def __init__(
        self,
        n_items: int,
        hidden_dims: list,
        time_size: int,
        dropout: float,
        normalize: bool,
    ):
        # pylint: disable = too-many-arguments, too-many-positional-arguments
        super().__init__()

        self.time_size = time_size
        self._normalize = normalize

        self.step_projection = nn.Linear(time_size, time_size)

        # The step description enters alongside the history rather than being
        # added to it, which is what the reference implementation does.
        widths = [n_items + time_size] + list(hidden_dims)
        self.encode = nn.ModuleList(
            nn.Linear(first, second) for first, second in zip(widths[:-1], widths[1:])
        )

        back = list(reversed(hidden_dims)) + [n_items]
        self.decode = nn.ModuleList(
            nn.Linear(first, second) for first, second in zip(back[:-1], back[1:])
        )

        self.drop = nn.Dropout(dropout)
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialise every layer the way the reference does."""
        layers: List[nn.Linear] = [
            *cast(Iterable[nn.Linear], self.encode),
            *cast(Iterable[nn.Linear], self.decode),
            self.step_projection,
        ]
        for layer in layers:
            fan_out, fan_in = layer.weight.size()
            layer.weight.data.normal_(0.0, math.sqrt(2.0 / (fan_in + fan_out)))
            layer.bias.data.normal_(0.0, 0.001)

    def forward(self, history: Tensor, steps: Tensor) -> Tensor:
        """Say what the history should be, given how noisy it is.

        Args:
            history (Tensor): The corrupted interaction vectors.
            steps (Tensor): The diffusion step each row sits at.

        Returns:
            Tensor: The denoised interaction vectors.
        """
        described = self.step_projection(
            timestep_embedding(steps, self.time_size).to(history.device)
        )

        if self._normalize:
            history = F.normalize(history)

        hidden = torch.cat([self.drop(history), described], dim=-1)

        for layer in self.encode:
            hidden = torch.tanh(layer(hidden))

        for position, layer in enumerate(self.decode):
            hidden = layer(hidden)
            if position != len(self.decode) - 1:
                hidden = torch.tanh(hidden)

        return hidden


@model_registry.register(name="DiffRec")
class DiffRec(IterativeRecommender):
    """Implementation of DiffRec algorithm from
        Diffusion Recommender Model (SIGIR 2023)

    A generative model that learns to recommend by learning to *repair*. Noise
    is added to a user's interaction vector in small steps until it is close to
    formless, and a network is trained to undo one step of that. Recommending is
    then running the repair: the user's real history is taken as a lightly
    corrupted signal and the network is asked what it should have been, and the
    items it puts back that were not there are the recommendation.

    The departure from the image diffusion this borrows from is that the
    corruption is kept deliberately mild. An interaction vector is not a
    photograph; destroying it entirely would destroy the personalisation along
    with the noise, so ``noise_scale`` is small and inference starts only a few
    steps in.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        hidden_dims (list): The widths of the denoiser's hidden layers.
        time_size (int): The width of the diffusion step description.
        steps (int): How many steps the forward corruption runs for.
        noise_scale (float): How much noise a step adds. Small on purpose.
        noise_min (float): The smallest noise level of the schedule.
        noise_max (float): The largest noise level of the schedule.
        sampling_steps (int): How far in to start when recommending.
        sampling_noise (bool): Whether to resample noise while repairing.
        dropout (float): The dropout applied to the input history.
        normalize (bool): Whether to scale each history to unit length.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        betas (Tensor): The noise added at each step of the schedule.
        alphas_cumprod (Tensor): How much signal survives to each step.
        alphas_cumprod_prev (Tensor): The same, shifted one step back.

    Raises:
        ValueError: If the training interactions were not provided, or the
            sampling starts further in than the schedule reaches.
    """

    DATALOADER_TYPE = DataLoaderType.INTERACTION_LOADER

    hidden_dims: list
    time_size: int
    steps: int
    noise_scale: float
    noise_min: float
    noise_max: float
    sampling_steps: int
    sampling_noise: bool
    dropout: float
    normalize: bool
    batch_size: int
    epochs: int
    learning_rate: float

    # Registered buffers, annotated so that they read as the tensors they are.
    betas: Tensor
    alphas_cumprod: Tensor
    alphas_cumprod_prev: Tensor

    def __init__(
        self,
        params: dict,
        info: dict,
        interactions: Interactions,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)

        if interactions is None:
            raise ValueError(
                "DiffRec repairs a user's interaction vector, so it needs the "
                "interactions at construction."
            )

        if self.sampling_steps > self.steps:
            raise ValueError(
                f"DiffRec was asked to start recommending {self.sampling_steps} "
                f"steps into a schedule that is only {self.steps} long. "
                "'sampling_steps' cannot exceed 'steps'."
            )

        self.train_matrix = interactions.get_sparse()

        self.denoiser = Denoiser(
            self.n_items,
            self.hidden_dims,
            self.time_size,
            self.dropout,
            self.normalize,
        )

        self._build_schedule()

    def _build_schedule(self) -> None:
        """Work out how much noise each step adds, once.

        Everything the forward and reverse processes need follows from the
        schedule, so it is derived here rather than at each use.
        """
        betas = torch.linspace(
            self.noise_scale * self.noise_min,
            self.noise_scale * self.noise_max,
            self.steps,
            dtype=torch.float64,
        )
        # The first step is held tiny, which is what stops the model overfitting
        # the least noisy end of the schedule.
        betas[0] = 1e-5

        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        previous = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", previous.float())

    def _corrupt(self, history: Tensor, steps: Tensor, noise: Tensor) -> Tensor:
        """Take the history the given number of steps towards formlessness.

        Args:
            history (Tensor): The clean interaction vectors.
            steps (Tensor): How far to take each row.
            noise (Tensor): The noise to mix in.

        Returns:
            Tensor: The corrupted interaction vectors.
        """
        if self.noise_scale == 0.0:
            # The degenerate schedule is a no-op: the model becomes a plain
            # denoising autoencoder over the untouched history.
            return history

        surviving = self.alphas_cumprod[steps].sqrt().unsqueeze(1)
        added = (1.0 - self.alphas_cumprod[steps]).sqrt().unsqueeze(1)

        return surviving * history + added * noise

    def _repair_once(self, history: Tensor, step: int) -> Tensor:
        """Undo one step of the corruption.

        Args:
            history (Tensor): The corrupted interaction vectors.
            step (int): The step they are currently at.

        Returns:
            Tensor: The mean of where they came from.
        """
        steps = torch.full(
            (history.size(0),), step, dtype=torch.long, device=history.device
        )
        predicted = self.denoiser(history, steps)

        if step == 0 or self.noise_scale == 0.0:
            return predicted

        # The posterior mean of the previous step, given where we are and what
        # the denoiser thinks the clean vector was.
        beta = self.betas[step]
        cumulative = self.alphas_cumprod[step]
        previous = self.alphas_cumprod_prev[step]

        from_clean = beta * previous.sqrt() / (1.0 - cumulative)
        from_current = (1.0 - previous) * (1.0 - beta).sqrt() / (1.0 - cumulative)

        mean = from_clean * predicted + from_current * history

        if not self.sampling_noise:
            return mean

        variance = beta * (1.0 - previous) / (1.0 - cumulative)
        return mean + variance.sqrt() * torch.randn_like(history)

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs: Any,
    ):
        return interactions.get_interaction_dataloader(
            batch_size=self.batch_size,
            **kwargs,
        )

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        history = batch[0]

        steps = torch.randint(0, self.steps, (history.size(0),), device=history.device)
        noise = torch.randn_like(history)
        corrupted = self._corrupt(history, steps, noise)

        # The network is asked for the clean vector rather than for the noise:
        # what matters here is the history itself, not the perturbation.
        predicted = self.denoiser(corrupted, steps)
        loss = ((history - predicted) ** 2).mean()

        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def forward(self, history: Tensor) -> Tensor:
        """Repair a batch of histories into a score over the catalogue.

        Args:
            history (Tensor): The users' interaction vectors.

        Returns:
            Tensor: One score per item.
        """
        if self.sampling_steps == 0:
            current = history
        else:
            steps = torch.full(
                (history.size(0),),
                self.sampling_steps - 1,
                dtype=torch.long,
                device=history.device,
            )
            current = self._corrupt(history, steps, torch.randn_like(history))

        # The walk itself always runs the whole schedule; sampling_steps only
        # says how far in the history is placed before it begins.
        for step in reversed(range(self.steps)):
            current = self._repair_once(current, step)

        return current

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction by repairing each user's own history.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        rows = self.train_matrix[user_indices.tolist(), :].toarray()
        history = torch.from_numpy(rows).float().to(self.device)

        predictions = self.forward(history)

        if item_indices is None:
            return predictions

        return predictions.gather(
            1, item_indices.to(predictions.device).clamp(max=self.n_items - 1)
        )
