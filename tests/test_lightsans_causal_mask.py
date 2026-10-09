"""LightSANs' position attention must look at the item itself and the past only.

The model gathers the state of each user's last real item, and every position
after it is padding, so a mask that hid the past and left the future open made
that state attend to padding alone.
"""

from typing import Any, List, cast

import torch

from warprec.data.dataset import Dataset
from warprec.recommenders.sequential_recommender.lightsans import (
    LightSANs,
    LightSANsLayer,
)

from conftest import make_model


def _position_probabilities(model: LightSANs) -> torch.Tensor:
    """Run the first layer once and capture its position attention probabilities.

    The layer passes the item attention and then the position attention
    through ``attn_dropout``; the second call carries the position one.

    Args:
        model (LightSANs): The model, in eval mode.

    Returns:
        torch.Tensor: The position attention, [n_heads, seq_len, seq_len].
    """
    layer = cast(LightSANsLayer, model.layers[0])
    seq_len = model.max_seq_len
    captured: List[torch.Tensor] = []

    def hook(_module: Any, inputs: Any, _output: Any) -> None:
        """Keep the input of the dropout call.

        Args:
            _module (Any): The dropout module.
            inputs (Any): Its positional inputs.
            _output (Any): Its output.
        """
        captured.append(inputs[0].detach().clone())

    handle = layer.attn_dropout.register_forward_hook(hook)
    try:
        torch.manual_seed(0)
        with torch.no_grad():
            layer(
                torch.randn(3, seq_len, model.embedding_size),
                pos_emb=model.position_embedding.weight,
                padding_mask=torch.zeros(3, seq_len, dtype=torch.bool),
                causal_mask=model.causal_mask,
            )
    finally:
        handle.remove()
    assert len(captured) == 2
    return captured[1]


def test_position_attention_is_causal(dataset: Dataset):
    """Each position attends to itself and the earlier positions, never later ones.

    Args:
        dataset (Dataset): The shared synthetic dataset.
    """
    model = make_model("LightSANs", dataset).eval()
    assert isinstance(model, LightSANs)
    seq_len = model.max_seq_len
    probs = _position_probabilities(model)

    assert probs.shape == (model.n_heads, seq_len, seq_len)
    future = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    assert torch.all(probs[:, future] == 0), "a position attends to the future"
    assert torch.all(probs[:, ~future] > 0), "a position ignores itself or the past"
    torch.testing.assert_close(probs.sum(-1), torch.ones(model.n_heads, seq_len))
    # The first position can only look at itself.
    torch.testing.assert_close(probs[:, 0, 0], torch.ones(model.n_heads))
