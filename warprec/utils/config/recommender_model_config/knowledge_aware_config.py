from typing import ClassVar

from pydantic import field_validator

from warprec.utils.config.common import (
    validate_between_zero_and_one,
    validate_layer_list,
    validate_str_list,
    validate_greater_equal_than_zero,
    validate_greater_than_zero,
)
from warprec.utils.config.model_configuration import (
    FLOAT_FIELD,
    INT_FIELD,
    LIST_INT_FIELD,
    STR_FIELD,
    RecomModel,
)
from warprec.utils.registry import params_registry


@params_registry.register("CKE")
class CKE(RecomModel):
    """Definition of the model CKE.

    Attributes:
        need_knowledge (ClassVar[bool]): The model scores from a knowledge graph.
        embedding_size (INT_FIELD): List of values for embedding_size.
        kg_embedding_size (INT_FIELD): List of values for kg_embedding_size.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        kg_reg_weight (FLOAT_FIELD): List of values for kg_reg_weight.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    need_knowledge: ClassVar[bool] = True

    embedding_size: INT_FIELD
    kg_embedding_size: INT_FIELD
    reg_weight: FLOAT_FIELD
    kg_reg_weight: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("kg_embedding_size")
    @classmethod
    def check_kg_embedding_size(cls, v: list):
        """Validate kg_embedding_size."""
        return validate_greater_than_zero(cls, v, "kg_embedding_size")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight."""
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

    @field_validator("kg_reg_weight")
    @classmethod
    def check_kg_reg_weight(cls, v: list):
        """Validate kg_reg_weight."""
        return validate_greater_equal_than_zero(cls, v, "kg_reg_weight")

    @field_validator("batch_size")
    @classmethod
    def check_batch_size(cls, v: list):
        """Validate batch_size."""
        return validate_greater_than_zero(cls, v, "batch_size")

    @field_validator("epochs")
    @classmethod
    def check_epochs(cls, v: list):
        """Validate epochs."""
        return validate_greater_than_zero(cls, v, "epochs")

    @field_validator("learning_rate")
    @classmethod
    def check_learning_rate(cls, v: list):
        """Validate learning_rate."""
        return validate_greater_than_zero(cls, v, "learning_rate")


@params_registry.register("KGAT")
class KGAT(RecomModel):
    """Definition of the model KGAT.

    Attributes:
        need_knowledge (ClassVar[bool]): The model scores from a knowledge graph.
        embedding_size (INT_FIELD): List of values for embedding_size.
        kg_embedding_size (INT_FIELD): List of values for kg_embedding_size.
        layers (LIST_INT_FIELD): List of propagation layer widths.
        dropout (FLOAT_FIELD): List of values for dropout.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    need_knowledge: ClassVar[bool] = True

    embedding_size: INT_FIELD
    kg_embedding_size: INT_FIELD
    layers: LIST_INT_FIELD
    dropout: FLOAT_FIELD
    reg_weight: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("kg_embedding_size")
    @classmethod
    def check_kg_embedding_size(cls, v: list):
        """Validate kg_embedding_size."""
        return validate_greater_than_zero(cls, v, "kg_embedding_size")

    @field_validator("layers")
    @classmethod
    def check_layers(cls, v: list):
        """Validate layers."""
        return validate_layer_list(cls, v, "layers")

    @field_validator("dropout")
    @classmethod
    def check_dropout(cls, v: list):
        """Validate dropout."""
        return validate_greater_equal_than_zero(cls, v, "dropout")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight."""
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

    @field_validator("batch_size")
    @classmethod
    def check_batch_size(cls, v: list):
        """Validate batch_size."""
        return validate_greater_than_zero(cls, v, "batch_size")

    @field_validator("epochs")
    @classmethod
    def check_epochs(cls, v: list):
        """Validate epochs."""
        return validate_greater_than_zero(cls, v, "epochs")

    @field_validator("learning_rate")
    @classmethod
    def check_learning_rate(cls, v: list):
        """Validate learning_rate."""
        return validate_greater_than_zero(cls, v, "learning_rate")


@params_registry.register("KGCN")
class KGCN(RecomModel):
    """Definition of the model KGCN.

    Attributes:
        need_knowledge (ClassVar[bool]): The model scores from a knowledge graph.
        embedding_size (INT_FIELD): List of values for embedding_size.
        neighbour_size (INT_FIELD): How many neighbours each entity is given.
        n_iter (INT_FIELD): How many hops out from an item to read.
        aggregator (STR_FIELD): How a node is combined with its neighbourhood.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    need_knowledge: ClassVar[bool] = True

    embedding_size: INT_FIELD
    neighbour_size: INT_FIELD
    n_iter: INT_FIELD
    aggregator: STR_FIELD = ["sum"]
    reg_weight: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("neighbour_size")
    @classmethod
    def check_neighbour_size(cls, v: list):
        """Validate neighbour_size."""
        return validate_greater_than_zero(cls, v, "neighbour_size")

    @field_validator("n_iter")
    @classmethod
    def check_n_iter(cls, v: list):
        """Validate n_iter."""
        return validate_greater_than_zero(cls, v, "n_iter")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight."""
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

    @field_validator("batch_size")
    @classmethod
    def check_batch_size(cls, v: list):
        """Validate batch_size."""
        return validate_greater_than_zero(cls, v, "batch_size")

    @field_validator("epochs")
    @classmethod
    def check_epochs(cls, v: list):
        """Validate epochs."""
        return validate_greater_than_zero(cls, v, "epochs")

    @field_validator("learning_rate")
    @classmethod
    def check_learning_rate(cls, v: list):
        """Validate learning_rate."""
        return validate_greater_than_zero(cls, v, "learning_rate")

    @field_validator("aggregator")
    @classmethod
    def check_aggregator(cls, v: list):
        """Validate aggregator."""
        return validate_str_list(cls, v, ["sum", "neighbour", "concat"], "aggregator")


@params_registry.register("RippleNet")
class RippleNet(RecomModel):
    """Definition of the model RippleNet.

    Attributes:
        need_knowledge (ClassVar[bool]): The model scores from a knowledge graph.
        embedding_size (INT_FIELD): List of values for embedding_size.
        n_hop (INT_FIELD): How many rings to spread out from the history.
        n_memory (INT_FIELD): How many facts each ring holds.
        kg_weight (FLOAT_FIELD): The weight of the fact-plausibility term.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    need_knowledge: ClassVar[bool] = True

    embedding_size: INT_FIELD
    n_hop: INT_FIELD
    n_memory: INT_FIELD
    kg_weight: FLOAT_FIELD
    reg_weight: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("n_hop")
    @classmethod
    def check_n_hop(cls, v: list):
        """Validate n_hop."""
        return validate_greater_than_zero(cls, v, "n_hop")

    @field_validator("n_memory")
    @classmethod
    def check_n_memory(cls, v: list):
        """Validate n_memory."""
        return validate_greater_than_zero(cls, v, "n_memory")

    @field_validator("kg_weight")
    @classmethod
    def check_kg_weight(cls, v: list):
        """Validate kg_weight."""
        return validate_greater_equal_than_zero(cls, v, "kg_weight")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight."""
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

    @field_validator("batch_size")
    @classmethod
    def check_batch_size(cls, v: list):
        """Validate batch_size."""
        return validate_greater_than_zero(cls, v, "batch_size")

    @field_validator("epochs")
    @classmethod
    def check_epochs(cls, v: list):
        """Validate epochs."""
        return validate_greater_than_zero(cls, v, "epochs")

    @field_validator("learning_rate")
    @classmethod
    def check_learning_rate(cls, v: list):
        """Validate learning_rate."""
        return validate_greater_than_zero(cls, v, "learning_rate")


@params_registry.register("KGIN")
class KGIN(RecomModel):
    """Definition of the model KGIN.

    Attributes:
        need_knowledge (ClassVar[bool]): The model scores from a knowledge graph.
        embedding_size (INT_FIELD): List of values for embedding_size.
        n_factors (INT_FIELD): How many intents to keep.
        n_hops (INT_FIELD): How many hops to propagate.
        node_dropout (FLOAT_FIELD): The share of edges dropped each pass.
        mess_dropout (FLOAT_FIELD): The dropout applied to each hop's output.
        independence (STR_FIELD): How intents are pushed apart.
        ind_weight (FLOAT_FIELD): The weight of that independence term.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
    """

    need_knowledge: ClassVar[bool] = True

    embedding_size: INT_FIELD
    n_factors: INT_FIELD
    n_hops: INT_FIELD
    node_dropout: FLOAT_FIELD
    mess_dropout: FLOAT_FIELD
    independence: STR_FIELD = ["distance"]
    ind_weight: FLOAT_FIELD
    reg_weight: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("n_factors")
    @classmethod
    def check_n_factors(cls, v: list):
        """Validate n_factors."""
        return validate_greater_than_zero(cls, v, "n_factors")

    @field_validator("n_hops")
    @classmethod
    def check_n_hops(cls, v: list):
        """Validate n_hops."""
        return validate_greater_than_zero(cls, v, "n_hops")

    @field_validator("node_dropout")
    @classmethod
    def check_node_dropout(cls, v: list):
        """Validate node_dropout."""
        return validate_between_zero_and_one(cls, v, "node_dropout")

    @field_validator("mess_dropout")
    @classmethod
    def check_mess_dropout(cls, v: list):
        """Validate mess_dropout."""
        return validate_between_zero_and_one(cls, v, "mess_dropout")

    @field_validator("ind_weight")
    @classmethod
    def check_ind_weight(cls, v: list):
        """Validate ind_weight."""
        return validate_greater_equal_than_zero(cls, v, "ind_weight")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight."""
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

    @field_validator("batch_size")
    @classmethod
    def check_batch_size(cls, v: list):
        """Validate batch_size."""
        return validate_greater_than_zero(cls, v, "batch_size")

    @field_validator("epochs")
    @classmethod
    def check_epochs(cls, v: list):
        """Validate epochs."""
        return validate_greater_than_zero(cls, v, "epochs")

    @field_validator("learning_rate")
    @classmethod
    def check_learning_rate(cls, v: list):
        """Validate learning_rate."""
        return validate_greater_than_zero(cls, v, "learning_rate")

    @field_validator("independence")
    @classmethod
    def check_independence(cls, v: list):
        """Validate independence."""
        return validate_str_list(cls, v, ["distance", "cosine"], "independence")
