from itertools import product
from typing import ClassVar

from pydantic import field_validator
from warprec.utils.config.model_configuration import (
    RecomModel,
    INT_FIELD,
    FLOAT_FIELD,
    BOOL_FIELD,
)
from warprec.utils.config.common import (
    validate_greater_than_zero,
    validate_greater_equal_than_zero,
    validate_bool_values,
    validate_between_zero_and_one,
)
from warprec.utils.registry import params_registry


@params_registry.register("Caser")
class Caser(RecomModel):
    """Definition of the model Caser.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        n_h (INT_FIELD):  List of values for n_h.
        n_v (INT_FIELD): List of values for n_h.
        dropout_prob (FLOAT_FIELD): List of values for dropout_prob.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        neg_samples (INT_FIELD): List of values for neg_samples.
        max_seq_len (INT_FIELD): List of values for max_seq_len.
        need_timestamp (ClassVar[bool]): Wether or not the model needs the timestamp.
    """

    embedding_size: INT_FIELD
    n_h: INT_FIELD
    n_v: INT_FIELD
    dropout_prob: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    neg_samples: INT_FIELD
    max_seq_len: INT_FIELD
    need_timestamp: ClassVar[bool] = True

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("n_h")
    @classmethod
    def check_n_h(cls, v: list):
        """Validate n_h."""
        return validate_greater_than_zero(cls, v, "n_h")

    @field_validator("n_v")
    @classmethod
    def check_n_v(cls, v: list):
        """Validate n_v."""
        return validate_greater_than_zero(cls, v, "n_v")

    @field_validator("dropout_prob")
    @classmethod
    def check_dropout_prob(cls, v: list):
        """Validate dropout_prob."""
        return validate_between_zero_and_one(cls, v, "dropout_prob")

    @field_validator("weight_decay")
    @classmethod
    def check_weight_decay(cls, v: list):
        """Validate weight_decay."""
        return validate_greater_equal_than_zero(cls, v, "weight_decay")

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

    @field_validator("neg_samples")
    @classmethod
    def check_neg_samples(cls, v: list):
        """Validate neg_samples."""
        return validate_greater_equal_than_zero(cls, v, "neg_samples")

    @field_validator("max_seq_len")
    @classmethod
    def check_max_seq_len(cls, v: list):
        """Validate max_seq_len."""
        return validate_greater_than_zero(cls, v, "max_seq_len")


@params_registry.register("FOSSIL")
class FOSSIL(RecomModel):
    """Definition of the model FOSSIL.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        order_len (INT_FIELD):  List of values for order_len.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        alpha (FLOAT_FIELD): List of values for alpha.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        neg_samples (INT_FIELD): List of values for neg_samples.
        max_seq_len (INT_FIELD): List of values for max_seq_len.
        need_timestamp (ClassVar[bool]): Wether or not the model needs the timestamp.
    """

    embedding_size: INT_FIELD
    order_len: INT_FIELD
    reg_weight: FLOAT_FIELD
    alpha: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    neg_samples: INT_FIELD
    max_seq_len: INT_FIELD
    need_timestamp: ClassVar[bool] = True

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("order_len")
    @classmethod
    def check_order_len(cls, v: list):
        """Validate order_len."""
        return validate_greater_than_zero(cls, v, "order_len")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight."""
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

    @field_validator("alpha")
    @classmethod
    def check_alpha(cls, v: list):
        """Validate alpha."""
        return validate_greater_equal_than_zero(cls, v, "alpha")

    @field_validator("weight_decay")
    @classmethod
    def check_weight_decay(cls, v: list):
        """Validate weight_decay."""
        return validate_greater_equal_than_zero(cls, v, "weight_decay")

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

    @field_validator("neg_samples")
    @classmethod
    def check_neg_samples(cls, v: list):
        """Validate neg_samples."""
        return validate_greater_equal_than_zero(cls, v, "neg_samples")

    @field_validator("max_seq_len")
    @classmethod
    def check_max_seq_len(cls, v: list):
        """Validate max_seq_len."""
        return validate_greater_than_zero(cls, v, "max_seq_len")

    def validate_all_combinations(self):
        """Validates if at least one valid combination of hyperparameters exists.
        This method should be called after all individual fields have been validated.

        Raises:
            ValueError: If no valid combination of hyperparameters can be formed.
        """
        # Extract parameters to check, removing searching strategy
        order_lengths = self._clean_param_list(self.order_len)
        max_lengths = self._clean_param_list(self.max_seq_len)

        # Check if any combination of parameters is a valid combination
        has_valid_combination = False
        for order_len, max_seq_len in product(order_lengths, max_lengths):
            if order_len <= max_seq_len:
                has_valid_combination = True
                break

        if not has_valid_combination:
            raise ValueError(
                "No valid hyperparameter combination found for FOSSIL. "
                "Ensure there's at least one combination of 'order_len' and "
                "'max_seq_len' that meets the criteria: "
                "1. Order_len must be <= max_seq_len. "
            )

    def validate_single_trial_params(self):
        """Validates the coherence of order_len and max_seq_len
        for a single trial's parameter set.

        Raises:
            ValueError: If the parameter values are not consistent for the model.
        """
        # Clean parameters from search space information
        order_len_clean = (
            self.order_len[1:]
            if self.order_len and isinstance(self.order_len[0], str)
            else self.order_len
        )
        max_seq_len_clean = (
            self.max_seq_len[1:]
            if self.max_seq_len and isinstance(self.max_seq_len[0], str)
            else self.max_seq_len
        )

        # Check is parameters are coherent
        if order_len_clean <= max_seq_len_clean:
            raise ValueError(
                f"Inconsistent order_len and max_seq_len configuration: "
                f"order_len_clean ({order_len_clean}) must be lesser or equal "
                f"of max_seq_len ({max_seq_len_clean})."
            )


@params_registry.register("GRU4Rec")
class GRU4Rec(RecomModel):
    """Definition of the model GRU4Rec.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        hidden_size (INT_FIELD):  List of values for hidden_size.
        num_layers (INT_FIELD): List of values for num_layers.
        dropout_prob (FLOAT_FIELD): List of values for dropout_prob.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        neg_samples (INT_FIELD): List of values for neg_samples.
        max_seq_len (INT_FIELD): List of values for max_seq_len.
        need_timestamp (ClassVar[bool]): Wether or not the model needs the timestamp.
    """

    embedding_size: INT_FIELD
    hidden_size: INT_FIELD
    num_layers: INT_FIELD
    dropout_prob: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    neg_samples: INT_FIELD
    max_seq_len: INT_FIELD
    need_timestamp: ClassVar[bool] = True

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("hidden_size")
    @classmethod
    def check_hidden_size(cls, v: list):
        """Validate hidden_size."""
        return validate_greater_than_zero(cls, v, "hidden_size")

    @field_validator("num_layers")
    @classmethod
    def check_num_layers(cls, v: list):
        """Validate num_layers."""
        return validate_greater_than_zero(cls, v, "num_layers")

    @field_validator("dropout_prob")
    @classmethod
    def check_dropout_prob(cls, v: list):
        """Validate dropout_prob."""
        return validate_between_zero_and_one(cls, v, "dropout_prob")

    @field_validator("weight_decay")
    @classmethod
    def check_weight_decay(cls, v: list):
        """Validate weight_decay."""
        return validate_greater_equal_than_zero(cls, v, "weight_decay")

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

    @field_validator("neg_samples")
    @classmethod
    def check_neg_samples(cls, v: list):
        """Validate neg_samples."""
        return validate_greater_equal_than_zero(cls, v, "neg_samples")

    @field_validator("max_seq_len")
    @classmethod
    def check_max_seq_len(cls, v: list):
        """Validate max_seq_len."""
        return validate_greater_than_zero(cls, v, "max_seq_len")


@params_registry.register("gSASRec")
class gSASRec(RecomModel):
    """Definition of the model gSASRec.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        n_layers (INT_FIELD):  List of values for n_layers.
        n_heads (INT_FIELD): List of values for n_heads.
        inner_size (INT_FIELD): List of values for inner_size.
        dropout_prob (FLOAT_FIELD): List of values for dropout_prob.
        attn_dropout_prob (FLOAT_FIELD): List of values for attn_dropout_prob.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        gbce_t (FLOAT_FIELD): List of values for gbce_t.
        neg_samples (INT_FIELD): List of values for neg_samples.
        max_seq_len (INT_FIELD): List of values for max_seq_len.
        reuse_item_embeddings (BOOL_FIELD): List of values for reuse_item_embeddings.
        need_timestamp (ClassVar[bool]): Wether or not the model needs the timestamp.
        need_single_trial_validation (ClassVar[bool]): Whether or not to check if a Ray Tune
            trial parameter are valid.
    """

    embedding_size: INT_FIELD
    n_layers: INT_FIELD
    n_heads: INT_FIELD
    inner_size: INT_FIELD
    dropout_prob: FLOAT_FIELD
    attn_dropout_prob: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    gbce_t: FLOAT_FIELD
    neg_samples: INT_FIELD
    max_seq_len: INT_FIELD
    reuse_item_embeddings: BOOL_FIELD
    need_timestamp: ClassVar[bool] = True
    need_single_trial_validation: ClassVar[bool] = True

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("n_layers")
    @classmethod
    def check_n_layers(cls, v: list):
        """Validate n_layers."""
        return validate_greater_than_zero(cls, v, "n_layers")

    @field_validator("n_heads")
    @classmethod
    def check_n_heads(cls, v: list):
        """Validate n_heads."""
        return validate_greater_than_zero(cls, v, "n_heads")

    @field_validator("inner_size")
    @classmethod
    def check_inner_size(cls, v: list):
        """Validate inner_size."""
        return validate_greater_than_zero(cls, v, "inner_size")

    @field_validator("dropout_prob")
    @classmethod
    def check_dropout_prob(cls, v: list):
        """Validate dropout_prob."""
        return validate_between_zero_and_one(cls, v, "dropout_prob")

    @field_validator("attn_dropout_prob")
    @classmethod
    def check_attn_dropout_prob(cls, v: list):
        """Validate attn_dropout_prob."""
        return validate_between_zero_and_one(cls, v, "attn_dropout_prob")

    @field_validator("weight_decay")
    @classmethod
    def check_weight_decay(cls, v: list):
        """Validate weight_decay."""
        return validate_greater_equal_than_zero(cls, v, "weight_decay")

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

    @field_validator("gbce_t")
    @classmethod
    def check_gbce_t(cls, v: list):
        """Validate gbce_t."""
        return validate_between_zero_and_one(cls, v, "gbce_t")

    @field_validator("neg_samples")
    @classmethod
    def check_neg_samples(cls, v: list):
        """Validate neg_samples."""
        return validate_greater_than_zero(cls, v, "neg_samples")

    @field_validator("max_seq_len")
    @classmethod
    def check_max_seq_len(cls, v: list):
        """Validate max_seq_len."""
        return validate_greater_than_zero(cls, v, "max_seq_len")

    @field_validator("reuse_item_embeddings")
    @classmethod
    def check_reuse_item_embeddings(cls, v: list):
        """Validate reuse_item_embeddings."""
        return validate_bool_values(v)

    def validate_all_combinations(self):
        """Validates if at least one valid combination of hyperparameters exists.
        This method should be called after all individual fields have been validated.

        Raises:
            ValueError: If no valid combination of hyperparameters can be formed.
        """
        # Extract parameters to check, removing searching strategy
        embedding_sizes = self._clean_param_list(self.embedding_size)
        num_heads = self._clean_param_list(self.n_heads)

        # Check if any combination of parameters is a valid combination
        has_valid_combination = False
        for emb_size, n_head in product(embedding_sizes, num_heads):
            if emb_size % n_head == 0:
                has_valid_combination = True
                break

        if not has_valid_combination:
            raise ValueError(
                "No valid hyperparameter combination found for gSASRec. "
                "Ensure there's at least one combination of 'embedding_size' and "
                "'n_heads' that meets the criteria: "
                "1. Embedding size must be divisible by n_heads. "
            )

    def validate_single_trial_params(self):
        """Validates the coherence of embedding_size and n_heads
        for a single trial's parameter set.

        Raises:
            ValueError: If the parameter values are not consistent for the model.
        """
        # Clean parameters from search space information
        embedding_size_clean = (
            self.embedding_size[1]
            if self.embedding_size and isinstance(self.embedding_size[0], str)
            else self.embedding_size[0]
        )
        n_heads_clean = (
            self.n_heads[1]
            if self.n_heads and isinstance(self.n_heads[0], str)
            else self.n_heads[0]
        )

        # Check if embedding size is divisible by n_heads, otherwise transformer
        # will raise an exception
        if embedding_size_clean % n_heads_clean != 0:
            raise ValueError(
                f"Inconsistent embedding and heads number configuration: "
                f"embedding_size ({embedding_size_clean}) must be divisible "
                f"by n_heads ({n_heads_clean})."
            )


@params_registry.register("SASRec")
class SASRec(RecomModel):
    """Definition of the model SASRec.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        n_layers (INT_FIELD):  List of values for n_layers.
        n_heads (INT_FIELD): List of values for n_heads.
        inner_size (INT_FIELD): List of values for inner_size.
        dropout_prob (FLOAT_FIELD): List of values for dropout_prob.
        attn_dropout_prob (FLOAT_FIELD): List of values for attn_dropout_prob.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        neg_samples (INT_FIELD): List of values for neg_samples.
        max_seq_len (INT_FIELD): List of values for max_seq_len.
        need_timestamp (ClassVar[bool]): Wether or not the model needs the timestamp.
        need_single_trial_validation (ClassVar[bool]): Whether or not to check if a Ray Tune
            trial parameter are valid.
    """

    embedding_size: INT_FIELD
    n_layers: INT_FIELD
    n_heads: INT_FIELD
    inner_size: INT_FIELD
    dropout_prob: FLOAT_FIELD
    attn_dropout_prob: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    neg_samples: INT_FIELD
    max_seq_len: INT_FIELD
    need_timestamp: ClassVar[bool] = True
    need_single_trial_validation: ClassVar[bool] = True

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("n_layers")
    @classmethod
    def check_n_layers(cls, v: list):
        """Validate n_layers."""
        return validate_greater_than_zero(cls, v, "n_layers")

    @field_validator("n_heads")
    @classmethod
    def check_n_heads(cls, v: list):
        """Validate n_heads."""
        return validate_greater_than_zero(cls, v, "n_heads")

    @field_validator("inner_size")
    @classmethod
    def check_inner_size(cls, v: list):
        """Validate inner_size."""
        return validate_greater_than_zero(cls, v, "inner_size")

    @field_validator("dropout_prob")
    @classmethod
    def check_dropout_prob(cls, v: list):
        """Validate dropout_prob."""
        return validate_between_zero_and_one(cls, v, "dropout_prob")

    @field_validator("attn_dropout_prob")
    @classmethod
    def check_attn_dropout_prob(cls, v: list):
        """Validate attn_dropout_prob."""
        return validate_between_zero_and_one(cls, v, "attn_dropout_prob")

    @field_validator("weight_decay")
    @classmethod
    def check_weight_decay(cls, v: list):
        """Validate weight_decay."""
        return validate_greater_equal_than_zero(cls, v, "weight_decay")

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

    @field_validator("neg_samples")
    @classmethod
    def check_neg_samples(cls, v: list):
        """Validate neg_samples."""
        return validate_greater_equal_than_zero(cls, v, "neg_samples")

    @field_validator("max_seq_len")
    @classmethod
    def check_max_seq_len(cls, v: list):
        """Validate max_seq_len."""
        return validate_greater_than_zero(cls, v, "max_seq_len")

    def validate_all_combinations(self):
        """Validates if at least one valid combination of hyperparameters exists.
        This method should be called after all individual fields have been validated.

        Raises:
            ValueError: If no valid combination of hyperparameters can be formed.
        """
        # Extract parameters to check, removing searching strategy
        embedding_sizes = self._clean_param_list(self.embedding_size)
        num_heads = self._clean_param_list(self.n_heads)

        # Check if any combination of parameters is a valid combination
        has_valid_combination = False
        for emb_size, n_head in product(embedding_sizes, num_heads):
            if emb_size % n_head == 0:
                has_valid_combination = True
                break

        if not has_valid_combination:
            raise ValueError(
                "No valid hyperparameter combination found for SASRec. "
                "Ensure there's at least one combination of 'embedding_size' and "
                "'n_heads' that meets the criteria: "
                "1. Embedding size must be divisible by n_heads. "
            )

    def validate_single_trial_params(self):
        """Validates the coherence of embedding_size and n_heads
        for a single trial's parameter set.

        Raises:
            ValueError: If the parameter values are not consistent for the model.
        """
        # Clean parameters from search space information
        embedding_size_clean = (
            self.embedding_size[1]
            if self.embedding_size and isinstance(self.embedding_size[0], str)
            else self.embedding_size[0]
        )
        n_heads_clean = (
            self.n_heads[1]
            if self.n_heads and isinstance(self.n_heads[0], str)
            else self.n_heads[0]
        )

        # Check if embedding size is divisible by n_heads, otherwise transformer
        # will raise an exception
        if embedding_size_clean % n_heads_clean != 0:
            raise ValueError(
                f"Inconsistent embedding and heads number configuration: "
                f"embedding_size ({embedding_size_clean}) must be divisible "
                f"by n_heads ({n_heads_clean})."
            )
