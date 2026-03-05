from itertools import product
from typing import ClassVar

from pydantic import field_validator

from warprec.utils.config.common import (
    validate_between_zero_and_one,
    validate_greater_equal_than_zero,
    validate_greater_than_zero,
    validate_str_list,
)
from warprec.utils.config.model_configuration import (
    FLOAT_FIELD,
    INT_FIELD,
    STR_FIELD,
    RecomModel,
)
from warprec.utils.registry import params_registry


@params_registry.register("BERT4RecJPQ")
class BERT4RecJPQ(RecomModel):
    embedding_size: INT_FIELD
    n_layers: INT_FIELD
    n_heads: INT_FIELD
    inner_size: INT_FIELD
    dropout_prob: FLOAT_FIELD
    attn_dropout_prob: FLOAT_FIELD
    mask_prob: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    max_seq_len: INT_FIELD
    pq_m: INT_FIELD
    centroid_strategy: STR_FIELD
    need_single_trial_validation: ClassVar[bool] = True

    @field_validator("embedding_size", "n_layers", "n_heads", "inner_size")
    @classmethod
    def check_positive_int_fields(cls, value: list, info):
        return validate_greater_than_zero(cls, value, info.field_name)

    @field_validator("batch_size", "epochs", "max_seq_len", "pq_m")
    @classmethod
    def check_runtime_positive_fields(cls, value: list, info):
        return validate_greater_than_zero(cls, value, info.field_name)

    @field_validator("dropout_prob", "attn_dropout_prob", "mask_prob")
    @classmethod
    def check_probability_fields(cls, value: list, info):
        return validate_between_zero_and_one(cls, value, info.field_name)

    @field_validator("weight_decay")
    @classmethod
    def check_weight_decay(cls, value: list):
        return validate_greater_equal_than_zero(cls, value, "weight_decay")

    @field_validator("learning_rate")
    @classmethod
    def check_learning_rate(cls, value: list):
        return validate_greater_than_zero(cls, value, "learning_rate")

    @field_validator("centroid_strategy")
    @classmethod
    def check_centroid_strategy(cls, value):
        return validate_str_list(cls, value, ["svd"], "centroid_strategy")

    def validate_all_combinations(self):
        embedding_sizes = self._clean_param_list(self.embedding_size)
        num_heads = self._clean_param_list(self.n_heads)
        pq_values = self._clean_param_list(self.pq_m)

        has_valid_combination = any(
            emb_size % n_head == 0 and emb_size % pq_m == 0
            for emb_size, n_head, pq_m in product(embedding_sizes, num_heads, pq_values)
        )
        if not has_valid_combination:
            raise ValueError(
                "No valid BERT4RecJPQ hyperparameter combination found. "
                "'embedding_size' must be divisible by both 'n_heads' and 'pq_m'."
            )

    def validate_single_trial_params(self):
        embedding_size = (
            self.embedding_size[1]
            if isinstance(self.embedding_size, list)
            and self.embedding_size
            and isinstance(self.embedding_size[0], str)
            else self.embedding_size[0]
            if isinstance(self.embedding_size, list)
            else self.embedding_size
        )
        n_heads = (
            self.n_heads[1]
            if isinstance(self.n_heads, list)
            and self.n_heads
            and isinstance(self.n_heads[0], str)
            else self.n_heads[0]
            if isinstance(self.n_heads, list)
            else self.n_heads
        )
        pq_m = (
            self.pq_m[1]
            if isinstance(self.pq_m, list)
            and self.pq_m
            and isinstance(self.pq_m[0], str)
            else self.pq_m[0]
            if isinstance(self.pq_m, list)
            else self.pq_m
        )

        if embedding_size % n_heads != 0:
            raise ValueError(
                f"Inconsistent configuration for BERT4RecJPQ: embedding_size ({embedding_size}) "
                f"must be divisible by n_heads ({n_heads})."
            )
        if embedding_size % pq_m != 0:
            raise ValueError(
                f"Inconsistent configuration for BERT4RecJPQ: embedding_size ({embedding_size}) "
                f"must be divisible by pq_m ({pq_m})."
            )


BERT4RecJPQParams = BERT4RecJPQ
