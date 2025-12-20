from itertools import product
from typing import ClassVar

from pydantic import field_validator
from warprec.utils.config.model_configuration import (
    RecomModel,
    INT_FIELD,
    LIST_INT_FIELD,
    FLOAT_FIELD,
    BOOL_FIELD,
)
from warprec.utils.config.common import (
    validate_greater_than_zero,
    validate_greater_equal_than_zero,
    validate_layer_list,
    validate_bool_values,
)
from warprec.utils.registry import params_registry


@params_registry.register("ConvNCF")
class ConvNCF(RecomModel):
    """Definition of the model ConvNCF.

    Attributes:
        embedding_size (INT_FIELD): List of values for embedding_size.
        cnn_channels (LIST_INT_FIELD): List of values for CNN channels.
        cnn_kernels (LIST_INT_FIELD): List of values for CNN kernels.
        cnn_strides (LIST_INT_FIELD): List of values for CNN strides.
        dropout_prob (FLOAT_FIELD): List of values for dropout_prob.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        batch_size (INT_FIELD): List of values for batch_size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        need_single_trial_validation (ClassVar[bool]): Whether or not to check if a Ray Tune
            trial parameter are valid.
    """

    embedding_size: INT_FIELD
    cnn_channels: LIST_INT_FIELD
    cnn_kernels: LIST_INT_FIELD
    cnn_strides: LIST_INT_FIELD
    dropout_prob: FLOAT_FIELD
    reg_weight: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    need_single_trial_validation: ClassVar[bool] = True

    @field_validator("embedding_size")
    @classmethod
    def check_embedding_size(cls, v: list):
        """Validate embedding_size."""
        return validate_greater_than_zero(cls, v, "embedding_size")

    @field_validator("cnn_channels")
    @classmethod
    def check_cnn_channels(cls, v: list):
        """Validate cnn_channels."""
        return validate_layer_list(cls, v, "cnn_channels")

    @field_validator("cnn_kernels")
    @classmethod
    def check_cnn_kernels(cls, v: list):
        """Validate cnn_kernels."""
        return validate_layer_list(cls, v, "cnn_kernels")

    @field_validator("cnn_strides")
    @classmethod
    def check_cnn_strides(cls, v: list):
        """Validate cnn_strides."""
        return validate_layer_list(cls, v, "cnn_strides")

    @field_validator("dropout_prob")
    @classmethod
    def check_dropout_prob(cls, v: list):
        """Validate dropout_prob."""
        return validate_greater_equal_than_zero(cls, v, "dropout_prob")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight"""
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

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

    def validate_all_combinations(self):
        """Validates if at least one valid combination of hyperparameters exists.
        This method should be called after all individual fields have been validated.

        Raises:
            ValueError: If no valid combination of hyperparameters can be formed.
        """
        # Extract parameters to check, removing searching strategy
        embedding_sizes = self._clean_param_list(self.embedding_size)
        cnn_channels_list = self._clean_param_list(self.cnn_channels)
        cnn_kernels_list = self._clean_param_list(self.cnn_kernels)
        cnn_strides_list = self._clean_param_list(self.cnn_strides)

        # Check if parameters are lists of lists
        cnn_channels_processed = [
            item if isinstance(item, list) else [item] for item in cnn_channels_list
        ]
        cnn_kernels_processed = [
            item if isinstance(item, list) else [item] for item in cnn_kernels_list
        ]
        cnn_strides_processed = [
            item if isinstance(item, list) else [item] for item in cnn_strides_list
        ]

        # Iter over all possible combinations and check if
        # any of them is valid.
        has_valid_combination = False
        for emb_size, channels_config, kernels_config, strides_config in product(
            embedding_sizes,
            cnn_channels_processed,
            cnn_kernels_processed,
            cnn_strides_processed,
        ):
            # Check for lengths
            if not (len(channels_config) == len(kernels_config) == len(strides_config)):
                continue
            # Check for embedding size
            if emb_size != channels_config[0]:
                continue

            # Found a valid combination
            has_valid_combination = True
            break

        if not has_valid_combination:
            raise ValueError(
                "No valid hyperparameter combination found for ConvNCF. "
                "Ensure there's at least one combination of 'embedding_size', "
                "'cnn_channels', 'cnn_kernels', and 'cnn_strides' that meets the criteria: "
                "1. The lengths of 'cnn_channels', 'cnn_kernels', and 'cnn_strides' must be equal. "
                "2. The dimension of the first CNN channel must be equal to 'embedding_size'."
            )

    def validate_single_trial_params(self):
        """Validates the coherence of cnn_channels, cnn_kernels, and cnn_strides
        for a single trial's parameter set.

        Raises:
            ValueError: If the parameter values are not consistent for the model.
        """
        # Clean parameters from search space information
        embedding_size_clean = (
            self.embedding_size[1:]
            if self.embedding_size and isinstance(self.embedding_size[0], str)
            else self.embedding_size
        )
        cnn_channels_clean = (
            self.cnn_channels[1:]
            if self.cnn_channels and isinstance(self.cnn_channels[0], str)
            else self.cnn_channels
        )
        cnn_kernels_clean = (
            self.cnn_kernels[1:]
            if self.cnn_kernels and isinstance(self.cnn_kernels[0], str)
            else self.cnn_kernels
        )
        cnn_strides_clean = (
            self.cnn_strides[1:]
            if self.cnn_strides and isinstance(self.cnn_strides[0], str)
            else self.cnn_strides
        )

        # Track the lengths of layers
        len_channels = len(cnn_channels_clean[0])
        len_kernels = len(cnn_kernels_clean[0])
        len_strides = len(cnn_strides_clean[0])

        # Check if this is a possible combination of parameters
        # if not, just raise an error.
        # RayTune will skip this trial
        if not (len_channels == len_kernels == len_strides):
            raise ValueError(
                f"Inconsistent CNN layer configuration: "
                f"cnn_channels length ({len_channels}), cnn_kernels length ({len_kernels}), "
                f"and cnn_strides length ({len_strides}) must be equal. "
            )

        emb_size = embedding_size_clean[0]
        first_layer_cnn = cnn_channels_clean[0][0]

        # Check if the first cnn_channel output layer
        # is the same as the embedding size
        if emb_size != first_layer_cnn:
            raise ValueError(
                f"Embedding size must be the same as the first layer of CNN. "
                f"embedding_size value ({emb_size}), first cnn layer ({first_layer_cnn}). "
            )


@params_registry.register("NeuMF")
class NeuMF(RecomModel):
    """Definition of the model NeuMF.

    Attributes:
        mf_embedding_size (INT_FIELD): List of mf embedding size.
        mlp_embedding_size (INT_FIELD): List of mlp embedding size.
        mlp_hidden_size (LIST_INT_FIELD): List of mlp_hidden_size values.
        mf_train (BOOL_FIELD): List of values for mf_train flag.
        mlp_train (BOOL_FIELD): List of values for mlp_train flag.
        dropout (FLOAT_FIELD): List of values for dropout.
        reg_weight (FLOAT_FIELD): List of values for reg_weight.
        weight_decay (FLOAT_FIELD): List of values for weight_decay.
        batch_size (INT_FIELD): List of values for batch size.
        epochs (INT_FIELD): List of values for epochs.
        learning_rate (FLOAT_FIELD): List of values for learning rate.
        neg_samples (INT_FIELD): List of values for negative sampling.
    """

    mf_embedding_size: INT_FIELD
    mlp_embedding_size: INT_FIELD
    mlp_hidden_size: LIST_INT_FIELD
    mf_train: BOOL_FIELD
    mlp_train: BOOL_FIELD
    dropout: FLOAT_FIELD
    reg_weight: FLOAT_FIELD
    weight_decay: FLOAT_FIELD
    batch_size: INT_FIELD
    epochs: INT_FIELD
    learning_rate: FLOAT_FIELD
    neg_samples: INT_FIELD

    @field_validator("mf_embedding_size")
    @classmethod
    def check_mf_embedding_size(cls, v: list):
        """Validate mf_embedding_size."""
        return validate_greater_than_zero(cls, v, "mf_embedding_size")

    @field_validator("mlp_embedding_size")
    @classmethod
    def check_mlp_embedding_size(cls, v: list):
        """Validate mlp_embedding_size."""
        return validate_greater_than_zero(cls, v, "mlp_embedding_size")

    @field_validator("mlp_hidden_size")
    @classmethod
    def check_mlp_hidden_size(cls, v: list):
        """Validate mlp_hidden_size."""
        return validate_layer_list(cls, v, "mlp_hidden_size")

    @field_validator("mf_train")
    @classmethod
    def check_mf_train(cls, v: list):
        """Validate mf_train."""
        return validate_bool_values(v)

    @field_validator("mlp_train")
    @classmethod
    def check_mlp_train(cls, v: list):
        """Validate mlp_train."""
        return validate_bool_values(v)

    @field_validator("dropout")
    @classmethod
    def check_dropout(cls, v: list):
        """Validate dropout."""
        return validate_greater_equal_than_zero(cls, v, "dropout")

    @field_validator("reg_weight")
    @classmethod
    def check_reg_weight(cls, v: list):
        """Validate reg_weight"""
        return validate_greater_equal_than_zero(cls, v, "reg_weight")

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
