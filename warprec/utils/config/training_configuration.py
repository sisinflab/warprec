from typing import Literal, Optional

from pydantic import BaseModel

NegativeSampling = Literal["uniform", "popularity"]
SequencePooling = Literal["mean", "sum", "max"]


class TrainingConfig(BaseModel):
    """Definition of the training configuration part of the configuration file.

    These options shape the signal a model learns from rather than the data the
    reader produces: the dataset is identical whichever value they take, only
    the examples drawn from it differ. That is why they live here and not under
    the reader, where they were first shipped.

    Attributes:
        negative_sampling (Optional[NegativeSampling]): How negatives are drawn during
            training. 'uniform' gives every item the same chance, 'popularity' draws
            proportionally to a dampened interaction count. Defaults to 'uniform'.
        sequence_pooling (Optional[SequencePooling]): How the values of a multi-valued
            field are combined into the single vector the field contributes. Defaults
            to 'mean', which matches the normalised multi-hot encoding the
            factorisation-machine literature defines these models over.
    """

    negative_sampling: Optional[NegativeSampling] = "uniform"
    sequence_pooling: Optional[SequencePooling] = "mean"
