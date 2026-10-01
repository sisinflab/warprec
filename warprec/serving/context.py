from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
from torch import Tensor

from warprec.data.entities.context import build_context_array
from warprec.serving.errors import ServingError


@dataclass(frozen=True)
class ContextSchema:
    """The context fields a model was trained on and the values each accepts.

    It turns the raw context of a request into the array the model reads, the
    way the dataset turned the context of every training row into it: a
    category becomes its index, a number stays a number, and a multi-valued
    field becomes the indices of its values, padded to the widest field.

    Attributes:
        labels (List[str]): The fields, in the order the model reads them.
        types (Dict[str, str]): Each field's kind: 'token', 'float' or 'seq'.
        maps (Dict[str, Dict[str, int]]): For each categorical or multi-valued
            field, the index of every value seen in training.
        max_len (int): The widest multi-valued field, or 1 when there is none.
    """

    labels: List[str]
    types: Dict[str, str]
    maps: Dict[str, Dict[str, int]]
    max_len: int

    @classmethod
    def from_info(
        cls, info: Dict[str, Any], maps: Dict[str, Dict[Any, int]]
    ) -> "ContextSchema":
        """Build the schema from a checkpoint's dataset information.

        Args:
            info (Dict[str, Any]): The dataset information saved with the model.
            maps (Dict[str, Dict[Any, int]]): The context vocabulary saved with it.

        Returns:
            ContextSchema: The schema.
        """
        labels = list(info.get("context_dims", {}))
        types = info.get("context_types", {})
        return cls(
            labels=labels,
            types={label: types.get(label, "token") for label in labels},
            # Values are matched as strings, whatever type the dataset read them as.
            maps={
                label: {
                    str(value): index for value, index in maps.get(label, {}).items()
                }
                for label in labels
            },
            max_len=max(info.get("context_max_len", {}).values(), default=1),
        )

    def describe(self) -> Dict[str, Dict[str, Any]]:
        """The fields a request must describe and the values each accepts.

        Returns:
            Dict[str, Dict[str, Any]]: Each field's kind and known values; None
                for a numeric field, which takes any number.
        """
        return {
            label: {
                "type": self.types[label],
                "values": None
                if self.types[label] == "float"
                else sorted(self.maps[label]),
            }
            for label in self.labels
        }

    def encode(self, context: Union[Dict[str, Any], List[Any]]) -> List[Any]:
        """Check a request's context and encode it as one row of the model's input.

        Args:
            context (Union[Dict[str, Any], List[Any]]): The value of every field,
                by name or as a list in the order of the labels.

        Returns:
            List[Any]: The encoded row, one entry per field.

        Raises:
            ServingError: If a field is missing, unexpected or holds a value the
                model was not trained on.
        """
        if isinstance(context, list):
            if len(context) != len(self.labels):
                raise ServingError(
                    422,
                    f"A context list holds one value per field, in the order "
                    f"{self.labels}; got {len(context)} values.",
                )
            context = dict(zip(self.labels, context))

        missing = [label for label in self.labels if label not in context]
        unexpected = [label for label in context if label not in self.labels]
        if missing or unexpected:
            raise ServingError(
                422,
                f"The context must describe exactly the fields {self.labels}: "
                f"missing {missing}, unexpected {unexpected}.",
            )
        return [self._encode_field(label, context[label]) for label in self.labels]

    def tensor(self, rows: List[List[Any]], device: Union[str, torch.device]) -> Tensor:
        """Stack encoded rows into the tensor the model's predict takes.

        Args:
            rows (List[List[Any]]): Rows returned by encode().
            device (Union[str, torch.device]): The model's device.

        Returns:
            Tensor: [rows, fields], or [rows, fields, max_len] with a multi-valued field.
        """
        array = build_context_array(
            np.array(rows, dtype=object),
            [self.types[label] for label in self.labels],
            self.max_len,
        )
        return torch.from_numpy(array).to(device)

    def _encode_field(self, label: str, value: Any) -> Any:
        """Encode one field's value as the dataset encoded it in training.

        Args:
            label (str): The field.
            value (Any): Its raw value; a list for a multi-valued field.

        Returns:
            Any: The value for a numeric field, the index for a categorical one,
                and the space-joined indices for a multi-valued one.

        Raises:
            ServingError: If the value does not fit the field.
        """
        kind = self.types[label]
        if kind == "float":
            try:
                return float(value)
            except (TypeError, ValueError):
                raise ServingError(
                    422, f"Context '{label}' is numeric, got '{value}'."
                ) from None

        values = value if isinstance(value, list) else [value]
        if kind != "seq" and len(values) != 1:
            raise ServingError(422, f"Context '{label}' takes one value, got {values}.")

        indices = []
        for item in values:
            index = self.maps[label].get(str(item))
            if index is None:
                raise ServingError(
                    422,
                    f"'{item}' is not a known value of context '{label}'. "
                    f"Known values: {sorted(self.maps[label])}.",
                )
            indices.append(index)

        if kind == "seq":
            return " ".join(str(index) for index in indices[: self.max_len])
        return indices[0]
