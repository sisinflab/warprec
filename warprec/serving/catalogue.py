import difflib
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd

from warprec.utils.config.serving_configuration import ItemMetadata


class Catalogue:
    """What the served items are called and what attributes they carry.

    Ids are kept as strings, which is how requests are matched against the
    model's mapping whatever type the dataset used for them. Names are matched
    exactly first and then without regard to case, so that a person or an agent
    typing a title does not have to reproduce its capitals.

    Args:
        names (Dict[str, str]): The name of every item, by its id.
        attributes (Dict[str, Dict[str, Any]]): The extra attributes of every
            item, by its id; a list for a multi-valued attribute.
    """

    def __init__(self, names: Dict[str, str], attributes: Dict[str, Dict[str, Any]]):
        self.names = names
        self.attributes = attributes
        self._by_name = {name: item for item, name in names.items()}
        # A folded name that two items share cannot pick one of them.
        folded = Counter(name.casefold() for name in names.values())
        self._by_folded = {
            name.casefold(): item
            for item, name in names.items()
            if folded[name.casefold()] == 1
        }
        self._folded_names = list(self._by_folded)

    def find(self, name: str) -> Optional[str]:
        """The id of the item with this name, exactly or ignoring case.

        Args:
            name (str): The name to look up.

        Returns:
            Optional[str]: The item id, or None when no single item has it.
        """
        item = self._by_name.get(name)
        if item is None:
            item = self._by_folded.get(name.casefold())
        return item

    def suggest(self, text: str, limit: int = 3) -> List[str]:
        """The names closest to some text, for a lookup that found nothing.

        Args:
            text (str): What was looked up.
            limit (int): The most names to return.

        Returns:
            List[str]: The closest names, best first.
        """
        close = difflib.get_close_matches(
            text.casefold(), self._folded_names, n=limit, cutoff=0.6
        )
        return [self.names[self._by_folded[name]] for name in close]

    def search(self, query: str) -> List[str]:
        """The ids of the items whose name contains the query, ignoring case.

        Args:
            query (str): Part of a name.

        Returns:
            List[str]: The matching ids: an exact name first, then names that
                start with the query, then the rest.
        """
        folded = query.casefold().strip()
        exact, starting, containing = [], [], []
        for item, name in self.names.items():
            candidate = name.casefold()
            if candidate == folded:
                exact.append(item)
            elif candidate.startswith(folded):
                starting.append(item)
            elif folded in candidate:
                containing.append(item)
        return exact + starting + containing

    def attribute_names(self) -> List[str]:
        """The attributes the items carry.

        Returns:
            List[str]: The attribute names, in file order.
        """
        names: Dict[str, None] = {}
        for attributes in self.attributes.values():
            names.update(dict.fromkeys(attributes))
        return list(names)

    def attribute_values(self, attribute: str) -> Counter:
        """How many items carry each value of an attribute.

        Args:
            attribute (str): The attribute.

        Returns:
            Counter: The number of items per value.
        """
        counts: Counter = Counter()
        for attributes in self.attributes.values():
            value = attributes.get(attribute)
            if value is None:
                continue
            counts.update(value if isinstance(value, list) else [value])
        return counts


def read_catalogue(metadata: ItemMetadata) -> Catalogue:
    """Read the names and attributes of the items from a delimited file.

    Args:
        metadata (ItemMetadata): Where the file is and how it is laid out.

    Returns:
        Catalogue: The names and attributes of the items.
    """
    frame = pd.read_csv(
        metadata.path,
        sep=metadata.sep,
        header=0 if metadata.header else None,
        dtype=str,
        encoding=metadata.encoding,
        # Multi-character separators such as '::' need the python engine.
        engine="python" if len(metadata.sep) > 1 else "c",
        keep_default_na=False,
    )

    def column(key):
        return frame.iloc[:, key] if isinstance(key, int) else frame[key]

    ids = column(metadata.id_column).tolist()
    names = dict(zip(ids, column(metadata.name_column)))

    attributes: Dict[str, Dict[str, Any]] = {item: {} for item in ids}
    for attribute, spec in metadata.columns.items():
        for item, cell in zip(ids, column(spec.column)):
            if spec.separator is None:
                attributes[item][attribute] = cell
            else:
                attributes[item][attribute] = [
                    value for value in cell.split(spec.separator) if value
                ]
    return Catalogue(names, attributes)
