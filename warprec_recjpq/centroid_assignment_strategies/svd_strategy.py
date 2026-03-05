from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from .centroid_strategy import CentroidAssignmentStragety
from .custom_discretizer import CustomKBinsDiscretizer


class SVDAssignmentStrategy(CentroidAssignmentStragety):
    def __init__(
        self,
        item_code_bytes: int,
        num_items: int,
        quantisation_strategy: str = "quantile",
    ) -> None:
        super().__init__(item_code_bytes, num_items)
        self.quantisation_strategy = quantisation_strategy

    def assign(self, train_users):
        rows = []
        cols = []
        vals = []
        item_pop = Counter()

        for user_idx, sequence in enumerate(train_users):
            for _timestamp, item_id in sequence:
                if item_id >= self.num_items:
                    raise ValueError(
                        f"Item_id={item_id} exceeds num_items={self.num_items}. Check item mappings."
                    )
                rows.append(user_idx)
                cols.append(item_id)
                vals.append(1)
                item_pop[item_id] += 1

        matrix = csr_matrix(
            (vals, (rows, cols)), shape=(len(train_users), self.num_items + 2)
        )
        svd = TruncatedSVD(n_components=self.item_code_bytes)
        svd.fit(matrix)
        item_embeddings = svd.components_

        item_pop_vector = np.zeros(self.num_items + 2)
        for item_idx in range(self.num_items + 2):
            item_pop_vector[item_idx] = item_pop[item_idx]

        assignments = []
        for component_idx in range(self.item_code_bytes):
            discretizer = CustomKBinsDiscretizer(
                n_bins=256,
                encode="ordinal",
                strategy=self.quantisation_strategy,
                item_popularity=item_pop_vector,
            )
            component = item_embeddings[component_idx]
            component = (component - np.min(component)) / (
                np.max(component) - np.min(component) + 1e-10
            )
            component += np.random.normal(0, 1e-5, self.num_items + 2)
            component = np.expand_dims(component, axis=1)
            component_assignments = discretizer.fit_transform(component).astype(
                "uint8"
            )[:, 0]
            assignments.append(component_assignments)

        return np.transpose(np.array(assignments))
