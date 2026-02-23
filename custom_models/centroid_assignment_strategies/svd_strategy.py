from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from .centroid_strategy import CentroidAssignmentStragety
from .custom_discretizer import CustomKBinsDiscretizer
import numpy as np

class SVDAssignmentStrategy(CentroidAssignmentStragety):
    def __init__(self, item_code_bytes, num_items, quantisation_strategy="quantile") -> None:
        super().__init__(item_code_bytes, num_items)
        self.quantisation_strategy = quantisation_strategy


    def assign(self, train_users):
     rows = []
     cols = []
     vals = []
     item_pop = Counter()

     for user_idx, sequence in enumerate(train_users):
        for (_, item_id) in sequence:
            if item_id >= self.num_items:
                raise ValueError(
                    f"Item_id={item_id} ≥ num_items={self.num_items} ➜ Errore di mapping (controlla item_id_to_index)"
                )
            rows.append(user_idx)
            cols.append(item_id)
            vals.append(1)
            item_pop[item_id] += 1

     print(f"[DEBUG] num_items: {self.num_items}, max(cols): {max(cols)}, shape: ({len(train_users)}, {self.num_items + 2})")
     assert max(cols) < self.num_items + 2, f"⚠️ Col index {max(cols)} ≥ matrix shape {self.num_items + 2}"

     matr = csr_matrix((vals, (rows, cols)), shape=(len(train_users), self.num_items + 2))

     print("Fitting SVD for initial centroids assignment")
     svd = TruncatedSVD(n_components=self.item_code_bytes)
     svd.fit(matr)
     item_embeddings = svd.components_
     print("SVD fit done.")

     item_pop_vector = np.zeros(self.num_items + 2)
     for i in range(self.num_items + 2):
        item_pop_vector[i] = item_pop[i]

     assignments = []
     for i in range(self.item_code_bytes):
        discretizer = CustomKBinsDiscretizer(
            n_bins=256,
            encode='ordinal',
            strategy=self.quantisation_strategy,
            item_popularity=item_pop_vector
        )
        ith_component = item_embeddings[i]
        ith_component = (ith_component - np.min(ith_component)) / (np.max(ith_component) - np.min(ith_component) + 1e-10)
        ith_component += np.random.normal(0, 1e-5, self.num_items + 2)
        ith_component = np.expand_dims(ith_component, axis=1)
        component_assignments = discretizer.fit_transform(ith_component).astype('uint8')[:, 0]
        assignments.append(component_assignments)

     return np.transpose(np.array(assignments))
