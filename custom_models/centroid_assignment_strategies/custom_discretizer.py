from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
#discretizzazione basata su popolarita di item
class CustomKBinsDiscretizer(KBinsDiscretizer):
    def __init__(self, n_bins=5, encode='onehot', strategy='quantile', item_popularity=None):
        super().__init__(n_bins=n_bins, encode=encode, strategy=strategy)
        self.item_popularity = item_popularity

    def fit(self, X, y=None):
        if self.strategy == 'quantile_popularity' and self.item_popularity is not None:
            self.bin_edges_ = []
            for feature in range(X.shape[1]):
                quantiles = np.linspace(0, 1, self.n_bins + 1)
                bin_edges = weighted_quantile(X[:, feature], quantiles, sample_weight=self.item_popularity)
                self.bin_edges_.append(bin_edges)
            self.bin_edges_ = np.array(self.bin_edges_, dtype=object)
        else:
            super().fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
#Nel quantile classico, i bin vengono creati in modo che ogni gruppo abbia lo stesso numero di elementi.
#Nel quantile_popularity, gli item più popolari hanno più peso, quindi i bin sono influenzati dalla distribuzione di popolarità.

def weighted_quantile(values, quantiles, sample_weight=None):
    """ Compute weighted quantiles. """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    sample_weight_sum = sum(sample_weight)
    range_per_quantile = sample_weight_sum / (len(quantiles) - 1)
    result = [values[0]]
    covered_weight = 0
    cnt = 0
    eps = 1e-8
    for i in range(1, len(quantiles)  -1):
        target_weight = range_per_quantile * i
        while covered_weight < target_weight - eps:
            covered_weight += sample_weight[cnt]
            cnt += 1
        result.append(values[cnt])
    result.append(values[-1])
    return result