# 📈 WarpRec Metrics

The WarpRec `Metrics` module offers a collection of state-of-the-art metrics for evaluating recommender systems. The subsequent sections will describe the different categories of **built-in** metrics already implemented inside WarpRec. The final section includes an easy-to-follow tutorial on how to implement custom metrics in the WarpRec framework.

## 📚 Table of Contents
- 🎯 [Accuracy](#🎯-accuracy)
- ⚖️ [Bias](#️⚖️-bias)
- 🌍 [Coverage](#🌍-coverage)
- 🧩 [Diversity](#🧩-diversity)
- 👥 [Fairness](#👥-fairness)
- ✨ [Novelty](#✨-novelty)
- ⭐ [Rating](#⭐-rating)
- 📘 [How to implement your metric](#📘-how-to-implement-your-metric)

## 🎯 Accuracy

Accuracy metrics quantify how well a recommender system predicts user preferences or identifies relevant items. They assess the correctness of recommendations by comparing predicted interactions or ratings against actual user behavior. High accuracy generally indicates that the system is effective at surfacing items users are likely to engage with.

**Available Accuracy Metrics**:

- [**Precision@K**](accuracy/precision.py): Measures the proportion of recommended items at rank K that are actually relevant.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [Precision]
...
```
- [**Recall@K**](accuracy/recall.py): Measures the proportion of relevant items that are successfully recommended within the top K items.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [Recall]
...
```
- [**F1-Score@K**](accuracy/f1.py): The harmonic mean of Precision@K and Recall@K, providing a balanced measure of accuracy.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [F1]
...
```
Extended-F1 is also available in WarpRec, allowing you to compute the harmonic mean of two metrics of your choice, as follows:
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: ["F1[nDCG, MAP]"]
...
```
- [**HitRate@K**](accuracy/hit_rate.py): Measures the percentage of users for whom at least one relevant item is found within the top K recommendations.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [HitRate]
...
```
- [**NDCG@K (Normalized Discounted Cumulative Gain)**](accuracy/ndcg.py): Evaluates the ranking quality of recommendations, giving higher scores to relevant items appearing at higher ranks.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [nDCG]
...
```
- [**MAP@K (Mean Average Precision)**](accuracy/map.py): Measures the mean of average precision scores across all users, rewarding correct recommendations ranked higher.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [MAP]
...
```
- [**MAR@K (Mean Average Recall)**](accuracy/mar.py): Measures the mean of average recall scores across all users, indicating how well the relevant items are retrieved on average.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [MAR]
...
```
- [**MRR@K (Mean Reciprocal Rank)**](accuracy/mrr.py): Measures the average of the reciprocal ranks of the first relevant item in the recommendations.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [MRR]
...
```
- [**AUC (Area Under the ROC Curve)**](accuracy/auc.py): Measures the probability that a randomly chosen relevant item is ranked higher than a randomly chosen irrelevant one.
```yaml
evaluation:
    metrics: [AUC]
...
```
- [**GAUC (Group Area Under the ROC Curve)**](accuracy/gauc.py): Computes AUC per user (or group), then averages the results; accounts for group-level ranking quality.
```yaml
evaluation:
    metrics: [GAUC]
...
```
- [**LAUC (Limited Area Under the ROC Curve)**](accuracy/lauc.py): AUC computed over a limited set of top-ranked items, focusing on ranking quality within the most relevant recommendations.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [LAUC]
...
```

## ⚖️ Bias

Bias metrics are designed to identify and measure systematic deviations or unfair tendencies in recommender system outputs. These metrics help uncover whether the system disproportionately favors or disfavors certain items, users, or groups, potentially leading to a lack of diversity or equitability in recommendations.

- [**ACLT (Average Coverage of Long-Tail items)**](bias/aclt.py): Measures the proportion of long-tail items recommended across all users, indicating the extent of long-tail exposure.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [ACLT]
...
```
- [**APLT (Average Proportion of Long-Tail items)**](bias/aplt.py): Measures the average proportion of long-tail items in each user's recommendation list, which captures individual-level diversity.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [APLT]
...
```
- [**ARP (Average Recommendation Popularity)**](bias/arp.py): Calculates the average popularity of recommended items, indicating the system’s tendency to favor popular content.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [ARP]
...
```
- [**PopREO (Popularity-based Ranking-based Equal Opportunity)**](bias/pop_reo.py): Measures whether users receive similar ranks for long-tail items regardless of their group membership, focusing on fairness in exposure.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [PopREO]
...
```
- [**PopRSP (Popularity-based Ranking-based Statistical Parity)**](bias/pop_rsp.py): Evaluates whether the average ranks of long-tail items are balanced across user groups, promoting fairness in recommendation ranking.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [PopRSP]
...
```

## 🌍 Coverage

Coverage metrics assess the extent to which a recommender system is able to recommend items from the entire catalog. They measure the diversity of the items recommended and the proportion of the item space that the system can effectively explore. High coverage suggests that the system can offer a wide range of recommendations beyond just the most popular items.

- [**ItemCoverage@k**](coverage/item_coverage.py): Measures the number of unique items recommended in the top-k positions across all users, indicating catalog coverage.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [ItemCoverage]
...
```
- [**UserCoverage@k**](coverage/user_coverage.py): Calculates the number of users with at least one recommended item in their top-k recommendations, indicating reach and usefulness.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [UserCoverage]
...
```
- [**NumRetrieved@k**](coverage/numretrieved.py): Counts the total number of distinct items retrieved in the top-k recommendations across all users.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [NumRetrieved]
...
```
- [**UserCoverageAtN**](coverage/user_coverage_at_n.py): Measures the number of users for whom the recommender retrieves at least N items, reflecting system responsiveness.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [UserCoverageAtN]
...
```

## 🧩 Diversity

Diversity metrics evaluate the variety of items within a user's recommendations or across the recommendations for a set of users. These metrics are crucial for preventing "filter bubbles" and ensuring that users are exposed to a broad range of items, potentially increasing serendipity and user satisfaction.

- [**GiniIndex**](diversity/gini_index.py): Measures the inequality in the distribution of recommended items; lower values indicate more equitable item exposure.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [Gini]
...
```
- [**Shannon Entropy**](diversity/shannon_entropy.py): Quantifies the diversity of recommended items using information entropy; higher values reflect greater item variety.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [ShannonEntropy]
...
```
- [**SRecall (Subtopic Recall)**](diversity/srecall.py): Measures how many distinct subtopics or categories are covered in the recommendations compared to the relevant ones, which reflects diversity across semantic dimensions. **This metric requires the user to provide side information.**
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [SRecall]
...
```

## 👥 Fairness

Fairness metrics aim to ensure that recommender systems provide equitable recommendations across different user groups, particularly those defined by sensitive attributes (e.g., gender, age, socioeconomic status). These metrics help detect and mitigate disparate impact or treatment in recommendation outcomes.

- [**BiasDisparityBD**](fairness/biasdisparitybd.py): Measures the difference in recommendation bias between user groups, indicating how much one group is favored over another. **This metric requires the user to provide clustering information.**
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [BiasDisparityBD]
...
```
- [**BiasDisparityBR (Bias Disparity – Bias Recommendations)**](fairness/biasdisparitybr.py): Quantifies the disparity in the frequency of biased (e.g., popular) items recommended to different user groups within their top-K recommendations. **This metric requires the user to provide clustering information.**
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [BiasDisparityBR]
...
```
- [**BiasDisparityBS (Bias Disparity – Bias Scores)**](fairness/biasdisparitybs.py): Measures the disparity in the average bias scores of recommended items across user groups, assessing score-level bias. **This metric requires the user to provide clustering information.**
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [BiasDisparityBS]
...
```
- [**Item MAD Ranking**](fairness/itemmadranking.py): Computes the Mean Absolute Deviation of item ranks across user groups, measuring fairness in item exposure in rankings. **This metric requires the user to provide clustering information.**
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [ItemMADRanking]
...
```
- [**Item MAD Rating**](fairness/itemmadrating.py): Computes the Mean Absolute Deviation of predicted item ratings across user groups, assessing fairness in predicted preferences. **This metric requires the user to provide clustering information.**
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [ItemMADRating]
...
```
- [**User MAD Ranking**](fairness/usermadranking.py): Measures the Mean Absolute Deviation of item ranking positions for each user group, focusing on rank consistency across users. **This metric requires the user to provide clustering information.**
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [UserMADRanking]
...
```
- [**User MAD Rating**](fairness/usermadrating.py): Measures the Mean Absolute Deviation of predicted item ratings for each user group, capturing disparities in predicted relevance. **This metric requires the user to provide clustering information.**
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [UserMADRating]
...
```
- [**REO (Ranking-based Equal Opportunity)**](fairness/reo.py): Assesses whether relevant items are ranked similarly across user groups, ensuring fair visibility of relevant content. **This metric requires the user to provide clustering information.**
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [REO]
...
```
- [**RSP (Ranking-based Statistical Parity)**](fairness/rsp.py): Measures whether the ranking positions of items (regardless of relevance) are equally distributed across user groups, ensuring fairness in exposure. **This metric requires the user to provide clustering information.**
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [RSP]
...
```

## ✨ Novelty

Novelty metrics assess the extent to which a recommender system suggests items that are new or unexpected to the user, beyond what is already popular or frequently consumed. These metrics are important for fostering exploration and serendipity, as highly novel recommendations can lead to delightful discoveries.

- [**EFD(Expected Free Discovery)**](novelty/efd.py): Estimates the likelihood that users discover relevant but less popular (unexpected) items in their top-K recommendations, promoting serendipity.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [EFD]
...
```
Extended-EFD is also available inside WarpRec, meaning you can compute the EFD score using a discounted relevance value, as follows:
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: ["EFD[discounted]"]
...
```
- [**EPC(Expected Popularity Complement)**](novelty/epc.py): Measures the average complement of item popularity in the top-K recommendations, encouraging exposure to less popular content.
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [EPC]
...
```
Extended-EPC is also available inside WarpRec, meaning you can compute the EPC score using a discounted relevance value, like such:
```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: ["EPC[discounted]"]
...
```

## ⭐ Rating

Rating metrics are specifically designed for recommender systems that predict explicit user ratings (e.g., 1-5 stars). These metrics quantify the accuracy of these numerical predictions by comparing them to the actual user ratings.

- [**MAE (Mean Absolute Error)**](rating/mae.py): The average absolute difference between predicted and actual ratings.
```yaml
evaluation:
    metrics: [MAE]
...
```
- [**MSE (Mean Squared Error)**](rating/mse.py): The average of the squared differences between predicted and actual ratings.
```yaml
evaluation:
    metrics: [MSE]
...
```
- [**RMSE (Root Mean Squared Error)**](rating/rmse.py): The square root of the MSE, providing an error measure in the same units as the ratings.
```yaml
evaluation:
    metrics: [RMSE]
...
```

## 📘 How to implement your metric

In this section we will walk you through how to implement your own metric inside WarpRec. First of all, let's import the main metric interface:

```python
from warprec.evaluation.metrics.base_metric import TopKMetric, BaseMetric
```

The **BaseMetric** interface is gonna be sufficient for metrics that do not depends on the cutoff, if your metric is based on ranking, then use the **TopKMetric** interface.

In this guide, we will implement a simple metric like the Precision@K using WarpRec interfaces. First of all, the Precision@K for a single user is defined as the number of relevant item actually recommended by the model, divided by the cutoff. The Precision@K of the whole system is the sum of all the Precision@K of all users, divided by the number of users.

Before starting with the implementation, we need to define what we need to *accumulate* during the batch iterations. The cutoff is actually a fixed value, so we don't need to accumulate it, on the other hand the number of **hits** inside the recommendations and the number of users it's not fixed. Now that we defined what we need, we can start the implementation.

**The number of users might seem a fixed value, but we will see the reasons why this might not be true later.**

```python
class MyMetric(TopKMetric):
    hits: Tensor
    users: Tensor

    def __init__(
        self,
        k: int,
        train_set: csr_matrix,
        *args: Any,
        item_cluster: Tensor = None,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state("hits", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")
```

And like this we have initialized our metric, it's that simple. After that, we need to define the two main components of the metric evaluation: the .update() method and the .compute() method.

The .update() takes as input the prediction tensor, which is gonna be of size `[batch_size, num_items]`. In this method we want to update the accumulators, based on the information of the predictions. Inside the kwargs passed to the .update() methods you can pass different information other than the predictions. For this computation we will need the ground_truth, another tensor of size `[batch_size, num_items]` that contains the relevance of each item for each user.

In order to compute Precision@K we need to retrieve the *binary_relevance*, in case the rating are not in the form of a [0, 1] tensor. Let's see how we can do it.

```python
def update(self, preds: Tensor, **kwargs: Any):
    """Updates the metric state with the new batch of predictions."""
    target: Tensor = kwargs.get("ground", torch.zeros_like(preds))
    target = self.binary_relevance(target)
```

Now we want to calculate a tensor of size `[batch_size, top_k]` which will contain, for each pair `(user, k)`, if in that position we actually retrieved a relevant item inside the recommendations. In order to do so, there is already a built-in functionality:

```python
def update(self, preds: Tensor, **kwargs: Any):
    """Updates the metric state with the new batch of predictions."""
    target: Tensor = kwargs.get("ground", torch.zeros_like(preds))
    target = self.binary_relevance(target)
    top_k_rel = self.top_k_relevance(preds, target, self.k)
```

And finally the last step of the .update() method will just accumulate the results. For the number of hits we already computed everything, the `top_k_rel` contains all the hits, so we just need to sum it. For the users, as anticipated before we need to check if these users are valid. For this there is also a functionality built-in. The number of user can change in two main scenarios:

1. A user that does not have at least one relevant item will not be considered inside evaluation.
2. We might want to compute an approximation of our metric, computing it only on the first ten batches, in that case the number of users evaluated need to be updated with each iteration.

To do so, we finalize the method like this:

```python
def update(self, preds: Tensor, **kwargs: Any):
    """Updates the metric state with the new batch of predictions."""
    target: Tensor = kwargs.get("ground", torch.zeros_like(preds))
    target = self.binary_relevance(target)
    top_k_rel = self.top_k_relevance(preds, target, self.k)

    # Accumulate
    self.hits = top_k_rel.sum()
    self.users = self.valid_users(target)
```

Now let's see how to create the .compute() method. This method will be called to compute the final value of the metric, whether or not it has been computed on all the batches. WarpRec expects metrics to return a dictionary, containing the name of the metric as the *key* and the computed value as the *value*, so we can finish the metric like this.

```python
def compute(self):
    """Computes the final metric value."""
    precision = (
        self.hits / (self.users * self.k)
        if self.users > 0
        else {"MyMetric": torch.tensor(0.0)}
    )
    return {"MyMetric": precision.item()}
```

And you are done! You Precision@K is completed and working as expected. You can add it to the *metric_registry* if you want and it will be available during training with a configuration file.

Optionally, you can add a .reset() method which will reset all the values of the accumulators. When using the add_state() method, Torchmetrics will handle the reset for you, but if you are not then a custom reset must be defined.
