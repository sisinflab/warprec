# Metrics Taxonomy

WarpRec includes **40 GPU-accelerated metrics** organized into 8 families. All metrics are implemented as PyTorch modules and support distributed evaluation.

| Family | Metric | Description | Type |
|---|---|---|---|
| **Accuracy** | [AUC](accuracy.md#auc) | Area Under the ROC Curve. | Global |
| | [F1@K](accuracy.md#f1) | Harmonic mean of Precision@K and Recall@K (or custom pair). | Top-K |
| | [GAUC](accuracy.md#gauc) | Per-user AUC averaged across all users. | Global |
| | [HitRate@K](accuracy.md#hitrate) | Fraction of users with at least one relevant item in top K. | Top-K |
| | [LAUC@K](accuracy.md#lauc) | AUC limited to the top-K ranked items. | Top-K |
| | [MAP@K](accuracy.md#map) | Mean Average Precision rewarding higher-ranked correct items. | Top-K |
| | [MAR@K](accuracy.md#mar) | Mean Average Recall indicating progressive retrieval quality. | Top-K |
| | [MRR@K](accuracy.md#mrr) | Mean Reciprocal Rank of the first relevant item. | Top-K |
| | [nDCG@K](accuracy.md#ndcg) | Normalized Discounted Cumulative Gain (exponential relevance). | Top-K |
| | [nDCGRendle2020@K](accuracy.md#ndcg) | nDCG with binary relevance following Rendle et al. (2020). | Top-K |
| | [Precision@K](accuracy.md#precision) | Proportion of relevant items in the top K. | Top-K |
| | [Recall@K](accuracy.md#recall) | Proportion of relevant items successfully retrieved in top K. | Top-K |
| **Rating** | [MAE](rating.md#mae) | Mean Absolute Error between predicted and actual ratings. | Global |
| | [MSE](rating.md#mse) | Mean Squared Error between predicted and actual ratings. | Global |
| | [RMSE](rating.md#rmse) | Root Mean Squared Error (same-scale error measure). | Global |
| **Coverage** | [ItemCoverage@K](coverage.md#itemcoverage) | Number of unique items recommended across all users. | Top-K |
| | [UserCoverage@K](coverage.md#usercoverage) | Number of users with at least one recommended item. | Top-K |
| | [NumRetrieved@K](coverage.md#numretrieved) | Average number of items retrieved per user in top K. | Top-K |
| | [UserCoverageAtN](coverage.md#usercoverageatn) | Number of users with at least N items retrieved. | Top-K |
| **Novelty** | [EFD@K](novelty.md#efd) | Expected Free Discovery (log-discounted novelty). | Top-K |
| | [EPC@K](novelty.md#epc) | Expected Popularity Complement (linear novelty). | Top-K |
| **Diversity** | [Gini@K](diversity.md#gini) | Gini Index measuring inequality of item exposure. | Top-K |
| | [ShannonEntropy@K](diversity.md#shannonentropy) | Information entropy over item recommendation frequencies. | Top-K |
| | [SRecall@K](diversity.md#srecall) | Subtopic Recall measuring feature/category coverage. | Top-K |
| **Bias** | [ACLT@K](bias.md#aclt) | Average Coverage of Long-Tail items in recommendations. | Top-K |
| | [APLT@K](bias.md#aplt) | Average Proportion of Long-Tail items per user. | Top-K |
| | [ARP@K](bias.md#arp) | Average Recommendation Popularity of top-K items. | Top-K |
| | [PopREO@K](bias.md#popreo) | Popularity-based Ranking-based Equal Opportunity. | Top-K |
| | [PopRSP@K](bias.md#poprsp) | Popularity-based Ranking-based Statistical Parity. | Top-K |
| **Fairness** | [BiasDisparityBD@K](fairness.md#biasdisparitybd) | Relative disparity between recommendation and training bias. | Top-K |
| | [BiasDisparityBR@K](fairness.md#biasdisparitybr) | Bias in recommendation frequency per user-item cluster pair. | Top-K |
| | [BiasDisparityBS](fairness.md#biasdisparitybs) | Bias in training data per user-item cluster pair. | Global |
| | [ItemMADRanking@K](fairness.md#itemmadranking) | Mean Absolute Deviation of discounted gain across item clusters. | Top-K |
| | [ItemMADRating@K](fairness.md#itemmadrating) | Mean Absolute Deviation of average ratings across item clusters. | Top-K |
| | [REO@K](fairness.md#reo) | Ranking-based Equal Opportunity across item clusters. | Top-K |
| | [RSP@K](fairness.md#rsp) | Ranking-based Statistical Parity across item clusters. | Top-K |
| | [UserMADRanking@K](fairness.md#usermadranking) | Mean Absolute Deviation of nDCG across user clusters. | Top-K |
| | [UserMADRating@K](fairness.md#usermadrating) | Mean Absolute Deviation of average scores across user clusters. | Top-K |
| **Multi-objective** | [EucDistance@K](multiobjective.md#eucdistance) | Euclidean distance from model performance to a Utopia Point. | Top-K |
| | [Hypervolume@K](multiobjective.md#hypervolume) | Volume of objective space dominated relative to a Nadir Point. | Top-K |
