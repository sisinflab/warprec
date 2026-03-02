# Metrics Taxonomy

WarpRec includes **40 GPU-accelerated metrics** organized into 8 families. All metrics are implemented as PyTorch modules and support distributed evaluation.

| Family | Metric | Description | Type |
|---|---|---|---|
| **Accuracy** | AUC | Area Under the ROC Curve. | Global |
| | F1@K | Harmonic mean of Precision@K and Recall@K (or custom pair). | Top-K |
| | GAUC | Per-user AUC averaged across all users. | Global |
| | HitRate@K | Fraction of users with at least one relevant item in top K. | Top-K |
| | LAUC@K | AUC limited to the top-K ranked items. | Top-K |
| | MAP@K | Mean Average Precision rewarding higher-ranked correct items. | Top-K |
| | MAR@K | Mean Average Recall indicating progressive retrieval quality. | Top-K |
| | MRR@K | Mean Reciprocal Rank of the first relevant item. | Top-K |
| | nDCG@K | Normalized Discounted Cumulative Gain (exponential relevance). | Top-K |
| | nDCGRendle2020@K | nDCG with binary relevance following Rendle et al. (2020). | Top-K |
| | Precision@K | Proportion of relevant items in the top K. | Top-K |
| | Recall@K | Proportion of relevant items successfully retrieved in top K. | Top-K |
| **Rating** | MAE | Mean Absolute Error between predicted and actual ratings. | Global |
| | MSE | Mean Squared Error between predicted and actual ratings. | Global |
| | RMSE | Root Mean Squared Error (same-scale error measure). | Global |
| **Coverage** | ItemCoverage@K | Number of unique items recommended across all users. | Top-K |
| | UserCoverage@K | Number of users with at least one recommended item. | Top-K |
| | NumRetrieved@K | Average number of items retrieved per user in top K. | Top-K |
| | UserCoverageAtN | Number of users with at least N items retrieved. | Top-K |
| **Novelty** | EFD@K | Expected Free Discovery (log-discounted novelty). | Top-K |
| | EPC@K | Expected Popularity Complement (linear novelty). | Top-K |
| **Diversity** | Gini@K | Gini Index measuring inequality of item exposure. | Top-K |
| | ShannonEntropy@K | Information entropy over item recommendation frequencies. | Top-K |
| | SRecall@K | Subtopic Recall measuring feature/category coverage. | Top-K |
| **Bias** | ACLT@K | Average Coverage of Long-Tail items in recommendations. | Top-K |
| | APLT@K | Average Proportion of Long-Tail items per user. | Top-K |
| | ARP@K | Average Recommendation Popularity of top-K items. | Top-K |
| | PopREO@K | Popularity-based Ranking-based Equal Opportunity. | Top-K |
| | PopRSP@K | Popularity-based Ranking-based Statistical Parity. | Top-K |
| **Fairness** | BiasDisparityBD@K | Relative disparity between recommendation and training bias. | Top-K |
| | BiasDisparityBR@K | Bias in recommendation frequency per user-item cluster pair. | Top-K |
| | BiasDisparityBS | Bias in training data per user-item cluster pair. | Global |
| | ItemMADRanking@K | Mean Absolute Deviation of discounted gain across item clusters. | Top-K |
| | ItemMADRating@K | Mean Absolute Deviation of average ratings across item clusters. | Top-K |
| | REO@K | Ranking-based Equal Opportunity across item clusters. | Top-K |
| | RSP@K | Ranking-based Statistical Parity across item clusters. | Top-K |
| | UserMADRanking@K | Mean Absolute Deviation of nDCG across user clusters. | Top-K |
| | UserMADRating@K | Mean Absolute Deviation of average scores across user clusters. | Top-K |
| **Multi-objective** | EucDistance@K | Euclidean distance from model performance to a Utopia Point. | Top-K |
| | Hypervolume@K | Volume of objective space dominated relative to a Nadir Point. | Top-K |
