# 13 · Cold-Start, Debiased and Re-Ranked Evaluation

Guide 12 evaluated models on the usual protocol: interactions held out, every
item rankable, every hit worth the same, the list ordered by score. This guide
changes each of those three assumptions in turn, on MovieLens-100K with the
19 genres as item side information:

- **cold start**: the `item_cold_start` and `user_cold_start` splits,
  `evaluation.candidates` to rank only the cold items, the random floor the
  evaluator logs, and why ties are broken at random;
- **debiased evaluation**: `evaluation.propensity` and the `IPSRecall`,
  `SNIPSRecall`, `IPSDCG` and `SNIPSDCG` estimators next to their naive
  counterparts, and what `clip` does to their variance;
- **re-ranking**: the top-level `rerank` section with `MMR` and `Calibration`,
  applied inside the evaluator, and the accuracy each one gives up for its
  own objective.

Everything runs through the `Evaluator` API (guide 12) with the configuration
section printed next to it.

[Open the notebook](https://github.com/sisinflab/warprec/blob/main/guides/13-cold-start-debiased-and-re-ranked-evaluation/cold-start-debiased-and-re-ranked-evaluation.ipynb){ .md-button .md-button--primary }

## Running it

`pip install warprec jupyter`. Runs in one to two minutes on a laptop CPU, most of it training two small sequential models; `guide_data.movielens_100k()` downloads and converts MovieLens-100K exactly as guide 1 shows. Clone the repository, or download the guide's folder together with `guides/guide_data.py`, and open the notebook from its own folder: the configuration files refer to the data relative to it.

## Files

- [`configs/debiased.yml`](https://github.com/sisinflab/warprec/blob/main/guides/13-cold-start-debiased-and-re-ranked-evaluation/configs/debiased.yml)
- [`configs/item-cold-start.yml`](https://github.com/sisinflab/warprec/blob/main/guides/13-cold-start-debiased-and-re-ranked-evaluation/configs/item-cold-start.yml)
- [`configs/rerank-calibration.yml`](https://github.com/sisinflab/warprec/blob/main/guides/13-cold-start-debiased-and-re-ranked-evaluation/configs/rerank-calibration.yml)
- [`configs/rerank-mmr.yml`](https://github.com/sisinflab/warprec/blob/main/guides/13-cold-start-debiased-and-re-ranked-evaluation/configs/rerank-mmr.yml)
- [`configs/user-cold-start.yml`](https://github.com/sisinflab/warprec/blob/main/guides/13-cold-start-debiased-and-re-ranked-evaluation/configs/user-cold-start.yml)

## Reference

- [Cold-Start Protocols](../evaluation/cold-start.md)
- [Debiased Evaluation](../evaluation/debiased.md)
- [Re-Ranking](../recommenders/reranking.md)
