# 6 · Knowledge graphs

Knowledge-aware models score from a graph of facts about the items as well as
from the interactions. This guide reads the LastFM data and the Freebase graph
released with KGCN, and covers:

- the two files `reader.knowledge` reads (triples and an item → entity
  alignment), through the Python API and through a configuration;
- **coverage**: only part of the catalogue is aligned to an entity, what WarpRec
  reports and warns about, what each model does with an uncovered item, and how
  to restrict the catalogue to the covered items;
- the `KnowledgeGraph` a `Dataset` carries: re-indexing, the item → entity
  alignment, neighbourhoods, the adjacency matrix and the first- and
  second-order item features the feature-based models read;
- a short training run of KaHFM next to BPR through the design pipeline.

[Open the notebook](https://github.com/sisinflab/warprec/blob/main/guides/06-knowledge-graphs/knowledge-graphs.ipynb){ .md-button .md-button--primary }

## Running it

`pip install warprec jupyter`. Runs in one to two minutes on a laptop CPU, most of it training; it downloads the KGCN music files (2 MB) into `guides/data/`. Clone the repository, or download the guide's folder together with `guides/guide_data.py`, and open the notebook from its own folder: the configuration files refer to the data relative to it.

## Files

- [`configs/knowledge.yml`](https://github.com/sisinflab/warprec/blob/main/guides/06-knowledge-graphs/configs/knowledge.yml)
- [`configs/train.yml`](https://github.com/sisinflab/warprec/blob/main/guides/06-knowledge-graphs/configs/train.yml)

## Reference

- [Knowledge-Aware Recommenders](../recommenders/knowledge.md)
- [Reader Configuration](../configuration/reader.md)
