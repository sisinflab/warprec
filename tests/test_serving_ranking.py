"""A served list is the list the run evaluated and wrote.

Serving masks seen items, breaks ties and re-ranks through the calls offline
evaluation and the recommendation writer make, from the seed the checkpoint
records. Popularity ties many items, so the tie break is exercised as well.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from conftest import make_model
from warprec.data.dataset import Dataset
from warprec.data.writer import LocalWriter
from warprec.recommenders.reranking import build_reranker
from warprec.serving.payload import build_serving_payload
from warprec.serving.servable import ServableModel, ServingPolicy
from warprec.utils.config import RerankConfig


@pytest.mark.parametrize("rerank", [None, "MMR", "Calibration"])
def test_a_served_list_is_the_list_the_run_wrote(
    tmp_path: Path, dataset: Dataset, rerank: Optional[str]
):
    """Every known user is served the items, in the order, written for them."""
    model = make_model("Pop", dataset)
    config = RerankConfig(name=rerank, pool=8, params={"weight": 0.7})
    k, seed = 6, 7

    writer = LocalWriter(dataset_name="served", local_path=str(tmp_path))
    writer.write_recs(
        model=model,
        dataset=dataset,
        k=k,
        reranker=build_reranker(config, dataset),
        seed=seed,
    )
    path = next(Path(writer.experiment_recommendation_path).glob("*.tsv"))
    written = pd.read_csv(path, sep="\t").groupby("user_id", sort=False)["item_id"]
    expected = {user: items.tolist() for user, items in written}

    payload = build_serving_payload(model, dataset, {"evaluation": {"seed": seed}})
    servable = ServableModel(model, payload, ServingPolicy(rerank=config))
    answers = servable.recommend(
        [servable.resolve(user_id=user, k=k) for user in expected]
    )
    served = {
        user: [entry["item_id"] for entry in answer]
        for user, answer in zip(expected, answers)
    }

    assert served == expected
