from copy import deepcopy
from ray import tune


# This function can for sure be done in a better way
def parse_params(params: dict):
    tune_params = {}
    strategy = params["optimization"]["strategy"]
    params_copy = deepcopy(params)
    params_copy.pop("meta")
    params_copy.pop("optimization")
    if strategy == "grid":
        for k, v in params_copy.items():
            tune_params[k] = tune.grid_search(v)

    elif strategy == "hopt":
        for k, v in params_copy.items():
            tune_params[k] = tune.uniform(v[0], v[1])

    return tune_params
