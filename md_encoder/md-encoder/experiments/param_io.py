import io
from typing import Any, Dict

import flax
import jax
import numpy as np


def flatten(d, parent_key='', sep='/'):
    ty = type(jax.tree_leaves(d)[0])
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if type(v) != ty:
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten(
    d: Dict[str, Any], 
    base: Dict[str, Any] = None,
    delim: str = "/",
) -> Dict[str, Any]:
    if base is None:
        base = {}

    for key, value in d.items():
        root = base

        ###
        # If a dotted path is encountered, create nested dicts for all but
        # the last level, then change root to that last level, and key to
        # the final key in the path.
        #
        # This allows one final setitem at the bottom of the loop.
        #
        if delim in key:
            *parts, key = key.split(delim)

            for part in parts:
                root.setdefault(part, {})
                root = root[part]

        if isinstance(value, dict):
            value = unflatten(value, root.get(key, {}))

        root[key] = value

    return base


def save(path, params):
    params = flatten(params)
    np.savez(path, **params)


def load(path):
    with open(path, "rb") as f:
        params = np.load(io.BytesIO(f.read()), allow_pickle=False)
        params = {k: v for k, v in params.items()}
    return flax.core.frozen_dict.freeze(unflatten(params))
