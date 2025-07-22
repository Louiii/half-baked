"""It would be great to use jax-md. However, for the example I created this is not a
good fit because the conformational changed can't be decomposed into energy functions.
"""

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax.random import PRNGKey
from md_encoder.atom_modules.modules import meshgrid
from tqdm import tqdm

TRAIN = ['AABBC', 'AABCC', 'AABBB', 'AACCB', 'AACBC', 'AABCB']
TEST = ['AACCC', 'AACBB']
A2I = dict(zip("ABC", range(3)))

spatial = 2
bond_length = 0.8
seq_len = 5


# @dataclasses.dataclass
# class Chain:
#     theta: jnp.array  # shape: [3,]
#     origin: jnp.array  # shape: [2,]
#     orientation: jnp.array  # shape: [2,]
#     bond_length: float
#     in_contact: bool


class Chain:
    theta: jnp.array  # shape: [3,]
    origin: jnp.array  # shape: [2,]
    orientation: jnp.array  # shape: [2,]
    bond_length: float
    in_contact: bool

    def energy(self, chain):
        return
    

def generate_sequence(key):
    tail = random.choice(
        key, jnp.arange(2), p=jnp.array([0.7, 0.3]), replace=True, shape=(seq_len - 2,)
    )
    return "AA" + "".join([["B", "C"][t] for t in tail])


def sample_rotation_directions(key, sequence):
    keys = iter(random.split(key, num=seq_len - 2))
    map_ = {
        "B": (lambda: 0),
        "C": (
            lambda: random.choice(
                next(keys), jnp.array([-1, 1]), p=jnp.array([0.95, 0.05])
                )
            ),
    }
    return [map_[char]() for char in sequence[2:]]


def get_rotation(orientation):
    # clockwise
    return jnp.array([
        [jnp.cos(orientation), jnp.sin(orientation)],
        [-jnp.sin(orientation), jnp.cos(orientation)],
    ])


def init():
    """
    state ['beginning', 'middle']:
      'beginning' -> all chains are initialised to a straight line.
      'middle' -> initialise uniformly within the stationary markov chain.
    """
    return box, 
