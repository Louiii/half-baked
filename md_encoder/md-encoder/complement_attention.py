"""Problem:

Define a set: S {1...N} with a corresponding fixed representation Z.
Sequentially, attempt to recover items from this S only using Z.
"""
import jax
import jax.numpy as jnp
from md_encoder.atom_modules.modules import meshgrid

spatial = 2
items = meshgrid([-1, 0, 1], num_dimensions=spatial).reshape(-1, spatial)
modes = items.shape[0]
weights = jnp.array([100] + [10] * 5 + [1] * 2 + [0.1])
weights /= weights.sum()

key = jax.random.PRNGKey(seed=0)


def sample_true_distribution(key, sd=0.1):
    key, subkey = jax.random.split(key)
    mode = jax.random.choice(subkey, items, replace=True, p=weights)
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, shape=(spatial,)) * sd
    return mode + noise


key, subkey = jax.random.split(key)
sample = sample_true_distribution(key, sd=0.1)

# take many samples, summarise into Z

# recover S, first s_i \in S, s_{i+1} \in S \ {i}, ...



# GOAL:
#   input: Fixed size, Z.
#   ouput: Set of items, Y, p(Y_i).
#   loss: Compare a set of items X and their probabilities P,
#         to predicted items Y and the predicted probabilities K.
#         Compute pairwise error E_ij of X_i and Y_j, then compute
#         permutation π:j->i then compute the loss as ∑_i E_{i,π_i}
#         and KL(P, K) = ∑_i P_i log(K_π_i)
"""
Training time:
    data: X = {(s_i, p_i): clustered states and epirical probabilities}
    data-transforms: randomly mask some states. Inital no masking, then increase.
    encoder_forward: map all states to a latent variable, Z.
    decoder_forward: map Z to predicted states and probabilities X'.
    loss: as above.

Inference time:
    data: X = {(s_1, [mask]), mask,..., mask}
    X' <- decoder_forward(encoder_forward(X))

Generating training data:
    - run MD and perform some sort of clustering (in torsion angle space cat
    coord space cat distogram space).
    - run some Boltzman / MCMC sampling procedure.
"""
