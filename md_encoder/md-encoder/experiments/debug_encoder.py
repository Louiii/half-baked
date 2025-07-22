import hydra
import jax
import jax.numpy as jnp
from md_encoder.atom_modules.decoder import AtomDecoder
from md_encoder.atom_modules.encoder_functions import prepare_encoder
from md_encoder.atom_modules.modules import meshgrid
from md_encoder.utils import termplot

# if __name__ == "__main__":
box_length = 1.
box_size = jnp.array([box_length, box_length, box_length])
num_atoms = 40
spatial_dims = 3
is_training = True

with hydra.initialize(config_path="../config", job_name="run_enc_dec"):
    config = hydra.compose(config_name="shared.yaml", overrides=[])

encoder_fn = prepare_encoder(config.encoder, box_length, spatial_dims)

points = jax.random.uniform(jax.random.PRNGKey(seed=0), (num_atoms, spatial_dims))


import matplotlib.pyplot as plt
import numpy as np
from md_encoder.atom_modules.encoder_functions import points_2_lattice

lt = points_2_lattice(points, jnp.ones(points.shape[0]), config.encoder, box_length, spatial_dims)

if spatial_dims == 2:
    print(lt.shape)
    plt.matshow(lt[..., 0].T)
    plt.scatter(*(points * np.array(lt.shape[:2])[None, :]).T, c="r")
    plt.show()


gap = box_length / (lt.shape[0] + 1)
# |. . . .|
vox_centers = jnp.linspace(gap * 0.5, box_length - gap * 0.5, lt.shape[0])
m3 = meshgrid(vox_centers, num_dimensions=3)
import numpy as np

val = np.array(lt[..., 0].reshape(-1))
coords = m3.reshape(-1, 3)
ix = val > np.percentile(val, 80)

# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# ax.scatter(*points.T)
# ax.scatter(*coords[ix].T, c=val[ix], alpha=val[ix] / np.max(val))
# plt.show()

key = jax.random.PRNGKey(seed=0)
enc_params = encoder_fn.init(key, points, jnp.ones(points.shape[0]))
enc_key, dec_key = jax.random.PRNGKey(seed=0), jax.random.PRNGKey(seed=0)


p = enc_params
mask = jnp.ones(points.shape[0])
latent_shape = encoder_fn.apply(p, enc_key, points, mask).shape
j = jax.random.normal(jax.random.PRNGKey(0), latent_shape)

def d(p):
    latent = encoder_fn.apply(p, enc_key, points, mask)
    return jnp.mean((latent - j) ** 2)

g = jax.jit(jax.value_and_grad(d, argnums=0))

losses = []
with termplot.DisplayProgress(15, 20, "|") as prog:
    for i in range(10):
        l, dp = g(p)
        losses.append(l)
        p = jax.tree_util.tree_map(lambda x, y: x - y * .01, p, dp)
        # print(l)
        prog.update(losses, prepend_string=f"current loss: {l}")
