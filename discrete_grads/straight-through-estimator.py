import functools
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

lr_step_queue = [
    (1.0, 0),
    (4e-3, 1_000),
    (1e-4, 10_000),
    (3e-5, 50_000),
    (1e-5, float('inf')),
]

steps = 100_000
interval = 100
n = 100
n_samples = 1
latent_size = 26
outdir = "straight_through"
os.system(f"rm -rf {outdir}")
os.makedirs(outdir, exist_ok=True)


x = jnp.linspace(-3, 3, n)[:, None]
y = jnp.sin(x)

def make_params(key, layer_sizes):
    params = []
    for h1, h2 in zip(layer_sizes[:-1], layer_sizes[1:]):
        init_sd = (0.2 * (h1 + h2)) ** -0.5
        key, sk = jax.random.split(key)
        w = jax.random.normal(key=sk, shape=(h1, h2)) * init_sd
        b = jnp.zeros(h2)
        params.append((w, b))
    return key, params


key = jax.random.PRNGKey(seed=0)
key, f_params = make_params(key, layer_sizes=[1, 32, latent_size])
key, g_params = make_params(key, layer_sizes=[latent_size, 128, 32, 128, 1])


def linear(act, w, b):
    return jnp.einsum("i,ij->j", act, w) + b


def f(x, params):
    act = x
    for w, b in params[:-1]:
        act = jax.nn.relu(linear(act, w, b))
    logits = linear(act, *params[-1])

    # p_unormalised = jnp.exp(logits - jnp.max(logits))
    # probs = p_unormalised / p_unormalised.sum()
    return jax.nn.sigmoid(logits)


def g(z, params):
    act = z
    for w, b in params[:-1]:
        act = jax.nn.relu(linear(act, w, b))
    y_hat = linear(act, *params[-1])
    return y_hat


def full_forward(key, x, params, n_samples):
    f_params, g_params = params
    prob = f(x, f_params)

    key, sk = jax.random.split(key)
    unif = jax.random.uniform(key=sk, shape=(n_samples, latent_size))
    z = (unif < prob[None, :]).astype(jnp.float32)

    # score function trick
    z = jax.lax.stop_gradient(z) + prob - jax.lax.stop_gradient(prob)

    y_hats = jax.vmap(g, in_axes=(0, None))(z, g_params)
    return key, y_hats, prob


def forward_and_loss(key, x, y, params, n_samples):
    key, y_hats, prob = full_forward(key, x, params, n_samples)
    return jnp.mean((y[None, :] - y_hats) ** 2), key


def _update(key, f_params, g_params, x, y, n_samples: int = 1):
    """_summary_

    Args:
        f_params (dict): param tree
        g_params (dict): param tree
        x (1-dimensional array): input vector
        y (1-dimensional array): target vector

    Returns:
        _type_: _description_
    """
    loss_fn = functools.partial(forward_and_loss, n_samples=n_samples)
    grad_fn = jax.value_and_grad(loss_fn, argnums=3, has_aux=True)
    params = (f_params, g_params)
    (loss, key), grads = grad_fn(key, x, y, params)
    params = jax.tree_map(lambda _p, _dp: _p - lr * _dp, params, grads)
    (f_params, g_params) = params
    return key, f_params, g_params, loss


def _inference(key, x, f_params, g_params, n_samples):
    key, y_hats, probs = full_forward(key, x, (f_params, g_params), n_samples)
    return key, y_hats, probs


def make_plot(ix, y_pred, probs):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    xrep = jnp.squeeze(x)[None, :].repeat(n_samples, axis=0).T.reshape(-1)
    ax1.plot(x, y, label="target")
    ax1.plot(xrep, y_pred.reshape(-1), marker=".", linestyle="None", label="samples")
    ax1.legend()

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    im = ax2.imshow(probs.T, cmap='bone')
    ax2.set_xlabel("x")
    ax2.set_xticks([])
    ax2.set_xticks([], minor=True)
    fig.colorbar(im, cax=cax, orientation='vertical')

    plt.suptitle(f"step: {ix}")

    plt.savefig(f"{outdir}/training_{str(ix).zfill(5)}.png")
    plt.clf()


update = jax.jit(functools.partial(_update, n_samples=n_samples))
inference = functools.partial(_inference, n_samples=n_samples)
inf = jax.jit(jax.vmap(inference, in_axes=(None, 0, None, None)))

losses = []
pbar = tqdm(range(steps))
lr, next_step = lr_step_queue.pop(0)
for i in pbar:
    if i == next_step:
        new_lr, next_step = lr_step_queue.pop(0)
        print(f"step: {i}; changing lr from {lr} to {new_lr}")
        lr = new_lr
    key, f_params, g_params, loss = update(
        key, f_params, g_params, x[i % n], y[i % n]
    )
    losses.append(np.array(loss))
    pbar.set_description(f"{np.mean(losses[-min(len(losses), 100):])}")
    if i % interval == 0:
        _key, y_pred, probs = inf(key, x, f_params, g_params)
        key = _key[0]
        make_plot(i // interval, y_pred, probs)