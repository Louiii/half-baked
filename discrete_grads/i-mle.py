import functools
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

lr = 6e-5
n = 100
samples = 1
latent_size = 32
scale_epsilon = jnp.array(0.05)
lambda_scale = jnp.array(1.)
outdir = "i_mle"
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
key, g_params = make_params(key, layer_sizes=[latent_size, 32, 1])


def linear(act, w, b):
    return jnp.einsum("i,ij->j", act, w) + b


def f(x, params):
    act = x
    for w, b in params[:-1]:
        act = jax.nn.relu(linear(act, w, b))
    logits = linear(act, *params[-1])

    safe_logits = logits - jax.lax.stop_gradient(jnp.max(logits))
    # p_unormalised = jnp.exp(safe_logits)
    # probs = p_unormalised / p_unormalised.sum()
    # return jax.nn.sigmoid(logits)
    return safe_logits


def g(z, params):
    act = z
    for w, b in params[:-1]:
        act = jax.nn.relu(linear(act, w, b))
    y_hat = linear(act, *params[-1])
    return y_hat


def loss_fn(y, z, param):
    return jnp.squeeze((g(z, param) - y) ** 2)


def _update(
    key,
    f_params,
    g_params,
    x,
    y,
    scale_epsilon,
    lambda_scale,
    latent_size: int,
    samples: int = 1
):
    """_summary_

    Args:
        f_params (dict): param tree
        g_params (dict): param tree
        x (1-dimensional array): input vector
        y (1-dimensional array): target vector

    Returns:
        _type_: _description_
    """
    # compute the gradient of each parameter wrt the output vector
    jac_f = jax.jacfwd(f, argnums=1)

    # compute the gradient of each parameter wrt the loss
    vloss = jax.vmap(loss_fn, in_axes=(None, 0, None))
    def multi_sample_loss(y, z_s, g_params):
        losses = vloss(y, z_s, g_params)
        return jnp.mean(losses)
    grad_g = jax.value_and_grad(multi_sample_loss, argnums=(1, 2), has_aux=False)

    # update g params
    safe_logits = f(x, f_params)

    # sample logits
    key, sk = jax.random.split(key)
    epsilon = jax.random.normal(key=sk, shape=(samples, latent_size)) * scale_epsilon
    noised_logits = safe_logits[None, :] + epsilon
    z = jax.nn.one_hot(jnp.argmax(noised_logits, axis=-1), noised_logits.shape[-1])

    loss, (dg_dz, dg_dp) = grad_g(y, z, g_params)

    # update g_params
    g_params = jax.tree_map(lambda _p, _dp: _p - lr * _dp, g_params, dg_dp)

    # compute a better version of the safe_logits
    better_logits = safe_logits[None, :] - lambda_scale * dg_dz

    grad_z_path = z - jax.nn.one_hot(
        jnp.argmax(better_logits + epsilon, axis=-1),
        better_logits.shape[-1]
    )
    HACK = 1.0
    # send grads through z via grad_z_path to f_param grads (instead of via dg_dz if there was no stochasticity)
    jac_tensors = jac_f(x, f_params)
    chain_rule = lambda _jac: jnp.mean(jnp.einsum("i...,bi->b...", _jac, grad_z_path), axis=0)
    df_dp = jax.tree_map(chain_rule, jac_tensors)
    f_params = jax.tree_map(lambda _p, _dp: _p - lr * HACK * _dp, f_params, df_dp)
    return key, f_params, g_params, loss


def inference(key, x, f_params, g_params, scale_epsilon, n_samples):
    safe_logits = f(x, f_params)
    # sample logits
    key, sk = jax.random.split(key)
    epsilon = jax.random.normal(key=sk, shape=(n_samples, latent_size)) * scale_epsilon
    noised_logits = safe_logits[None, :] + epsilon
    z = jax.nn.one_hot(jnp.argmax(noised_logits, axis=-1), noised_logits.shape[-1])

    samples = jax.vmap(g, in_axes=(0, None))(z, g_params)
    return samples, safe_logits


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
    # plt.close(fig)


update = jax.jit(functools.partial(
    _update,
    samples=samples,
    latent_size=latent_size,
))
n_samples = 10
inf = functools.partial(inference, n_samples=n_samples)
inf = jax.vmap(inf, in_axes=(None, 0, None, None, None))

interval = 1000
losses = []
max_step = 100000
part_way = 1000
psuedo_steps = 400
end_length = (psuedo_steps + max_step - part_way)
pbar = tqdm(range(max_step))
for i in pbar:
    if i > part_way:
        # lr = 5e-4
        prop = 1 - ((i - part_way) / end_length)
        scale_epsilon = scale_epsilon * prop
    key, f_params, g_params, loss = update(
        key, f_params, g_params, x[i % n], y[i % n], scale_epsilon, lambda_scale
    )
    losses.append(np.array(loss))
    pbar.set_description(f"{np.mean(losses[-min(len(losses), 100):])}")
    if i % interval == 0:
        y_pred, probs = inf(key, x, f_params, g_params, scale_epsilon)
        make_plot(i // interval, y_pred, probs)

# ffmpeg -framerate 5 -pattern_type glob -i 'i_mle/*.png' -c:v libx264 -pix_fmt yuv420p out.mp4