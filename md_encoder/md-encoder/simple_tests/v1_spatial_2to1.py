from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
from md_encoder.atom_modules.modules import (
    MLP,
    Transition,
    get_initializer_scale,
    glorot_uniform,
)
from tqdm import tqdm


def enc(x):
    """x: [scope, hidden] -> y: [hidden]"""
    n_head, qk_dim, v_dim = 4, 128, 128
    pos_enc_dim, out_dim = 64, 128
    class E(hk.Module):
        def __init__(self, name):
            super().__init__(name=name)

        def __call__(self, x: jnp.array):
            """
            """
            s, c = x.shape
            weight_init = get_initializer_scale("linear", (c,))

            positional_encoding = hk.get_parameter(
                "positional_encoding",
                (s, pos_enc_dim),
                x.dtype,
                weight_init
            )
            data = jnp.concatenate([x, positional_encoding], axis=-1)
            h = data.shape[-1]
            memory = data.reshape(-1, h)
            # query: [h,], memory: [s**3, h]
            qw = hk.get_parameter("q", shape=(h, n_head, qk_dim), init=glorot_uniform())
            kw = hk.get_parameter("k", shape=(h, n_head, qk_dim), init=glorot_uniform())
            vw = hk.get_parameter("v", shape=(h, n_head, v_dim), init=glorot_uniform())
            # print(data.shape)
            # print(qw.shape)
            q = jnp.einsum("a,ahc->hc", data[0], qw)
            k = jnp.einsum("ka,ahc->khc", memory, kw)
            v = jnp.einsum("ka,ahc->khc", memory, vw)
            _logits = jnp.einsum("hc,khc->hk", q, k)
            # import pdb; pdb.set_trace()
            logits = _logits
            scores = jax.nn.softmax(logits, axis=-1)
            mixed_v = jnp.einsum("hk,khc->hc", scores, v)

            init = glorot_uniform()
            o_weights = hk.get_parameter("out_w", shape=(n_head, v_dim, out_dim), init=init)
            o_bias = hk.get_parameter("out_b", shape=(out_dim,), init=hk.initializers.Constant(0.0))
            output = jnp.einsum("hc,hco->o", mixed_v, o_weights) + o_bias
            return output
    return E("enc")(x)

def dec(x):
    """x: [hidden] -> y: [scope, hidden]"""
    s = 2
    pos_enc_dim, h = 64, 128
    class D(hk.Module):
        def __init__(self, name):
            super().__init__(name=name)

        def __call__(self, x: jnp.array):
            """
            """
            (c,) = x.shape
            weight_init = get_initializer_scale("linear", (c,))

            pos = hk.get_parameter(
                "positional_encoding",
                (s, pos_enc_dim),
                x.dtype,
                weight_init
            )
            data = jnp.concatenate([x[None, :].repeat(pos.shape[0], 0), pos], axis=-1)
            
            return MLP([2 * h, 2 * h, 2 * h, 10], "mlp_in")(data, no_final_nonlin=True)
    return D("dec")(x)


key = jax.random.PRNGKey(seed=0)
fwd = hk.transform(lambda x: dec(enc(x)))

def loss(params, key, data):
    return ((data - fwd.apply(params, key, data)) ** 2).mean()

lr = 3e-2

@jax.jit
def overfit(params, key, data):
    l, dps = jax.value_and_grad(loss)(params, key, data)
    params = jax.tree_map(lambda p, dp: p - lr * dp, params, dps)
    return l, params


s = 10
data = jax.random.normal(key, shape=(2, 10))
params = fwd.init(key, data)
pbar = tqdm(range(1, 10_001))
for t in pbar:
    l, params = overfit(params, key, data)
    pbar.set_description(f"{l.item():.8f}")
