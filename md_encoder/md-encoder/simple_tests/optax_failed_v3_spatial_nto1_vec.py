from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from md_encoder.atom_modules.modules import (
    MLP,
    Transition,
    get_initializer_scale,
    glorot_uniform,
)
from tqdm import tqdm

CHANNELS = 6
SCOPE = 30
n_head = 8
qk_dim = 256
v_dim = 256
pos_enc_dim = 64
out_dim = 128
STEPS = 10_000
h = 256
BATCH = 100
DEC_MLP_HIDDEN = [512, 256, 512, 256, CHANNELS]
clip_global_grad_norm = 1.0
lr_decay = 1.0
warm_up_n_steps = 1000
lr_decay_after_n_steps = 1.0
b1 = .9
b2 = .999
num_grad_acc_steps = 1
lr = 3e-2


def enc(x):
    """x: [batch, scope, hidden] -> y: [batch, hidden]"""
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
    return Transition("tr")(jax.vmap(E("enc"))(x))

def dec(x):
    """x: [batch, hidden] -> y: [batch, scope, hidden]"""
    class D(hk.Module):
        def __init__(self, name):
            super().__init__(name=name)

        def __call__(self, x: jnp.array):
            """"""
            (c,) = x.shape
            weight_init = get_initializer_scale("linear", (c,))

            pos = hk.get_parameter(
                "positional_encoding",
                (SCOPE, pos_enc_dim),
                x.dtype,
                weight_init
            )
            data = jnp.concatenate([x[None, :].repeat(pos.shape[0], 0), pos], axis=-1)
            
            return MLP(DEC_MLP_HIDDEN, "mlp_in")(data, no_final_nonlin=True)
    return jax.vmap(D("dec"))(x)


key = jax.random.PRNGKey(seed=0)
fwd = hk.transform(lambda x: dec(enc(x)))

def loss(params, key, data):
    return ((data - fwd.apply(params, key, data)) ** 2).mean()

# @jax.jit
# def overfit(params, key, data):
#     l, dps = jax.value_and_grad(loss)(params, key, data)
#     params = jax.tree_map(lambda p, dp: p - lr * dp, params, dps)
#     return l, params


def get_single_sample_gradient_transformation():
    optimizer = optax.clip_by_global_norm(clip_global_grad_norm)
    return optimizer

def get_optimizer():
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0,
                end_value=lr,
                transition_steps=warm_up_n_steps,
            ),
            optax.constant_schedule(value=lr_decay * lr),
        ],
        boundaries=[lr_decay_after_n_steps],
    )

    optimizer = optax.chain(
        optax.scale_by_adam(b1=b1, b2=b2),
        optax.scale_by_schedule(lr_schedule),
        # optax.clip_by_global_norm(clip_global_grad_norm),
        # Scale updates by -1 since optax.apply_updates is *additive*
        # and we want to descend on the loss.
        optax.scale(-1.0),
        # optax.apply_every(num_grad_acc_steps),
    )
    optimizer = optax.MultiSteps(
        optimizer,
        every_k_schedule=num_grad_acc_steps,
        use_grad_mean=True,
    )
    return optimizer


s = 10
data = jax.random.normal(key, shape=(BATCH, SCOPE, CHANNELS))
params = fwd.init(key, data)
optimizer_state = get_optimizer().init(params)

@jax.jit
def overfit(params, key, data, optimizer_state):
    l, grads = jax.value_and_grad(loss)(params, key, data)
    updates, optimizer_state = get_optimizer().update(grads, optimizer_state)
    params = optax.apply_updates(params, updates)
    return l, params, optimizer_state


pbar = tqdm(range(1, STEPS + 1))
for t in pbar:
    l, params, optimizer_state = overfit(params, key, data, optimizer_state)
    pbar.set_description(f"{l.item():.8f}")
