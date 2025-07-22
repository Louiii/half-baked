import io
import os
from dataclasses import dataclass
from functools import partial
from typing import Mapping, MutableMapping

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from md_encoder.atom_modules.modules import (
    MLP,
    Transition,
    get_initializer_scale,
    glorot_uniform,
    pad3,
)
from tqdm import tqdm


@dataclass
class EncoderConfig:
    stride: int
    scope: int
    channels: int
    pos_enc_dim: int
    n_head: int
    qk_dim: int
    v_dim: int
    out_dim: int
    zero_init: bool


def spatial_attention_encoder_layer(data: jnp.array, encoder_config: EncoderConfig):
    scope = encoder_config.scope
    pos_enc_dim = encoder_config.pos_enc_dim
    stride = encoder_config.stride
    n_head = encoder_config.n_head
    qk_dim = encoder_config.qk_dim
    v_dim = encoder_config.v_dim
    out_dim = encoder_config.out_dim
    zero_init = encoder_config.zero_init

    s = 2 * scope + 1
    total_size = data.shape[0]

    h = data.shape[-1]
    data += MLP([2 * h, 2 * h, 2 * h, h], "mlp_in")(data)

    s1, s2, s3, c = data.shape
    padded_data = pad3(data, pad=scope)

    class ConvolutionKernel(hk.Module):
        def __init__(self, name):
            super().__init__(name=name)

        def __call__(self, x: jnp.array, bias: jnp.array):
            """Attend from the central voxel in x to all of the others, using a predefi-
            ned positional encoding. TODO: Possibly add simple symmetries by concat.

            Args:
                x (jnp.array): shape [s, s, s, c], where s = 2*scope+1, c = `channel`
                bias (jnp.array): shape [s, s, s], used as a mask

            Returns:
                _type_: _description_
            """
            s, s, s, c = x.shape
            weight_init = get_initializer_scale("linear", (c,))

            positional_encoding = hk.get_parameter(
                "positional_encoding",
                (s, s, s, pos_enc_dim),
                x.dtype,
                weight_init
            )
            data = jnp.concatenate([x, positional_encoding], axis=-1)
            query = data[scope, scope, scope]
            h = data.shape[-1]
            memory = data.reshape(-1, h)
            # query: [h,], memory: [s**3, h]
            qw = hk.get_parameter("q", shape=(h, n_head, qk_dim), init=glorot_uniform())
            kw = hk.get_parameter("k", shape=(h, n_head, qk_dim), init=glorot_uniform())
            vw = hk.get_parameter("v", shape=(h, n_head, v_dim), init=glorot_uniform())
            q = jnp.einsum("a,ahc->hc", query, qw)
            k = jnp.einsum("ka,ahc->khc", memory, kw)
            v = jnp.einsum("ka,ahc->khc", memory, vw)
            _logits = jnp.einsum("hc,khc->hk", q, k)
            _bias = bias.reshape(-1, 1).T
            # import pdb; pdb.set_trace()
            logits = _logits + _bias
            scores = jax.nn.softmax(logits, axis=-1)
            mixed_v = jnp.einsum("hk,khc->hc", scores, v)

            init = hk.initializers.Constant(0.0) if zero_init else glorot_uniform()
            o_weights = hk.get_parameter("out_w", shape=(n_head, v_dim, out_dim), init=init)
            o_bias = hk.get_parameter("out_b", shape=(out_dim,), init=hk.initializers.Constant(0.0))
            output = jnp.einsum("hc,hco->o", mixed_v, o_weights) + o_bias
            return output


    # get_mask = lambda i: (jnp.arange(s) + i >= scope) * (jnp.arange(s) + i < total_size + scope).astype(int)
    def wrap_conv(central_index):
        # perform the slice, make mask from indices
        away_from_start = jnp.arange(s)[None, :] + central_index[:, None] >= scope
        away_from_end = jnp.arange(s)[None, :] + central_index[:, None] < total_size + scope
        mask = away_from_start * away_from_end
        mask = mask[0][:, None, None] + mask[1][None, :, None] + mask[2][None, None, :]
        bias = 1e9 * (mask - 1.0)[..., None]  # the bias is added in log space
        central_index = jnp.concatenate([central_index, jnp.zeros(1)]).astype(jnp.int32)
        _x = jax.lax.dynamic_slice(
            padded_data, start_indices=central_index, slice_sizes=[s, s, s, c]
        )
        conv = ConvolutionKernel("attention_ConvolutionKernel")(_x, bias)
        return Transition("transition")(conv)

    # done in numpy to avoid the jax tracers
    ss = np.array([s1, s2, s3], dtype=jnp.int32)
    nps = 1 + ((ss - 1) // stride)
    assert np.all((nps - 1) * stride == ss - 1), "Make sure the final stride fits in"
    nps = nps[::-1]
    total_points = np.prod(nps)
    stagger = np.concatenate([np.ones(1), np.cumprod(nps)[:-1]])[::-1]

    stagger = jnp.array(stagger)

    def _1d_ix_to_central_ixs(i):
        _3d_ix = (i // stagger) % nps[::-1]
        return scope + stride * _3d_ix

    _1d_ixs = jnp.arange(total_points)
    run = jax.vmap(lambda i: wrap_conv(_1d_ix_to_central_ixs(i)), in_axes=(0,))
    out = run(_1d_ixs)
    return out.reshape(*nps, -1)


def spatial_attention_decoder_layer(data: jnp.array, encoder_config: EncoderConfig):
    """Perform the reverse of the encoder layer by attending to the empty positional
    encodings

    Args:
        data (jnp.array): _description_
        encoder_config (EncoderConfig): _description_
    """
    scope = encoder_config.scope
    pos_enc_dim = encoder_config.pos_enc_dim
    stride = encoder_config.stride
    n_head = encoder_config.n_head
    qk_dim = encoder_config.qk_dim
    v_dim = encoder_config.v_dim
    out_dim = encoder_config.channels
    zero_init = encoder_config.zero_init

    s = 2 * scope + 1
    total_size = data.shape[0]

    class DeConvolutionKernel(hk.Module):
        """Currently only supporting non-overlapping deconvs"""
        def __init__(self, name):
            super().__init__(name=name)

        def __call__(self, x: jnp.array, bias: jnp.array):
            """Since a deconvolution which maps a single voxel to the neighbour voxels
            would produce a overlapping neighbour outputs they need to be combined in
            some way... this is a little tricky to variably tune the sequential and
            parallel parts.
            Big mem case: consider all the corner cross points to collate, i.e. (s-1)^3
            points which have 8 voxels adjacent. Then vmap a gather-mean op over these.
            Would need to repeat for non-corner edge collation.
            Can then generalise this to small mem by making one voxel axis sequential.
            This actually should be simple with the sharding function.
            The outer-most datastructures would be very big- ((s-1) * cube_len)^3

            Args:
                x (jnp.array): shape [c]
                bias (jnp.array): shape [s, s, s] represents a mask to ensure that the
                  outer voxel activations are not written to

            Returns:
                _type_: _description_
            """
            

            weight_init = get_initializer_scale("linear", (c,))

            positional_encoding = hk.get_parameter(
                "positional_encoding",
                (s, s, s, x.shape[-1]),
                x.dtype,
                weight_init
            )
            pos = positional_encoding.reshape(-1, x.shape[-1])
            # the positional encoding will act as the memory
            # it will be directly used for the keys
            # import pdb; pdb.set_trace()
            data = jnp.concatenate([x[None, :].repeat(pos.shape[0], 0), pos], axis=-1)


            # query = data[scope, scope, scope]
            h = data.shape[-1]
            qw = hk.get_parameter("q", shape=(h, n_head, qk_dim), init=glorot_uniform())
            kw = hk.get_parameter("k", shape=(x.shape[-1], n_head, qk_dim), init=glorot_uniform())
            vw = hk.get_parameter("v", shape=(h, n_head, v_dim), init=glorot_uniform())
            q = jnp.einsum("qa,ahc->qhc", data, qw)
            k = jnp.einsum("ka,ahc->khc", pos, kw)
            v = jnp.einsum("ka,ahc->khc", data, vw)
            _logits = jnp.einsum("qhc,khc->qhk", q, k)
            # _bias = bias.reshape(-1, 1).T
            # # import pdb; pdb.set_trace()
            logits = _logits # + _bias
            scores = jax.nn.softmax(logits, axis=-1)
            mixed_v = jnp.einsum("qhk,khc->qhc", scores, v)

            init = hk.initializers.Constant(0.0) if zero_init else glorot_uniform()
            o_weights = hk.get_parameter("out_w", shape=(n_head, v_dim, out_dim), init=init)
            o_bias = hk.get_parameter("out_b", shape=(out_dim,), init=hk.initializers.Constant(0.0))
            output = jnp.einsum("qhc,hco->qo", mixed_v, o_weights) + o_bias
            output = output.reshape(s, s, s, -1)
            return output

    bias = 0

    h = data.shape[-1]
    data += MLP([2 * h, 2 * h, 2 * h, h], "mlp_in")(data)
    deconvolution_kernel = DeConvolutionKernel("deconvolution")
    vdeconv = jax.vmap(deconvolution_kernel, in_axes=(0, None))
    *voxels, c = data.shape
    out = vdeconv(data.reshape(-1, c), bias)
    full_out = out.reshape(*voxels, *out.shape[1:])
    n = s * data.shape[0]
    out = jnp.moveaxis(
        full_out,
        (0, 3, 1, 4, 2, 5, 6),
        (0, 1, 2, 3, 4, 5, 6),
    ).reshape(n, n, n, -1)
    out = out[scope:-scope, scope:-scope, scope:-scope]
    return Transition("transition_out")(out)


def loss(params, key, data):
    return ((data - fwd.apply(params, key, data)) ** 2).mean()


# @jax.jit
def overfit(params, key, data):
    l, dps = jax.value_and_grad(loss)(params, key, data)
    params = jax.tree_map(lambda p, dp: p - lr * dp, params, dps)
    return l, params

# fwd = hk.transform(encoder)

def save(filepath, params):
    _p = jax.tree_map(np.asarray, params)

    def flatten(d, parent_key="", sep="//"):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                items.extend(flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    _p = hk.data_structures.to_mutable_dict(_p)
    np.savez_compressed(filepath, **flatten(_p, parent_key="", sep="//"))


def load(path):
    with open(path, "rb") as f:
        params = np.load(io.BytesIO(f.read()), allow_pickle=False)

    def flat_params_to_haiku(params: Mapping[str, np.ndarray]) -> hk.Params:
        """Convert a dictionary of NumPy arrays to Haiku parameters."""
        hk_params = {}
        for path, array in params.items():
            scope, name = path.split("//")
            if scope not in hk_params:
                hk_params[scope] = {}
            hk_params[scope][name] = jnp.array(array)

        return hk_params
    return flat_params_to_haiku(params)


if __name__ == "__main__":
    
    encoder_config = EncoderConfig(
        stride=3,
        scope=1,
        channels=3,
        pos_enc_dim=32,
        n_head=4,
        qk_dim=128,
        v_dim=128,
        out_dim=128,
        zero_init=False,
    )
    encoder = partial(spatial_attention_encoder_layer, encoder_config=encoder_config)
    decoder = partial(spatial_attention_decoder_layer, encoder_config=encoder_config)
    key = jax.random.PRNGKey(seed=0)
    fwd = hk.transform(lambda x: decoder(encoder(x)))
    lr = 3e-2

    s = 10
    data = jax.random.normal(key, shape=(s, s, s, encoder_config.channels))
    params = fwd.init(key, data)
    # z = fwd.apply(params, key, data)
    param_path = "tparams.npz"
    possible_path = input("load from checkpoint? Give path or 'N'.")
    if possible_path != "N":
        assert os.path.isfile(possible_path)
        params = load(possible_path)
        param_path = possible_path
    pbar = tqdm(range(1, 10_001))
    for t in pbar:
        l, params = overfit(params, key, data)
        pbar.set_description(f"{l.item():.8f}")
        if t % 100 == 0:
            save(param_path, params)
