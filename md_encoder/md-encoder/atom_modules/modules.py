import collections
import numbers
import time
from functools import partial
from re import I
from typing import List, Optional, Sequence, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

SPATIAL_DIMENSIONS = 3

# Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
TRUNCATED_NORMAL_STDDEV_FACTOR = np.asarray(0.87962566103423978, dtype=np.float32)

# R3_NEIGHBOURS = jnp.stack(np.meshgrid(*[np.array([0, 1, 2])]*3)).reshape(3, -1).T


def get_initializer_scale(initializer_name, input_shape):
    """Get Initializer for weights and scale to multiply activations by."""

    if initializer_name == "zeros":
        w_init = hk.initializers.Constant(0.0)
    else:
        # fan-in scaling
        scale = 1.0
        for channel_dim in input_shape:
            scale /= channel_dim
        if initializer_name == "relu":
            scale *= 2

        noise_scale = scale

        stddev = np.sqrt(noise_scale)
        # Adjust stddev for truncation.
        stddev = stddev / TRUNCATED_NORMAL_STDDEV_FACTOR
        w_init = hk.initializers.TruncatedNormal(mean=0.0, stddev=stddev)

    return w_init


def glorot_uniform():
    return hk.initializers.VarianceScaling(
        scale=1.0, mode="fan_avg", distribution="uniform"
    )


class Linear(hk.Module):
    """Protein folding specific Linear module.

    This differs from the standard Haiku Linear in a few ways:
      * It supports inputs and outputs of arbitrary rank
      * Initializers are specified by strings
    """

    def __init__(
        self,
        num_output: Union[int, Sequence[int]],
        initializer: str = "linear",
        num_input_dims: int = 1,
        use_bias: bool = True,
        bias_init: float = 0.0,
        precision=None,
        name: str = "linear",
    ):
        """Constructs Linear Module.

        Args:
          num_output: Number of output channels. Can be tuple when outputting
              multiple dimensions.
          initializer: What initializer to use, should be one of {'linear', 'relu',
            'zeros'}
          num_input_dims: Number of dimensions from the end to project.
          use_bias: Whether to include trainable bias
          bias_init: Value used to initialize bias.
          precision: What precision to use for matrix multiplication, defaults
            to None.
          name: Name of module, used for name scopes.
        """
        super().__init__(name=name)
        if isinstance(num_output, numbers.Integral):
            self.output_shape = (num_output,)
        else:
            self.output_shape = tuple(num_output)
        self.initializer = initializer
        self.use_bias = use_bias
        self.bias_init = bias_init
        self.num_input_dims = num_input_dims
        self.num_output_dims = len(self.output_shape)
        self.precision = precision

    def __call__(self, inputs):
        """Connects Module.

        Args:
          inputs: Tensor with at least num_input_dims dimensions.

        Returns:
          output of shape [...] + num_output.
        """

        if self.num_input_dims > 0:
            in_shape = inputs.shape[-self.num_input_dims :]
        else:
            in_shape = ()

        weight_init = get_initializer_scale(self.initializer, in_shape)

        in_letters = "abcde"[: self.num_input_dims]
        out_letters = "hijkl"[: self.num_output_dims]

        weight_shape = in_shape + self.output_shape
        weights = hk.get_parameter("weights", weight_shape, inputs.dtype, weight_init)

        equation = f"...{in_letters}, {in_letters}{out_letters}->...{out_letters}"

        output = jnp.einsum(equation, inputs, weights, precision=self.precision)

        if self.use_bias:
            bias = hk.get_parameter(
                "bias",
                self.output_shape,
                inputs.dtype,
                hk.initializers.Constant(self.bias_init),
            )
            output += bias

        return output


class LayerNorm(hk.Module):
    """LayerNorm module.
    See: https://arxiv.org/abs/1607.06450.
    """

    def __init__(
        self,
        axis: Union[int, Sequence[int], slice],
        create_scale: bool,
        create_offset: bool,
        eps: float = 1e-5,
        scale_init: Optional[hk.initializers.Initializer] = None,
        offset_init: Optional[hk.initializers.Initializer] = None,
        use_fast_variance: bool = False,
        name: Optional[str] = None,
    ):
        """Constructs a LayerNorm module.
        Args:
          axis: Integer, list of integers, or slice indicating which axes to
            normalize over.
          create_scale: Bool, defines whether to create a trainable scale
            per channel applied after the normalization.
          create_offset: Bool, defines whether to create a trainable offset
            per channel applied after normalization and scaling.
          eps: Small epsilon to avoid division by zero variance. Defaults ``1e-5``,
            as in the paper and Sonnet.
          scale_init: Optional initializer for gain (aka scale). By default, one.
          offset_init: Optional initializer for bias (aka offset). By default, zero.
          use_fast_variance: If true, use a faster but less numerically stable
            formulation for computing variance.
          name: The module name.
        """
        super().__init__(name=name)
        if not create_scale and scale_init is not None:
            raise ValueError("Cannot set `scale_init` if `create_scale=False`.")
        if not create_offset and offset_init is not None:
            raise ValueError("Cannot set `offset_init` if `create_offset=False`.")

        if isinstance(axis, slice):
            self.axis = axis
        elif isinstance(axis, int):
            self.axis = (axis,)
        elif isinstance(axis, collections.abc.Iterable) and all(
            isinstance(ax, int) for ax in axis
        ):
            self.axis = tuple(axis)
        else:
            raise ValueError("`axis` should be an int, slice or iterable of ints.")

        self.eps = eps
        self.create_scale = create_scale
        self.create_offset = create_offset
        self.scale_init = scale_init or jnp.ones
        self.offset_init = offset_init or jnp.zeros
        self.use_fast_variance = use_fast_variance

    def __call__(
        self,
        inputs: jnp.ndarray,
        scale: Optional[jnp.ndarray] = None,
        offset: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Connects the layer norm.
        Args:
          inputs: An array, where the data format is ``[N, ..., C]``.
          scale: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the scale applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_scale=True``.
          offset: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the offset applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_offset=True``.
        Returns:
          The array, normalized.
        """
        if self.create_scale and scale is not None:
            raise ValueError("Cannot pass `scale` at call time if `create_scale=True`.")
        if self.create_offset and offset is not None:
            raise ValueError(
                "Cannot pass `offset` at call time if `create_offset=True`."
            )

        axis = self.axis
        if isinstance(axis, slice):
            axis = tuple(range(inputs.ndim)[axis])

        mean = jnp.mean(inputs, axis=axis, keepdims=True)
        if self.use_fast_variance:
            mean_of_squares = jnp.mean(jnp.square(inputs), axis=axis, keepdims=True)
            variance = mean_of_squares - jnp.square(mean)
        else:
            variance = jnp.var(inputs, axis=axis, keepdims=True)

        param_shape = inputs.shape[-1:]
        if self.create_scale:
            scale = hk.get_parameter(
                "scale", param_shape, inputs.dtype, init=self.scale_init
            )
        elif scale is None:
            scale = np.array(1.0, dtype=inputs.dtype)

        if self.create_offset:
            offset = hk.get_parameter(
                "offset", param_shape, inputs.dtype, init=self.offset_init
            )
        elif offset is None:
            offset = np.array(0.0, dtype=inputs.dtype)

        scale = jnp.broadcast_to(scale, inputs.shape)
        offset = jnp.broadcast_to(offset, inputs.shape)
        mean = jnp.broadcast_to(mean, inputs.shape)

        eps = jax.lax.convert_element_type(self.eps, variance.dtype)
        inv = scale * jax.lax.rsqrt(variance + eps)
        return inv * (inputs - mean) + offset


class Attention(hk.Module):
    """Multihead attention."""

    def __init__(self, config, global_config, output_dim, name="attention"):
        super().__init__(name=name)

        self.config = config
        self.global_config = global_config
        self.output_dim = output_dim

    def __call__(self, q_data, m_data, bias, nonbatched_bias=None):
        """Builds Attention module.

        Arguments:
          q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
          m_data: A tensor of memories from which the keys and values are
            projected, shape [batch_size, N_keys, m_channels].
          bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
          nonbatched_bias: Shared bias, shape [N_queries, N_keys].

        Returns:
          A float32 tensor of shape [batch_size, N_queries, output_dim].
        """
        # Sensible default for when the config keys are missing
        key_dim = self.config.get("key_dim", int(q_data.shape[-1]))
        value_dim = self.config.get("value_dim", int(m_data.shape[-1]))
        num_head = self.config.num_head
        assert key_dim % num_head == 0
        assert value_dim % num_head == 0
        key_dim = key_dim // num_head
        value_dim = value_dim // num_head

        q_weights = hk.get_parameter(
            "query_w",
            shape=(q_data.shape[-1], num_head, key_dim),
            init=glorot_uniform(),
        )
        k_weights = hk.get_parameter(
            "key_w", shape=(m_data.shape[-1], num_head, key_dim), init=glorot_uniform()
        )
        v_weights = hk.get_parameter(
            "value_w",
            shape=(m_data.shape[-1], num_head, value_dim),
            init=glorot_uniform(),
        )

        q = jnp.einsum("bqa,ahc->bqhc", q_data, q_weights) * key_dim ** (-0.5)
        k = jnp.einsum("bka,ahc->bkhc", m_data, k_weights)
        v = jnp.einsum("bka,ahc->bkhc", m_data, v_weights)
        logits = jnp.einsum("bqhc,bkhc->bhqk", q, k) + bias
        if nonbatched_bias is not None:
            logits += jnp.expand_dims(nonbatched_bias, axis=0)
        weights = jax.nn.softmax(logits)
        weighted_avg = jnp.einsum("bhqk,bkhc->bqhc", weights, v)

        if self.global_config.zero_init:
            init = hk.initializers.Constant(0.0)
        else:
            init = glorot_uniform()

        if self.config.gating:
            gating_weights = hk.get_parameter(
                "gating_w",
                shape=(q_data.shape[-1], num_head, value_dim),
                init=hk.initializers.Constant(0.0),
            )
            gating_bias = hk.get_parameter(
                "gating_b",
                shape=(num_head, value_dim),
                init=hk.initializers.Constant(1.0),
            )

            gate_values = (
                jnp.einsum("bqc, chv->bqhv", q_data, gating_weights) + gating_bias
            )

            gate_values = jax.nn.sigmoid(gate_values)

            weighted_avg *= gate_values

        o_weights = hk.get_parameter(
            "output_w", shape=(num_head, value_dim, self.output_dim), init=init
        )
        o_bias = hk.get_parameter(
            "output_b", shape=(self.output_dim,), init=hk.initializers.Constant(0.0)
        )

        output = jnp.einsum("bqhc,hco->bqo", weighted_avg, o_weights) + o_bias

        return output


class RelativePositionalScope(hk.Module):
    def __init__(self, scope, capacity):
        super().__init__(name="encoder")
        """
        args:
            scope: Number of points to the left and right of the center.
        """
        self.scope = scope
        self.capacity = capacity

    def __call__(self, spatial_embeddings: jnp.array):
        """_summary_

        Args:
            spatial_embeddings (jnp.array[2 * self.scope + 1, ...(SPATIAL_DIMENSIONS)...]): 
        """
        width = 2 * self.scope + 1
        assert all(width == n for n in spatial_embeddings.shape[:-1])
        shape = (width, self.capacity,)
        single_dimension_relpos = hk.get_parameter(
            name="single_dimension_relpos",
            shape=shape,
            dtype=spatial_embeddings.dtype,
            init=get_initializer_scale(initializer_name="", input_shape=shape),
        )
        # n = spatial_embeddings.shape[0]
        # ix = jnp.arange(width)
        relpos = []
        for j in range(SPATIAL_DIMENSIONS):
            slc = tuple(slice(None) if i == j else None for i in range(SPATIAL_DIMENSIONS))
            data = single_dimension_relpos[slc]
            for i in range(SPATIAL_DIMENSIONS):
                if i != j:
                    data = data.repeat(width, axis=i)
            relpos.append(data)
        cube_relpos = jnp.concatenate(relpos + [spatial_embeddings], axis=-1)
        return cube_relpos.reshape(-1, cube_relpos.shape[-1])


class AttentionElement(hk.Module):
    def __init__(self, scope, capacity, in_channel, out_channel, final_channel):
        super().__init__(name="single_query_attention")
        self.scope = scope
        self.capacity = capacity
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.final_channel = final_channel

    def __call__(
        self,
        central_embedding: jnp.array,
        spatial_embeddings: jnp.array,
        mask: jnp.array = None
    ):
        """_summary_

        Args:
            central_embedding (jnp.array): _description_
            spatial_embeddings (jnp.array): _description_
        """
        memory = RelativePositionalScope(self.scope, self.capacity)(spatial_embeddings)
        query = Linear("query", self.channel)(central_embedding)
        keys = Linear("key", self.in_channel)(memory)
        values = Linear("value", self.out_channel)(memory)
        logits = jnp.einsum("i,ji->j", query, keys)
        if mask is not None:
            mask = mask.reshape(-1)
            logits -= (1. - mask) * 1e9
        scores = jax.nn.softmax(logits, axis=0)
        reweighted = jnp.einsum("i,ij->j", scores, values)
        out = Linear("out", self.final_channel)(reweighted)
        return out


class CubeAttention(hk.Module):
    def __init__(self, within_edges, scope):
        super().__init__(name="cube_attention")
        self.within_edges = within_edges
        self.scope = scope

    def __call__(self, spatial_embeddings: jnp.array):
        """_summary_

        Args:
            spatial_embeddings (jnp.array): _description_

        Returns:
            box_emb: _description_
        """
        n, n, n, d = spatial_embeddings.shape
        # mask = jnp.ones_like(spatial_embeddings)
        width = 2 * self.scope + 1
        # inner = jnp.arange(n - width)[None, :] + jnp.arange(width)[:, None]
        central_res = self.scope + jnp.arange(n - width)

        if not self.within_edges:
            m = n + 2 * self.scope

            def cat(se, sh, axis):
                return jnp.concatenate([jnp.zeros(sh), se, jnp.zeros(sh)], axis=axis)

            def cat3axes(se):
                se = cat(se, sh=(self.scope, n, n, d), axis=0)
                se = cat(se, sh=(m, self.scope, n, d), axis=1)
                se = cat(se, sh=(m, m, self.scope, d), axis=2)
                return se

            spatial_embeddings = cat3axes(spatial_embeddings)
            central_res = self.scope + jnp.arange(n)
            # mask = cat3axes(mask)

        def make_ixs(i):
            return jnp.arange(i - self.scope, i + self.scope + 1)

        def make_mask(ixs):
            return (ixs > self.scope) * (ixs < m - self.scope)

        def triple_broadcast(a, b, c):
            return a[:, None, None], b[None, :, None], c[None, None, :]

        def index_and_attend(central_res_i, central_res_j, central_res_k):
            centralemb = spatial_embeddings[central_res_i, central_res_j, central_res_k]
            ni = make_ixs(central_res_i)
            nj = make_ixs(central_res_j)
            nk = make_ixs(central_res_k)
            neighbour_embeddings = spatial_embeddings[tuple(jnp.meshgrid(ni, nj, nk))]
            # make the neighbour embedding mask live!
            if not self.within_edges:
                m1, m2, m3 = triple_broadcast(make_mask(ni), make_mask(nj), make_mask(nk))
                mask = m1 * m2 * m3
            else:
                mask = None
            return AttentionElement()(centralemb, neighbour_embeddings, mask)
        vee = jax.vmap(index_and_attend, in_axes=(0, None, None))
        vve = jax.vmap(vee, in_axes=(None, 0, None))
        vvv = jax.vmap(vve, in_axes=(None, None, 0))
        box_emb = vvv(central_res, central_res, central_res)
        return box_emb


class MLP(hk.Module):
    def __init__(self, dims, name="mlp"):
        super().__init__(name=name)
        self.dims = dims

    def __call__(self, act, no_final_nonlin=False):
        layers = []
        for i, d in enumerate(self.dims, start=1):
            layers += [
                Linear(d, initializer="relu", name=f"transition{i}"),
                jax.nn.relu,
            ]
        return hk.Sequential(layers[:-1] if no_final_nonlin else layers)(act)

class Transition(hk.Module):
    def __init__(self, name="transition_block"):
        super().__init__(name=name)

    def __call__(self, act):
        nc = act.shape[-1]

        num_intermediate = 4
        zero_init = "linear"

        act = hk.LayerNorm(
            axis=[-1], create_scale=True, create_offset=True, name="input_layer_norm"
        )(act)

        transition_module = hk.Sequential(
            [
                Linear(
                    num_intermediate, initializer="relu", name="transition1"
                ),
                jax.nn.relu,
                Linear(
                    nc,
                    initializer="zeros" if zero_init else "linear",
                    name="transition2",
                ),
            ]
        )

        return transition_module(act)


def get_neighbour_voxels(data, voxel_index, leave_n_axes=1):
    """Assume the data is padded on each axis by 1 at each end.
    args:
        x: [N_VOX+2, N_VOX+2, N_VOX+2, C]
    returns:
        y: [27, C] 27 neighbours
    """
    slc = jax.lax.dynamic_slice(
        data,
        jnp.concatenate([voxel_index, jnp.array([0] * leave_n_axes)]),
        (3, 3, 3,) + data.shape[-leave_n_axes:]
    )
    return slc.reshape(int(3**3), *data.shape[-leave_n_axes:])


def get_ndim_neighbours(data, voxel_index, leave_n_axes=1):
    """Assume the data is padded on each axis by 1 at each end.
    args:
        x: [N_VOX+2, N_VOX+2, N_VOX+2, C]
    returns:
        y: [27, C] 27 neighbours
    """
    vox_dims = data.ndim - leave_n_axes
    slc = jax.lax.dynamic_slice(
        data,
        jnp.concatenate([voxel_index, jnp.array([0] * leave_n_axes)]),
        (3,) * vox_dims + data.shape[-leave_n_axes:]
    )
    return slc.reshape(int(3 ** vox_dims), *data.shape[-leave_n_axes:])


# def meshgrid3(_1d: jnp.array):
#     """_summary_

#     Args:
#         _1d (jnp.array[N,]): an array with just one axis.

#     Returns:
#        (jnp.array[N, N, N, 3]): item (x, y, z, i) is _1d[x] for i==0, _1d[y] for i==1,
#        _1d[z] for i==2.
#     """
#     n = _1d.shape[0]
#     arrs = [
#         _1d[tuple(
#             [slice(None) if i == j else None for j in [0, 1, 2]]
#         ) + (None,)].repeat(n, axis=(i + 1) % 3).repeat(n, axis=(i + 2) % 3)
#         for i in [0, 1, 2]
#     ]
#     return jnp.concatenate(arrs, axis=-1)


def meshgrid(
    arrays_1d: Union[List[jnp.array], jnp.array],
    num_dimensions: Optional[int] = None,
):
    """_summary_

    Args:
        _1d (jnp.array[N,]): an array with just one axis.

    Returns:
       (jnp.array[N, N, N, 3]): item (x, y, z, i) is _1d[x] for i==0, _1d[y] for i==1,
       _1d[z] for i==2.
    """
    arrays_1d = jnp.array(arrays_1d)
    if arrays_1d.ndim == 2:
        assert num_dimensions is None
    else:
        assert type(num_dimensions) is int and num_dimensions >= 1
        arrays_1d = arrays_1d[None].repeat(num_dimensions, axis=0)

    dimensions = len(arrays_1d)
    lengths = [a.shape[0] for a in arrays_1d]
    expanded_arrays = []
    for dim in range(dimensions):
        _1d_arr = arrays_1d[dim]
        slices = [slice(None) if dim == j else None for j in range(dimensions)]
        expanded = _1d_arr[tuple(slices) + (None,)]
        for r in range(dimensions):
            if dim != r:
                expanded = expanded.repeat(lengths[r], axis=r)
        expanded_arrays.append(expanded)
    return jnp.concatenate(expanded_arrays, axis=-1)


def pad3(data, pad=1):
    s1, s2, s3, *c = data.shape
    dt = data.dtype
    sh = (pad, s2, s3, *c)
    _data = jnp.concatenate([jnp.zeros(sh, dt), data, jnp.zeros(sh, dt)], axis=0)
    sh = (s1 + 2 * pad, pad, s3, *c)
    _data = jnp.concatenate([jnp.zeros(sh, dt), _data, jnp.zeros(sh, dt)], axis=1)
    sh = (s1 + 2 * pad, s2 + 2 * pad, pad, *c)
    _data = jnp.concatenate([jnp.zeros(sh, dt), _data, jnp.zeros(sh, dt)], axis=2)
    padded_data = _data
    return padded_data


def pad_edges_ndim(data: jnp.array, pad: int = 1, leave_n_dims: int = 1):
    """_summary_

    Args:
        data (jnp.array): _description_
        pad (int, optional): _description_. Defaults to 1.
        leave_n_dims (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    voxel_dims, other_dims = data.shape[:-leave_n_dims], data.shape[-leave_n_dims:]
    dt = data.dtype

    def cat(data, sh, axis):
        return jnp.concatenate([jnp.zeros(sh, dt), data, jnp.zeros(sh, dt)], axis=axis)

    padded_dims = []
    for i, vd in enumerate(voxel_dims, start=1):
        sh = (*padded_dims, pad, *voxel_dims[i:], *other_dims)
        data = cat(data, sh, i - 1)
        padded_dims.append(vd + 2 * pad)
    return data


def spatial_hash_to_lattice(
    coords,
    atom_mask,
    domain: jnp.array,
    num_points_1d_per_bin_voxel: int,
    gauss_width: float,
    number_of_atom_types: int,
):
    assert coords.ndim == 5
    assert atom_mask.ndim == 4
    num_bin_voxels = coords.shape[0]
    lattice_voxel_size = domain / num_bin_voxels
    gap = lattice_voxel_size / num_points_1d_per_bin_voxel
    # this is within a voxel, imagine two neighbouring voxels: |. .|. .|
    within_voxel_lattice_1d = jnp.linspace(
        gap * 0.5, lattice_voxel_size - gap * 0.5, num_points_1d_per_bin_voxel
    )
    within_voxel_lattice = meshgrid(within_voxel_lattice_1d, num_dimensions=3)

    # for static shapes after indexing-- hmm maybe this can be done automatically.
    *vsh, buf, ch = coords.shape
    padded_coords = pad3(coords.reshape(*vsh, -1), pad=1)
    *pvsh, _ = padded_coords.shape
    padded_coords = padded_coords.reshape(*pvsh, buf, ch)
    padded_mask = pad3(atom_mask, pad=1)

    ix2coord = domain / num_bin_voxels
    a, b, c, _ = within_voxel_lattice.shape
    wvl = within_voxel_lattice.reshape(a * b * c, -1)

    def compute_lattice_signal(voxel_index):
        """vox_ix[i,j,k]-->[num_lattice_x, num_lattice_y, num_lattice_z]"""
        nx = get_neighbour_voxels(padded_coords, voxel_index, leave_n_axes=2)
        nm = get_neighbour_voxels(padded_mask, voxel_index)
        # [within_lattice ** 3, buffer, coord]
        vl = wvl + (ix2coord * (voxel_index - 1))[None, :]
        # nx: [neigh_vox, buffer, coord]
        norms = jnp.linalg.norm(vl[None, :, None, :] - nx[:, None, :, :], axis=-1) ** 2
        # norms: [neigh_vox, within_lattice ** 3, buffer]
        channels = jax.nn.one_hot((nm - 1) * (nm > 0), number_of_atom_types) * (nm > 0)[..., None]
        # mask = nm[:, None, :] > 0
        expd = jnp.exp(-norms / (gauss_width ** 2))[..., None] * channels[:, None, :]
        return expd.sum(2).sum(0).reshape(a, b, c, number_of_atom_types)

    voxel_indices_1d = 1 + jnp.arange(num_bin_voxels)  # 1 for the padding
    voxel_indices = meshgrid(voxel_indices_1d, num_dimensions=3)

    f = jax.vmap(
        jax.vmap(
            jax.vmap(
                compute_lattice_signal, in_axes=0
            ), in_axes=0
        ), in_axes=0
    )
    out_ = f(voxel_indices)
    n = num_bin_voxels * num_points_1d_per_bin_voxel

    # out = out.reshape(*voxels, *out.shape[1:])
    out = jnp.moveaxis(
        out_,
        (0, 3, 1, 4, 2, 5, 6),
        (0, 1, 2, 3, 4, 5, 6),
    )
    out = out_.reshape(n, n, n, number_of_atom_types)
    # out = jnp.einsum("abcdefg->adbecfg", out_).reshape(n, n, n, number_of_atom_types)
    return out


def ndim_spatial_hash_to_lattice(
    coords: jnp.array,
    atom_mask: jnp.array,
    domain: Union[List[float], float],
    num_points_1d_per_bin_voxel: Union[List[int], int],
    gauss_width: float,
    number_of_atom_types: int,
):
    """_summary_

    Args:
        coord (jnp.array[dim_1_vox, ..., dim_n_vox, buffer, n]):
        atom_mask (jnp.array[dim_1_vox, ..., dim_n_vox, buffer]): _description_
        domain (jnp.array[dim_1_range, ..., dim_n_range]): _description_
        num_points_1d_per_bin_voxel (int): _description_
        gauss_width (float): _description_
        number_of_atom_types (int): _description_

    Returns:
        _type_: _description_
    """
    *voxel_dimensions, buffer_size, num_dim = coords.shape
    assert atom_mask.shape == (tuple(voxel_dimensions) + (buffer_size,))
    assert len(voxel_dimensions) == num_dim
    if type(domain) is float:
        domain = [domain] * num_dim
    domain = np.array(domain)
    if type(num_points_1d_per_bin_voxel) is int:
        num_points_1d_per_bin_voxel = [num_points_1d_per_bin_voxel] * num_dim
    num_points_1d_per_bin_voxel = np.array(num_points_1d_per_bin_voxel)

    num_bin_voxels = np.array(voxel_dimensions)
    num_points_1d_per_bin_voxel = np.array(num_points_1d_per_bin_voxel)
    ix2coord = domain / num_bin_voxels
    gap = ix2coord / num_points_1d_per_bin_voxel
    # this is within a voxel, imagine two neighbouring voxels: |. .|. .|
    within_voxel_lattices_1d = [
        np.linspace(g * 0.5, i2c - g * 0.5, n)
        for g, i2c, n in zip(gap, ix2coord, num_points_1d_per_bin_voxel)
    ]
    within_voxel_lattice = meshgrid(within_voxel_lattices_1d)

    padded_coords = pad_edges_ndim(coords, pad=1, leave_n_dims=2)
    padded_mask = pad_edges_ndim(atom_mask, pad=1, leave_n_dims=1)

    wvl = within_voxel_lattice.reshape(
        np.prod(within_voxel_lattice.shape[:num_dim]), -1
    )

    def compute_lattice_signal(voxel_index):
        """vox_ix[i,j,k]-->[num_lattice_x, num_lattice_y, num_lattice_z]"""
        nx = get_ndim_neighbours(padded_coords, voxel_index, leave_n_axes=2)
        nm = get_ndim_neighbours(padded_mask, voxel_index, leave_n_axes=1)
        # [within_lattice ** ndim, coord]
        vl = wvl + (ix2coord * voxel_index)[None, :]
        # nx: [neigh_vox, buffer, coord]
        norms = jnp.linalg.norm(vl[None, :, None, :] - nx[:, None, :, :], axis=-1) ** 2
        # norms: [neigh_vox, within_lattice ** ndim, buffer]
        channels = jax.nn.one_hot((nm - 1) * (nm > 0), number_of_atom_types) * (nm > 0)[..., None]
        # mask = nm[:, None, :] > 0
        expd = jnp.exp(-norms / (gauss_width ** 2))[..., None] * channels[:, None, :]
        summed_neighbours = expd.sum(0)
        summed_buffer = summed_neighbours.sum(1)
        return summed_buffer.reshape(
            *within_voxel_lattice.shape[:num_dim], number_of_atom_types
        )

    voxel_indices = meshgrid([jnp.arange(n) for n in num_bin_voxels])

    f = compute_lattice_signal
    for _ in range(num_dim):
        f = jax.vmap(f, in_axes=0)
    out = f(voxel_indices)
    n = num_bin_voxels * num_points_1d_per_bin_voxel

    ax = []
    for i in range(num_dim):
        ax += [i, num_dim + i]
    start = (*ax, 2 * num_dim)
    distination = tuple(range(2 * num_dim + 1))

    out = jnp.moveaxis(out, start, distination)
    out = out.reshape(*n, number_of_atom_types)
    return out


def variable_number_of_points_to_fixed_density(
    points: jnp.array,
    mask: jnp.array,
    domain: jnp.array,
    num_points_1d: int,
    gauss_width: float,
    radius: float = None,
    max_signal_limit: float = 10.0,
    cos_enc_num: int = 10,
):
    """Alternatively could do an initial message passing step attending to the neighbours
    """
    if radius is None:
        radius = 10. * gauss_width
    fixed_den_1d = jnp.linspace(*domain, num_points_1d)
    fixed_latice = jnp.stack(jnp.meshgrid(fixed_den_1d, fixed_den_1d, ))#fixed_den_1d))
    fixed_latice = fixed_latice.reshape(2, -1).T

    def signal(lattice_pt):
        norms = jnp.linalg.norm(lattice_pt[None] - points, axis=-1)
        radial_mask = norms < radius
        return (jnp.exp(-norms / gauss_width) * mask * radial_mask).sum(0)

    expd_pts = jax.vmap(signal)(fixed_latice)
    max_signal = jnp.max(expd_pts)
    expd_pts /= max_signal

    angle = 2 * jnp.pi * jnp.minimum(max_signal / max_signal_limit, 1.)
    encoded_max_signal = jnp.cos(jnp.arange(1, cos_enc_num + 1) * angle)
    return expd_pts, encoded_max_signal, max_signal


def reconstruct_points(signal: jnp.array, num_points: int):
    """_summary_

    Args:
        signal (jnp.array): _description_

    Returns:
        points (jnp.array): _description_
    """
    mask = jnp.arange(MAX_POINTS) < num_points
    sample_propto_signal = ...
    new_signal = generate(sample_propto_signal)
    # remove some points which are oversampled
    
    return points


def old_main():
    ####
    N = 100
    x = jnp.array(np.random.rand(N, 2))
    ex, encoded_max_signal, max_signal = variable_number_of_points_to_fixed_density(
        points=x, mask=jnp.ones(N), domain=(0, 1), num_points_1d=100, gauss_width=0.01,
    )
    print((min(ex), max(ex)))

    import matplotlib.pyplot as plt


    def plot(e, bm=None):
        if bm is None:
            bm = e >= e.sort()[-N]
        plt.matshow(e.reshape(100, 100))
        ixs = jnp.arange(100)
        ixs = jnp.stack(jnp.meshgrid(ixs, ixs)).reshape(2, -1).T
        plt.scatter(*ixs[bm].T, c="r", s=0.3)
        plt.show()

    plot(ex)

    bmask = ex >= ex.sort()[-N]
    ls = jnp.linspace(0, 1, N)
    allpts = jnp.stack(jnp.meshgrid(ls, ls)).reshape(2, -1).T
    pts = allpts[bmask]
    ex1, encoded_max_signal1, max_signal1 = variable_number_of_points_to_fixed_density(
        points=pts, mask=jnp.ones(N), domain=(0, 1), num_points_1d=N, gauss_width=0.01,
    )
    plot(ex1)
    diff = ex * max_signal - ex1 * max_signal1
    plot(diff)
    """
    Architecture:

    Encoder[first stage]
        RadialBasis
        EqualSample
        CubeAttention(within_edges=True, scope=10)
        Transition
        CubeAttention(within_edges=True, scope=3)
        Transition
        CubeAttention(within_edges=False, scope=3)
        Transition
    Encoder[second stage]
        Block:
            CubeAttention(within_edges=False, scope=5)
            Transition
            CubeAttention(within_edges=False, scope=3)
            Transition
            CubeAttention(within_edges=False, scope=3, stride=2)
            Transition
        :
        (weight share loop Block until the cube width is 1?)
        --> in doing so, communicate to a track somehow?


    Randomly decide on MD sample rate durinig training
    """


    # r=hk.transform(lambda x: RelativePositionalScope(3, 9)(x))
    # p=r.init(jax.random.PRNGKey(0), x)
    # c = r.apply(p, jax.random.PRNGKey(0), x)


def main_2():
    from atom_modules.spatial_datastructure_parallel import (
        hash_3d_data,
        load_data,
        paths,
    )

    T = 100
    num_streams = 16

    data, atom_mask, a2i, box, n = load_data(
        paths[0], permute=True, pad=True, num_streams=num_streams
    )

    spatial_dimension = 0
    box_length = box[spatial_dimension]

    hash_3d_data = partial(
        hash_3d_data,
        num_divisions=15,
        num_streams=num_streams,
        box_length=box_length,
        buffer_factor=5.0,
        merged_buffer_factor=1.2,
        n=n,
    )
    hash_3d_data = jax.jit(hash_3d_data)

    jspatial_hash_to_lattice = partial(
        spatial_hash_to_lattice,
        domain=box_length,
        num_points_1d_per_bin_voxel=2,
        gauss_width=0.2,
        number_of_atom_types=len(a2i) + 1,
    )
    jspatial_hash_to_lattice = jax.jit(jspatial_hash_to_lattice)

    lattices = []
    for t in tqdm(range(T)):
        start = time.time()
        _data, a_mask, num_lost_points = hash_3d_data(data[t], atom_mask)
        print(f"Time: {time.time() - start}s")

        start = time.time()
        o = jspatial_hash_to_lattice(_data, a_mask,)
        print(f"Time: {time.time() - start}s")
        _plot(o); break

        lattices.append(o)
    lattices = jnp.stack(lattices)
    # print(o.shape)
    # old_main()


def _plot(out_data):
    import matplotlib.pyplot as plt
    slice_ = out_data[0]
    seq = [
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd',
        'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn',
        'YlGn'
    ]
    n = out_data.shape[-1]
    for i in range(n):
        plt.imshow(slice_[..., i], alpha=1/n, cmap=seq[i])
    plt.savefig("temp")


def ground_truth(data, atom_mask, n_channels, box_length, n_lattice, width, spatial_dim):
    gap = box_length / n_lattice
    _1d = np.linspace(gap * 0.5, box_length - gap * 0.5, n_lattice)
    pos = meshgrid(_1d, num_dimensions=spatial_dim).reshape(-1, spatial_dim)
    diff = data[:, None, :] - pos[None, :, :]
    diffs = np.exp(-((diff ** 2)/(width ** 2)).sum(-1)) # [n_pts, n_lat]
    m = np.zeros((data.shape[0], n_channels)) # [n_pts, chan]
    for i, j in enumerate(atom_mask):
        m[i, j - 1] = j > 0
    sig = m[:, None, :] * diffs[:, :, None]
    # sig *= m[:, None, :] > 0
    sig = sig.sum(0) # [n_lat, chan]
    sig = sig.reshape(*tuple([n_lattice] * spatial_dim), n_channels)
    return sig


def main_3():
    from atom_modules.spatial_datastructure_parallel import (
        hash_3d_data,
        load_data,
        paths,
    )
    num_streams = 16
    gauss_width = 0.03
    num_divisions = 3
    num_points_1d_per_bin_voxel = 5
    n_lattice = num_divisions * num_points_1d_per_bin_voxel

    data, atom_mask, a2i, box, n = load_data(
        paths[0], permute=True, pad=True, num_streams=num_streams
    )
    # box seems to be in units of 0.1 angstrom, I found that about 0.5A is good voxel size
    voxel_width = 0.05
    num_vox = box[0] / voxel_width
    # box_dist_angstrom = box[0] * 10

    data = data[0]
    (take_this,) = np.where(atom_mask > 0)
    data = data[take_this]

    spatial_dimension = 0
    box_length = box[spatial_dimension] * 0.08
    (take_this,) = np.where(np.prod(data < box_length, axis=-1))
    data = data[take_this]
    atom_mask = atom_mask[take_this]

    n = (atom_mask > 0).sum()

    num_per_shard = jnp.ceil(n / num_streams).astype(int)
    padded_size = num_per_shard * num_streams
    pad = jnp.zeros((padded_size - n, 3), data.dtype)

    def plot_scatter(direc):
        import os

        import matplotlib.pyplot as plt
        os.makedirs(direc, exist_ok=True)
        for j, angle in tqdm(list(enumerate(np.linspace(0, 60, 100)))):
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            for i in set(a2i.values()):
                ax.scatter(*data[atom_mask==i].T, alpha=0.4, s=2.9)
            ax.view_init(40, angle)
            plt.savefig(f"{direc}/plot_{j}.png", dpi=500)
            plt.clf()
        # os.system(f"ffmpeg -r 10 -i {direc}/plot_%d.png -y {direc}.gif")
    # plot_scatter(direc="scatter")

    data = jnp.concatenate([data, pad], axis=0)
    atom_mask = jnp.concatenate(
        [atom_mask, jnp.zeros((padded_size - n,), data.dtype)], axis=0
    )

    hash_3d_data = partial(
        hash_3d_data,
        num_divisions=num_divisions,
        num_streams=int(num_streams),
        box_length=float(box_length),
        buffer_factor=5.0,
        merged_buffer_factor=1.2,
        n=int(n),
    )
    hash_3d_data = jax.jit(hash_3d_data)

    _data, a_mask, num_lost_points = hash_3d_data(data, atom_mask)


    print(f"num_lost_points: {num_lost_points}")

    jspatial_hash_to_lattice = partial(
        spatial_hash_to_lattice,
        domain=box_length,
        num_points_1d_per_bin_voxel=num_points_1d_per_bin_voxel,
        gauss_width=gauss_width,
        number_of_atom_types=len(a2i) + 1,
    )
    # jspatial_hash_to_lattice = jax.jit(jspatial_hash_to_lattice)

    start = time.time()
    o = jspatial_hash_to_lattice(_data, a_mask,)
    print(f"Time: {time.time() - start}s")
    s = ground_truth(data, atom_mask, len(a2i)+1, box_length, n_lattice=n_lattice, width=gauss_width)
    # _plot(o)


    def plot_scatter_out(out, direc):
        colours = [
            ["firebrick", "coral"],
            ["darkblue", "cornflowerblue"],
            ["forestgreen", "limegreen"],
            ["darkviolet", "mediumorchid"],
            ["gold", "lemonchiffon"],
            ["chocolate", "peru"],
        ]
        import os

        import matplotlib.pyplot as plt
        num = out.shape[0]
        gap = box_length / num
        _1d = np.linspace(gap * 0.5, box_length - gap * 0.5, num)
        pos = meshgrid(_1d, num_dimensions=3)
        out = out.reshape(-1, out.shape[-1]) / np.max(out)
        pos = pos.reshape(-1, 3)
        os.makedirs(direc, exist_ok=True)
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for i in range(out.shape[-1]):
            # ax.scatter(*pos.T, alpha=out[..., i], s=2, c=colours[i][1])
            # ax.scatter(*data[atom_mask==i+1].T, alpha=0.6, s=3.9, c=colours[i][0])
            oi = out[..., i]
            a = oi > np.percentile(oi, 95)
            ax.scatter(*pos[a].T, alpha=0.3, s=2, c=colours[i][1]) # alpha=oi[a]
            ax.scatter(*data[atom_mask==i+1].T, alpha=0.6, s=3.9, c=colours[i][0])
        for j, angle in tqdm(list(enumerate(np.linspace(0, 60, 100)))):
            ax.view_init(40, angle)
            plt.savefig(f"{direc}/plot_{j}.png", dpi=500)
        plt.clf()
        # os.system(f"ffmpeg -r 10 -i {direc}/plot_%d.png -y {direc}.gif")
    # plot_scatter_out(o, direc="lattice")

    out_data = o

    import matplotlib.pyplot as plt
    seq = [
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd',
        'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn',
        'YlGn'
    ]
    n = out_data.shape[-1]
    # cs = jnp.mean(o, axis=(0, 1, 2))
    # cs /= cs.sum()
    h, w = 3, 5
    fig, axes = plt.subplots(h, w)
    for j in range(h):
        for k in range(w):
            slice_ = out_data[k * h + j]
            for i in range(n):
                axes[j, k].imshow(slice_[..., i], alpha=0.5, cmap=seq[i])
                axes[j, k].axis("off")
    plt.tight_layout()
    plt.savefig("temp")


if __name__ == '__main__':
    # data = np.random.randn(4, 5, 6, 3)
    # a = pad3(data, pad=1)
    # b = pad_edges_ndim(data, pad=1, leave_n_dims=1)
    # print(f"err: {np.abs(a - b).sum()}")
    kw = {
        "coords": np.random.rand(4, 4, 4, 7, 3),
        "atom_mask": np.ones((4, 4, 4, 7)),
        "domain": 1.0,
        "num_points_1d_per_bin_voxel": 2,
        "gauss_width": 0.1,
        "number_of_atom_types": 1,
    }
    l = ndim_spatial_hash_to_lattice(**kw)
    ll = spatial_hash_to_lattice(**kw)
    print(f"err: {np.abs(l - ll).sum()}")
