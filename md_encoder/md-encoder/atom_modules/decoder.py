from functools import partial
from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
from absl import logging
from md_encoder.atom_modules.modules import (
    MLP,
    Attention,
    LayerNorm,
    Linear,
    Transition,
    get_initializer_scale,
    get_neighbour_voxels,
    glorot_uniform,
    ground_truth,
    meshgrid,
    pad3,
    spatial_hash_to_lattice,
)
from md_encoder.atom_modules.spatial_datastructure_parallel import spatial_hash
from md_encoder.utils import prng

SPATIAL_DIMS = 3


def initialise_points(key, num_points, buffer_size):
    points = jax.random.uniform(key=key, shape=(buffer_size, SPATIAL_DIMS))
    mask = jnp.arange(buffer_size) < num_points
    return points, mask


def point_query_points():
    return


def interpolate_and_pos_enc(
    latent_encoding: jnp.array,
    atom_positions: jnp.array,
    box_size: jnp.array,
    atom_latent_dim: int = 64,
    p_enc_dim: int = 4,
):
    """Create a latent representation of each atom.

    Args:
        latent_encoding (jnp.array[N_lat, N_lat, N_lat, c_l]): _description_
        atom_positions (jnp.array[N_atoms, 3]): _description_

    Returns:
        atom_representation (jnp.array[N_atoms, c_a]): _description_
    """
    lat_xyz = jnp.array(latent_encoding.shape[:-1])
    len_xyz = box_size / (lat_xyz - 1)
    # remainder = jnp.stack([
    #     jnp.mod(atom_positions[:, 0], len_xyz[0]),
    #     jnp.mod(atom_positions[:, 1], len_xyz[1]),
    #     jnp.mod(atom_positions[:, 2], len_xyz[2])
    # ]).T  # [N_atoms, 3]

    unitised_atoms = atom_positions / box_size[None, :]
    index_sized_atoms = (lat_xyz - 1)[None, :] * unitised_atoms
    index = jnp.floor(index_sized_atoms).astype(jnp.int32)  # [N_atoms, 3]

    neighbour_index_shifts = meshgrid(jnp.arange(2), num_dimensions=3)  # [2, 2, 2, 3]
    neighbour_index_shifts = neighbour_index_shifts.reshape(-1, 3)

    c_l = latent_encoding.shape[-1]
    num_atoms = atom_positions.shape[0]
    num_neighbours = 2 ** 3

    # indices: [num_atoms, num_neighbours, 3]
    indices = index[:, None, :] + neighbour_index_shifts[None, :, :]

    # diff: [num_atoms, num_neighbours, 3], range[-1, 1]
    diff = indices - atom_positions[:, None, :] / len_xyz[None, None, :]
    shifted_diff = (diff + 1) * .5
    angle_multipliers = jnp.arange(1, p_enc_dim + 1)
    pos_enc = jnp.cos(shifted_diff[..., None] * angle_multipliers[None, None, None, :])
    # pos_enc: [num_atoms, num_neighbours, 3, channel]
    pos_enc = pos_enc.reshape(num_atoms, num_neighbours, -1)

    x_ix, y_ix, z_ix = indices.reshape(num_neighbours * num_atoms, 3).T

    def gather_item(x, y, z):
        return latent_encoding[x, y, z]
    gather_items = jax.vmap(gather_item)

    per_atom_and_neigh_latents = gather_items(x_ix, y_ix, z_ix)
    atom_latents = per_atom_and_neigh_latents.reshape(num_atoms, num_neighbours, c_l)

    per_atom_latents = jnp.concatenate([atom_latents, pos_enc], axis=-1)
    per_atom_latents = per_atom_latents.reshape(num_atoms, -1)

    proj_weights = hk.get_parameter(
        "proj_weights",
        shape=(per_atom_latents.shape[-1], atom_latent_dim),
        init=glorot_uniform(),
    )

    return jnp.einsum("ij,jk->ik", per_atom_latents, proj_weights)


def compute_neighbours(atom_positions):
    """_summary_

    Args:
        atom_positions (jnp.array[N_atom, 3]): _description_

    Returns:
        _type_: _description_
    """
    num_atoms = atom_positions.shape[0]
    index = jnp.arange(num_atoms)
    # it could be worth concatenating the index along the channel dimension
    n_streams = 4
    num_divisions = 6
    buffer_factor = 3.5
    box_length = 1.0
    radius = 0.1
    num_dimensions = 3

    mask = jnp.ones(num_atoms).reshape(n_streams, -1)
    shard = jnp.concatenate(
        [atom_positions, index[:, None]], axis=-1
    ).reshape(n_streams, -1, 4)
    buffer_, buffer_mask, counts = spatial_hash(
        shard, mask, num_dimensions, num_divisions, n_streams, buffer_factor, box_length
    )
    num_voxels = buffer_.shape[0]
    # for each voxel in the buffer, loop over each atom in the buffer, if the mask is 1
    # compute the distance to every atom in the adjacent voxels, record the index of
    # the neighbours in the index for the current atom.
    *vsh, buf, ch = buffer_.shape
    padded_buffer = pad3(buffer_.reshape(*vsh, -1), pad=1)
    *pvsh, _ = padded_buffer.shape
    padded_buffer = padded_buffer.reshape(*pvsh, buf, ch)
    padded_mask = pad3(buffer_mask, pad=1)

    def write_neighbours_for_all_atoms_in_voxel(voxel_index: jnp.array):
        """_summary_

        Args:
            voxel_index (jnp.array[3]): _description_

        Returns:
            _type_: _description_
        """
        # ax: [buffer_size, 4]
        ax = padded_buffer[voxel_index[0], voxel_index[1], voxel_index[2]]
        atom_coords, atom_indices = ax[:, :-1], ax[:, -1]
        # mx: [buffer_size]
        atom_mask = padded_mask[voxel_index[0], voxel_index[1], voxel_index[2]]
        # nx: [27, buffer_size, 4]
        nx = get_neighbour_voxels(padded_buffer, voxel_index, leave_n_axes=2)
        neighbours = nx.reshape(-1, 4)
        neighbour_coords, neighbour_indices = neighbours[:, :-1], neighbours[:, -1]
        # nx: [27, buffer_size]
        nm = get_neighbour_voxels(padded_mask, voxel_index)
        mask_neighbours = nm.reshape(-1)

        # [buffer_size, 27 * buffer_size]
        diffs = atom_coords[:, None, :] - neighbour_coords[None, :, :]
        dists = jnp.linalg.norm(diffs, axis=-1)

        within_rad = dists < radius

        neighbour_mask = atom_mask[:, None] * within_rad * mask_neighbours[None, :]

        return {
            "atom_indices": atom_indices,
            "atom_mask": atom_mask,
            "neighbour_indices": neighbour_indices,
            "neighbour_mask": neighbour_mask,
            "diffs": diffs
        }

    voxel_indices_1d = 1 + jnp.arange(num_voxels)  # 1 for the padding
    voxel_indices = meshgrid(voxel_indices_1d, num_dimensions=3)

    f = jax.vmap(
        jax.vmap(
            jax.vmap(
                write_neighbours_for_all_atoms_in_voxel, in_axes=0
            ), in_axes=0
        ), in_axes=0
    )
    ret = f(voxel_indices)
    ret["neighbour_indices"] = ret["neighbour_indices"][:, :, :, None, :].repeat(
        buf, axis=3
    )
    print(f"buffer_size: {buf}")
    jax.tree_util.tree_map(lambda x: print(x.shape), ret)
    ret = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[4:]), ret)
    atom_mask = ret.pop("atom_mask")

    ret = jax.tree_util.tree_map(
        lambda x: x[jnp.where(atom_mask, size=num_atoms)], ret
    )
    # reorganise neighbour_indices such that the first axis corresponds to the
    # atom_indices
    atom_indices = ret.pop("atom_indices").astype(jnp.int32)
    permutation = jnp.empty(num_atoms, dtype=jnp.int32)
    permutation = permutation.at[atom_indices].set(jnp.arange(num_atoms))

    ret = jax.tree_util.tree_map(lambda x: x[permutation], ret)

    # return atom_indices, atom_mask, neighbour_indices, neighbour_mask, diffs
    return ret["neighbour_indices"], ret["neighbour_mask"], ret["diffs"]


class LocalPointAttention(hk.Module):
    """Iteration of the Iterative Point Attention Decoder.

    This .
      * Compare to the sparse denisty latent representation and a dense activation track
    """

    def __init__(self, config, global_config, encode_fn: Callable):
        super().__init__(name="local_point_attention")
        self.config = config
        self.global_config = global_config
        self.encode = encode_fn

    def __call__(
        self,
        enc_params,
        atom_representation: jnp.array,
        latent_encoding: jnp.array,
        atom_positions: jnp.array,
        box_size: jnp.array,
    ):
        """Update to the atom representation.

        Args:
            atom_representation (jnp.array[N_atoms, c_a]): _description_
            latent_encoding (jnp.array[N_lat, N_lat, N_lat, c_l]): _description_
            atom_positions (jnp.array[N_atoms, 3]): _description_
        """
        neighbour_indices, neighbour_mask, neighbour_displacements = compute_neighbours(
            atom_positions
        )
        for k in ["neighbour_indices", "neighbour_mask", "neighbour_displacements"]:
            print(f"{k}: {eval(f'{k}.shape')}")

        # Treat this the same way as the pair bias in Invariant Point Attention.
        neighbour_dists = jnp.linalg.norm(neighbour_displacements, axis=-1)
        # no self-interaction
        neighbour_dists += (neighbour_dists < 1e-6) * 1e9

        neighbour_sq_inv_dists = neighbour_dists ** -2
        neighbour_dists /= jnp.max(neighbour_dists)

        # 
        alpha = interpolate_and_pos_enc(latent_encoding, atom_positions, box_size)

        # 
        atom_mask = jnp.ones(atom_positions.shape[0])
        key = jax.random.PRNGKey(seed=0)
        corresponding_latent = self.encode(enc_params, key, atom_positions, atom_mask)
        corresponding_latent = jax.lax.stop_gradient(corresponding_latent)
        corresponding_alpha = interpolate_and_pos_enc(
            corresponding_latent, atom_positions, box_size
        )

        d_alpha = corresponding_alpha - alpha

        # just concatenating for now, for ease of implementation.
        a = jnp.concatenate(
            [atom_representation, alpha, corresponding_alpha, d_alpha], axis=-1
        )
        print(f"a.shape: {a.shape}")
        print(f"neighbour_indices: {neighbour_indices.shape}")
        memory = jax.vmap(lambda _a, ni: _a[ni])(a, neighbour_indices.astype(jnp.int32))
        print(memory.shape)

        # bias = Linear()(neighbour_sq_inv_dists)
        bias = 1e9 * (neighbour_mask[:, None, None, :] - 1.0)
        print(
            "query: a, memory: [memory, neighbour_sq_inv_dists]\n"
            f"a: {a.shape}, memory: {memory.shape}, bias: {bias.shape}, "
            f"neighbour_sq_inv_dists: {neighbour_sq_inv_dists.shape}"
        )
        all_ = jnp.array(a)
        memory = jax.vmap(lambda nix: all_[nix])(neighbour_indices.astype(jnp.int32))
        print(
            f"Pre Attention UPDATE: [query: {str(all_)[:20]}|"
            f"|\nall_: {str(all_)[:20]}|\nmemory: {str(memory)[:20]}"
            f"|\nneighbour_sq_inv_dists: {str(neighbour_sq_inv_dists)[:20]}]"
        )
        memory = memory * neighbour_sq_inv_dists[:, :, None]
        query = a[:, None, :]
        print(f"Memory: {memory.shape}")
        print(
            f"Attention UPDATE: [query: {str(query)[:20]}|\n{str(memory)[:20]}"
            f"|\n{str(bias)[:20]}]"
        )
        # import pdb; pdb.set_trace()
        update = Attention(
            self.config.attention, self.global_config, self.config.output_dim
        )(query, memory, bias)
        print(f"Attn out: {update.shape}")
        update = update[:, 0, :]
        return update


class AtomDecoderIteration(hk.Module):
    """Iteration of the Iterative Point Attention Decoder.

    This .
      * Compare to the sparse denisty latent representation and a dense activation track
    """

    def __init__(self, config, global_config, encode_fn):
        super().__init__(name="atom_decoder_iteration")
        self.config = config
        self.global_config = global_config
        self.encode = encode_fn

    def __call__(
        self,
        enc_params,
        latent_encoding,
        atom_representation,
        atom_positions,
        box_size,
        safe_key,
        is_training,
    ):
        """Roughly follows the structure of the structure module, shared weights...

        Args:
          inputs: Tensor with at least num_input_dims dimensions.

        Returns:
          output of shape [...] + num_output.
        """

        print(
            f"START ITER: [atom_positions: {atom_positions.shape}, "
            f"atom_representation: {atom_representation.shape}, "
            f"\n{str(atom_representation)[:100]}]"
        )

        def safe_dropout_fn(tensor, safe_key):
            return prng.safe_dropout(
                tensor=tensor,
                safe_key=safe_key,
                rate=self.config.dropout,
                is_deterministic=self.global_config.deterministic,
                is_training=is_training,
            )

        lpa = LocalPointAttention(
            self.config.local_point_attention, self.global_config, self.encode
        )

        update = lpa(
            enc_params, atom_representation, latent_encoding, atom_positions, box_size
        )
        print(f"LPA UPDATE: [update: {update.shape}]\n{str(update)[:100]}]")
        atom_representation += update

        safe_key, *sub_keys = safe_key.split(3)
        sub_keys = iter(sub_keys)
        atom_representation = safe_dropout_fn(atom_representation, next(sub_keys))
        atom_representation = hk.LayerNorm(
            axis=[-1],
            create_scale=True,
            create_offset=True,
            name="attention_layer_norm",
        )(atom_representation)

        # Transition
        final_init = "zeros" if self.config.zero_init else "linear"
        input_act = atom_representation
        for i in range(self.config.num_layer_in_transition):
            init = "relu" if i < self.config.num_layer_in_transition - 1 else final_init
            atom_representation = Linear(
                self.config.num_channel, initializer=init, name="transition"
            )(atom_representation)
            if i < self.config.num_layer_in_transition - 1:
                atom_representation = jax.nn.relu(atom_representation)
        atom_representation += input_act

        safe_key, *sub_keys = safe_key.split(3)
        sub_keys = iter(sub_keys)
        atom_representation = safe_dropout_fn(atom_representation, next(sub_keys))
        atom_representation = hk.LayerNorm(
            axis=[-1],
            create_scale=True,
            create_offset=True,
            name="attention_layer_norm",
        )(atom_representation)

        # Affine update
        delta_atom_positions = Linear(
            3, initializer=final_init, name="affine_update"
        )(atom_representation)

        atom_positions += delta_atom_positions

        atom_positions = jnp.maximum(0, atom_positions)
        atom_positions = jnp.minimum(box_size[None, :], atom_positions)

        print(
            f"END ITER: [atom_positions: {atom_positions.shape}, "
            f"atom_representation: {atom_representation.shape}, "
            f"\n{str(atom_representation)[:100]}]"
        )
        return atom_representation, atom_positions


class AtomDecoder(hk.Module):
    """Iterative Point Attention Decoder.

    This module:
      * Sequentially improve the predicted points
      * Compare to the sparse denisty latent representation and a dense activation track
    """

    def __init__(self, config, encode: Callable):
        super().__init__(name="atom_decoder")
        self.config = config
        self.global_config = config.global_config
        self.encode = encode

    def __call__(
        self,
        enc_params,
        latent_encoding: jnp.array,
        box_size: jnp.array,
        num_atoms: int,
        safe_key=None,
        is_training=False,
    ):
        if safe_key is None:
            safe_key = prng.SafeKey(hk.next_rng_key())
        elif isinstance(safe_key, jnp.ndarray):
            safe_key = prng.SafeKey(safe_key)

        atom_representation = jnp.zeros((num_atoms, self.config.atom_rep_channel))
        safe_key, sub_key = safe_key.split()
        atom_positions = jax.random.uniform(sub_key.get(), shape=(num_atoms, 3))

        act = (latent_encoding, atom_representation, atom_positions, box_size)

        iteration = AtomDecoderIteration(self.config, self.global_config, self.encode)

        def wrap_iteration(i, x):
            del i
            act, safe_key = x
            safe_key1, safe_key2 = (
                safe_key.split()
                if self.config.stochastic_folding
                else safe_key.duplicate()
            )
            atom_representation, atom_positions = iteration(
                enc_params, *act, safe_key=safe_key2, is_training=is_training,
            )
            act = (latent_encoding, atom_representation, atom_positions, box_size)
            return act, safe_key1

        act, safe_key = hk.fori_loop(
            0, self.config.num_iter, wrap_iteration, (act, safe_key)
        )
        latent_encoding, atom_representation, atom_positions, box_size = act
        return atom_positions


class DecoderIteration(hk.Module):
    """Iteration of the Iterative Point Attention Decoder.

    This .
      * Compare to the sparse denisty latent representation and a dense activation track
    """

    def __init__(self, config):
        """Constructs Decoder Module.

        Args:
          config: 
        """
        primed_r3_spatial_hash = partial(
            spatial_hash,
            num_dimensions=3,
            num_divisions=config.lattice_hash.num_divisions,
            n_streams=config.lattice_hash.n_streams,
            buffer_factor=config.lattice_hash.buffer_factor,
            box_length=config.lattice_hash.box_length,
        )
        primed_spatial_hash_to_lattice = partial(
            spatial_hash_to_lattice,
            domain=config.lattice_hash.box_length,
            num_points_1d_per_bin_voxel=config.lattice_hash.num_points_1d_per_bin_voxel,
            gauss_width=config.lattice_hash.gauss_width,
            number_of_atom_types=config.lattice_hash.number_of_atom_types,
        )
        # # primed_r3_spatial_hash returns (buffer_, buffer_mask, counts)
        # # primed_spatial_hash_to_lattice args are (buffer_, buffer_mask,)
        # self.points2lattice = lambda data, mask: primed_spatial_hash_to_lattice(
        #     *primed_r3_spatial_hash(data, mask)[:2]
        # )
        def points2lattice(data, mask):
            buffer_, buffer_mask, _counts = primed_r3_spatial_hash(data, mask)
            dense_lattice = primed_spatial_hash_to_lattice()
            return dense_lattice, buffer_, buffer_mask

        self.points2lattice = points2lattice

    def __call__(
        self, key, low_density_latent_representation, points, mask, latent_points
    ):
        """Connects Module.

        Args:
          inputs: Tensor with at least num_input_dims dimensions.

        Returns:
          output of shape [...] + num_output.
        """
        dense_lattice, buffer_, buffer_mask = self.points2lattice(points, mask)

        points = ...

        return points


class Decoder(hk.Module):
    """Iterative Point Attention Decoder.

    This module:
      * Sequentially improve the predicted points
      * Compare to the sparse denisty latent representation and a dense activation track
    """

    def __init__(self, config):
        """Constructs Decoder Module.

        Args:
          config: 
        """
        self.config = config

    def __call__(self, key, low_density_latent_representation, num_points, config):
        """Connects Module.

        Args:
          inputs: Tensor with at least num_input_dims dimensions.

        Returns:
          output of shape [...] + num_output.
        """
        points, mask = partial(initialise_points, self.config.buffer_size)(
            key, num_points
        )
        iteration = DecoderIteration(self.config)

        points, latent_points = iteration(
            low_density_latent_representation, points, mask, latent_points
        )

        dense_lattice = iteration.points2lattice(points, mask)
        return points, dense_lattice


if __name__ == "__main__":
    pass
