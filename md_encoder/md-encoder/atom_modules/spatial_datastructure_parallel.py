import math
import os
import time
from functools import partial
from typing import List, Tuple, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from md_encoder.utils.read_md_utils import read_frames
from tqdm import tqdm

# from utils import make_vid


def stream(
    shard: jnp.array,
    shard_mask: jnp.array,
    num_divisions: int,
    box_length: float,
    spatial_dimension: int,
    buffer_factor: float = 2.0,
):# -> Tuple[jnp.array, jnp.array]:
    """note to self: could do an add op in the buffer and increment the counter by mask

    Args:
        shard (jnp.array[num_per_shard, 3]): _description_
        shard_mask (jnp.array[num_per_shard]): _description_
        num_divisions (int): _description_
        box_length (float): _description_
        spatial_dimension (int): _description_
        buffer_factor (float): _description_

    Returns:
        buffer (jnp.array[num_divisions, division_buffer_size, 3]): _description_
        buffer_mask (jnp.array[num_divisions, division_buffer_size]): _description_
        counts (jnp.array[num_divisions]): _description_
    """
    data_dim = shard.shape[-1]
    num_in_shard = shard.shape[0]
    division_buffer_size = int(buffer_factor * num_in_shard / num_divisions)
    buffer_ = jnp.zeros(
        (num_divisions, division_buffer_size, data_dim), dtype=shard.dtype
    )
    buffer_mask = jnp.zeros((num_divisions, division_buffer_size), dtype=jnp.int32)
    counts = jnp.zeros(num_divisions, dtype=jnp.int32)

    def loop_body(i, args):
        shard, shard_mask, buffer_, buffer_mask, counts = args
        x = shard[i]
        m = shard_mask[i]
        j = jnp.floor(
            x[spatial_dimension] * num_divisions / box_length
        ).astype(jnp.int32)
        c = counts[j]
        buffer_ = buffer_.at[j, c].set(x)
        buffer_mask = buffer_mask.at[j, c].set(m)
        counts = counts.at[j].set(c + (m > 0))
        return shard, shard_mask, buffer_, buffer_mask, counts

    shard, shard_mask, buffer_, buffer_mask, counts = jax.lax.fori_loop(
        lower=0,
        upper=num_in_shard,
        body_fun=loop_body,
        init_val=(shard, shard_mask, buffer_, buffer_mask, counts),
    )
    return buffer_, buffer_mask, counts


def merge_stream_dimension(
    buffer_: jnp.array,
    buffer_mask: jnp.array,
    counts: jnp.array,
    new_buffer_size: int = -1,
):
    """_summary_

    Args:
        buffer (jnp.array[num_streams, num_divisions, division_buffer_size, 3]):
        buffer_mask (jnp.array[num_streams, num_divisions, division_buffer_size]):
        counts (jnp.array[num_streams, num_divisions]):
        new_buffer_size (int):
    """
    num_streams, num_divisions, division_buffer_size = buffer_mask.shape
    def push_stream_into_buffer_dim(arr):
        right_sh = arr.shape[3:]
        arr = jnp.swapaxes(arr, 1, 0)
        return arr.reshape(num_divisions, num_streams * division_buffer_size, *right_sh)

    buffer_mask = push_stream_into_buffer_dim(buffer_mask)
    buffer_ = push_stream_into_buffer_dim(buffer_)

    if new_buffer_size != -1:
        def naive_gather(div_buffer: jnp.array, div_mask: jnp.array):
            """_summary_

            Args:
                div_buffer (jnp.array[num_streams * division_buffer_size, 3]): 
                div_mask (jnp.array[num_streams * division_buffer_size]): 
                div_counts (jnp.array[num_streams * division_buffer_size]): 

            Returns:
                _type_: _description_
            """
            (ixs,) = jnp.where(div_mask > 0, size=new_buffer_size, fill_value=-1)
            mask = ixs != -1
            return div_buffer[ixs], div_mask[ixs]

        buffer_, buffer_mask = jax.vmap(naive_gather, in_axes=(0, 0))(
            buffer_, buffer_mask
        )
        # def gather(div_buffer: jnp.array, div_mask: jnp.array, div_counts: jnp.array):
        #     """_summary_

        #     Args:
        #         div_buffer (jnp.array[num_streams, division_buffer_size, 3]): 
        #         div_mask (jnp.array[num_streams, division_buffer_size]): 
        #         div_counts (jnp.array[num_streams, division_buffer_size]): 

        #     Returns:
        #         _type_: _description_
        #     """
        #     new_buffer = jnp.zeros((new_buffer_size, 3), dtype=buffer_.dtype)
        #     new_mask = jnp.zeros((new_buffer_size,), dtype=buffer_mask.dtype)
        #     # stream_ixs = jnp.arange(num_streams)
        #     upper_ixs = jnp.cumsum(div_counts, axis=0)
        #     lower_ixs = jnp.concatenate([jnp.zeros((1,), jnp.int32), upper_ixs[:-1]])

        #     def loop_body(i, args):
        #         new_buffer, new_mask = args
        #         l = lower_ixs[i]
        #         u = upper_ixs[i]
        #         c = div_counts[i]
        #         data = jax.lax.dynamic_slice(div_buffer[i], (0, 0), (c, 3))
        #         # data = div_buffer[i, :c]
        #         # mask = div_mask[i, :c]
        #         # new_buffer = new_buffer.at[l:u].set(div_buffer[i, :c])
        #         # new_mask = new_mask.at[l:u].set(mask)
        #         new_buffer = jax.lax.dynamic_update_slice(new_buffer, data, (l,))
        #         return new_buffer, new_mask

        #     new_buffer, new_mask, *_ = jax.lax.fori_loop(
        #         lower=0,
        #         upper=num_streams,
        #         body_fun=loop_body,
        #         init_val=(new_buffer, new_mask),
        #     )
        #     # new_buffer, new_mask = jax.vmap(
        #     #     slice_gather, in_axes=(None, None, 0, 0, 0, 0, 0)
        #     # )(new_buffer, new_mask, buff, mask, count, lower_ixs, upper_ixs)
        #     return new_buffer, new_mask

        # buffer_ = jnp.swapaxes(buffer_, 1, 0)
        # buffer_mask = jnp.swapaxes(buffer_mask, 1, 0)
        # buffer_, buffer_mask = jax.vmap(gather, in_axes=(1, 1, 1))(
        #     buffer_, buffer_mask, counts
        # )
        # gather = partial(
        #     jax.lax.gather,
        #     # operand,
        #     # start_indices,
        #     dimension_numbers=jax.lax.GatherDimensionNumbers(
        #         offset_dims=,
        #         collapsed_slice_dims=,
        #         start_index_map=,
        #     ),
        #     # slice_sizes,
        #     # *,
        #     unique_indices=True,
        #     indices_are_sorted=True,
        #     mode=None,
        #     fill_value=None,
        # )

        # def masked_gather(l, u, buffer_row, mask):
        #     """Collect the points in the row of the buffer and place contiguously in a
        #     smaller buffer.

        #     Args:
        #         buffer_row (_type_): 1d array of position data
        #         mask (_type_): repeating structure every `division_buffer_size` where
        #             there will potentially be points contiguously from the left
        #             corresponding to the count size of that row. This could really have
        #             two levels of gathering, the first a (1) dynamic slice because the
        #             xla code may be able to do that quicker than a gather based on an
        #             index array; then doing the second with an (2) index array along a
        #             smaller axis (note this is just to collect the padded points in the
        #             original buffer, or any points that are masked in the original
        #             buffer). Currently I just implement (1).

        #     Returns:
        #         _type_: _description_
        #     """
        #     return jax.lax.gather(
        #         operand,
        #         start_indices,
        #         dimension_numbers,
        #         slice_sizes,
        #         *,
        #         unique_indices=False,
        #         indices_are_sorted=False,
        #         mode=None,
        #         fill_value=None,
        #     )
        #     bool gather into new buffer dimension
        #     return jax.lax.gather(buffer_, )
        #     new_buffer = new_buffer.at[l:u].set(buffer_row[:u-l])
        #     new_mask = new_mask.at[l:u].set(mask[:u-l])

        # buffer_, buffer_mask = jax.vmap(
        #     masked_gather, in_axes=(0, 0, 0, 0)
        # )(lower_ixs, upper_ixs, buffer_, buffer_mask)
    # else:
    #     def push_stream_into_buffer_dim(arr):
    #         right_sh = arr.shape[3:]
    #         arr = jnp.swapaxes(arr, 1, 0)
    #         return arr.reshape(num_divisions, num_streams * division_buffer_size, *right_sh)

    #     buffer_mask = push_stream_into_buffer_dim(buffer_mask)
    #     buffer_ = push_stream_into_buffer_dim(buffer_)

    return buffer_, buffer_mask, counts.sum(axis=0)


def buffer_data_contigously(
    data: jnp.array,
    atom_mask: jnp.array,
    spatial_dimension: int,
    new_buffer_size: int,
    num_divisions: int,
    num_streams: int,
    buffer_factor: float,
    box_length: float,
):
    """_summary_

    Args:
        data (jnp.array): _description_
        atom_mask (jnp.array): _description_
        spatial_dimension (int): _description_
        new_buffer_size (int): _description_
        num_divisions (int): _description_
        num_streams (int): _description_
        buffer_factor (float): _description_
        box_length (float): _description_

    Returns:
        _type_: _description_
    """
    straw = partial(
        stream,
        num_divisions=num_divisions,
        box_length=box_length,
        spatial_dimension=spatial_dimension,
        buffer_factor=buffer_factor
    )
    data = data.reshape(num_streams, -1, 3)
    atom_mask = atom_mask.reshape(num_streams, -1)
    buffer_, buffer_mask, counts = jax.vmap(straw, in_axes=(0, 0))(data, atom_mask)
    buffer_, buffer_mask, counts = merge_stream_dimension(
        buffer_, buffer_mask, counts, new_buffer_size=new_buffer_size
    )
    return buffer_, buffer_mask


def spatial_hash(
    data: jnp.array,
    mask: jnp.array,
    num_dimensions: int,
    num_divisions: Union[List[int], int],
    n_streams: Union[List[int], int],
    buffer_factors: Union[List[float], float],
    box_size: Union[List[float], float],
):
    """You can append along the last dim of data, to hash by the first three indices of
    the channel dimension.

    Args:
        data (jnp.array): _description_
        mask (jnp.array): _description_
        num_dimensions (int): _description_
        num_divisions (Union[List[int], int]): _description_
        n_streams (Union[List[int], int]): each voxels buffer must be written to sequen-
          tially, with a counter, n_streams allows for the same voxel to have points
          written to in parallel. Ensure that [n1, ..., ni, ...] n1 / ni is an int.
        buffer_factors (Union[List[float], float]): _description_
        box_size (Union[List[float], float]): _description_

    Returns:
        _type_: _description_
    """
    channel_dims = data.shape[-1]

    if type(box_size) is not list:
        box_size = [box_size] * num_dimensions
    if type(buffer_factors) is not list:
        buffer_factors = [buffer_factors] * num_dimensions
    if type(n_streams) is not list:
        n_streams = [n_streams] * num_dimensions
    if type(num_divisions) is not list:
        num_divisions = [num_divisions] * num_dimensions
    n_streams += [1]

    data = data.reshape(n_streams[0], -1, channel_dims)
    mask = mask.reshape(n_streams[0], -1)
    # loops are in python, but only depend on static args, so shouldn't affect jax jit
    for i in range(num_dimensions):
        f = partial(
            stream,
            num_divisions=num_divisions[i],
            box_length=box_size[i],
            buffer_factor=buffer_factors[i],
            spatial_dimension=i,
        )
        for _ in range(i + 1):  # vmap leading dims [stream, *divisions]
            f = jax.vmap(f)
        data, mask, counts = f(data, mask)

        # put stream axis next to the buffer axis
        stream_axis = i
        new_hash_dim = i + 1
        data = jnp.swapaxes(data, stream_axis, new_hash_dim)
        mask = jnp.swapaxes(mask, stream_axis, new_hash_dim)

        # re-adjust buffer and streams
        current_n_streams = n_streams[i]
        next_n_streams = n_streams[i + 1]
        curr_buff_size = data.shape[-2]
        mult = current_n_streams // next_n_streams
        assert current_n_streams % next_n_streams == 0
        new_shape = tuple(
            num_divisions[:i + 1] + [next_n_streams, mult * curr_buff_size]
        )
        data = data.reshape(*new_shape, channel_dims)
        mask = mask.reshape(*new_shape)

    # merge stream and buffer dimensions
    data = data.reshape(*data.shape[:num_dimensions], -1, channel_dims)
    mask = mask.reshape(*data.shape[:num_dimensions], -1)
    return data, mask, counts


def hash_3d_data(
    data: jnp.array,
    atom_mask: jnp.array,
    num_divisions: int,
    num_streams: int,
    box_length: float,
    buffer_factor: float,
    merged_buffer_factor: float,
    n: int,
):
    """_summary_

    Args:
        data (jnp.array): _description_
        mask (jnp.array): _description_
        num_divisions (int): _description_
        num_streams (int): _description_
        box_length (float): _description_
        buffer_factor (float): _description_
        merged_buffer_factor (float): _description_
        n (int): _description_

    Returns:
        _type_: _description_
    """
    # data = data.reshape(num_streams, -1, 3)
    # atom_mask = atom_mask.reshape(num_streams, -1)

    new_buffer_size = lambda i: math.ceil(
        (merged_buffer_factor ** i) * n / (num_streams * num_divisions ** i)
    ) * num_streams

    bdc = partial(
        buffer_data_contigously,
        num_divisions=num_divisions,
        num_streams=num_streams,
        buffer_factor=buffer_factor,
        box_length=box_length,
    )
    data, atom_mask = bdc(
        data=data,
        atom_mask=atom_mask,
        spatial_dimension=0,
        new_buffer_size=new_buffer_size(1)
    )

    vbdc = jax.vmap(bdc, in_axes=(0, 0, None, None))
    data, atom_mask = vbdc(data, atom_mask, 1, new_buffer_size(2))

    vvbdc = jax.vmap(vbdc, in_axes=(0, 0, None, None))
    data, atom_mask = vvbdc(data, atom_mask, 2, new_buffer_size(3))

    memory_redundancy_factor = np.prod(atom_mask.shape) / n
    num_lost_points = n - (atom_mask > 0).sum()
    print(
        f"buffer_factor: {buffer_factor}\n"
        f"merged_buffer_factor: {merged_buffer_factor}\n"
        f"memory_redundancy_factor: {memory_redundancy_factor}\n"
        f"num_lost_points: {num_lost_points}"
    )
    return data, atom_mask, num_lost_points


def load_data(path, permute=True, pad=True, num_streams=None):
    X, resname, atom_name, box = read_frames(path)

    #####
    # improve this later
    _atom_name = [a[0] for a in atom_name]
    atom_names = set(_atom_name)
    a2i = {a: i for i, a in enumerate(atom_names, start=1)}
    atom_type = jnp.array([a2i[a] for a in _atom_name]).astype(jnp.int32)
    #####

    #####
    # The data is a bit messy
    X -= jnp.min(X.reshape(-1, 3), axis=0)[None, None, :]
    box = jnp.maximum(jnp.max(X.reshape(-1, 3), axis=0), box)
    #####

    ts, n, _ = X.shape

    if permute:
        print("Permuting...")
        ti = time.time()
        key = jax.random.PRNGKey(seed=0)
        X = jax.random.permutation(key, X, axis=-2, independent=True)
        print(f"Done in {time.time() - ti}s")

    if pad:
        assert num_streams is not None
        num_per_shard = jnp.ceil(n / num_streams).astype(int)
        padded_size = num_per_shard * num_streams
        pad = jnp.zeros((ts, padded_size - n, 3), X.dtype)
        atom_mask = jnp.concatenate(
            [atom_type, jnp.zeros(padded_size - n, jnp.int32)], axis=-1
        )
        X = jnp.concatenate([X, pad], axis=1)
    else:
        atom_mask = atom_type

    return X, atom_mask, a2i, box, n


paths = [
    # "/home/louis/Desktop/md_traj/md_0_1.trr",
    "/data/md_0_1.trr",
    "/Users/louisrobinson/Desktop/md/mac_mount/pmhc/md_0_1.trr",
    "/home/louis/Dropbox/Cool/md/md_0_1.trr"
]


def main():
    run_all_frames = False

    if run_all_frames:
        outpath = "frames/buffer"
        os.makedirs(outpath, exist_ok=True)

        num_streams = 16
        num_divisions = 8
        spatial_dimension = 0
        buffer_factor = 4.0
        merged_buffer_factor = 1.5

        X, atom_mask, a2i, box, n = load_data(
            paths[0], permute=False, pad=True, num_streams=num_streams
        )
        box_length = box[spatial_dimension]

        straw = partial(
            stream,
            num_divisions=num_divisions,
            box_length=box_length,
            spatial_dimension=spatial_dimension,
            buffer_factor=buffer_factor
        )
        vstraw = jax.jit(jax.vmap(straw, in_axes=(0, 0)))
        for t in tqdm(range(X.shape[0])):
            xp = X[t]
            xp = xp.reshape(num_streams, -1, 3)

            buffer_, buffer_mask, counts = vstraw(xp, atom_mask)

            print(f"num_per_shard: {num_per_shard}")
            print(f"buffer.shape: {buffer_.shape}")
            # print(f"buffer_mask.sum(-1): {buffer_mask.sum(-1)}")
            plt.matshow(buffer_mask.sum(-1).T, vmin=0, vmax=buffer_.shape[2])
            plt.colorbar()
            plt.xlabel("num_streams")
            plt.ylabel("num_divisions")
            plt.savefig(f"{outpath}/t_{t}")
            plt.close()
            print(f"n: {n}")
            print(f"buffer_mask.sum(): {buffer_mask.sum()}")

        make_vid(outpath, video_name=f"{outpath}/video.mp4v")
    else:
        # at t=0 there is a data and memory dependency
        # I can use this config to get no lost points:
        # mem-redun: 5.86
        #     num_streams = 16
        #     num_divisions = 15
        #     spatial_dimension = 0
        #     buffer_factor = 7.0
        #     merged_buffer_factor = 1.8
        # With shuffling (50 seconds!) we can use to get no lost points:
        # mem-redun: 1.76
        #     num_streams = 16
        #     num_divisions = 15
        #     spatial_dimension = 0
        #     buffer_factor = 5.0
        #     merged_buffer_factor = 1.2
        t = 0

        num_streams = 16
        num_divisions = 15
        spatial_dimension = 0
        buffer_factor = 5.0
        merged_buffer_factor = 1.2

        data, atom_mask, a2i, box, n = load_data(
            paths[0], permute=True, pad=True, num_streams=num_streams
        )
        box_length = box[spatial_dimension]

        hash_3d_data = partial(
            hash_3d_data,
            num_divisions=num_divisions,
            num_streams=num_streams,
            box_length=box_length,
            buffer_factor=buffer_factor,
            merged_buffer_factor=merged_buffer_factor,
            n=n,
        )
        hash_3d_data = jax.jit(hash_3d_data)
        for t in range(3):
            start = time.time()
            _data, a_mask, num_lost_points = hash_3d_data(data[t], atom_mask)
            print(f"Time: {time.time() - start}s")

        _mask = a_mask > 0
        empty_space = jnp.argmax(_mask.reshape(-1, _mask.shape[-1])[:, ::-1], axis=-1).min()
        for m in _mask.sum(-1):
            plt.matshow(m)
            plt.colorbar()
            plt.show()
        print(f"lost {int(num_lost_points)} points")


if __name__ == "__main__":
    main()
