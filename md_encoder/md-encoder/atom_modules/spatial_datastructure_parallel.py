import math
import os
import time
from functools import partial
from typing import List, Tuple, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from utils.read_md_utils import read_frames


def bin_particles_into_cells(
    particle_shard: jnp.array,
    particle_mask: jnp.array,
    num_cells: int,
    box_size: float,
    dimension_index: int,
    buffer_scale_factor: float = 2.0,
) -> Tuple[jnp.array, jnp.array, jnp.array]:
    """Bins particles from a shard into spatial cells along a specific dimension.
    
    This function assigns particles to cells based on their position in the specified
    dimension and stores them in buffers. It uses a loop to sequentially place particles
    into the appropriate cell's buffer, updating counts and masks accordingly.
    
    Note: Could potentially optimize by using an add operation in the buffer and
    incrementing the counter by the mask value.

    Args:
        particle_shard (jnp.array[num_per_shard, data_dim]): Array of particle data (e.g., positions).
        particle_mask (jnp.array[num_per_shard]): Mask indicating valid particles (1 for valid, 0 otherwise).
        num_cells (int): Number of spatial cells to divide the box into.
        box_size (float): Length of the simulation box along the binning dimension.
        dimension_index (int): Index of the dimension (e.g., 0 for x, 1 for y) to bin along.
        buffer_scale_factor (float, optional): Factor to scale the buffer size per cell. Defaults to 2.0.

    Returns:
        cell_buffers (jnp.array[num_cells, cell_buffer_size, data_dim]): Buffers holding particles per cell.
        cell_buffer_masks (jnp.array[num_cells, cell_buffer_size]): Masks for the buffers.
        cell_counts (jnp.array[num_cells]): Number of valid particles per cell.
    """
    data_dim = particle_shard.shape[-1]
    num_particles_in_shard = particle_shard.shape[0]
    cell_buffer_size = int(buffer_scale_factor * num_particles_in_shard / num_cells)
    cell_buffers = jnp.zeros(
        (num_cells, cell_buffer_size, data_dim), dtype=particle_shard.dtype
    )
    cell_buffer_masks = jnp.zeros((num_cells, cell_buffer_size), dtype=jnp.int32)
    cell_counts = jnp.zeros(num_cells, dtype=jnp.int32)

    def assign_particle_to_cell_loop_body(i, args):
        particle_shard, particle_mask, cell_buffers, cell_buffer_masks, cell_counts = args
        position = particle_shard[i]
        mask_value = particle_mask[i]
        cell_index = jnp.floor(
            position[dimension_index] * num_cells / box_size
        ).astype(jnp.int32)
        current_count = cell_counts[cell_index]
        cell_buffers = cell_buffers.at[cell_index, current_count].set(position)
        cell_buffer_masks = cell_buffer_masks.at[cell_index, current_count].set(mask_value)
        cell_counts = cell_counts.at[cell_index].set(current_count + (mask_value > 0))
        return particle_shard, particle_mask, cell_buffers, cell_buffer_masks, cell_counts

    particle_shard, particle_mask, cell_buffers, cell_buffer_masks, cell_counts = jax.lax.fori_loop(
        lower=0,
        upper=num_particles_in_shard,
        body_fun=assign_particle_to_cell_loop_body,
        init_val=(particle_shard, particle_mask, cell_buffers, cell_buffer_masks, cell_counts),
    )
    return cell_buffers, cell_buffer_masks, cell_counts


def merge_streams_into_cells(
    cell_buffers: jnp.array,
    cell_buffer_masks: jnp.array,
    cell_counts: jnp.array,
    target_buffer_size: int = -1,
) -> Tuple[jnp.array, jnp.array, jnp.array]:
    """Merges multiple streams of binned data into contiguous buffers per cell.
    
    This function consolidates data from multiple streams into a single buffer dimension
    per cell. If a target buffer size is provided, it gathers valid particles into a
    smaller, contiguous buffer.

    Args:
        cell_buffers (jnp.array[num_streams, num_cells, cell_buffer_size, data_dim]): 
            Buffers from multiple streams.
        cell_buffer_masks (jnp.array[num_streams, num_cells, cell_buffer_size]): 
            Masks for the buffers.
        cell_counts (jnp.array[num_streams, num_cells]): Counts per stream and cell.
        target_buffer_size (int, optional): Desired size for the merged buffer. If -1, no resizing. Defaults to -1.

    Returns:
        merged_buffers (jnp.array[num_cells, merged_buffer_size, data_dim]): Merged particle buffers.
        merged_buffer_masks (jnp.array[num_cells, merged_buffer_size]): Merged masks.
        total_cell_counts (jnp.array[num_cells]): Total counts per cell across streams.
    """
    num_streams, num_cells, cell_buffer_size = cell_buffer_masks.shape
    
    def flatten_streams_into_buffer_dim(array):
        right_shape = array.shape[3:]
        array = jnp.swapaxes(array, 1, 0)
        return array.reshape(num_cells, num_streams * cell_buffer_size, *right_shape)

    merged_buffer_masks = flatten_streams_into_buffer_dim(cell_buffer_masks)
    merged_buffers = flatten_streams_into_buffer_dim(cell_buffers)

    if target_buffer_size != -1:
        def gather_valid_particles(cell_buffer: jnp.array, cell_mask: jnp.array):
            """Gathers valid (masked) particles into a contiguous array.
            
            Args:
                cell_buffer (jnp.array[flattened_buffer_size, data_dim]): Flattened buffer for one cell.
                cell_mask (jnp.array[flattened_buffer_size]): Mask for the buffer.

            Returns:
                gathered_buffer (jnp.array[target_buffer_size, data_dim]): Contiguous valid particles.
                gathered_mask (jnp.array[target_buffer_size]): Corresponding mask.
            """
            (valid_indices,) = jnp.where(cell_mask > 0, size=target_buffer_size, fill_value=-1)
            valid_mask = valid_indices != -1
            return cell_buffer[valid_indices], cell_mask[valid_indices] * valid_mask.astype(jnp.int32)

        merged_buffers, merged_buffer_masks = jax.vmap(gather_valid_particles, in_axes=(0, 0))(
            merged_buffers, merged_buffer_masks
        )
    
    total_cell_counts = cell_counts.sum(axis=0)
    return merged_buffers, merged_buffer_masks, total_cell_counts


def bin_and_merge_particles_along_dimension(
    particle_positions: jnp.array,
    particle_masks: jnp.array,
    dimension_index: int,
    target_buffer_size: int,
    num_cells: int,
    num_streams: int,
    buffer_scale_factor: float,
    box_size: float,
) -> Tuple[jnp.array, jnp.array]:
    """Bins particles along a single dimension and merges streams into contiguous buffers.
    
    This is a wrapper that applies binning to sharded data and then merges the results.

    Args:
        particle_positions (jnp.array): Particle position data.
        particle_masks (jnp.array): Masks for valid particles.
        dimension_index (int): Dimension to bin along.
        target_buffer_size (int): Size for the merged buffer per cell.
        num_cells (int): Number of cells along the dimension.
        num_streams (int): Number of parallel streams for processing.
        buffer_scale_factor (float): Scaling factor for initial buffer sizes.
        box_size (float): Box size along the dimension.

    Returns:
        merged_buffers (jnp.array): Merged particle buffers.
        merged_buffer_masks (jnp.array): Merged masks.
    """
    bin_particles_partial = partial(
        bin_particles_into_cells,
        num_cells=num_cells,
        box_size=box_size,
        dimension_index=dimension_index,
        buffer_scale_factor=buffer_scale_factor
    )
    particle_positions = particle_positions.reshape(num_streams, -1, 3)
    particle_masks = particle_masks.reshape(num_streams, -1)
    cell_buffers, cell_buffer_masks, cell_counts = jax.vmap(bin_particles_partial, in_axes=(0, 0))(particle_positions, particle_masks)
    merged_buffers, merged_buffer_masks, _ = merge_streams_into_cells(
        cell_buffers, cell_buffer_masks, cell_counts, target_buffer_size=target_buffer_size
    )
    return merged_buffers, merged_buffer_masks


def spatial_hash_particles(
    particle_positions: jnp.array,
    particle_masks: jnp.array,
    num_dimensions: int,
    num_cells_per_dim: Union[List[int], int],
    num_streams_per_dim: Union[List[int], int],
    buffer_scale_factors: Union[List[float], float],
    box_sizes: Union[List[float], float],
) -> Tuple[jnp.array, jnp.array, jnp.array]:
    """Performs spatial hashing of particles into a multi-dimensional grid.
    
    This function bins particles sequentially along each dimension, using streams for
    parallel processing. It supports appending along the last dimension of data for
    hashing by the first three channels.

    Args:
        particle_positions (jnp.array): Particle position data.
        particle_masks (jnp.array): Masks for valid particles.
        num_dimensions (int): Number of spatial dimensions to hash (e.g., 3 for 3D).
        num_cells_per_dim (Union[List[int], int]): Number of cells per dimension.
        num_streams_per_dim (Union[List[int], int]): Number of streams per dimension.
        buffer_scale_factors (Union[List[float], float]): Buffer scaling per dimension.
        box_sizes (Union[List[float], float]): Box sizes per dimension.

    Returns:
        hashed_buffers (jnp.array): Hashed particle buffers in grid shape.
        hashed_buffer_masks (jnp.array): Corresponding masks.
        cell_counts (jnp.array): Counts per final cell.
    """
    channel_dims = particle_positions.shape[-1]

    if not isinstance(box_sizes, list):
        box_sizes = [box_sizes] * num_dimensions
    if not isinstance(buffer_scale_factors, list):
        buffer_scale_factors = [buffer_scale_factors] * num_dimensions
    if not isinstance(num_streams_per_dim, list):
        num_streams_per_dim = [num_streams_per_dim] * num_dimensions
    if not isinstance(num_cells_per_dim, list):
        num_cells_per_dim = [num_cells_per_dim] * num_dimensions
    num_streams_per_dim += [1]

    particle_positions = particle_positions.reshape(num_streams_per_dim[0], -1, channel_dims)
    particle_masks = particle_masks.reshape(num_streams_per_dim[0], -1)
    
    # Sequentially bin along each dimension
    for dim in range(num_dimensions):
        bin_partial = partial(
            bin_particles_into_cells,
            num_cells=num_cells_per_dim[dim],
            box_size=box_sizes[dim],
            buffer_scale_factor=buffer_scale_factors[dim],
            dimension_index=dim,
        )
        # Vmap over leading dimensions (streams and previous divisions)
        for _ in range(dim + 1):
            bin_partial = jax.vmap(bin_partial)
        particle_positions, particle_masks, cell_counts = bin_partial(particle_positions, particle_masks)

        # Swap stream axis to be next to buffer axis
        stream_axis = dim
        new_hash_dim = dim + 1
        particle_positions = jnp.swapaxes(particle_positions, stream_axis, new_hash_dim)
        particle_masks = jnp.swapaxes(particle_masks, stream_axis, new_hash_dim)

        # Reshape for next dimension's streams and buffers
        current_num_streams = num_streams_per_dim[dim]
        next_num_streams = num_streams_per_dim[dim + 1]
        current_buffer_size = particle_positions.shape[-2]
        multiplier = current_num_streams // next_num_streams
        assert current_num_streams % next_num_streams == 0
        new_shape = tuple(
            num_cells_per_dim[:dim + 1] + [next_num_streams, multiplier * current_buffer_size]
        )
        particle_positions = particle_positions.reshape(*new_shape, channel_dims)
        particle_masks = particle_masks.reshape(*new_shape)

    # Final merge of streams and buffers
    particle_positions = particle_positions.reshape(*particle_positions.shape[:num_dimensions], -1, channel_dims)
    particle_masks = particle_masks.reshape(*particle_positions.shape[:num_dimensions], -1)
    return particle_positions, particle_masks, cell_counts


def hash_3d_particles(
    particle_positions: jnp.array,
    particle_masks: jnp.array,
    num_cells: int,
    num_streams: int,
    box_size: float,
    initial_buffer_scale: float,
    merged_buffer_scale: float,
    num_particles: int,
) -> Tuple[jnp.array, jnp.array, int]:
    """Hashes particles into a 3D grid for efficient spatial queries.
    
    This function applies binning and merging sequentially along x, y, z dimensions.
    It calculates memory redundancy and lost points due to buffering.

    Args:
        particle_positions (jnp.array): Particle positions.
        particle_masks (jnp.array): Particle masks.
        num_cells (int): Number of cells per dimension.
        num_streams (int): Number of streams for parallel processing.
        box_size (float): Simulation box size (assumed cubic for simplicity).
        initial_buffer_scale (float): Initial buffer scaling factor.
        merged_buffer_scale (float): Scaling for merged buffers.
        num_particles (int): Total number of particles.

    Returns:
        hashed_buffers (jnp.array): 3D hashed buffers.
        hashed_buffer_masks (jnp.array): Masks for hashed buffers.
        num_lost_points (int): Number of points lost (should be 0 if buffers are sufficient).
    """
    def calculate_target_buffer_size(dim: int) -> int:
        return math.ceil(
            (merged_buffer_scale ** dim) * num_particles / (num_streams * num_cells ** dim)
        ) * num_streams

    bin_and_merge_partial = partial(
        bin_and_merge_particles_along_dimension,
        num_cells=num_cells,
        num_streams=num_streams,
        buffer_scale_factor=initial_buffer_scale,
        box_size=box_size,
    )
    
    # Bin along x (dimension 0)
    particle_positions, particle_masks = bin_and_merge_partial(
        particle_positions=particle_positions,
        particle_masks=particle_masks,
        dimension_index=0,
        target_buffer_size=calculate_target_buffer_size(1)
    )

    # Vmap for y (dimension 1)
    v_bin_and_merge = jax.vmap(bin_and_merge_partial, in_axes=(0, 0, None, None))
    particle_positions, particle_masks = v_bin_and_merge(particle_positions, particle_masks, 1, calculate_target_buffer_size(2))

    # Double vmap for z (dimension 2)
    vv_bin_and_merge = jax.vmap(v_bin_and_merge, in_axes=(0, 0, None, None))
    particle_positions, particle_masks = vv_bin_and_merge(particle_positions, particle_masks, 2, calculate_target_buffer_size(3))

    memory_redundancy = np.prod(particle_masks.shape) / num_particles
    num_lost_points = num_particles - (particle_masks > 0).sum()
    print(
        f"initial_buffer_scale: {initial_buffer_scale}\n"
        f"merged_buffer_scale: {merged_buffer_scale}\n"
        f"memory_redundancy: {memory_redundancy}\n"
        f"num_lost_points: {num_lost_points}"
    )
    return particle_positions, particle_masks, num_lost_points


def load_trajectory_data(path, permute=True, pad=True, num_streams=None) -> Tuple[jnp.array, jnp.array, dict, jnp.array, int]:
    """Loads molecular dynamics trajectory data and prepares it for processing.
    
    This function reads frames, assigns atom types, shifts positions to origin,
    and optionally permutes and pads the data.

    Args:
        path (str): Path to the trajectory file.
        permute (bool, optional): Whether to randomly permute particles along the particle axis. Defaults to True.
        pad (bool, optional): Whether to pad data to be divisible by num_streams. Defaults to True.
        num_streams (int, optional): Required if pad=True; number of streams for sharding.

    Returns:
        positions (jnp.array[ts, n, 3]): Particle positions over timesteps.
        atom_masks (jnp.array[ts, n]): Atom type masks (non-zero for valid atoms).
        atom_type_map (dict): Mapping from atom names to integer types.
        box_sizes (jnp.array[3]): Simulation box sizes.
        num_particles (int): Original number of particles per frame.
    """
    positions, resnames, atom_names, box_sizes = read_frames(path)

    # Simple atom type encoding (improve later if needed)
    atom_names_flat = [a[0] for a in atom_names]
    unique_atom_names = set(atom_names_flat)
    atom_type_map = {a: i for i, a in enumerate(unique_atom_names, start=1)}
    atom_types = jnp.array([atom_type_map[a] for a in atom_names_flat]).astype(jnp.int32)

    # Shift positions to origin and update box
    positions -= jnp.min(positions.reshape(-1, 3), axis=0)[None, None, :]
    box_sizes = jnp.maximum(jnp.max(positions.reshape(-1, 3), axis=0), box_sizes)

    timesteps, num_particles, _ = positions.shape

    if permute:
        print("Permuting...")
        start_time = time.time()
        key = jax.random.PRNGKey(seed=0)
        positions = jax.random.permutation(key, positions, axis=-2, independent=True)
        print(f"Done in {time.time() - start_time}s")

    if pad:
        assert num_streams is not None
        particles_per_stream = jnp.ceil(num_particles / num_streams).astype(int)
        padded_size = particles_per_stream * num_streams
        padding = jnp.zeros((timesteps, padded_size - num_particles, 3), positions.dtype)
        atom_masks = jnp.concatenate(
            [atom_types, jnp.zeros(padded_size - num_particles, jnp.int32)], axis=-1
        )
        positions = jnp.concatenate([positions, padding], axis=1)
    else:
        atom_masks = atom_types

    return positions, atom_masks, atom_type_map, box_sizes, num_particles


def visualize_trajectory(positions, atom_masks, atom_type_map, box_sizes, num_cells=15, output_file='trajectory.mp4'):
    positions = np.array(positions)
    atom_masks = np.array(atom_masks)
    box_sizes = np.array(box_sizes)
    
    element_map = {v: k for k, v in atom_type_map.items()}
    element_colors = {'N': 'blue', 'C': 'black', 'O': 'red', 'S': 'yellow'}
    
    valid_non_h = (atom_masks > 0) & (atom_masks != atom_type_map['H']) & (atom_masks != atom_type_map['O'])
    
    filtered_positions = positions[:, valid_non_h, :]
    filtered_types = atom_masks[valid_non_h]
    
    cell_sizes = box_sizes / num_cells
    central_cell = np.array([num_cells // 2] * 3)
    
    shift = cell_sizes * 1.5
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.tight_layout()

    def plot_box(pos1, pos2, c):
        x, y, z = pos1
        x1, y1, z1 = pos2
        ax.plot([x, x1], [y, y], [z, z], c=c)
        ax.plot([x, x], [y, y1], [z, z], c=c)
        ax.plot([x, x], [y, y], [z, z1], c=c)
        ax.plot([x1, x], [y1, y1], [z1, z1], c=c)
        ax.plot([x1, x1], [y1, y], [z1, z1], c=c)
        ax.plot([x1, x1], [y1, y1], [z1, z], c=c)

        ax.plot([x1, x], [y, y], [z1, z1], c=c)
        ax.plot([x1, x], [y1, y1], [z, z], c=c)
        ax.plot([x, x], [y1, y], [z1, z1], c=c)
        ax.plot([x1, x1], [y1, y], [z, z], c=c)
        ax.plot([x1, x1], [y, y], [z1, z], c=c)
        ax.plot([x, x], [y1, y1], [z1, z], c=c)

    def update(t):
        ax.clear()
        for offset in [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [-1, -1, 1]]:
            plot_box(cell_sizes * (central_cell + offset), cell_sizes * (central_cell + 1 + offset), c="gray")

        plot_box(cell_sizes * central_cell, cell_sizes * (central_cell + 1), c="orange")

        pos_t = filtered_positions[t]
        cell_idx = np.floor(pos_t / cell_sizes).astype(int)
        is_central = np.all(cell_idx == central_cell, axis=1)
        
        unique_types = np.unique(filtered_types)
        for typ in unique_types:
            elem = element_map[typ]
            color = element_colors.get(elem, 'green')  # Default color for unknown elements
            type_mask = filtered_types == typ
            central_mask = type_mask & is_central
            other_mask = type_mask & ~is_central
            
            if np.any(central_mask):
                ax.scatter(
                    pos_t[central_mask, 0],
                    pos_t[central_mask, 1],
                    pos_t[central_mask, 2],
                    c=color,
                    alpha=0.8,
                    s=28
                )
            if np.any(other_mask):
                ax.scatter(
                    pos_t[other_mask, 0],
                    pos_t[other_mask, 1],
                    pos_t[other_mask, 2],
                    c=color,
                    alpha=0.2,
                    s=12
                )
        
        ax.set_xlim(shift[0], box_sizes[0]-shift[0])
        ax.set_ylim(shift[1], box_sizes[1]-shift[1])
        ax.set_zlim(shift[2], box_sizes[2]-shift[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(25, t)
        # ax.set_title(f'Frame {t}')
        return ax,
    
    ani = FuncAnimation(fig, update, frames=range(positions.shape[0]), blit=False, interval=200)
    ani.save(output_file, writer='ffmpeg', fps=12)
    plt.close(fig)


def main():
    paths = [
        "../mac_mount/pmhc/md_0_1.trr",
        "/home/louis/Dropbox/Cool/md/md_0_1.trr"
    ]
    process_all_frames = False

    if process_all_frames:
        output_dir = "frames/buffer"
        os.makedirs(output_dir, exist_ok=True)

        num_streams = 16
        num_cells = 8
        dimension_index = 0
        initial_buffer_scale = 4.0
        merged_buffer_scale = 1.5  # Not used in this branch, but kept for consistency

        positions, atom_masks, atom_type_map, box_sizes, num_particles = load_trajectory_data(
            paths[0], permute=False, pad=True, num_streams=num_streams
        )
        box_size = box_sizes[dimension_index]

        bin_particles_partial = partial(
            bin_particles_into_cells,
            num_cells=num_cells,
            box_size=box_size,
            dimension_index=dimension_index,
            buffer_scale_factor=initial_buffer_scale
        )
        v_bin_particles = jax.jit(jax.vmap(bin_particles_partial, in_axes=(0, 0)))
        for t in tqdm(range(positions.shape[0])):
            frame_positions = positions[t]
            frame_positions = frame_positions.reshape(num_streams, -1, 3)

            cell_buffers, cell_buffer_masks, cell_counts = v_bin_particles(frame_positions, atom_masks)

            particles_per_stream = frame_positions.shape[1]
            print(f"particles_per_stream: {particles_per_stream}")
            print(f"cell_buffers.shape: {cell_buffers.shape}")
            plt.matshow(cell_buffer_masks.sum(-1).T, vmin=0, vmax=cell_buffers.shape[2])
            plt.colorbar()
            plt.xlabel("num_streams")
            plt.ylabel("num_cells")
            plt.savefig(f"{output_dir}/t_{t}")
            plt.close()
            print(f"num_particles: {num_particles}")
            print(f"cell_buffer_masks.sum(): {cell_buffer_masks.sum()}")

        # make_vid(output_dir, video_name=f"{output_dir}/video.mp4v")  # Commented out as make_vid is not defined
    else:
        # Configuration for no lost points with permutation
        timestep = 0

        num_streams = 16
        num_cells = 15
        dimension_index = 0
        initial_buffer_scale = 5.0
        merged_buffer_scale = 1.2

        positions, atom_masks, atom_type_map, box_sizes, num_particles = load_trajectory_data(
            paths[0], permute=False, pad=True, num_streams=num_streams
        )
        visualize_trajectory(positions, atom_masks, atom_type_map, box_sizes, num_cells=8, output_file='cells_trajectory.mp4')
        box_size = box_sizes[dimension_index]

        hash_3d_partial = partial(
            hash_3d_particles,
            num_cells=num_cells,
            num_streams=num_streams,
            box_size=box_size,
            initial_buffer_scale=initial_buffer_scale,
            merged_buffer_scale=merged_buffer_scale,
            num_particles=num_particles,
        )
        hash_3d_jitted = jax.jit(hash_3d_partial)
        for t in range(3):
            start = time.time()
            hashed_buffers, hashed_masks, num_lost_points = hash_3d_jitted(positions[t], atom_masks)
            print(f"Time: {time.time() - start}s")

        # valid_mask = hashed_masks > 0
        # # Find the minimum empty space in the buffers (reversed argmax for trailing zeros)
        # min_empty_space = jnp.argmax(valid_mask.reshape(-1, valid_mask.shape[-1])[:, ::-1], axis=-1).min()
        # for slice_mask in valid_mask.sum(-1):
        #     plt.matshow(slice_mask)
        #     plt.colorbar()
        #     plt.show()
        print(f"lost {int(num_lost_points)} points")


if __name__ == "__main__":
    main()