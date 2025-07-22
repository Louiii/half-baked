import jax
import jax.numpy as jnp
import numpy as np
from md_encoder.read_md_utils import read_frames
from tqdm import tqdm


####################
def collect_points(points, n_lattice_1d=11, factor=1.5):
    """assume box [0, 1]^3"""
    p = points * (n_lattice_1d - 1)
    num_lattice_pts = n_lattice_1d ** 3
    num_per_voxel = num_lattice_pts / len(p)
    f = jnp.floor(p)  # map from point ix -> lattice coord
    neigh = jnp.stack(np.meshgrid(*[[0, 1]] * 3)).reshape(3, -1).T
    allp = f[:, None, :] + neigh[None, :, :]
    _mult = n_lattice_1d ** jnp.arange(3)
    ixs = (allp * _mult[None, None, :]).sum(-1)  # [num_pts, 8]

    buffer_size = num_per_voxel * 8 * factor
    # def make_row(_1d_voxel_indices, i):
    #     bool_mask = _1d_voxel_indices == i
    #     (ixs,) = jnp.where(bool_mask, size=buffer_size, fill_value=-1)
    #     return ixs

    neighbour_buffer = jnp.zeros((len(points), buffer_size), dtype=np.int32)
    
    # pts = points[:, None, :].repeat(1, int(2 ** 3), 1)
    # ixs = allp.reshape(-1)
    # pts = pts.reshape(-1, 3)
    # p = pts[ixs].reshape()
    return
####################

def compute_num_in_sphere(position, radius):
    return (np.linalg.norm(X - position, axis=-1) < radius).sum(-1)


def get_position(radius):
    return radius + np.random.rand(3) * (xyz - 2 * radius)


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
    fixed_latice = jnp.stack(jnp.meshgrid(fixed_den_1d, fixed_den_1d, fixed_den_1d))
    fixed_latice = fixed_latice.reshape(3, -1).T

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


def get_radius_from_apx_nn(box, apx_nn=100):
    p = box * .5
    diff = np.linalg.norm(X[0] - p, axis=-1)
    diff.sort()
    r = diff[apx_nn]
    return r


path = "/Users/louisrobinson/Desktop/md/mac_mount/pmhc/md_0_1.trr"
X, resname, atom_name, box = read_frames(path)

from jax_md import partition, space

box_size = box[0]
displacement_fn, shift_fn = space.periodic(box_size)

r = get_radius_from_apx_nn(box, apx_nn=100) * 1.2
dr_threshold = r * 0.1
neighbour_list_fn = partition.neighbor_list(
    displacement_or_metric=displacement_fn,  # A function `d(R_a, R_b)` that computes the displacement between pairs of points.
    box_size=box,  # Either a float specifying the size of the box or an array of shape `[spatial_dim]` specifying the box size in each spatial dimension.
    r_cutoff=r,  # A scalar specifying the neighborhood radius.
    dr_threshold=dr_threshold,  # A scalar specifying the maximum distance particles can move before rebuilding the neighbor list.
    capacity_multiplier=1.25,  # A floating point scalar specifying the fractional increase in maximum neighborhood occupancy we allocate compared with the maximum in the example positions.
    disable_cell_list=False,  # An optional boolean. If set to `True` then the neighbor list is constructed using only distances. This can be useful for debugging but should generally be left as `False`.
    mask_self=True,  # An optional boolean. Determines whether points can consider themselves to be their own neighbors.
    custom_mask_function=None,  # An optional function. Takes the neighbor array and masks selected elements. Note: The input array to the function is `(n_particles, m)` where the index of particle 1 is in index in the first dimension of the array, the index of particle 2 is given by the value in the array
    fractional_coordinate=False,  # An optional boolean. Specifies whether positions will be supplied in fractional coordinates in the unit cube, :math:`[0, 1]^d`. If this is set to True then the `box_size` will be set to `1.0` and the cell size used in the cell list will be set to `cutoff / box_size`.
    format=partition.NeighborListFormat.Dense,  # The format of the neighbor list; see the :meth:`NeighborListFormat` enum for details about the different choices for formats. Defaults to `Dense`.
    # **static_kwargs: kwargs that get threaded through the calculation of example positions.
)
neighbours = neighbour_list_fn.allocate(X[0])  # Create a new neighbour list.
mask = neighbours.idx != neighbours.reference_position.shape[0]
t = 5
# neighbours_history = 
for x in tqdm(X[1:t]):
    # takes about 25s per iteration to allocate 100k points, with r = 0.7, dr = 0.07
    neighbours = neighbours.update(x)  # Update the neighbour list without resizing.
    if neighbours.did_buffer_overflow:  # Couldn't fit all the neighbours into the list.
        print("WARNING: List overflowed.")
        neighbours = neighbour_list_fn.allocate(X)  # So create a new neighbour list.



ex, encoded_max_signal, max_signal = variable_number_of_points_to_fixed_density(
    points=x,
    mask=mask,
    domain=(0, box[0]),
    num_points_1d=100,
    gauss_width=r / 10,
    radius=r,
)


def get_neighbours(
    box_size,
    hash_cell_width,
    radius,
    positions,
):
    # num_points = positions.shape[0]
    positions = positions / box_size[None, :]
    n1d_voxels = jnp.ceil(1 / hash_cell_width)
    num_voxels = int(n1d_voxels ** 3)
    # dr = 1 / (n1d_voxels - 1)

    # voxels_1d = jnp.linspace(0, 1, n1d_voxels)
    voxels = jnp.floor(positions * (n1d_voxels - 1)).astype(jnp.int32)
    num_voxels = len(voxels)
    _u, c = jnp.unique(voxels, return_counts=True, size=num_voxels, fill_value=-1)

    max_points_in_cell = jnp.max(c)
    buffer_size = jnp.ceil(max_points_in_cell * 1.2).astype(int)
    # cell_buffer = -jnp.ones((num_points, buffer_size, 3))
    def _3d_voxel_ixs_to_1d(_3d_voxel_ixs):
        r3 = n1d_voxels ** jnp.arange(3)
        r1 = (_3d_voxel_ixs * r3[None, :]).sum(-1).astype(jnp.int32)
        return r1

    def get_voxel_neighbour_indices(_1d_voxel_index):
        z_component, rest = jnp.divmod(_1d_voxel_index, n1d_voxels ** 2)
        y_component, x_component = jnp.divmod(rest, n1d_voxels)
        adjacent = jnp.stack(jnp.meshgrid(
            jnp.concatenate([x_component - 1, x_component, x_component + 1]),
            jnp.concatenate([y_component - 1, y_component, y_component + 1]),
            jnp.concatenate([z_component - 1, z_component, z_component + 1]),
        ))
        msk = ((adjacent >= 0) * (adjacent < n1d_voxels)).prod(-1).reshape(-1)
        ixs = _3d_voxel_ixs_to_1d(_3d_voxel_ixs=adjacent.reshape(-1, 3))
        return ixs, msk

    all_voxels = jnp.arange(num_voxels)
    voxel_ix2neighbour_ixs, voxel_neighbour_mask = jax.vmap(
        get_voxel_neighbour_indices
    )(all_voxels)

    def make_row(_1d_voxel_indices, i):
        bool_mask = _1d_voxel_indices == i
        (ixs,) = jnp.where(bool_mask, size=buffer_size, fill_value=-1)
        return ixs
        # p = pos[ixs]
        # m = ixs != -1
        # return p, m

    def fill_voxels(positions):
        """Complexity: num_voxels * num_points"""
        vi = _3d_voxel_ixs_to_1d(_3d_voxel_ixs=jnp.floor(positions * n1d_voxels))
        # make_row with vi checks every other point... not just the neighbouring voxels
        make_rows = jax.vmap(make_row, in_axes=(None, 0))
        pos_ixs = make_rows(vi, jnp.arange(num_voxels))
        return pos_ixs, pos_ixs != -1
        # return binned_pos, binned_masks
    voxel2pos_ixs, mask = fill_voxels(positions)  # shape: [num_voxels, buffer_size] indexing positions

    def update_voxel(voxel):
        """map point in R3 -> neighbour cell indices -> all points in those voxels"""
        vi = _3d_voxel_ixs_to_1d(_3d_voxel_ixs=jnp.floor(positions * n1d_voxels))
        neighbouring_voxels = voxel_ix2neighbour_ixs[voxel]
        neighbouring_voxels_mask = voxel_neighbour_mask[voxel]
        m = mask * neighbouring_voxels_mask
        prev_pos_ixs = voxel2pos_ixs[neighbouring_voxels] * m
        prev_pos_ixs -= (1 - m)
        prev_pos_ixs = prev_pos_ixs.reshape(-1)
        vi_subset = vi[prev_pos_ixs]
        new_pos_ixs = prev_pos_ixs[make_row(vi_subset, voxel)]
        return new_pos_ixs

    def update_voxels():
        pos_ixs = jax.vmap(update_voxel)(jnp.arange(num_voxels))
        return pos_ixs, pos_ixs != -1

    return jax.jit(fill_voxels), jax.jit(update_voxels)



# n = 10
# reps = 10
# d = {}
# for r in tqdm(np.linspace(0.1, 4, n)):
#     d[str(r)] = [
#         list(map(int, compute_num_in_sphere(get_position(r), r))) for _ in range(reps)
#     ]
# with open("data/rad.json", "w") as f:
#     f.write(json.dumps(d))

# with open("data/rad.json") as f:
#     d = json.load(f)
# xs = [(float(k), min(map(min, vs)), max(map(max, vs))) for k, vs in d.items()]
# rs, mns, mxs = list(zip(*xs))

# plt.plot(rs, mns)
# plt.plot(rs, mxs)
# plt.show()

# t = 30
# x = X[:t]
# mean_pos = x.mean(0).mean(0)
# diffs = np.linalg.norm(x - mean_pos, axis=-1) < 1.0

# points = []
# for x_, d in zip(x, diffs):
#     points.append(x_[d.astype(bool)])



# image_folder = "plots"
# for i, p in enumerate(points):
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(projection="3d")
#     ax.scatter(*p.T, marker="^")
#     plt.tight_layout()
#     plt.savefig(f"{image_folder}/frame_{i}.png")
#     plt.close(fig)


# make_vid(image_folder, video_name=f"{image_folder}/video.mp4v")
