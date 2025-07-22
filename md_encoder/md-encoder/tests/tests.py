import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from md_encoder.atom_modules.encoder_functions import points_2_lattice
from md_encoder.atom_modules.modules import (
    get_neighbour_voxels,
    ground_truth,
    meshgrid,
    pad3,
    spatial_hash_to_lattice,
)
from md_encoder.atom_modules.spatial_datastructure_parallel import spatial_hash, stream
from tqdm import tqdm

# import sys; sys.path.append("/".join(sys.path[0].split("/")[:-1]))  # noqa: 


def test_stream():
    n = 1000
    shard = np.random.rand(n, 3)
    y = int(n * 0.29)
    mask = np.array([1] * (n - y) + [0] * y)
    buffer_, buffer_mask, counts = stream(
        shard,
        mask,
        num_divisions=8,
        box_length=1,
        spatial_dimension=0,
        buffer_factor=3.0,
    )
    print("counts")
    print(counts)
    b = buffer_[..., :2]
    for division, m in zip(b, buffer_mask):
        points = division[np.where(m)]
        plt.scatter(*points.T)
    plt.savefig("images/debug.png")
    plt.clf()


def test_stream2():
    n = 1000
    shard = np.random.rand(n, 3)
    y = int(n * 0.29)
    mask = np.array([1] * (n - y) + [0] * y)
    buffer_, buffer_mask, counts = stream(
        shard,
        mask,
        num_divisions=5,
        box_length=1,
        spatial_dimension=0,
        buffer_factor=3.0,
    )
    print("counts")
    print(counts)
    buffer_, buffer_mask, counts = jax.vmap(
        partial(
            stream,
            num_divisions=5,
            box_length=1,
            spatial_dimension=1,
            buffer_factor=3.0,
        )
    )(buffer_, buffer_mask)
    print("counts")
    print(counts)
    s1, s2 = buffer_.shape[:2]
    b = buffer_[..., :2].reshape(s1 * s2, -1, 2)
    bm = buffer_mask.reshape(s1 * s2, -1)
    for i, (division, m) in enumerate(zip(b, bm)):
        points = division[np.where(m)]
        p = np.mean(points, axis=0)
        plt.scatter(*points.T, alpha=0.5)
        plt.text(*p, str(i))
    plt.savefig("images/debug2.png")
    plt.clf()


def test_streams3d(direc="images/test_streams3d"):
    n = 1200
    n_streams = 4
    num_divisions = 2
    buffer_factor = 3
    box_length = 1
    shard = box_length * np.random.rand(n, 3).reshape(n_streams, -1, 3)
    y = int(n * 0.11)
    mask = np.array([1] * (n - y) + [0] * y).reshape(n_streams, -1)

    buffer_, buffer_mask, counts = spatial_hash(
        shard, mask, 3, num_divisions, n_streams, buffer_factor, box_length
    )

    print("counts")
    print(counts.sum(-2))
    s1, s2, s3 = buffer_.shape[:3]
    b = buffer_.reshape(s1 * s2 * s3, -1, 3)
    bm = buffer_mask.reshape(s1 * s2 * s3, -1)

    os.makedirs(direc, exist_ok=True)
    for j, angle in tqdm(list(enumerate(np.linspace(0, 270, 200)))):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        for i, (division, m) in enumerate(zip(b, bm)):
            points = division[np.where(m)]
            p = np.mean(points, axis=0)
            ax.scatter(*points.T, alpha=0.5)
            ax.text(*p, str(i))
        ax.view_init(40, angle)
        plt.savefig(f"{direc}/plot_{j}.png", dpi=500)
        plt.clf()


def test_streams():
    n = 1200
    n_streams = 4
    num_divisions = 5
    shard = np.random.rand(n, 3).reshape(n_streams, -1, 3)
    y = int(n * 0.11)
    mask = np.array([1] * (n - y) + [0] * y).reshape(n_streams, -1)
    st = partial(
        stream,
        num_divisions=num_divisions,
        box_length=1,
        buffer_factor=3.0,
    )
    buffer_, buffer_mask, counts = jax.vmap(partial(st, spatial_dimension=0))(shard, mask)
    buffer_ = jnp.swapaxes(buffer_, 0, 1).reshape(num_divisions, n_streams, -1, 3)
    buffer_mask = jnp.swapaxes(buffer_mask, 0, 1).reshape(num_divisions, n_streams, -1)
    print("counts")
    print(counts.sum(-2))
    within = jax.vmap(partial(st, spatial_dimension=1))
    buffer_, buffer_mask, counts = jax.vmap(within)(buffer_, buffer_mask)
    buffer_ = jnp.swapaxes(buffer_, 1, 2).reshape(num_divisions, num_divisions, -1, 3)
    buffer_mask = jnp.swapaxes(buffer_mask, 1, 2).reshape(num_divisions, num_divisions, -1)
    import matplotlib.pyplot as plt
    print("counts")
    print(counts.sum(-2))
    s1, s2 = buffer_.shape[:2]
    b = buffer_[..., :2].reshape(s1 * s2, -1, 2)
    bm = buffer_mask.reshape(s1 * s2, -1)
    for i, (division, m) in enumerate(zip(b, bm)):
        points = division[np.where(m)]
        p = np.mean(points, axis=0)
        plt.scatter(*points.T, alpha=0.5)
        plt.text(*p, str(i))
    plt.savefig("images/debug3.png")
    plt.clf()


def test_neighbours(direc="images/test_neighbours"):
    n = 2000
    n_streams = 4
    num_divisions = 6
    buffer_factor = 3
    box_length = 1
    shard = box_length * np.random.rand(n, 3).reshape(n_streams, -1, 3)
    y = int(n * 0.11)
    mask = np.array([1] * (n - y) + [0] * y).reshape(n_streams, -1)

    buffer_, buffer_mask, counts = spatial_hash(
        shard, mask, 3, num_divisions, n_streams, buffer_factor, box_length
    )

    print("counts")
    print(counts.sum(-2))
    s1, s2, s3 = buffer_.shape[:3]
    b = buffer_.reshape(s1 * s2 * s3, -1, 3)
    bm = buffer_mask.reshape(s1 * s2 * s3, -1)
    all_ = b[np.where(bm)]

    lattice = pad3(buffer_)
    lattice_mask = pad3(buffer_mask[..., None])[..., 0]
    mg = meshgrid(np.arange(1, num_divisions+1), num_dimensions=3).reshape(-1, 3)

    def get_pts(ix):
        l = lattice[ix[0], ix[1], ix[2]]
        lm = lattice_mask[ix[0], ix[1], ix[2]]
        points = l[np.where(lm)]
        neigh = get_neighbour_voxels(lattice, ix-1, leave_n_axes=2)
        neigh_mask = get_neighbour_voxels(lattice_mask, ix-1, leave_n_axes=1)
        neigh = neigh.reshape(-1, 3)
        neigh_mask = neigh_mask.reshape(-1)
        return points, neigh[np.where(neigh_mask)]

    os.makedirs(direc, exist_ok=True)
    cells = num_divisions ** 3
    for j, angle in tqdm(list(enumerate(np.linspace(0, 360, cells * 2)))):
        cell = j // 4
        ix = mg[cell]
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        points, neigh = get_pts(ix)
        ax.scatter(*all_.T, alpha=0.1, c="grey")
        ax.scatter(*neigh.T, alpha=0.3, c="orange")
        ax.scatter(*points.T, alpha=0.5, c="red")
        ax.view_init(40, angle)
        plt.savefig(f"{direc}/plot_{j}.png", dpi=300)
        plt.clf()


def test_lattice(direc="images/test_lattice"):
    """Also make sure that the Gaussian width is much smaller than the voxel size."""
    n = 2000
    gauss_width = 0.01
    number_of_atom_types = 2
    n_streams = 4
    num_divisions = 6
    num_points_1d_per_bin_voxel = 5
    buffer_factor = 1.2
    box_length = 1
    shard = box_length * np.random.rand(n, 3).reshape(n_streams, -1, 3)
    y = int(n * 0.11)
    z = int(n * 0.19)
    mask = np.array([1] * (n - y - z) + [2] * y + [0] * z).reshape(n_streams, -1)

    buffer_, buffer_mask, counts = spatial_hash(
        shard, mask, 3, num_divisions, n_streams, buffer_factor, box_length
    )
    t1 = time.time()
    lattice = spatial_hash_to_lattice(
        coords=buffer_,
        atom_mask=buffer_mask,
        domain=box_length,
        num_points_1d_per_bin_voxel=num_points_1d_per_bin_voxel,
        gauss_width=gauss_width,
        number_of_atom_types=number_of_atom_types,
    )
    t2 = time.time()
    signals = ground_truth(
        data=shard.reshape(-1, 3),
        atom_mask=mask.reshape(-1),
        n_channels=number_of_atom_types,
        box_length=box_length,
        n_lattice=num_points_1d_per_bin_voxel * num_divisions,
        width=gauss_width,
    )
    t3 = time.time()
    error = np.mean(np.abs(signals - lattice))
    print(
        f"error: {error}; gt: {np.abs(signals).sum()}; lat: {np.abs(lattice).sum()}  "
        f"hash time: {t2-t1}s; gt time: {t3-t2}s"
    )

    # print("counts")
    # print(counts.sum(-2))
    # s1, s2, s3 = buffer_.shape[:3]
    # b = buffer_.reshape(s1 * s2 * s3, -1, 3)
    # bm = buffer_mask.reshape(s1 * s2 * s3, -1)
    # all_ = b[np.where(bm)]

    # lattice = pad3(buffer_)
    # lattice_mask = pad3(buffer_mask[..., None])[..., 0]
    # mg = meshgrid(np.arange(1, num_divisions+1), num_dimensions=3).reshape(-1, 3)

    # def get_pts(ix):
    #     l = lattice[ix[0], ix[1], ix[2]]
    #     lm = lattice_mask[ix[0], ix[1], ix[2]]
    #     points = l[np.where(lm)]
    #     neigh = get_neighbour_voxels(lattice, ix-1, leave_n_axes=2)
    #     neigh_mask = get_neighbour_voxels(lattice_mask, ix-1, leave_n_axes=1)
    #     neigh = neigh.reshape(-1, 3)
    #     neigh_mask = neigh_mask.reshape(-1)
    #     return points, neigh[np.where(neigh_mask)]

    # os.makedirs(direc, exist_ok=True)
    # cells = num_divisions ** 3
    # for j, angle in tqdm(list(enumerate(np.linspace(0, 270, cells * 2)))):
    #     cell = j // 4
    #     ix = mg[cell]
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection="3d")
    #     points, neigh = get_pts(ix)
    #     ax.scatter(*all_.T, alpha=0.1, c="grey")
    #     ax.scatter(*neigh.T, alpha=0.3, c="orange")
    #     ax.scatter(*points.T, alpha=0.5, c="red")
    #     ax.view_init(40, angle)
    #     plt.savefig(f"{direc}/plot_{j}.png", dpi=500)
    #     plt.clf()


def test_spatial_hash_2d():
    n = 40
    points = np.random.rand(n, 2)
    mask = np.ones(n)
    b, bm, c = spatial_hash(
        points,
        mask,
        num_dimensions=2,
        num_divisions=[3, 2],
        n_streams=[4, 2],
        buffer_factors=[1.5, 2.],
        box_size=[1., 1.],
    )
    for row, m in zip(b, bm):
        for col, mm in zip(row, m):
            plt.scatter(*col[mm==1].T)
    plt.show()


def test_spatial_hash_1d():
    n = 40
    points = np.random.rand(n, 1)
    mask = np.ones(n)
    b, bm, c = spatial_hash(
        points,
        mask,
        num_dimensions=1,
        num_divisions=[3,],
        n_streams=[4,],
        buffer_factors=[1.5,],
        box_size=[1.,],
    )
    for row, m in zip(b, bm):
        x = row[m == 1]
        plt.scatter(x, np.zeros_like(x))
    plt.show()


def test_ndim_lattice():
    """Also make sure that the Gaussian width is much smaller than the voxel size."""
    import hydra

    spatial_dim = 3
    n = 100
    points = np.random.rand(n, spatial_dim)
    mask = np.ones(n)
    box_length = 1
    with hydra.initialize(config_path="../config", job_name="run_enc_dec"):
        config = hydra.compose(config_name="shared.yaml", overrides=[]).encoder
    t1 = time.time()
    lattice = points_2_lattice(points, mask, config, box_length, spatial_dim)
    t2 = time.time()
    cfg = config.spatial_hash
    signals = ground_truth(
        data=points,
        atom_mask=mask.astype(int),
        n_channels=cfg.number_of_atom_types,
        box_length=box_length,
        n_lattice=cfg.num_points_1d_per_bin_voxel * cfg.num_divisions,
        width=cfg.gauss_width,
        spatial_dim=spatial_dim,
    )
    t3 = time.time()
    error = np.mean(np.abs(signals - lattice))
    print(
        f"error: {error}; gt: {np.mean(np.abs(signals))}; lat: {np.mean(np.abs(lattice))}  "
        f"hash time: {t2-t1}s; gt time: {t3-t2}s"
    )
    # print(signals.shape)
    # print(lattice.shape)
    # si = signals[..., 0]
    # la = lattice[..., 0]
    # dd = (si.reshape(-1)[None, :] - la.reshape(-1)[:, None]) ** 2
    # close = dd < 1e-6
    # plt.matshow(close)
    # plt.show()
    # import pdb;pdb.set_trace()
    if spatial_dim <= 2:
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        ax1.matshow(np.atleast_2d(signals[..., 0]))
        ax1.set_title("ground truth")
        ax2.matshow(np.atleast_2d(lattice[..., 0]))
        ax2.set_title("hash")
        plt.show()


if __name__ == "__main__":
    # test_spatial_hash_2d()
    # test_ndim_lattice()
    # test_streams3d()
    # test_lattice(direc="images/test_lattice")
    test_neighbours(direc="images/test_neighbours")
