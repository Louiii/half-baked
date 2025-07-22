from functools import partial

import haiku as hk
from md_encoder.atom_modules.modules import ndim_spatial_hash_to_lattice
from md_encoder.atom_modules.simple_encoder_decoder import (
    spatial_attention_encoder_layer,
)
from md_encoder.atom_modules.spatial_datastructure_parallel import spatial_hash


def points_2_lattice(points, mask, config, box_length, spatial_dims=3):
    """[N_atoms, 3] -> [N_dlat, N_dlat, N_dlat, channels]"""
    cfg = config.spatial_hash
    buffer_, buffer_mask, counts = spatial_hash(
        points,
        mask,
        num_dimensions=spatial_dims,
        num_divisions=cfg.num_divisions,
        n_streams=cfg.n_streams,
        buffer_factors=cfg.buffer_factor,
        box_size=box_length,
    )
    lost_pts = mask.sum() - buffer_mask.sum()
    # print(f"Lost points: {int(lost_pts)}")
    lattice = ndim_spatial_hash_to_lattice(
        coords=buffer_,
        atom_mask=buffer_mask,
        domain=box_length,
        num_points_1d_per_bin_voxel=cfg.num_points_1d_per_bin_voxel,
        gauss_width=cfg.gauss_width,
        number_of_atom_types=cfg.number_of_atom_types,
    )
    return lattice, lost_pts


def prepare_encoder(config, box_length, spatial_dims):
    encoder = partial(
        spatial_attention_encoder_layer,
        encoder_config=config,
    )
    p2l = partial(
        points_2_lattice,
        config=config,
        box_length=box_length,
        spatial_dims=spatial_dims,
    )

    def encoder_fn(points, mask):
        """[N_atoms, 3] -> [N_lat, N_lat, N_lat, channels]"""
        lattice = p2l(points, mask)
        return encoder(lattice)
    return hk.transform(encoder_fn)


if __name__ == '__main__':
    import hydra
    import numpy as np
    n = 40
    points = np.random.rand(n, 3)
    mask = np.ones(n)
    box_length = 1
    with hydra.initialize(config_path="../config", job_name="run_enc_dec"):
        config = hydra.compose(config_name="shared.yaml", overrides=[])
    points_2_lattice(points, mask, config.encoder, box_length)
