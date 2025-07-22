import haiku as hk
import hydra
import jax
import jax.numpy as jnp
from md_encoder.atom_modules.decoder import AtomDecoder
from md_encoder.atom_modules.encoder_functions import prepare_encoder
from md_encoder.utils import termplot

if __name__ == "__main__":
    box_length = 1.
    box_size = jnp.array([box_length, box_length, box_length])
    num_atoms = 40
    is_training = True

    with hydra.initialize(config_path="../config", job_name="run_enc_dec"):
        config = hydra.compose(config_name="shared.yaml", overrides=[])

    encoder_fn = prepare_encoder(config.encoder, box_length)

    def decoder_fn(enc_params, latent_enc):
        """[N_lat, N_lat, N_lat, channels] -> [N_atoms, 3]"""
        return AtomDecoder(
            config.decoder, encoder_fn.apply
        )(enc_params, latent_enc, box_size, num_atoms, is_training=is_training)

    decoder_fn = hk.transform(decoder_fn)

    def reconstruction_loss(params, points, enc_key, dec_key):
        mask = jnp.ones(points.shape[0])
        latent = encoder_fn.apply(params["encoder"], enc_key, points, mask)
        recon = decoder_fn.apply(params["decoder"], dec_key, params["encoder"], latent)
        target = points_2_lattice(points, mask)
        recon_lattice = points_2_lattice(recon, mask)
        err_accumulated_channel = ((recon_lattice - target) ** 2).sum(-1)
        lattice_agg = jnp.mean
        loss = lattice_agg(err_accumulated_channel.reshape(-1))
        return loss

    points = jax.random.uniform(jax.random.PRNGKey(seed=0), (num_atoms, 3))

    key = jax.random.PRNGKey(seed=0)
    enc_params = encoder_fn.init(key, points, jnp.ones(points.shape[0]))
    enc_key, dec_key = jax.random.PRNGKey(seed=0), jax.random.PRNGKey(seed=0)

    latent = encoder_fn.apply(enc_params, key, points, jnp.ones(points.shape[0]))
    dec_params = decoder_fn.init(key, enc_params, latent)

    params = {"encoder": enc_params, "decoder": dec_params}

    grad_fn = jax.value_and_grad(reconstruction_loss, argnums=0)

    steps = 10
    lr = 0.001
    for t in range(steps):
        value, grads = grad_fn(params, points, enc_key, dec_key)
        params = jax.tree_util.tree_map(lambda x, y: x - y * lr, params, grads)
        import pdb; pdb.set_trace()

    # key = jax.random.PRNGKey(seed=0)
    # fwd = hk.transform(lambda x: decoder(encoder(x)))
    # lr = 3e-2

    # s = 10
    # data = jax.random.normal(key, shape=(s, s, s, encoder_config.channels))
    # params = fwd.init(key, data)
    # # z = fwd.apply(params, key, data)
    # param_path = "tparams.npz"
    # possible_path = input("load from checkpoint? Give path or 'N'.")
    # if possible_path != "N":
    #     assert os.path.isfile(possible_path)
    #     params = load(possible_path)
    #     param_path = possible_path
    # pbar = tqdm(range(1, 10_001))
    # for t in pbar:
    #     l, params = overfit(params, key, data)
    #     pbar.set_description(f"{l.item():.8f}")
    #     if t % 100 == 0:
    #         save(param_path, params)
