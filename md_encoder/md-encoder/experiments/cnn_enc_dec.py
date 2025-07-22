import os
from functools import partial
from time import time

import flax
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from flax import linen as nn
from jax import random
from md_encoder.atom_modules.encoder_functions import points_2_lattice
from tqdm import tqdm

import param_io

print("Device:", jax.devices()[0])

class Encoder(nn.Module):
    depth : int
    c_hid : list
    latent_dims : list
    nonlin_last: bool
    kernel_widths : list
    kernel_strides : list
    spatial_dims : int

    def _conv(self, x, layer):
        c = nn.Conv(
            features=self.c_hid[layer],
            kernel_size=(self.kernel_widths[layer],) * self.spatial_dims,
            strides=(self.kernel_strides[layer],) * self.spatial_dims,
            name=f"layer{layer + 1}",
        )(x)
        a = nn.gelu(c)
        a = nn.Dense(features=self.c_hid[layer] * 4, name=f"post_conv_layer4x_{layer + 1}")(a)
        a = nn.relu(a)
        a = nn.Dense(features=self.c_hid[layer], name=f"post_conv_layer_{layer + 1}")(a)
        return a

    def _mlp(self, act, _layer, _pre="", latent_dims=None, nl_last=None):
        dims = self.latent_dims if latent_dims is None else latent_dims
        nonlin_last = self.nonlin_last if nl_last is None else nl_last
        for i, d in enumerate(dims, start=1):
            act = nn.Dense(features=d, name=f"{_pre}mlp_out_layer_{i}")(act)
            if i == len(dims) and not nonlin_last:
                pass
            else:
                act = nn.gelu(act)
        return act

    def atn(self, act, _i):
        # the next thing to try would be self attend without reduction, followed by reduction
        pos_channels = 32
        # x: [b, x, y, c]
        b, x, y, c = act.shape
        w = self.kernel_widths[2]
        act = act.reshape(b, w, x // w, w, y // w, c)
        query_act = act[:, w // 2, :, w // 2, :, :]
        positional_enc = self.param(
            "pos_enc",
            flax.linen.initializers.lecun_normal(),
            (w, w, pos_channels)
        )
        pos = positional_enc[None, :, None, :, None, :]
        memory = jnp.concatenate(
            [act, pos.repeat(b, axis=0).repeat(x // w, axis=2).repeat(y // w, axis=4)],
            axis=-1
        )
        # query_act: [b, x, y, c]
        # memory: [b, X, x, Y, y, c]
        memory = jnp.moveaxis(
            memory,
            source=(0, 2, 4, 1, 3, 5),
            destination=(0, 1, 2, 3, 4, 5),
        )
        memory = memory.reshape(b, x // w, y // w, w * w, c + pos_channels)
        # query_act: [b, x, y, c]
        # memory: [b, x, y, mem, c]
        keydim = 32
        valuedim = 128
        query = self._mlp(
            query_act,
            _layer=None,
            _pre="query_",
            latent_dims=[256, keydim],
            nl_last=False,
        )
        keys = self._mlp(
            memory,
            _layer=None,
            _pre="key_",
            latent_dims=[256, keydim],
            nl_last=False,
        )
        values = self._mlp(
            memory,
            _layer=None,
            _pre="value_",
            latent_dims=[256, valuedim],
            nl_last=False,
        )
        qk = jnp.einsum("...c,...mc->...m", query, keys) / w
        # qk: [b, x, y, mem]
        # values: [b, x, y, mem, C]
        qk = jax.nn.softmax(qk, axis=-1)
        weighted_values = jnp.einsum("...m,...mc->...c", qk, values)
        # weighted_values: [b, x, y, C]
        outdim = self.c_hid[2]
        out = self._mlp(
            weighted_values,
            _layer=None,
            _pre="out_",
            latent_dims=[256, outdim],
            nl_last=False,
        )
        return out

    def __getitem__(self, i):
        return {
            0: self._conv,
            1: self._conv,
            2: self.atn,#self._conv,
            3: self._mlp,
        }[i]

    @nn.compact
    def __call__(self, x, shallow=0, deep=None):
        if deep is None:
            deep = self.depth
        acts = [x]
        for i in range(shallow, deep):
            acts.append(self[i](acts[-1], i))
        return acts

class Decoder(nn.Module):
    depth : int
    c_hid : list
    latent_dims : list
    kernel_widths : list
    kernel_strides : list
    spatial_dims : int

    def _mlp_inv(self, x):
        x = nn.Dense(features=self.latent_dims[-1], name="mlp_inv")(x)
        x = nn.gelu(x)
        return x

    # def _layer3_inv(self, x):
    #     x = nn.ConvTranspose(
    #         features=self.c_hid[2],
    #         kernel_size=(self.kernel_widths[2],) * self.spatial_dims,
    #         strides=(self.kernel_strides[2],) * self.spatial_dims,
    #         name="_layer3_inv_conv_transpose",
    #     )(x)
    #     x = nn.gelu(x)
    #     x = nn.Conv(
    #         features=self.c_hid[2],
    #         kernel_size=(self.kernel_widths[2],) * self.spatial_dims,
    #         name="_layer3_inv_conv",
    #     )(x)
    #     x = nn.gelu(x)
    #     return x

    def _layer2_inv(self, x):
        x = nn.ConvTranspose(
            features=self.c_hid[1],
            kernel_size=(self.kernel_widths[1],) * self.spatial_dims,
            strides=(self.kernel_strides[1],) * self.spatial_dims,
            name="_layer2_inv_conv_transpose",
        )(x)
        x = nn.gelu(x)
        x = nn.Conv(
            features=self.c_hid[1],
            kernel_size=(self.kernel_widths[1],) * self.spatial_dims,
            name="_layer2_inv_conv",
        )(x)
        x = nn.gelu(x)
        return x

    def _layer1_inv(self, x):
        x = nn.ConvTranspose(
            features=self.c_hid[0],
            kernel_size=(self.kernel_widths[0],) * self.spatial_dims,
            strides=(self.kernel_strides[0],) * self.spatial_dims,
            name="_layer1_inv_conv",
        )(x)
        x = nn.tanh(x)
        return x

    def _layer3_inv(self, x):
        """This is just separate MLPs for each new item to expand to."""
        # x: [b, x, y, c]
        assert self.kernel_strides[2] == self.kernel_widths[2]
        num_expand_x = self.kernel_strides[2]
        num_expand_y = self.kernel_strides[2]
        # multi_lin = self.param(
        #     f"expand_lin_{i}",
        #     flax.linen.initializers.lecun_normal(),
        #     (num_expand_x, num_expand_y, prev_channels, new_channels)
        # )
        # multi_bias = self.param(
        #     f"expand_bias_{i}",
        #     lambda _key, shape: jnp.zeros(shape),
        #     (num_expand_x, num_expand_y, new_channels)
        # )
        # def go(x):
        #     act = jnp.einsum("bxyc,nmch->bxnymh", x, multi_lin)
        #     act += multi_bias[None, None, :, None, :, :]
        #     return act
        def create_multi_linear(i, prev_channels, new_channels):
            multi_lin = self.param(
                f"expand_lin_{i}",
                flax.linen.initializers.lecun_normal(),
                (num_expand_x, num_expand_y, prev_channels, new_channels)
            )
            multi_bias = self.param(
                f"expand_bias_{i}",
                lambda _key, shape: jnp.zeros(shape),
                (num_expand_x, num_expand_y, new_channels)
            )
            def go(x, expand):
                einstr = "bxyc,nmch->bxnymh" if expand else "bxnymc,nmch->bxnymh"
                print(f"x: {x.shape}; multi_lin: {multi_lin.shape}")
                act = jnp.einsum(einstr, x, multi_lin)
                print(f"act: {act.shape}")
                act += multi_bias[None, None, :, None, :, :]
                return act
            return go

        channels = [x.shape[-1], 128, 256, self.c_hid[2]]
        act = x
        for i, (p, n) in enumerate(zip(channels[:-1], channels[1:])):
            act = create_multi_linear(i, p, n)(act, i == 0)
            act = nn.gelu(act)
        _b, _x, _n, _y, _m, _h = act.shape
        act = act.reshape(_b, _x * _n, _y * _m, _h)
        return act

    def __getitem__(self, i):
        return {
            0: self._layer1_inv,
            1: self._layer2_inv,
            2: self._layer3_inv,
            3: self._mlp_inv,
        }[i]

    @nn.compact
    def __call__(self, z, shallow=0, deep=None):
        if deep is None:
            deep = self.depth
        acts = [z]
        for i in reversed(range(shallow, deep)):
            acts.append(self[i](acts[-1]))
        return acts[::-1]


def update_layers(zs, shallow, deep, intermediate_losses=False, lr=3e-4):
    _dp = decoder.init(rng, zs[deep], shallow=shallow, deep=deep)['params']
    # _ep = encoder.init(rng, zs[0], shallow=0, deep=deep)['params']
    # _zs = encoder.apply({'params': _ep}, zs[0], shallow=0, deep=deep)
    relevant = encoder.init(rng, zs[shallow], shallow=shallow, deep=deep)['params'].keys()

    # .init(rng, z, start=start)['params']
    def _loss(params, x):
        zs = encoder.apply({'params': params["encoder"]}, x, shallow=0, deep=deep)
        emb = zs[deep]
        # print(f"dp.keys(): {_dp.keys()}")
        # print(f"params['decoder'].keys(): {params['decoder'].keys()}")
        recs = decoder.apply({'params': {k: params["decoder"][k] for k in _dp}}, emb, shallow=shallow, deep=deep)
        recs = [None] * shallow + recs
        # assert len(recs) == len(zs), f"len(recs) : {len(recs)}, len(zs) : {len(zs)}"
        if intermediate_losses:
            return sum(jnp.mean((recs[i] - zs[i]) ** 2) for i in range(shallow, deep))
        else:
            return jnp.mean((recs[shallow] - zs[shallow]) ** 2)
    opt = optax.adam(learning_rate=lr, b1=0.9, b2=0.999)
    def _extract(data):
        return {"decoder": data["decoder"], "encoder": {k: data["encoder"][k] for k in relevant}}
    def _update(x, params, opt_state):
        loss, grad = jax.value_and_grad(_loss)(params, x)
        # don't update the start encoder params
        subset_p = _extract(params)
        subset_g = _extract(grad)
        updates, opt_state = opt.update(subset_g, opt_state)
        new_params = optax.apply_updates(subset_p, updates)
        for k, v in params["encoder"].items():
            if k not in relevant:
                new_params["encoder"][k] = v
        return new_params, opt_state, loss
    return jax.jit(_update), lambda p: opt.init(_extract(p))


def prepare_dataloader(precompute):
    data = np.load('/Users/louisrobinson/Desktop/md/jaxmd_plot/npt_nvt_full_chains.npz')
    trajectory = data["log_position"]

    # data = np.load("../artificial_data/npt_nvt_full_chains.npy", allow_pickle=True).item()
    chain_type = data['molecule_type']
    colours = data["colors"]
    c2i = {tuple(c): i for i, c in enumerate(np.unique(colours, axis=0), start=1)}
    atom_types = np.array([c2i[tuple(c.tolist())] for c in colours])
    bonds = data["bonds"]
    bonds_for_angle = data["bonds_for_angle"]
    # trajectory = data["log"]["position"]
    box_length = 56.377

    with hydra.initialize(config_path="../config", job_name="run_enc_dec"):
        config = hydra.compose(config_name="worm_cnn.yaml", overrides=[])

    ptl = jax.jit(
        partial(
            points_2_lattice,
            mask=atom_types,
            config=config.encoder,
            box_length=box_length,
            spatial_dims=spatial_dims,
        )
    )
    batch_size = 4

    if not precompute:
        class Dataloader:
            def __iter__(self):
                ixs = np.arange(len(trajectory))
                # np.random.shuffle(ixs)
                # for i in range(len(ixs) // self.batch_size):
                while True:
                    np.random.shuffle(ixs)
                    for i in range(len(ixs) // batch_size):
                        start_time = time()
                        batch = np.stack([
                            ptl(trajectory[j])[0]
                            for j in ixs[i * batch_size : (i + 1) * batch_size]
                        ])
                        yield batch, time() - start_time

        dl = iter(Dataloader())
    else:
        cs = config.encoder.spatial_hash
        cache_path = f"precomputed_lattices_pbv{cs.num_points_1d_per_bin_voxel}_nd{cs.num_divisions}.npy"
        if os.path.isfile(cache_path):
            lattices = np.load(cache_path)
        else:
            print(f"creating cache: {cache_path}")
            ls = []
            lost_ixs = []
            for ix, pts in tqdm(enumerate(trajectory), total=len(trajectory)):
                lat, lost = ptl(pts)
                ls.append(lat)
                if lost > 0:
                    print((lost, ix))
                    lost_ixs.append(ix)

            lattices = np.stack(ls)
            np.save(cache_path, lattices, allow_pickle=False)
        def _dl():
            while True:
                ixs = np.arange(len(trajectory))
                np.random.shuffle(ixs)
                for i in range(len(ixs) // batch_size):
                    yield lattices[i * batch_size : (i + 1) * batch_size], 0.
        dl = _dl()
    return dl, config


def default(zs, params):
    shallow_upd, sh_init = update_layers(zs, shallow=0, deep=1, lr=3e-4)
    # core_upd, init = update_layers(zs, shallow=1, deep=3, intermediate_losses=True,
    # lr=1e-5)
    mid_upd, md_init = update_layers(zs, shallow=1, deep=2, intermediate_losses=False, lr=3e-4)
    core_upd, cr_init = update_layers(zs, shallow=2, deep=3, intermediate_losses=False, lr=1e-3)

    return {
        "shallow": {"num": 5, "upd": shallow_upd, "opt": sh_init(params)},
        "middle": {"num": 1, "upd": mid_upd, "opt": md_init(params)},
        "core": {"num": 1, "upd": core_upd, "opt": cr_init(params)},
    }


def uproot(zs, params):
    shallow_upd, sh_init = update_layers(zs, shallow=0, deep=1, lr=3e-4)
    # core_upd, init = update_layers(zs, shallow=1, deep=3, intermediate_losses=True,
    # lr=1e-5)
    mid_upd, md_init = update_layers(zs, shallow=0, deep=2, intermediate_losses=False, lr=3e-4)
    core_upd, cr_init = update_layers(zs, shallow=0, deep=4, intermediate_losses=False, lr=3e-4)

    return {
        "shallow": {"num": 5, "upd": shallow_upd, "opt": sh_init(params)},
        "middle": {"num": 1, "upd": mid_upd, "opt": md_init(params)},
        "core": {"num": 1, "upd": core_upd, "opt": cr_init(params)},
    }


def _perform_update(upd_fn, dl, params, opt_state):
    x, t = next(dl)
    start = time()
    # upd_fn = {"shallow": shallow_upd, "core": core_upd}[update_fn]
    params, opt_state, loss = upd_fn(x, params, opt_state)
    # print(f"({update_fn}) update time: {time() - start:.3f}")
    return params, opt_state, loss, time() - start, t


def _perform_updates(upd, dl, params, opt_state, num):
    _loss = []
    utimes = []
    btimes = []
    for _ in range(num):
        params, opt_state, l, utime, btime = _perform_update(upd, dl, params, opt_state)
        _loss.append(l)
        utimes.append(utime)
        btimes.append(btime)
    return params, opt_state, _loss, utimes, btimes


def run_updates(dl, params, update_suite):
    btimes = []
    _losses = {}
    utimes = {}
    for name, _data in update_suite.items():
        params, o, loss, utime, bt = _perform_updates(
            _data["upd"], dl, params, _data["opt"], num=_data["num"]
        )
        update_suite[name]["opt"] = o
        btimes += bt
        _losses[name] = np.mean(loss)
        utimes[name] = (np.mean(utime), np.std(utime))
    _tmp = {k: f'{a:.3f}±{b:.3f}' for k, (a, b) in utimes.items()}
    print(f"update times | {_tmp}")
    if not precompute:
        print(f"preparing batch took: {np.mean(btimes):.3f}±{np.std(btimes):.3f}(s)")
    return params, update_suite, _losses


def plot_recons(recons, target, im_losses, save=""):
    fig, axes = plt.subplots(ncols=len(recons) + 1, figsize=(15, 5))
    for d, (ax, r, l) in enumerate(zip(axes, recons, im_losses), start=1):
        ax.imshow(r)
        ax.set_title(f"Depth: {d}; loss: {l:.6f}")
        ax.set_axis_off()
    axes[-1].imshow(target)
    axes[-1].set_title("Target")
    axes[-1].set_axis_off()
    plt.tight_layout()
    plt.savefig(save, dpi=200) if save else plt.show()



rng = random.PRNGKey(0)

spatial_dims = 2  # 3
steps = 100_000
lr = 3e-3
window_stride = (3,) * spatial_dims
kernel_shape = (3,) * spatial_dims
conv_channels = 16

precompute = True
dl, config = prepare_dataloader(precompute=precompute)
x, t = next(dl)


# print(z.shape)
# print(tree_structure(eparams))

encoder = Encoder(
    depth=config.encoder.depth,
    c_hid=config.encoder.conv_channels,
    latent_dims=[64, 256, 64],
    nonlin_last=True,
    kernel_widths=config.encoder.kernel_widths,
    kernel_strides=config.encoder.kernel_strides,
    spatial_dims=2,
)
eparams = encoder.init(rng, x)['params']
zs = encoder.apply({'params': eparams}, x)
# (x, z1, z2, z3, z) = zs
decoder = Decoder(
    depth=config.encoder.depth,
    c_hid=[x.shape[-1]] + config.encoder.conv_channels[:-1],
    latent_dims=[config.encoder.conv_channels[-1]],
    kernel_widths=config.encoder.kernel_widths,
    kernel_strides=config.encoder.kernel_strides,
    spatial_dims=2,
)
dparams = decoder.init(rng, zs[-1])['params']

recs = decoder.apply({'params': dparams}, zs[-1])

print([(a.shape, b.shape) for a, b in zip(zs, recs)])


params = {"encoder": eparams, "decoder": dparams}

update_suite = default(zs, params)
# update_suite = uproot(zs, params)

savedir_base = "plots/cnn_multi"
os.makedirs(savedir_base, exist_ok=True)
name = "cnn_enc_"
savedir = os.path.join(savedir_base, name)
continue_from = 0
resume = input("resume (y/n)") in ['y', 'Y']
if not resume:
    exp = max([int(d.split("_")[-1]) for d in os.listdir(savedir_base) if d.startswith(name)] + [0]) + 1
    param_path = f"params/multicnn_params_{exp}.npz"
    os.makedirs(os.path.split(param_path)[0], exist_ok=True)
    print(f"creating new checkpoint: '{param_path}'")
    savedir = savedir + str(exp)
else:
    exp = input("Enter resume params index:")
    param_path = f"params/multicnn_params_{exp}.npz"
    if not os.path.isfile(param_path):
        print(f"could not find: '{param_path}'")
        raise FileNotFoundError
    else:
        print(f"loading {param_path}")
        params = param_io.load(param_path)
        savedir = savedir + exp
        continue_from = max([int(fn.split("img_")[1].split(".")[0]) for fn in os.listdir(savedir)]) + 1
        print(f"continuing images from {continue_from}")
os.makedirs(savedir, exist_ok=True)
movie_str = f"ffmpeg -v info -r 5 -i {savedir_base}/cnn_enc_{exp}/img_%5d.png -y {savedir_base}/enc{exp}.mp4"

image_x, _ = next(dl)

steps = 2_000
every = 20
losses = []
shallow_on = True
for _t in range(steps):
    params, update_suite, _losses = run_updates(dl, params, update_suite)
    print(f"[{_t}] loss: {jax.tree_map(lambda a: np.round(a, 5), _losses)}")
    losses.append(_losses)
    if _t % every == 0:
        param_io.save(param_path, params)
        zs = encoder.apply({'params': params["encoder"]}, image_x[:1], deep=config.encoder.depth)
        recons = []
        for d in range(1, config.encoder.depth):
            _dp = decoder.init(rng, zs[d], shallow=0, deep=d)['params']
            recs = decoder.apply({'params': {k: params["decoder"][k] for k in _dp}}, zs[d], shallow=0, deep=d)
            recons.append(recs[0][0])
        im_losses = [np.mean((image_x[0] - r) ** 2) for r in recons]
        step = _t // every
        step += continue_from
        plot_recons(recons, zs[0][0], im_losses, save=os.path.join(savedir, f"img_{step:05d}.png"))
    if shallow_on and _losses["shallow"] < 0.001:
        print("Turning off shallow updates.")
        update_suite["shallow"]["num"] = 0
        update_suite["middle"]["num"] = 10
        update_suite["core"]["num"] = 1
        shallow_on = False
    if _losses["middle"] < 0.05:
        print("Turning off middle updates.")
        update_suite["middle"]["num"] = 0
        update_suite["core"]["num"] = 5

os.system(movie_str)

df = pd.DataFrame().from_records(losses)
plt.clf()
for k in update_suite:
    plt.plot(df[k], label=k)
plt.legend()
plt.title("Losses")
plt.savefig(f"{savedir_base}/loss_{exp}.png")
