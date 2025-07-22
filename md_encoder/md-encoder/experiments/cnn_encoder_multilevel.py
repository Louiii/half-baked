import os
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from md_encoder.atom_modules.encoder_functions import points_2_lattice
from md_encoder.atom_modules.image_conv_ndim import (
    compute_padding,
    conv_forward,
    conv_transpose_forward,
    default_kernel_init,
)
from tqdm import tqdm


def encoder_levels(params, data, layers, level):
    act = data
    for i in range(level):
        act = layers[i](params[i], act)
    return act

def decoder_levels(params, data, layers, level):
    act = data
    for i in range(level, len(layers)):
        act = layers[i](key, params[i], act)
    return act

def layer(key, params, x):
    if "k" not in params:
        image_channels = x.shape[-1]
        full_kernel_shape = kernel_shape + (image_channels, conv_channels)
        key, skey = jax.random.split(key)
        params["k"] = default_kernel_init(skey, full_kernel_shape, jnp.float32)
    z = conv(x, params["k"])
    if "lin" not in params:
        key, skey = jax.random.split(key)
        params["lin"] = {
            "w": jax.random.normal(),
            "b": jnp.zeros(out_size)
        }
    return z
    # kt = jnp.swapaxes(k, -2, -1)
    # recon = conv_transpose(y, kt)
    # loss = jnp.mean((x - recon) ** 2, axis=(0, 1, 2)).sum()
    # return loss, recon


data = np.load("../artificial_data/npt_nvt_full_chains.npy", allow_pickle=True).item()


chain_type = data['molecule_type']
colours = data["colors"]
c2i = {tuple(c): i for i, c in enumerate(np.unique(colours, axis=0), start=1)}
atom_types = np.array([c2i[tuple(c.tolist())] for c in colours])
bonds = data["bonds"]
bonds_for_angle = data["bonds_for_angle"]
trajectory = data["log"]["position"]
box_length = 56.377

spatial_dims = 2  # 3
steps = 100_000
lr = 3e-3
window_stride = (3,) * spatial_dims
kernel_shape = (3,) * spatial_dims
conv_channels = 16

def activation_reduction_per_layer(image_shape, window_stride, num_layers):
    def pr(n):
        def a(i, n):
            if n > 1000:
                return a(i+1, n / 1000)
            else:
                return i, n
        i, n = a(0, n)
        n = int(n)
        return str(n) + {0: "", 1: "K", 2: "M", 3: "G", 4: "T"}[i]
    act = np.array(image_shape).astype(np.float64)
    window_stride = np.array(window_stride).astype(np.float64)
    sizes = [(np.array(act), pr(np.prod(act)))]
    for _ in range(num_layers):
        act /= window_stride
        num_spatial = np.prod(act)
        sizes.append((int(np.array(act)[0]), pr(num_spatial)))
    print(sizes)


points = trajectory[0]
# num_atoms = 40
# points = jax.random.uniform(jax.random.PRNGKey(seed=0), (num_atoms, spatial_dims))

key = jax.random.PRNGKey(seed=0)

with hydra.initialize(config_path="../config", job_name="run_enc_dec"):
    config = hydra.compose(config_name="worm_cnn.yaml", overrides=[])

x = points_2_lattice(
    points, atom_types, config.encoder, box_length, spatial_dims
)

activation_reduction_per_layer(x.shape[:-1], window_stride, num_layers=5)

import matplotlib.pyplot as plt


def pl(d, title, other=None, save=""):
    if other is None:
        print(np.max(d))
        plt.imshow(d)
        plt.title(title)
    else:
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        print((np.max(d), np.max(other)))
        ax1.imshow(d)
        ax2.imshow(other)
        plt.suptitle(title)
    plt.savefig(save) if save else plt.show()

# pl(x, "target")

print(f"x.shape: {x.shape}, spatial-dim-prod: {np.prod(x.shape[:spatial_dims])}")
batch = 1
x = x[None]  # batch, spatial_x, spatial_y, spatial_z, image_channels
# x = x[:, :30, :30, :]
image_channels = x.shape[-1]

in_features = image_channels
full_kernel_shape = kernel_shape + (image_channels, conv_channels)

padding = "VALID" #'CIRCULAR'
padding_lax = compute_padding(padding, kernel_shape, (1,) * spatial_dims, x)

conv = partial(
    conv_forward,
    num_kernel_dims=spatial_dims,
    strides=window_stride,
    padding_lax=padding_lax,
)

kernel = default_kernel_init(key, full_kernel_shape, jnp.float32)
# bias = zeros

y = conv(x, kernel)

print(f"y.shape: {y.shape}, spatial-dim-prod: {np.prod(y.shape[1:1+spatial_dims])}")

conv_transpose = partial(
    conv_transpose_forward,
    num_kernel_dims=spatial_dims,
    strides=window_stride,
    padding=padding,
)

import haiku as hk


class Conv(hk.Module):
    def __init__(self, kernel_shape, window_stride, padding="VALID"):
        super().__init__(name="conv")
        self.padding = padding # 'CIRCULAR'
        self.kernel_shape = kernel_shape
        self.window_stride = window_stride

    def __call__(self, x):
        image_channels = x.shape[-1]
        self.spatial_dims = x.ndim - 1
        # in_features = image_channels
        full_kernel_shape = self.kernel_shape + (image_channels, conv_channels)

        padding_lax = compute_padding(
            self.padding, kernel_shape, (1,) * self.spatial_dims, x
        )

        conv = partial(
            conv_forward,
            num_kernel_dims=self.spatial_dims,
            strides=self.window_stride,
            padding_lax=padding_lax,
        )
        # kernel = default_kernel_init(key, full_kernel_shape, jnp.float32)
        self.kernel = hk.get_parameter(
            name="kernel",
            shape=full_kernel_shape,
            init=jnp.zeros#default_kernel_init,
        )
        return conv(x, self.kernel)

    def transpose(self, z):
        conv_transpose = partial(
            conv_transpose_forward,
            num_kernel_dims=self.spatial_dims,
            strides=self.window_stride,
            padding=self.padding,
        )
        kt = jnp.swapaxes(self.kernel, -2, -1)
        return conv_transpose(z, kt)

def te(z):
    return hk.Linear(z.shape[-1])(jax.nn.relu(hk.Linear(z.shape[-1])(z)))
# rng = jax.random.PRNGKey(seed=0)
# tr = hk.transform(te)
# params = tr.init(rng, x)

# def forward_hk(x):
#     conv_module = Conv(kernel_shape, window_stride)
#     z = conv_module(x)
#     z = hk.Linear(z.shape[-1])(jax.relu(hk.Linear(z.shape[-1]))(z))
#     recon = conv_module.transpose(z)
#     loss = jnp.mean((x - recon) ** 2, axis=(0, 1, 2)).sum()
#     return loss

# rng = jax.random.PRNGKey(seed=0)
# tr = hk.transform(forward_hk)
# params = tr.init(rng, x)


def f_same_kernel(x, k):
    y = conv(x, k)
    kt = jnp.swapaxes(k, -2, -1)
    recon = conv_transpose(y, kt)
    loss = jnp.mean((x - recon) ** 2, axis=(0, 1, 2)).sum()
    return loss, recon


def f_diff_kernel(x, k):
    k1, k2 = k
    y = conv(x, k1)
    recon = conv_transpose(y, k2)
    loss = jnp.mean((x - recon) ** 2, axis=(0, 1, 2)).sum()
    return loss, recon

k = kernel
f = f_same_kernel
# k = (jnp.array(kernel), jnp.array(jnp.swapaxes(kernel, -2, -1)))
# f = f_diff_kernel

def inference(input_points, k):
    input_x = points_2_lattice(
        input_points, atom_types, config.encoder, box_length, spatial_dims
    )
    loss, recon = f(input_x, k)
    return loss, recon, input_x


savedir_base = "plots/single_layer_cnn"
os.makedirs(savedir_base, exist_ok=True)
name = "cnn_enc_"
savedir = os.path.join(savedir_base, name)
i = max([int(d.split("_")[-1]) for d in os.listdir(savedir_base) if d.startswith(name)] + [0])
savedir = savedir + str(i + 1)
os.makedirs(savedir, exist_ok=True)

def create_update(f):
    def update(x, k):
        grad, recon = jax.grad(f, argnums=1, has_aux=True)(x, k)
        k = jax.tree_map(lambda z, g: z - lr * g, k, grad)
        return k, recon
    return jax.jit(update)


def random_crop(key, x):
    key, s1, s2 = jax.random.split(key, num=3)
    start_x = jax.random.randint(
        minval=0, maxval=x.shape[1] - crop_size, key=s1, shape=(1,)
    )[0]
    start_y = jax.random.randint(
        minval=0, maxval=x.shape[1] - crop_size, key=s2, shape=(1,)
    )[0]
    x_crop = x[:, start_x:start_x + crop_size, start_y:start_y + crop_size, :]
    return key, x_crop


update = create_update(f)
plot_every = steps // 10
key = jax.random.PRNGKey(0)
crop_size = 30
fixed_crop = (slice(None), slice(0, crop_size), slice(0, crop_size))

def crop_fixed(key, x):
    return key, x[fixed_crop]

crop_x = crop_fixed
# crop_x = random_crop
# crop_x = lambda *a: a

for t in tqdm(range(steps)):
    key, x_crop = crop_x(key, x)
    k, recon = update(x_crop, k)
    if t % 100 == 0:
        print(f(x[fixed_crop], k)[0])
    if t % plot_every == 0:
        pl(recon[0], "recon", x_crop[0], save=os.path.join(savedir, f"step_{t//plot_every}.png"))
    if t == int(steps / 2):
        lr *= 0.2

key, skey = jax.random.split(key)
ixs = jax.random.randint(
    minval=0, maxval=trajectory.shape[0], key=skey, shape=(5,)
)
for i, j in enumerate(ixs, start=1):
    loss, recon, target = inference(trajectory[j], k)
    pl(recon, "recon", target, save=os.path.join(savedir, f"inf_{i}.png"))
