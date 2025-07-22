from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
from md_encoder.atom_modules.image_conv_ndim import (
    compute_padding,
    conv_forward,
    conv_transpose_forward,
    default_kernel_init,
)


class ConvLayer(hk.Module):
    def __init__(self, kernel_shape, conv_channels, window_stride, spatial_dims):
        super(ConvLayer, self).__init__()
        self.kernel_shape = kernel_shape
        self.conv_channels = conv_channels
        self.window_stride = window_stride
        self.spatial_dims = spatial_dims

    def __call__(self, x):
        in_features = x.shape[-1]
        full_kernel_shape = self.kernel_shape + (in_features, self.conv_channels)

        padding = 'CIRCULAR'
        padding_lax = compute_padding(
            padding, self.kernel_shape, (1,) * self.spatial_dims, x
        )

        conv = partial(
            conv_forward,
            num_kernel_dims=self.spatial_dims,
            strides=self.window_stride,
            padding_lax=padding_lax,
        )

        kernel = hk.get_parameter(
            name="kernel",
            shape=full_kernel_shape,
            init=default_kernel_init,
        )
        # kernel = default_kernel_init(key, full_kernel_shape, jnp.float32)
        # bias = zeros

        y = conv(x, kernel)
        return y


def create_update(n_layers, model):
    def get_single_comp_graph(x, params, rng, depth):
        layers = range(depth)
        for l in layers:
            layer_params = params[l]
            x = model[l].conv_forward(layer_params["encoder"], rng, x)
        z = x
        for l in reversed(layers):
            layer_params = params[l]
            x = model[l].conv_transpose(layer_params["decoder"], rng, x)
        return x, z

    def get_single_update(x, params, rng, depth):
        def loss_fn(params, rng, x):
            recon, enc = get_single_comp_graph(x, params, rng, depth)
            loss = jnp.mean((x - recon) ** 2, axis=(0, 1, 2)).sum()
            return loss, (enc, recon)
        grad_fn = jax.grad(loss_fn, argnums=0, has_aux=True)
        grad, (enc, recon) = grad_fn(params, rng, x)
        
        return params

    updates = [partial(get_update_single, depth=l) for l in range(1, n_layers + 1)]

    # have a dict mapping depth to encoder, decoder, and update function
    depth[1] = (encoder_layer, decoder_layer, update, param_partition)
    return updates, encoder_decoder_pairs