from functools import partial
from typing import Any, Iterable, Optional, Tuple, Union

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import lecun_normal
from md_encoder.atom_modules.encoder_functions import points_2_lattice
from tqdm import tqdm

Dtype = Any
Array = Any

default_kernel_init = lecun_normal()


def _conv_dimension_numbers(input_shape):
    """Computes the dimension numbers based on the input shape."""
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


def compute_padding(padding, kernel_size, kernel_dilation, inputs):
    """_summary_

    Args:
      padding: either the string `'SAME'`, the string `'VALID'`, the string
        `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
        high)` integer pairs that give the padding to apply before and after each
        spatial dimension.
      kernel_size: shape of the convolutional kernel. For 1D convolution,
        the kernel size can be passed as an integer. For all other cases, it must
        be a sequence of integers.
      kernel_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel (default: 1). Convolution with kernel dilation
        is also known as 'atrous convolution'.

    Returns:
        _type_: _description_
    """
    if padding == 'CIRCULAR':
        kernel_size_dilated = [
            (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
        ]
        pads = [(0, 0)] + [
            ((k - 1) // 2, k // 2) for k in kernel_size_dilated
        ] + [(0, 0)]
        inputs = jnp.pad(inputs, pads, mode='wrap')
        padding_lax = 'VALID'
    else:
        padding_lax = padding
    return padding_lax


def conv_forward(
    inputs: Array,
    kernel,
    num_kernel_dims: Iterable[int],
    padding_lax,
    strides: Union[None, int, Iterable[int]] = 1,
    input_dilation: Union[None, int, Iterable[int]] = 1,
    kernel_dilation: Union[None, int, Iterable[int]] = 1,
    feature_group_count: int = 1,
    dtype: Dtype = jnp.float32,
    precision: Any = None,
) -> Array:
    """Applies a convolution to the inputs.

    Args:
      inputs: input data with dimensions (batch, spatial_dims..., features).
        This is the channels-last convention, i.e. NHWC for a 2d convolution
        and NDHWC for a 3D convolution. Note: this is different from the input
        convention used by `lax.conv_general_dilated`, which puts the spatial
        dimensions last.
      kernel:
      num_kernel_dims:
      strides: an integer or a sequence of `n` integers, representing the
        inter-window strides (default: 1).
      input_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of `inputs` (default: 1).
        Convolution with input dilation `d` is equivalent to transposed
        convolution with stride `d`.
      kernel_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel (default: 1). Convolution with kernel dilation
        is also known as 'atrous convolution'.
      feature_group_count: integer, default 1. If specified divides the input
        features into groups.
      dtype: the dtype of the computation (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.

    Returns:
    The convolved data.
    """

    inputs = jnp.asarray(inputs, dtype)

    def maybe_broadcast(x):
        if x is None:
            # backward compatibility with using None as sentinel for
            # broadcast 1
            x = 1
        if isinstance(x, int):
            return (x,) * num_kernel_dims
        return x

    is_single_input = False
    if inputs.ndim == num_kernel_dims + 1:
        is_single_input = True
        inputs = jnp.expand_dims(inputs, axis=0)

    strides = maybe_broadcast(strides)  # strides or (1,) * (inputs.ndim - 2)
    input_dilation = maybe_broadcast(input_dilation)
    kernel_dilation = maybe_broadcast(kernel_dilation)

    dimension_numbers = _conv_dimension_numbers(inputs.shape)
    y = jax.lax.conv_general_dilated(
        inputs,
        kernel,
        strides,
        padding_lax,
        lhs_dilation=input_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        precision=precision,
    )

    if is_single_input:
        y = jnp.squeeze(y, axis=0)
    # if use_bias:
    #     bias = param('bias', bias_init, (features,), param_dtype)
    #     bias = jnp.asarray(bias, dtype)
    #     y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y


def conv_transpose_forward(
    inputs: Array,
    kernel,
    num_kernel_dims,
    strides: Optional[Iterable[int]] = None,
    padding: Union[str, Iterable[Tuple[int, int]]] = 'SAME',
    kernel_dilation: Optional[Iterable[int]] = None,
    dtype: Dtype = jnp.float32,
    precision: Any = None,
) -> Array:
    """Applies a transposed convolution to the inputs. Behaviour mirrors of
    `jax.lax.conv_transpose`.

    Args:
      inputs: input data with dimensions (batch, spatial_dims..., features).
        This is the channels-last convention, i.e. NHWC for a 2d convolution
        and NDHWC for a 3D convolution. Note: this is different from the input
        convention used by `lax.conv_general_dilated`, which puts the spatial
        dimensions last.
      kernel:
      num_kernel_dims:
      strides: a sequence of `n` integers, representing the inter-window strides.
      padding: either the string `'SAME'`, the string `'VALID'`, the string
        `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
        high)` integer pairs that give the padding to apply before and after each
        spatial dimension.
      kernel_dilation: `None`, or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel. Convolution with kernel dilation is also known as 'atrous
        convolution'.
      dtype: the dtype of the computation (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.

    Returns:
      The convolved data.
    """
    inputs = jnp.asarray(inputs, dtype)

    is_single_input = False
    if inputs.ndim == num_kernel_dims + 1:
        is_single_input = True
        inputs = jnp.expand_dims(inputs, axis=0)

    strides = strides or (1,) * (inputs.ndim - 2)

    if padding == 'CIRCULAR':
        padding_lax = 'VALID'
    else:
        padding_lax = padding

    y = jax.lax.conv_transpose(
        inputs,
        kernel,
        strides,
        padding_lax,
        rhs_dilation=kernel_dilation,
        precision=precision,
    )

    if padding == "CIRCULAR":
        # For circular padding, we need to identify the size of the final output
        # ("period") along each spatial dimension, pad each dimension to an
        # integer number of periods, and wrap the array periodically around each
        # dimension. Padding should be done in such a way that the start of the
        # original input data inside the padded array is located at integer
        # number of periods - otherwise the result would be circularly shifted.

        # Compute period along each spatial dimension - it's input size scaled
        # by the stride.
        scaled_x_dims = [
            x_dim * stride for x_dim, stride in zip(inputs.shape[1:-1], strides)
        ]
        # Compute difference between the current size of y and the final output
        # size, and complement this difference to 2 * period - that gives how
        # much we need to pad.
        size_diffs = [
            -(y_dim - x_dim) % (2 * x_dim)
            for y_dim, x_dim in zip(y.shape[1:-1], scaled_x_dims)
        ]
        # Divide the padding equaly between left and right. The choice to put
        # "+1" on the left (and not on the right) represents a convention for
        # aligning even-sized kernels.
        total_pad = [((size_diff + 1) // 2, size_diff // 2) for size_diff in size_diffs]
        y = jnp.pad(y, [(0, 0)] + total_pad + [(0, 0)])
        # Wrap the result periodically around each spatial dimension,
        # one by one.
        for i in range(1, y.ndim - 1):
            y = y.reshape(y.shape[:i] + (-1, scaled_x_dims[i - 1]) + y.shape[i + 1:])
            y = y.sum(axis=i)

    if is_single_input:
        y = jnp.squeeze(y, axis=0)
    # if self.use_bias:
    #     bias = self.param('bias', self.bias_init, (self.features,), self.param_dtype)
    #     bias = jnp.asarray(bias, self.dtype)
    #     y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y


if __name__ == "__main__":
    spatial_dims = 2  # 3
    steps = 100_000
    lr = 3e-3
    window_stride = (2,) * spatial_dims
    kernel_shape = (3,) * spatial_dims
    conv_channels = 128

    box_length = 1.
    num_atoms = 40
    points = jax.random.uniform(jax.random.PRNGKey(seed=0), (num_atoms, spatial_dims))

    key = jax.random.PRNGKey(seed=0)

    with hydra.initialize(config_path="../config", job_name="run_enc_dec"):
        config = hydra.compose(config_name="shared.yaml", overrides=[])

    x = points_2_lattice(
        points, jnp.ones(points.shape[0]), config.encoder, box_length, spatial_dims
    )

    print(f"x.shape: {x.shape}, spatial-dim-prod: {np.prod(x.shape[:3])}")
    batch = 1
    x = x[None]  # batch, spatial_x, spatial_y, spatial_z, image_channels
    image_channels = x.shape[-1]

    in_features = image_channels
    full_kernel_shape = kernel_shape + (image_channels, conv_channels)

    padding = 'SAME'
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

    print(f"y.shape: {y.shape}, spatial-dim-prod: {np.prod(y.shape[1:4])}")

    conv_transpose = partial(
        conv_transpose_forward,
        num_kernel_dims=spatial_dims,
        strides=window_stride,
        padding=padding,
    )

    k = kernel

    def f(x, k):
        y = conv(x, k)
        kt = jnp.swapaxes(k, -2, -1)
        recon = conv_transpose(y, kt)
        return jnp.mean((x - recon) ** 2, axis=(0, 1, 2)).sum()

    df_dk = jax.jit(jax.grad(f, argnums=1))
    for t in tqdm(range(steps)):
        k -= lr * df_dk(x, k)
        if t % 100 == 0:
            print(f(x, k))
        if t == int(steps / 2):
            lr *= 0.5
