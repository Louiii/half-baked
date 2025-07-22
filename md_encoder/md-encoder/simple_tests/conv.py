import jax
import jax.numpy as jnp

# n, c, h, w = 2, 32, 100, 100
# o, i = 32, 32
window_stride = (2, 2)

# conv4 3x3 window strides=2
batch = 1
kernel_width = 3
kernel_height = 3
image_channels = 1
conv_channels = 10
x = jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch, 32, 32, image_channels))
k = jax.random.normal(
    key=jax.random.PRNGKey(0),
    shape=(kernel_width, kernel_height, image_channels, conv_channels),
)

def conv4(x, k):
    return jax.lax.conv_general_dilated(
        x,
        k,
        window_strides=window_stride,
        padding='SAME',
        dimension_numbers=jax.lax.ConvDimensionNumbers((0,3,1,2), (3,2,0,1), (0,3,1,2))
    )


def conv_transpose(y, k):
    return jax.lax.conv_transpose(
        y,  # a rank n+2 dimensional input array.
        k,  # a rank n+2 dimensional array of kernel weights.
        window_stride,  # sequence of n integers, sets fractional stride.
        padding="SAME",
        rhs_dilation=None,
        dimension_numbers=jax.lax.ConvDimensionNumbers((0,3,1,2), (3,2,0,1), (0,3,1,2)),
        transpose_kernel=True,
        precision=None,
        preferred_element_type=None,
    )


def f(x, k):
    y = conv4(x, k)
    recon = conv_transpose(y, k)
    return jnp.mean((x - recon) ** 2, axis=(0, 1, 2)).sum()


df_dk = jax.jit(jax.grad(f, argnums=1))
for t in range(10_000):
    k -= 3e-3 * df_dk(x, k)
    if t % 10 == 0:
        print(f(x, k))
# print(y.shape)
# print(x.shape)
# print(x-recon)
