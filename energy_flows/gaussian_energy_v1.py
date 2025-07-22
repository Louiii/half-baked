import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as st
from distributions import ust_points_sampler
from matplotlib.animation import FuncAnimation
from scipy.optimize import linear_sum_assignment


def sample_data_distribution(key, size, n, m):
    return np.stack([ust_points_sampler(n=n, m=n) for _ in range(size)])


def gaussian_pdf(x, mu, std):
    r = ((x - mu)**2).sum(-1)
    const = (2 * jnp.pi * (std**2))**(-0.5*x.shape[-1])
    return jnp.exp(-0.5 * r / (std**2)) * const


def mog_pdf(x, mus, stds):
    return jax.vmap(gaussian_pdf, in_axes=(None, 0, None if stds.ndim == 0 else 0))(x, mus, stds).mean()


def gaussian_convolution(x, delta_functions, std):
    return jax.vmap(mog_pdf, in_axes=(0, None, None))(x, delta_functions, std)


def negative_log_likelihood(data):
    """Compute the negative log-likelihood of 'data' under the 2D standard normal dist.
    
    Args:
        data: np.ndarray of shape (N, 2)

    Returns:
        float: NLL
    """
    N = data.shape[0]
    # Squared norms of rows
    squared_norms = np.sum(data**2, axis=1)  # shape (N,)

    # log p(x_i) for each sample x_i under N(0, I):
    # log p(x_i) = -0.5 * ||x_i||^2 - log(2*pi)
    log_likelihoods = -0.5 * squared_norms - np.log(2 * np.pi)

    # Mean log-likelihood
    mean_log_likelihood = np.mean(log_likelihoods)

    # Negative log-likelihood
    return -mean_log_likelihood


def qq_plot_chi_square(data, out_path):
    """QQ-plot of squared norms of data vs chi-square with 2 degrees of freedom."""
    # Squared Mahalanobis distances under covariance=I, mean=0
    d_sq = np.sum(data**2, axis=1)
    d_sq_sorted = np.sort(d_sq)

    # Theoretical chi-square quantiles
    probs = (np.arange(1, len(d_sq_sorted) + 1) - 0.5) / len(d_sq_sorted)
    chi2_quantiles = st.chi2.ppf(probs, df=2)

    plt.figure(figsize=(6, 6))
    plt.plot(chi2_quantiles, d_sq_sorted, 'o', markersize=2, label='Empirical')
    plt.plot([0, max(chi2_quantiles)], [0, max(d_sq_sorted)], 'r--', label='Diagonal')
    plt.xlabel('Theoretical χ²(2) quantiles')
    plt.ylabel('Empirical squared distance')
    plt.title('Q-Q plot vs. χ²(2)')
    plt.legend()
    plt.savefig(out_path)
    plt.close()
    plt.clf()


def inv_sigmoid(x):
    return -jnp.log((1.0 / x) - 1)


def apx_inv_gaussian_cdf(x):
    return inv_sigmoid(x) / 1.7


def gaussian_energy(
    points,
    time,
    shift: float = 0.01,
    threshold: float = 0.7,
    attr_coeff: float = 0.15,
    repl_coeff: float = 20.0,
):
    """Create an energy function to repel samples from each other, but also fit them to
    a Gaussian PDF."""
    gaussian_factor = 0.05 * jnp.log(gaussian_pdf(jnp.zeros(2), points, 1.0)).sum() + 0.5 * gaussian_pdf(jnp.zeros(2), points, 1.0).sum()
    distances = ((points[:, None, :] - points[None, :, :])**2).sum(-1) + jnp.eye(len(points)) * 1e6
    repulsion = ((shift + distances) ** -1).mean()
    attraction = -attr_coeff * gaussian_factor.mean()
    return (time > threshold) * attraction + (time < threshold) * repulsion * repl_coeff


def minimise_gaussian_energy(samples, step_size=1.0, num_steps: int = 100):
    def step(x, t):
        dstd = jax.grad(gaussian_energy)(x, t)
        x -= dstd * step_size * t
        return x, (x, dstd)
    ts = jnp.linspace(min_t, max_t, num_steps, endpoint=False)
    _, (path, grad_path) = jax.lax.scan(step, init=samples, xs=ts)
    return path[-1], path, grad_path, ts


def generate_training_batch(batch_size):
    y = sample_data_distribution(key, batch_size)
    x, _ = minimise_gaussian_energy(y)
    return x, y


def animate(path, output_path, num_hist_bins=30):
    print("Writing animation.")
    num_points = path.shape[1]
    xmin, xmax = np.min(path[:, :, 0]), np.max(path[:, :, 0])
    ymin, ymax = np.min(path[:, :, 1]), np.max(path[:, :, 1])

    # Prepare the figure and subplots
    rows = 1
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    scatter_ax, hist_ax = axs

    # Scatter plot setup
    sc = scatter_ax.scatter([], [], s=10, c='blue')
    trail_lines = [scatter_ax.plot([], [], alpha=0.5, lw=1, color='blue')[0] for _ in range(num_points)]
    scatter_ax.set_xlim(xmin, xmax)
    scatter_ax.set_ylim(ymin, ymax)

    # Histogram setup
    xedges = np.linspace(xmin, xmax, num_hist_bins + 1)
    yedges = np.linspace(ymin, ymax, num_hist_bins + 1)
    X, Y = np.meshgrid(xedges, yedges)
    hist_data = np.zeros((num_hist_bins, num_hist_bins))  # Initialize empty data for the histogram
    quadmesh = hist_ax.pcolormesh(X, Y, hist_data, shading="auto", cmap="viridis")
    hist_ax.set_xlim(xmin, xmax)
    hist_ax.set_ylim(ymin, ymax)

    for ax in axs:
        ax.set_axis_off()
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)

    # Trail length
    trail_length = len(path)

    # Initialize the animation
    def init():
        sc.set_offsets(np.zeros((num_points, 2)))
        for line in trail_lines:
            line.set_data([], [])
        quadmesh.set_array(hist_data.ravel())  # Initialize histogram data as zeros
        return [sc, *trail_lines, quadmesh]

    # Update function for animation
    def update(frame):
        # Update scatter plot and trails
        sc.set_offsets(path[frame, :, :])
        for i, line in enumerate(trail_lines):
            start = max(0, frame - trail_length)
            line.set_data(path[start:frame, i, 0], path[start:frame, i, 1])

        # Update histogram
        x = path[frame, :, 0]
        y = path[frame, :, 1]
        hist, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
        quadmesh.set_array(hist.T.ravel())  # Update histogram data
        quadmesh.set_clim(0, hist.max())  # Adjust color scale

        return [sc, *trail_lines, quadmesh]

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(path), init_func=init, blit=True)

    # Save the animation as an mp4 file
    ani.save(output_path, writer='ffmpeg', fps=20)
    plt.close(fig)

    print(f"Animation saved as {output_path}.")


def plot(output_path, trajectory, ot_plan, xmin=-2.0, xmax=2.0, bins=30):
    rows = 2
    cols = 2
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6*cols, 6*rows))
    for ax in axes.flatten():
        ax.set_axis_off()
    
    # Our sample
    ax = axes[0, 0]
    ax.hist2d(*trajectory[-1].T, bins=bins, range=[[xmin, xmax], [xmin, xmax]])

    # Gaussian sample
    ax = axes[0, 1]
    ax.hist2d(*np.random.randn(2, 10000), bins=bins, range=[[xmin, xmax], [xmin, xmax]])

    # Our assignment
    ax = axes[1, 0]
    start = trajectory[0]
    end = trajectory[-1]
    x_coords = np.stack([start[:, 0], end[:, 0]])
    y_coords = np.stack([start[:, 1], end[:, 1]])
    ax.plot(x_coords, y_coords, marker="o", c="k", lw=1, alpha=0.5, ms=1)

    # # Optimal transport assignment
    ax = axes[1, 1]
    ax.plot(ot_plan[..., 0], ot_plan[..., 1], marker="o", c="k", lw=1, alpha=0.5, ms=1)

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    plt.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    grad_std = jax.grad(mog_pdf, argnums=2)
    direction = jax.vmap(jax.vmap(grad_std, in_axes=(None, None, 0)), in_axes=(0, None, None))

    key = jr.key(0)
    data_sample = sample_data_distribution(key, 1, n=10, m=12)[0]
    data_sample -= data_sample.mean(axis=0, keepdims=True)
    data_sample /= data_sample.std(axis=0, keepdims=True)
    print(f"{data_sample.shape=}")


    xmin = -2.0
    xmax = 2.0
    numx = 160
    show_n_paths = 100
    num_time = 200
    min_t = 0.01
    max_t = 1.0
    run_traversal = False
    xs = np.linspace(xmin, xmax, numx)
    stds = jnp.linspace(min_t, max_t, num_time)

    sample, path, grad_path, path_ts = minimise_gaussian_energy(data_sample + 1e-6 * jr.normal(key, data_sample.shape))


    print(f"Negative Log-Likelihood of start: {negative_log_likelihood(path[-1]):.4f}")
    print(f"Negative Log-Likelihood of end: {negative_log_likelihood(path[0]):.4f}")

    qq_plot_chi_square(path[-1], "qq_plot.png")

    # Example Mardia's Test usage:
    data_df = pd.DataFrame(path[-1], columns=["x1", "x2"])
    test_results = pg.multivariate_normality(data_df, alpha=0.05)
    print(test_results)

    subset = data_sample
    gaussian = jr.normal(jr.key(0), subset.shape)


    # Compute the cost matrix
    cost_matrix = np.linalg.norm(subset[:, None, :] - gaussian[None, :, :], axis=2)

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)


    name = "v1"
    plot(output_path=f"{name}.png", trajectory=path, ot_plan=np.stack([subset[row_ind], gaussian[col_ind]]))
    # ibreakpoint()

    data_sample = sample_data_distribution(key, 1, n=10, m=8)[0]
    # data_sample = sample_data_distribution(key, 1, n=5, m=8)[0]
    data_sample -= data_sample.mean(axis=0, keepdims=True)
    data_sample /= data_sample.std(axis=0, keepdims=True)

    sample, path, grad_path, path_ts = minimise_gaussian_energy(data_sample + 1e-6 * jr.normal(key, data_sample.shape))

    animate(path, output_path=f'{name}.mp4')
