import glob
import math
import os

import imageio
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import pingouin as pg
import scipy.stats as st
from distributions import ust_points_sampler
from jynx import PyTree, TrainState
from jynx import layers as nn
from jynx import make_train_step
from jynx.pytree import static
from matplotlib.animation import FuncAnimation
from scipy.optimize import linear_sum_assignment
from scipy.special import gamma
from tqdm import tqdm


def sample_data_distribution(key, size, n, m):
    return np.stack([ust_points_sampler(n=n, m=m) for _ in range(size)])


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
    # Squared norms of rows
    squared_norms = (data**2).sum(axis=1)  # shape (N,)

    # log p(x_i) for each sample x_i under N(0, I):
    # log p(x_i) = -0.5 * ||x_i||^2 - log(2*pi)
    log_likelihoods = -0.5 * squared_norms - jnp.log(2 * jnp.pi)

    # Mean log-likelihood
    mean_log_likelihood = log_likelihoods.mean()

    # Negative log-likelihood
    return -mean_log_likelihood


def qq_plot_chi_square(data, out_path):
    """QQ-plot of squared norms of data vs chi-square with 2 degrees of freedom."""
    # Squared Mahalanobis distances under covariance=I, mean=0
    d_sq = np.sum(data**2, axis=1)
    
    # Sort distances
    d_sq_sorted = np.sort(d_sq)

    # Theoretical chi-square quantiles
    probs = (np.arange(1, len(d_sq_sorted) + 1) - 0.5) / len(d_sq_sorted)
    chi2_quantiles = st.chi2.ppf(probs, df=2)

    plt.figure(figsize=(6, 6))
    plt.plot(chi2_quantiles, d_sq_sorted, 'o', markersize=2, label='Empirical')
    # Plot the diagonal
    plt.plot([0, max(chi2_quantiles)], [0, max(d_sq_sorted)], 'r--', label='Diagonal')
    plt.xlabel('Theoretical $\chi^2$(2) quantiles')
    plt.ylabel('Empirical squared distance')
    plt.title('Q-Q plot vs. $\chi^2$(2)')
    plt.legend()
    plt.savefig(out_path)
    plt.close()
    plt.clf()


def inv_sigmoid(x):
    return -jnp.log((1.0 / x) - 1)


def apx_inv_gaussian_cdf(x):
    return inv_sigmoid(x) / 1.7


def chi_squared_loss(x):
    d_sq = jnp.sum(x**2, axis=-1)
    d_sq_sorted = jax.vmap(lambda ix: d_sq[ix])(jax.lax.stop_gradient(jnp.argsort(d_sq)))
    return (jnp.abs(d_sq_sorted - chi2_quantiles)).mean()


def energy(
    points,
    time,
    key,
    nll_attraction: bool = False,
    use_covariance_term: bool = True,
    num_random_projections: int = 64,
    shift: float = 0.01,
    threshold: float = 0.7,
    attr_coeff: float = 20.0,
    repl_coeff: float = 20.0,
):
    """Create an energy function to repel samples from each other, but also fit them to
    a Gaussian PDF."""
    distances = (
        (points[:, None, :] - points[None, :, :])**2
    ).sum(-1) + jnp.eye(len(points)) * 1e6

    repulsion = ((shift + distances) ** -1).mean()

    if nll_attraction:
        attraction = attr_coeff * negative_log_likelihood(points)
    else:
        attraction = 0.5 * attr_coeff * chi_squared_loss(points)

    if use_covariance_term:
        residuals = points - points.mean(axis=0, keepdims=True)
        covariance = (1.0 / (points.shape[0] - 1)) * jnp.einsum("ni, nj -> ij", residuals, residuals)
        attraction += attr_coeff * jnp.abs(jnp.eye(points.shape[-1]) - covariance).sum()

    if num_random_projections:
        # This is to enforce the shape.
        directions = jr.normal(key, shape=(num_random_projections, points.shape[-1]))
        directions /= jnp.linalg.norm(directions, axis=-1, keepdims=True)
        projection = jnp.einsum("ni, pi -> pn", points - points.mean(axis=0, keepdims=True), directions)
        proj_loss = jax.vmap(chi_squared_loss)(projection[..., None]).mean()
        repulsion += proj_loss

    return (time > threshold) * attraction + (time < threshold) * repulsion * repl_coeff


def minimise_gaussian_energy(samples, step_size=1.0, num_steps: int = 100, *, key):
    def step(x, tk):
        t, key = tk
        dstd = jax.grad(energy)(x, t, key)
        x -= dstd * step_size * t
        return x, (x, dstd)
    ts = jnp.linspace(min_t, max_t, num_steps, endpoint=False)
    _, (path, grad_path) = jax.lax.scan(step, init=samples, xs=(ts, jr.split(key, num_steps)))
    return path[-1], path, grad_path, ts


def expected_nearest_neighbour_distance_at_radius(radii, num_dims, sample_size):
    """Approximate the expected distance to the nearest neighbour, given one sample
    is at radius r from the origin in an M-dimensional standard normal distribution.
    Uses the local-Poisson approximation.

    Args:
        radii : float or array-like of floats
            One or more radii (distance from the origin). 
            If an array is provided, the return is an array of the same shape.
        num_dims : int
            Dimension M of the ambient space.
        sample_size : int, optional (default=1000)
            Total sample size N (including the point at radius r).

    Returns:
        float or np.ndarray
            The approximate expected nearest-neighbor distance(s).
    """
    # Convert the input radii to a NumPy array for vectorized computation.
    r = np.atleast_1d(radii).astype(float)

    # Volume of the unit ball in M dimensions
    #     c_M = pi^(M/2) / Gamma(M/2 + 1)
    c_M = math.pi**(num_dims / 2.0) / gamma(num_dims / 2.0 + 1.0)

    # Factor from the known mean nearest-neighbor distance in an M-dim Poisson process
    #     Gamma(1 + 1/M)
    gamma_factor = gamma(1.0 + 1.0 / num_dims)

    # Evaluate the standard M-dim normal density at radius r:
    #     p(r) = (1 / (2π)^(M/2)) * exp(-r^2 / 2).
    # We'll do this elementwise for the array r.
    normal_const = (2.0 * math.pi)**(-0.5 * num_dims)
    p_r = normal_const * np.exp(-0.5 * r**2)

    # Local Poisson intensity near radius r:
    #     λ = (N - 1) * p(r)
    # Multiply by c_M for the formula of the nearest-neighbor distance in a Poisson process.
    lam_times_cM = (sample_size - 1) * p_r * c_M

    # Finally, the local-Poisson approximation:
    #     E[D_min | r] ≈ Gamma(1 + 1/M) * [ λ * c_M ]^(-1/M).
    # That is the formula for the mean nearest-neighbor distance in an M-dim homogeneous Poisson process.
    E = gamma_factor * lam_times_cM**(-1.0 / num_dims)

    # Return a float if the input was a single radius, or an ndarray otherwise.
    return E[0] if E.size == 1 else E


def apply_post_noising(key, sample):
    sample -= sample.mean(axis=0, keepdims=True)
    radii = jnp.linalg.norm(sample, axis=-1)
    stds = expected_nearest_neighbour_distance_at_radius(
        radii, num_dims=sample.shape[-1], sample_size=sample.shape[0]
    )
    noise = jr.normal(key, sample.shape) * stds[:, None]
    return sample + noise


def sample_post_noise_distribution(key, sample, batch_dims, total_iterations, noise_factor=1.0):
    sample -= sample.mean(axis=0, keepdims=True)
    radii = jnp.linalg.norm(sample, axis=-1)
    stds = expected_nearest_neighbour_distance_at_radius(
        radii, num_dims=sample.shape[-1], sample_size=sample.shape[0]
    )[:, None]
    @jax.jit
    def next_batch(key):
        key, subkey = jr.split(key)
        noise = jr.normal(subkey, batch_dims + sample.shape) * stds
        return key, sample + noise * noise_factor
    for _ in range(total_iterations):
        key, out = next_batch(key)
        yield out


def destination_consistency_plot(samples, output_path):
    nrows = 1
    ncols = 2
    fig, (ax1, ax2) = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    for x in samples:
        ax1.scatter(*x.T, s=0.1, alpha=0.7)
    for x in noised_samples:
        ax2.scatter(*x.T, s=0.1, alpha=0.7)
    for x, nx in zip(samples, noised_samples):
        ax2.plot(*np.stack([x.T, nx.T], axis=1), lw=0.1, alpha=0.7, c="k")
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    for ax in [ax1, ax2]:
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([xmin, xmax])
        ax.set_axis_off()
    plt.savefig(output_path, dpi=400)
    plt.close()
    plt.clf()


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
    sc = scatter_ax.scatter([], [], s=10, c='k')
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


def inference_plot(key, train_state, data_sample, output_path, num_rows=3, num_cols=4):
    batch_size = num_rows * num_cols - 1
    # Inference / generation / model-sampling
    prior = jr.normal(key, (batch_size,) + data_sample.shape)
    predicted = train_state.params(prior)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    for ax, data in zip(axes.flatten(), [data_sample] + list(np.array(predicted))):
        ax.scatter(*data.T, s=2, c="k", alpha=0.5)
        ax.set_axis_off()

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    plt.savefig(output_path)
    plt.close(fig)


def generate_movie(folder_path, output_path):
    image_files = sorted(glob.glob(os.path.join(folder_path, '*')))
    writer = imageio.get_writer(output_path, fps=30)
    for image_file in image_files:
        image = imageio.imread(image_file)
        writer.append_data(image)
    writer.close()


def train_mlp(
    batch_size=8,
    total_iterations=10_000,
    every=10,
    plot_every=100,
    movie_every=300,
    noise_factor=1.0,
    plotdir="plots/generated_samples",
    movie_path=None,
):
    """Fit an MLP using flow matching to the assignments given by the energy function.

    Then integrate the flow. Try doing single step prediction.
    """
    if movie_path is None:
        movie_path = plotdir + ".mp4"
    key = jr.key(0)
    data_sample = sample_data_distribution(key, 1, n=10, m=10)[0]
    data_sample -= data_sample.mean(axis=0, keepdims=True)
    data_sample /= data_sample.std(axis=0, keepdims=True)

    guassian_centroids = minimise_gaussian_energy(
        samples=data_sample + 1e-6 * jr.normal(key, data_sample.shape),
        key=key,
    )[0]

    def sin_cos_embedding(x, num):
        freqs = 2**np.arange(num)
        angles = 2 * jnp.pi * freqs * x[..., None]
        return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


    class MLP(PyTree):
        mlp: nn.Module
        pe_dim: int =  static()

        def __call__(self, x):
            x = (x + 4.0) / 8  # Should probably just use  the unbounded PE...
            x = jnp.concatenate([
                sin_cos_embedding(x[..., 0], self.pe_dim),
                sin_cos_embedding(x[..., 1], self.pe_dim)
            ], axis=-1)
            return self.mlp(x)

    pe_dim = 8
    mlp = MLP(mlp=nn.mlp([4 * pe_dim, 512, 32, 256, 2], activation=jax.nn.gelu, key=jr.key(0)), pe_dim=pe_dim)
    def loss_fn(params, batch, key):
        pred = params(batch["gaussian"])
        return (jnp.abs(pred - batch["data"])).sum(-1).mean()
    optimiser = optax.adam(learning_rate=1e-4)
    train_step = make_train_step(loss_fn=loss_fn, optimizer=optimiser)
    train_state = TrainState(
        params=mlp, grads=jax.tree.map(jnp.zeros_like, mlp), opt_state=optimiser.init(mlp)
    )
    pbar = tqdm(total=total_iterations)
    counter = 0
    losses = []
    os.system(f"rm -rf {plotdir}")
    os.makedirs(plotdir, exist_ok=True)
    for gaussian_sample in sample_post_noise_distribution(
        key=key,
        sample=guassian_centroids,
        batch_dims=(batch_size,),
        total_iterations=total_iterations,
        noise_factor=noise_factor,
    ):
        train_state, loss, aux = train_step(train_state, {"gaussian": gaussian_sample, "data": data_sample}, None)
        losses.append(float(np.array(loss)))
        if counter % every == 0:
            pbar.set_description(f"{np.mean(losses)=:.5f}")
            pbar.update(every)
            losses = []
        if counter % plot_every == 0:
            inference_plot(key, train_state, data_sample, output_path=f"{plotdir}/{counter // plot_every:05d}.png")
        if counter % movie_every == 0:
            generate_movie(folder_path=plotdir, output_path=movie_path)
        counter += 1
    inference_plot(key, train_state, data_sample, output_path=f"{plotdir}/{counter // plot_every:05d}.png")
    generate_movie(folder_path=plotdir, output_path=movie_path)


def train_mlps():
    plotdir = f"plots/{prefix}_generated_samples"
    moviedir = f"movies/{prefix}_generated_samples"
    os.makedirs(moviedir, exist_ok=True)
    for noise_factor in [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.8, 0.9, 1.0, 1.1, 1.2, 2.0, 3.0, 5.0]:
        train_mlp(
            batch_size=8,
            total_iterations=10_000,
            every=10,
            plot_every=100,
            movie_every=1000,
            noise_factor=noise_factor,
            plotdir=plotdir,
            movie_path=f"{moviedir}/noise_factor_{str(noise_factor).replace('.', '_')}.mp4",
        )


if __name__ == "__main__":
    prefix = "outputs"
    grad_std = jax.grad(mog_pdf, argnums=2)
    direction = jax.vmap(jax.vmap(grad_std, in_axes=(None, None, 0)), in_axes=(0, None, None))


    key = jr.key(0)
    data_sample = sample_data_distribution(key, 1, n=10, m=10)[0]
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


    # Compute the Chi-squared quantiles for this sample size,
    probs = (np.arange(1, len(data_sample) + 1) - 0.5) / len(data_sample)
    chi2_quantiles = st.chi2.ppf(probs, df=data_sample.shape[-1])


    # Minimise the Gaussian energy.
    sample, path, grad_path, path_ts = minimise_gaussian_energy(
        samples=data_sample + 1e-6 * jr.normal(key, data_sample.shape),
        key=jr.key(0),
    )

    # Run a number of samples can check if they are correlated / similar at the end of the
    # trajectories.
    # Note: Stochastisicty comes from the random directions, and small offsets at the start.
    samples = [
        minimise_gaussian_energy(
            samples=data_sample + 1e-6 * jr.normal(key, data_sample.shape),
            key=k,
        )[0] for k in tqdm(jr.split(jr.key(0), num=10))
    ]

    # Apply some noising to the final state of the trajectory.
    noised_samples = [apply_post_noising(k, s) for k, s in zip(jr.split(jr.key(1), num=len(samples)), samples)]
    destination_consistency_plot(samples, output_path=f"{prefix}_endpoint_consistency_plot.png")



    print(f"Negative Log-Likelihood of start: {negative_log_likelihood(path[-1]):.4f}")
    print(f"Negative Log-Likelihood of end: {negative_log_likelihood(path[0]):.4f}")

    qq_plot_chi_square(path[-1], f"{prefix}_qq_plot.png")

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


    name = "gaussian_energy_generation"
    plot(output_path=f"{prefix}_{name}.png", trajectory=path, ot_plan=np.stack([subset[row_ind], gaussian[col_ind]]))

    data_sample = sample_data_distribution(key, 1, n=10, m=10)[0]
    # data_sample = sample_data_distribution(key, 1, n=5, m=8)[0]
    data_sample -= data_sample.mean(axis=0, keepdims=True)
    data_sample /= data_sample.std(axis=0, keepdims=True)

    sample, path, grad_path, path_ts = minimise_gaussian_energy(
        data_sample + 1e-6 * jr.normal(key, data_sample.shape),
        key=jr.key(0),
    )

    animate(path, output_path=f'{prefix}_{name}.mp4')
