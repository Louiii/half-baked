import shutil
from pathlib import Path
from typing import Callable, Dict, Tuple

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dpp import DPP
from sklearn.decomposition import PCA
from tqdm import tqdm

# Configuration
np.random.seed(42)
sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif')
N_PARTICLES = 40


def superimpose_coordinates(
    reference_coords: np.ndarray, moving_coords: np.ndarray
) -> np.ndarray:
    """
    Superimposes a set of moving coordinates onto a reference set using the
    Kabsch algorithm.
    """
    ref_center = np.mean(reference_coords, axis=0)
    mov_center = np.mean(moving_coords, axis=0)
    ref_centered = reference_coords - ref_center
    mov_centered = moving_coords - mov_center
    covariance_matrix = np.dot(mov_centered.T, ref_centered)
    u, _, vt = np.linalg.svd(covariance_matrix)
    rotation_matrix = np.dot(u, vt)
    if np.linalg.det(rotation_matrix) < 0:
        vt[-1, :] *= -1
        rotation_matrix = np.dot(u, vt)
    transformed_coords = np.dot(moving_coords, rotation_matrix) + (
        ref_center - np.dot(mov_center, rotation_matrix)
    )
    return transformed_coords


def chain_chain_error(chain1: np.ndarray, chain2: np.ndarray) -> np.ndarray:
    """
    Calculates the Mean Squared Error (MSE) between two or more chains
    after optimal superposition. This serves as our distance metric.
    """
    # Base case: compare two single chains
    if chain1.ndim == 2 and chain2.ndim == 2:
        chain2_aligned = superimpose_coordinates(chain1, chain2)
        return np.mean((chain1 - chain2_aligned) ** 2)
    
    # Batch case: compare one chain to many, or many to many
    elif chain1.ndim == 3 and chain2.ndim == 3:
        # Handle broadcasting if one of the inputs is a single chain in a batch
        if chain1.shape[0] == 1 and chain2.shape[0] > 1:
            chain1 = np.repeat(chain1, chain2.shape[0], axis=0)
        if chain2.shape[0] == 1 and chain1.shape[0] > 1:
            chain2 = np.repeat(chain2, chain1.shape[0], axis=0)
            
        errors = [chain_chain_error(c1, c2) for c1, c2 in zip(chain1, chain2)]
        return np.array(errors)
        
    raise ValueError("Inputs must be 2D (single chain) or 3D (batch of chains) arrays.")


def create_mode_generator(n_particles: int = N_PARTICLES) -> Tuple[Callable, int]:
    """Creates a function that generates particle chains based on sine curves."""
    t = np.linspace(0, 1, n_particles)
    modes = np.array([[0.0], [np.pi * 0.5], [np.pi]])

    def _create_curve(coeffs: np.ndarray) -> np.ndarray:
        (a,) = coeffs
        return np.stack([t, np.sin(2 * np.pi * t + a)], axis=-1)

    def _interpolate_and_center(m1_idx: int, m2_idx: int, prog: float) -> np.ndarray:
        start_coeffs = modes[m1_idx]
        end_coeffs = modes[m2_idx]
        interp_coeffs = start_coeffs + (end_coeffs - start_coeffs) * prog
        positions = _create_curve(interp_coeffs)
        start_mean = np.mean(_create_curve(modes[m1_idx]), axis=0)
        end_mean = np.mean(_create_curve(modes[m2_idx]), axis=0)
        interp_mean = start_mean + (end_mean - start_mean) * prog
        return positions - interp_mean

    return _interpolate_and_center, len(modes)


def generate_correlated_noise(
    prev_scales: np.ndarray, amplitude: float = 0.01, n_particles: int = N_PARTICLES
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates smooth, time-correlated noise for the particle chain."""
    new_scales = prev_scales + np.random.randn(4, 2) * amplitude
    new_scales *= 0.5
    ts = np.linspace(0, 1, n_particles)
    basis_functions = np.sin(np.arange(4)[:, None] * np.pi * ts[None, :])
    chain_noise = (basis_functions[:, :, None] * new_scales[:, None, :]).sum(0)
    return chain_noise, new_scales


class TrajectoryGenerator:
    """A generator that simulates a molecular trajectory using a Markov chain."""
    def __init__(self, mode_gen: Callable, trans_prob: np.ndarray, init_probs: np.ndarray,
                 noise_amp: float, trans_dur: int):
        self.mode_generator = mode_gen
        self.transition_prob = trans_prob
        self.noise_amplitude = noise_amp
        self.transition_duration = trans_dur
        self.n_states = init_probs.shape[0]
        self.state_indices = np.arange(self.n_states)
        self.current_state = np.random.choice(self.state_indices, p=init_probs)
        self.previous_state = self.current_state
        self.transition_counter = 0
        self.noise_scales = np.zeros((4, 2))
        self.global_rotation = 0.0
        self.global_translation = np.zeros(2)

    def step(self) -> Tuple[np.ndarray, Dict]:
        """Generates the next frame in the trajectory."""
        if self.transition_counter == 0:
            new_state = np.random.choice(self.state_indices, p=self.transition_prob[self.current_state])
            if new_state != self.current_state:
                self.previous_state = self.current_state
                self.current_state = new_state
                self.transition_counter = self.transition_duration
        else:
            self.transition_counter -= 1
        
        progress = 1.0 - (self.transition_counter / self.transition_duration) if self.transition_counter > 0 else 0.0
        base_positions = self.mode_generator(self.previous_state, self.current_state, progress)
        chain_noise, self.noise_scales = generate_correlated_noise(self.noise_scales, amplitude=self.noise_amplitude)
        center = np.mean(base_positions, axis=0)
        self.global_rotation += (np.random.rand() - 0.5) * 0.02
        self.global_translation += np.random.randn(2) * 0.02
        rot_matrix = np.array([[np.cos(self.global_rotation), -np.sin(self.global_rotation)],
                               [np.sin(self.global_rotation), np.cos(self.global_rotation)]])
        rotated_pos = (base_positions - center).dot(rot_matrix)
        final_positions = rotated_pos + (center + self.global_translation) + chain_noise
        
        state_info = {"state": self.current_state, "previous_state": self.previous_state}
        return final_positions, state_info


def setup_trajectory_generator(noise_amp: float, trans_dur: int, trans_prob: np.ndarray) -> TrajectoryGenerator:
    """
    Configures and initializes the TrajectoryGenerator.

    Args:
        noise_amp (float): The amplitude of the correlated noise.
        trans_dur (int): The number of frames for a state transition.
        trans_prob (np.ndarray): The pre-computed transition probability matrix.

    Returns:
        TrajectoryGenerator: An initialized TrajectoryGenerator instance.
    """
    mode_generator, n_modes = create_mode_generator()

    # The initial state is always state 0.
    initial_probs = np.zeros(n_modes)
    initial_probs[0] = 1.0
    
    # Ensure the provided transition matrix is valid
    assert np.allclose(trans_prob.sum(axis=-1), 1.0), "Probabilities in transition matrix must sum to 1"

    return TrajectoryGenerator(
        mode_gen=mode_generator,
        trans_prob=trans_prob,
        init_probs=initial_probs,
        noise_amp=noise_amp,
        trans_dur=trans_dur
    )


# Kernel Graph Sampling Code

class DisjointSet:
    """A data structure for tracking disjoint sets, for MST."""
    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}

    def find(self, item):
        if self.parent[item] == item:
            return item
        self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1, item2):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root1] = root2
                if self.rank[root1] == self.rank[root2]:
                    self.rank[root2] += 1

def build_graph_from_kernel(kernel: np.ndarray, dist_thresh: float) -> Dict[int, set]:
    """Builds a graph from kernel points using an MST plus short edges."""
    n_points = len(kernel)
    if n_points < 2:
        return {i: set() for i in range(n_points)}
    indices = np.arange(n_points)
    pairs = np.transpose(np.triu_indices(n_points, k=1))
    distances = chain_chain_error(kernel[pairs[:, 0]], kernel[pairs[:, 1]])
    sorted_indices = np.argsort(distances)
    sorted_edges = pairs[sorted_indices]
    sorted_distances = distances[sorted_indices]
    disjoint_set = DisjointSet(indices)
    k2n = {i: set() for i in indices}
    for (u, v), w in zip(sorted_edges, sorted_distances):
        if disjoint_set.find(u) != disjoint_set.find(v):
            disjoint_set.union(u, v)
            k2n[u].add(v)
            k2n[v].add(u)
    short_edge_mask = sorted_distances < dist_thresh
    for u, v in sorted_edges[short_edge_mask]:
        k2n[u].add(v)
        k2n[v].add(u)
    return k2n

def graph_search(x, k2n, kernel, start_nodes, rounds, closest_per_round):
    """Performs a limited-breadth graph search to find nearest neighbors."""
    candidate_nodes = set(start_nodes)
    for _ in range(rounds):
        node_list = list(candidate_nodes)
        # The [0] is removed here
        distances = chain_chain_error(np.array([x]), np.array(kernel)[node_list])
        sorted_indices = np.argsort(distances)
        closest_nodes = np.array(node_list)[sorted_indices[:closest_per_round]]
        for node_idx in closest_nodes:
            candidate_nodes.update(k2n[node_idx])
    final_candidates = list(candidate_nodes)
    # The [0] is removed here too
    final_distances = chain_chain_error(np.array([x]), np.array(kernel)[final_candidates])
    sorted_final_indices = np.argsort(final_distances)
    return np.array(final_candidates)[sorted_final_indices], final_distances[sorted_final_indices]


class Config:
    """Class to hold hyperparameters."""
    SEARCH_ROUNDS = 5
    SEARCH_CLOSEST_PER_ROUND = 8
    SEARCH_START_SUBSET = 32
    MAX_DISTANCE_CUTOFF = 0.01
    MIN_DISTANCE_CUTOFF = 0.0005
    WEIGHT_DECAY_FACTOR = 1.0
    INITIAL_WEIGHT = np.array([0.1])
    WEIGHT_LENGTH_SCALE = 0.2
    INITIAL_KERNEL_SIZE = 10
    CLEAN_GRAPH_EVERY = 200
    MAX_STEPS = 5000


def build_conformation_kernel(config: Config, trajectory_generator: TrajectoryGenerator):
    """
    Main loop to build the kernel of conformations by sampling from the trajectory.
    """
    kernel = []
    weights = np.array([])
    k2n = {}
    all_states = []
    
    deposit_weight = lambda d: config.INITIAL_WEIGHT * ((d * config.WEIGHT_LENGTH_SCALE) ** -0.5)
    
    pbar = tqdm(range(config.MAX_STEPS), desc="Initialising Kernel")
    for t in pbar:
        x, state_info = trajectory_generator.step()
        all_states.append(state_info['state'])

        # Initial Kernel Population
        if len(kernel) < config.INITIAL_KERNEL_SIZE:
            if not kernel or np.all(chain_chain_error(np.array([x]), np.array(kernel)) > config.MIN_DISTANCE_CUTOFF):
                kernel.append(x)
                weights = np.append(weights, config.INITIAL_WEIGHT)
            if len(kernel) == config.INITIAL_KERNEL_SIZE:
                k2n = build_graph_from_kernel(np.array(kernel), config.MAX_DISTANCE_CUTOFF)
            continue

        # Add New Point to Existing Kernel
        start_nodes = np.random.choice(len(kernel), size=min(len(kernel), config.SEARCH_START_SUBSET), replace=False)
        neighbors_idx, distances = graph_search(
            x, k2n, np.array(kernel), start_nodes, config.SEARCH_ROUNDS, config.SEARCH_CLOSEST_PER_ROUND
        )
        min_dist, closest_neighbor_idx = distances[0], neighbors_idx[0]

        if min_dist < config.MIN_DISTANCE_CUTOFF:
            pbar.set_description(f"Step {t}: Too Close")
            # --- FIX: Deposit weight ONLY to the single closest neighbor ---
            closest_neighbor_idx = neighbors_idx[0]
            weights[closest_neighbor_idx] += deposit_weight(min_dist)
            # ----------------------------------------------------------------
        else:
            new_node_idx = len(kernel)
            kernel.append(x)
            weights = np.append(weights, config.INITIAL_WEIGHT)
            if min_dist < config.MAX_DISTANCE_CUTOFF:
                pbar.set_description(f"Step {t}: Adding New Neighbor")
                valid_neighbors = neighbors_idx[distances < config.MAX_DISTANCE_CUTOFF]
                k2n[new_node_idx] = set(valid_neighbors)
                for neighbor in valid_neighbors:
                    k2n[neighbor].add(new_node_idx)
                weights[valid_neighbors] += deposit_weight(distances[distances < config.MAX_DISTANCE_CUTOFF])
            else:
                pbar.set_description(f"Step {t}: Adding Remote Node")
                k2n[new_node_idx] = {closest_neighbor_idx}
                k2n[closest_neighbor_idx].add(new_node_idx)
                weights[closest_neighbor_idx] += deposit_weight(min_dist)

        if t > 0 and t % config.CLEAN_GRAPH_EVERY == 0:
            pbar.set_description(f"Step {t}: Cleaning Graph")
            k2n = build_graph_from_kernel(np.array(kernel), config.MAX_DISTANCE_CUTOFF)
            
        weights *= config.WEIGHT_DECAY_FACTOR
        
    return np.array(kernel), weights, k2n, np.array(all_states)


# def estimate_mode_probabilities(modes, kernel, weights, closeness_threshold):
#     """Estimates the probability mass of each mode based on the kernel."""
#     dists = chain_chain_error(
#         np.repeat(modes, len(kernel), axis=0),
#         np.tile(kernel, (len(modes), 1, 1))
#     ).reshape(len(modes), len(kernel))
    
#     mode_assignment = np.argmin(dists, axis=0)
#     mode_distances = np.min(dists, axis=0)
    
#     current_mass = np.zeros(len(modes))
#     for i in range(len(modes)):
#         mask = (mode_assignment == i) & (mode_distances < closeness_threshold)
#         current_mass[i] = weights[mask].sum()
        
#     return current_mass


def estimate_mode_probabilities(modes, kernel, weights, sigma=None):
    """
    Estimates the probability mass of each mode using a soft, Gaussian-based
    assignment of kernel points.

    Args:
        modes (np.ndarray): The reference conformations for each mode.
        kernel (np.ndarray): The array of kernel conformations.
        weights (np.ndarray): The weight of each kernel point.
        sigma (float, optional): The bandwidth of the Gaussian kernel. A smaller
            sigma leads to "harder" assignments. If None, it's estimated
            from the data as the mean distance to the 5th nearest neighbor,
            which is a robust heuristic.

    Returns:
        np.ndarray: The estimated probability mass for each mode.
    """
    # 1. Calculate all-vs-all distances
    dists = chain_chain_error(
        np.repeat(modes, len(kernel), axis=0),
        np.tile(kernel, (len(modes), 1, 1))
    ).reshape(len(modes), len(kernel))

    # 2. Heuristically determine sigma if not provided.
    #    A good sigma is crucial for good results.
    if sigma is None:
        # Sort distances for each kernel point and find the 5th nearest neighbor
        # to estimate a reasonable local density scale.
        kernel_dists = chain_chain_error(
            np.repeat(kernel, len(kernel), axis=0),
            np.tile(kernel, (len(kernel), 1, 1))
        ).reshape(len(kernel), len(kernel))
        np.fill_diagonal(kernel_dists, np.inf) # Ignore self-distance
        k_nearest_dists = np.sort(kernel_dists, axis=1)[:, :5]
        sigma = np.mean(k_nearest_dists)
        print(f"Heuristically determined sigma: {sigma:.4f}")

    # 3. Convert distances to affinities using a Gaussian kernel
    #    The `T` transposes the matrix to shape (n_kernel, n_modes)
    affinities = np.exp(-dists.T**2 / (2 * sigma**2))

    # 4. Normalize affinities so each row (each kernel point) sums to 1.
    #    This gives the probability of a kernel point belonging to each mode.
    #    Add a small epsilon to avoid division by zero for remote points.
    row_sums = affinities.sum(axis=1, keepdims=True) + 1e-9
    assignment_probs = affinities / row_sums # Shape: (n_kernel, n_modes)

    # 5. Calculate the final probability mass for each mode by taking the
    #    weighted sum of the assignment probabilities.
    #    The weights are shape (n_kernel,), so we broadcast them.
    mode_mass = np.sum(assignment_probs * weights[:, np.newaxis], axis=0)

    return mode_mass


def get_analytical_ground_truth(transition_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the analytical stationary distribution (ground truth probabilities)
    for a given Markov chain transition matrix.

    Args:
        transition_matrix (np.ndarray): The P matrix for the Markov chain.

    Returns:
        np.ndarray: The vector of stationary probabilities for each state.
    """
    # For the stationary distribution pi, we have pi * P = pi.
    # This is equivalent to finding the eigenvector of P.T with eigenvalue 1.
    eigenvals, eigenvecs = np.linalg.eig(transition_matrix.T)
    
    # Find the eigenvector corresponding to the eigenvalue closest to 1
    stationary_vector = np.real(eigenvecs[:, np.isclose(eigenvals, 1)])
    
    # The eigenvector is unique up to a scaling factor, so we normalise it
    # to make it a probability distribution (sums to 1).
    stationary_probs = stationary_vector / stationary_vector.sum()
    
    return stationary_probs.flatten()


def plot_mode_centric_dpp_samples(sampled_kernel, dpp_indices, all_modes, kernel, weights, output_dir):
    """
    Generates a plot where each column is dedicated to a reference mode.
    Sampled conformations are plotted in the column of their closest mode,
    aligned to that mode, and colored by their kernel weight. The title
    includes counts and normalised weights for each mode.

    Args:
        sampled_kernel (np.ndarray): The subset of kernel points selected by DPP.
        dpp_indices (np.ndarray): The indices of the sampled_kernel points in the original kernel.
        all_modes (np.ndarray): The reference conformations for each mode.
        kernel (np.ndarray): The full array of kernel conformations.
        weights (np.ndarray): The weight of each point in the full kernel.
        output_dir (Path): The directory to save the plot in.
    """
    n_modes = len(all_modes)
    fig, axes = plt.subplots(1, n_modes, figsize=(6 * n_modes, 7), squeeze=False) # Increased height for title
    axes = axes.flatten()

    # 1. Assign each point in the DPP sample to its single closest reference mode.
    dists_to_modes = chain_chain_error(
        np.repeat(sampled_kernel, n_modes, axis=0),
        np.tile(all_modes, (len(sampled_kernel), 1, 1))
    ).reshape(len(sampled_kernel), n_modes)
    
    closest_mode_assignments = np.argmin(dists_to_modes, axis=1)

    # 2. Calculate counts and summed weights for each mode from the DPP sample.
    mode_counts = []
    mode_weight_sums = []
    sampled_weights = weights[dpp_indices]

    for mode_idx in range(n_modes):
        assigned_mask = (closest_mode_assignments == mode_idx)
        mode_counts.append(np.sum(assigned_mask))
        mode_weight_sums.append(np.sum(sampled_weights[assigned_mask]))

    mode_counts = np.array(mode_counts)
    mode_weight_sums = np.array(mode_weight_sums)

    # 3. Normalise the summed weights to get a probability distribution.
    total_weight_sum = np.sum(mode_weight_sums)
    if total_weight_sum > 0:
        normalised_weights = mode_weight_sums / total_weight_sum
    else:
        normalised_weights = np.zeros_like(mode_weight_sums)

    # 4. Format the subtitle string with the new information.
    counts_str = ", ".join([f"Mode {i}: {c}" for i, c in enumerate(mode_counts)])
    weights_str = ", ".join([f"Mode {i}: {w:.2f}" for i, w in enumerate(normalised_weights)])
    main_title = "DPP Sampled Conformations Aligned to Closest Mode"
    subtitle = f"Counts: [{counts_str}]  |  Normalised Weights: [{weights_str}]"
    fig.suptitle(f"{main_title}\n{subtitle}", fontsize=16, y=0.98)


    # 5. Create a color normaliser based on the min/max weights of the *sampled* points
    if len(sampled_weights) > 0:
        norm = mcolors.Normalize(vmin=np.min(sampled_weights), vmax=np.max(sampled_weights))
        cmap = cm.get_cmap('viridis')
    else:
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cmap = cm.get_cmap('viridis')

    # 6. Iterate through each mode/subplot column to plot the data.
    for mode_idx, ax in enumerate(axes):
        ref_mode = all_modes[mode_idx]
        ax.plot(*ref_mode.T, 'r-', lw=2.5, zorder=100)

        assigned_mask = (closest_mode_assignments == mode_idx)
        confs_for_this_mode = sampled_kernel[assigned_mask]
        weights_for_this_mode = sampled_weights[assigned_mask]

        for conf, weight in zip(confs_for_this_mode, weights_for_this_mode):
            aligned_conf = superimpose_coordinates(ref_mode, conf)
            color = cmap(norm(weight))
            ax.plot(*aligned_conf.T, color=color, lw=1.5, alpha=0.8)

        ax.set_title(f"Reference Mode {mode_idx}")
        ax.axis("off")

    # 7. Add a single, shared colorbar.
    fig.subplots_adjust(right=0.85, top=0.85, bottom=0.1)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, cax=cbar_ax, label='Kernel Point Weight')

    plt.savefig(output_dir / "DPP_samples_mode_centric.png")
    plt.close(fig)


def plot_kernel_pca(kernel, weights, all_modes, output_dir):
    """
    Calculates a pairwise distance matrix of the kernel, performs PCA to get a
    2D embedding, and creates a scatter plot of the results.

    Args:
        kernel (np.ndarray): The array of kernel conformations.
        weights (np.ndarray): The weight of each kernel point.
        all_modes (np.ndarray): The reference conformations for each mode.
        output_dir (Path): The directory to save the plot in.
    """
    n_kernel = len(kernel)
    if n_kernel < 3:
        print("Kernel too small for PCA plot, skipping.")
        return

    print("Generating PCA plot of the kernel landscape...")

    # 1. Calculate the all-vs-all distance matrix for the kernel.
    #    This matrix describes the "location" of each point relative to all others.
    dist_matrix = chain_chain_error(
        np.repeat(kernel, n_kernel, axis=0),
        np.tile(kernel, (n_kernel, 1, 1))
    ).reshape(n_kernel, n_kernel)

    # 2. Use PCA to reduce the dimensionality of the distance matrix to 2D.
    #    We treat each row of the distance matrix as a feature vector.
    pca = PCA(n_components=2)
    embedding_2d = pca.fit_transform(dist_matrix)

    # 3. For coloring, assign each kernel point to its closest reference mode.
    dists_to_modes = chain_chain_error(
        np.repeat(kernel, len(all_modes), axis=0),
        np.tile(all_modes, (n_kernel, 1, 1))
    ).reshape(n_kernel, len(all_modes))
    mode_assignments = np.argmin(dists_to_modes, axis=1)

    # 4. Prepare for plotting.
    plt.figure(figsize=(12, 10))
    
    # Normalise weights for scaling the dot sizes. Add a minimum size.
    # We scale by the square root to make area proportional to weight.
    scaled_sizes = np.sqrt(weights / np.max(weights)) * 200 + 10

    # Get colors for each mode.
    colors = list(mcolors.TABLEAU_COLORS.values())
    point_colors = [colors[i] for i in mode_assignments]

    # 5. Create the scatter plot.
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        s=scaled_sizes,
        c=point_colors,
        alpha=0.7,
        edgecolors='w',
        linewidth=0.5
    )

    # Create a legend for the mode colors.
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Mode {i}',
                                  markerfacecolor=colors[i], markersize=10)
                       for i in range(len(all_modes))]

    plt.legend(handles=legend_elements, title="Closest Mode")
    
    plt.title("2D PCA Embedding of the Conformational Kernel", fontsize=16)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.savefig(output_dir / "kernel_pca_embedding.png")
    plt.close()


def plot_dpp_grid_by_mode(sampled_kernel, all_modes, output_dir):
    """
    Generates a grid plot where each column is dedicated to a reference mode.
    DPP-sampled conformations are assigned to their closest mode and then
    plotted in the corresponding column, ordered by distance to the mode.

    Args:
        sampled_kernel (np.ndarray): The subset of kernel points selected by DPP.
        all_modes (np.ndarray): The reference conformations for each mode.
        output_dir (Path): The directory to save the plot in.
    """
    n_modes = len(all_modes)
    if len(sampled_kernel) == 0:
        print("No samples in kernel to plot.")
        return

    # 1. Assign each sampled conformation to its closest reference mode.
    dists_to_modes = chain_chain_error(
        np.repeat(sampled_kernel, n_modes, axis=0),
        np.tile(all_modes, (len(sampled_kernel), 1, 1))
    ).reshape(len(sampled_kernel), n_modes)
    
    assignments = np.argmin(dists_to_modes, axis=1)

    # 2. Group the conformations and their distances by their assigned mode.
    confs_by_mode = [[] for _ in range(n_modes)]
    dists_by_mode = [[] for _ in range(n_modes)]

    for i, conf in enumerate(sampled_kernel):
        mode_idx = assignments[i]
        confs_by_mode[mode_idx].append(conf)
        dists_by_mode[mode_idx].append(dists_to_modes[i, mode_idx])

    # 3. **NEW**: Sort each group by distance (ascending).
    for mode_idx in range(n_modes):
        if confs_by_mode[mode_idx]:  # Check if the list is not empty
            # Zip distances and conformations, sort by distance, then unzip.
            sorted_pairs = sorted(zip(dists_by_mode[mode_idx], confs_by_mode[mode_idx]))
            dists_by_mode[mode_idx], confs_by_mode[mode_idx] = zip(*sorted_pairs)

    # 4. Determine grid dimensions. The number of rows is determined by the
    #    mode that has the most samples assigned to it.
    n_cols = n_modes
    n_rows = max(len(confs) for confs in confs_by_mode) if sampled_kernel.size > 0 else 1

    # 5. Create the plot grid.
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

    # 6. Iterate through the grid and plot each conformation.
    for mode_idx in range(n_cols):
        ax_title = axes[0, mode_idx]
        ax_title.set_title(f"Closest to Mode {mode_idx}", fontsize=14)
        
        for row_idx in range(n_rows):
            ax = axes[row_idx, mode_idx]
            
            # Check if there is a conformation to plot in this cell
            if row_idx < len(confs_by_mode[mode_idx]):
                conf = confs_by_mode[mode_idx][row_idx]
                dist = dists_by_mode[mode_idx][row_idx]
                ref_mode = all_modes[mode_idx]

                # Plot the reference mode in red
                ax.plot(*ref_mode.T, "r-", label=f"Mode {mode_idx}")
                
                # Align and plot the sampled conformation in blue
                aligned_conf = superimpose_coordinates(ref_mode, conf)
                ax.plot(*aligned_conf.T, "b-", label="Sampled Conf")
                
                # Add the distance as text
                ax.text(0.05, 0.95, f"Dist: {dist:.4f}",
                        transform=ax.transAxes, ha='left', va='top', fontsize=10)
                ax.axis("off")
            else:
                # If no conformation for this cell, turn off the axis
                ax.axis("off")

    fig.suptitle("DPP Sampled Conformations Grouped and Ordered by Closest Mode", fontsize=18, y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.93]) # Adjust layout for suptitle
    plt.savefig(output_dir / "DPP_samples_grid_ordered.png")
    plt.close(fig)


def main():
    """Main function to run the simulation, analysis, and visualisation."""
    # Setup
    OUTPUT_DIR = Path("conformation_sampling_plots")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()

    config = Config()
    
    # MD simulation parameters
    TRANSITION_DURATION = 30
    ESCAPE_TIME = 300
    NOISE_AMPLITUDE = 0.05

    # Define retain_prob here so you can build the matrix
    retain_prob = np.exp(np.log(0.5) / ESCAPE_TIME)
    transition_prob_matrix = np.array([
        [retain_prob, 1 - retain_prob, 0],
        [0, retain_prob, 1 - retain_prob],
        [1 - retain_prob, 0, retain_prob],
    ])

    # Pass it to the generator setup function (you'll need to modify setup_trajectory_generator slightly
    # to accept the matrix instead of building it inside)
    generator = setup_trajectory_generator(
        noise_amp=NOISE_AMPLITUDE,
        trans_dur=TRANSITION_DURATION,
        trans_prob=transition_prob_matrix # Pass the matrix
    )

    # Run Kernel Builder
    kernel, weights, k2n, states = build_conformation_kernel(config, generator)
    print(f"\nSimulation finished. Final kernel size: {len(kernel)} conformations.")

    # Analysis
    print("Analysing results...")
    mode_generator, n_modes = create_mode_generator()
    all_modes = np.array([mode_generator(i, i, 0) for i in range(n_modes)])
    ground_truth = get_analytical_ground_truth(transition_prob_matrix)

    # Visualisation 1: DPP Sampled Conformations
    print("Visualising diverse conformations using DPP...")
    # Use DPP to select a diverse subset for visualisation
    if len(kernel) > 1:
        # Calculate full distance matrix for DPP
        dist_matrix = chain_chain_error(
            np.repeat(kernel, len(kernel), axis=0),
            np.tile(kernel, (len(kernel), 1, 1))
        ).reshape(len(kernel), len(kernel))
        
        # Quality is based on weights, similarity is based on distance
        quality = weights / weights.mean()
        similarity = np.exp(-dist_matrix / np.mean(dist_matrix))
        
        dpp = DPP(quality, similarity)
        num_samples = min(18, len(kernel))
        dpp_indices = dpp.sample(k=num_samples)
        sampled_kernel = kernel[dpp_indices]
    else:
        sampled_kernel = kernel

    plot_mode_centric_dpp_samples(sampled_kernel, dpp_indices, all_modes, kernel, weights, OUTPUT_DIR)

    plot_dpp_grid_by_mode(sampled_kernel, all_modes, OUTPUT_DIR)

    # PCA Embedding of the Kernel
    plot_kernel_pca(kernel, weights, all_modes, OUTPUT_DIR)

    # Mode Probability Convergence
    print("Visualising mode probability estimates...")
    closeness_thresholds = np.linspace(0.0, config.MAX_DISTANCE_CUTOFF * 1.5, 40)
    prob_estimates = np.array([
        estimate_mode_probabilities(all_modes, kernel, weights, c) for c in closeness_thresholds
    ])

    # Normalise to get probabilities
    prob_estimates_norm = prob_estimates / np.sum(prob_estimates, axis=1, keepdims=True)

    fig = plt.figure(figsize=(10, 6))
    colors = mcolors.TABLEAU_COLORS
    for i in range(n_modes):
        plt.plot(closeness_thresholds, prob_estimates_norm[:, i], color=list(colors)[i],
                    label=f"Est. Mode {i}")
        plt.axhline(ground_truth[i], color=list(colors)[i], linestyle="--",
                    label=f"GT Mode {i}")
    
    plt.title("Estimated Mode Probabilities vs. Closeness Threshold")
    plt.xlabel("Closeness Threshold (Max distance to be included in a mode)")
    plt.ylabel("Probability Mass")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.savefig(OUTPUT_DIR / "mode_prob_estimates.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
