import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configuration
np.random.seed(42)


# Helper Classes and Functions

class DisjointSet:
    """
    A data structure for tracking a set of elements partitioned into a
    number of disjoint (non-overlapping) subsets. Used here for Kruskal's
    algorithm to build a Minimum Spanning Tree.
    """
    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}

    def find(self, item):
        """Find the root of the set containing item, with path compression."""
        if self.parent[item] == item:
            return item
        else:
            # Recursively find the root and update the parent (path compression)
            self.parent[item] = self.find(self.parent[item])
            return self.parent[item]

    def union(self, item1, item2):
        """Merge the sets containing item1 and item2, using union by rank."""
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            # Attach smaller rank tree under root of high rank tree
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root1] = root2
                if self.rank[root1] == self.rank[root2]:
                    self.rank[root2] += 1


def euclidean_distance(x, y):
    """Calculates the Euclidean distance between points."""
    return np.linalg.norm(x - y, axis=-1)


def build_graph_from_kernel(kernel, distance_threshold):
    """
    Builds a graph from a set of kernel points using a Minimum Spanning Tree (MST)
    and adds back edges below a certain distance threshold.

    This creates a graph that is guaranteed to be connected (via the MST) but also
    captures local neighborhood information with the additional short edges.

    Args:
        kernel (np.ndarray): An array of (N, D) points.
        distance_threshold (float): The threshold for adding extra edges.

    Returns:
        dict: An adjacency list representation of the graph (k2n).
    """
    n_points = len(kernel)
    if n_points < 2:
        return {i: set() for i in range(n_points)}

    # 1. Calculate all pairwise distances and create a list of all possible edges.
    indices = np.arange(n_points)
    pairs = np.transpose(np.triu_indices(n_points, k=1))
    p1, p2 = pairs[:, 0], pairs[:, 1]
    distances = euclidean_distance(kernel[p1], kernel[p2])

    # 2. Sort edges by distance for Kruskal's algorithm.
    sorted_indices = np.argsort(distances)
    sorted_edges = pairs[sorted_indices]
    sorted_distances = distances[sorted_indices]

    # 3. Build the Minimum Spanning Tree (MST) using Kruskal's algorithm.
    disjoint_set = DisjointSet(indices)
    k2n = {i: set() for i in indices} # kernel-to-neighbours adjacency list
    for (u, v), w in zip(sorted_edges, sorted_distances):
        if disjoint_set.find(u) != disjoint_set.find(v):
            disjoint_set.union(u, v)
            k2n[u].add(v)
            k2n[v].add(u)

    # 4. Add back any additional edges that are below the distance threshold.
    short_edge_mask = sorted_distances < distance_threshold
    for u, v in sorted_edges[short_edge_mask]:
        k2n[u].add(v)
        k2n[v].add(u)

    return k2n


def graph_search(x, k2n, kernel, start_nodes, rounds, closest_per_round):
    """
    Performs a limited-breadth graph search to find the nearest neighbors to a
    point `x` within the kernel.

    Args:
        x (np.ndarray): The point to search from.
        k2n (dict): The graph's adjacency list.
        kernel (np.ndarray): The array of kernel points.
        start_nodes (np.ndarray): A random subset of nodes to start the search from.
        rounds (int): The number of search expansion rounds.
        closest_per_round (int): How many of the closest nodes to expand in each round.

    Returns:
        tuple: A tuple containing the sorted neighbor indices and their distances.
    """
    candidate_nodes = set(start_nodes)
    for _ in range(rounds):
        # Find the closest nodes from the current candidates
        distances = euclidean_distance(x, kernel[list(candidate_nodes)])
        sorted_indices = np.argsort(distances)
        
        # Get the actual indices of the closest nodes in the kernel
        closest_nodes = np.array(list(candidate_nodes))[sorted_indices[:closest_per_round]]

        # Expand the search to include the neighbors of these closest nodes
        for node_idx in closest_nodes:
            candidate_nodes.update(k2n[node_idx])

    # After all rounds, calculate final distances for all candidates and sort
    final_candidates = np.array(list(candidate_nodes))
    final_distances = euclidean_distance(x, kernel[final_candidates])
    sorted_final_indices = np.argsort(final_distances)

    return final_candidates[sorted_final_indices], final_distances[sorted_final_indices]


# Main Simulation

def run_simulation():
    """
    Main function to run the kernel graph simulation, visualize the results,
    and compare the reconstructed distribution to the original.
    """
    # Parameters
    # Data distribution to learn (2D Gaussian)
    distribution = lambda: np.random.randn(2)
    
    # Kernel building parameters
    MAX_STEPS = 10_000
    INITIAL_KERNEL_SIZE = 10
    MIN_DISTANCE_CUTOFF = 0.2  # Points closer than this are not added
    MAX_DISTANCE_CUTOFF = 0.4  # Points further than this are "remote"
    
    # Graph search parameters
    SEARCH_ROUNDS = 4
    SEARCH_START_SUBSET = 32
    SEARCH_CLOSEST_PER_ROUND = 8

    # Graph maintenance parameters
    CLEAN_GRAPH_EVERY = 200 # Timesteps
    
    # Weighting parameters
    INITIAL_WEIGHT = np.array([0.1])
    WEIGHT_DECAY_FACTOR = 1.0 # No decay in this setup
    deposit_weight = lambda d: INITIAL_WEIGHT * ((d * 0.2) ** -0.5)

    # State Initialization
    kernel = []
    weights = np.array([])
    k2n = {}
    all_samples = []

    print("Running simulation...")
    for t in range(MAX_STEPS):
        x = distribution()
        all_samples.append(x)

        # Initial Kernel Population
        if len(kernel) < INITIAL_KERNEL_SIZE:
            # Add point only if it's not too close to existing points
            if not kernel or np.all(euclidean_distance(x, np.array(kernel)) > MIN_DISTANCE_CUTOFF):
                kernel.append(x)
                weights = np.append(weights, INITIAL_WEIGHT)
            # Once the initial kernel is full, build the first graph
            if len(kernel) == INITIAL_KERNEL_SIZE:
                k2n = build_graph_from_kernel(np.array(kernel), MAX_DISTANCE_CUTOFF)
            continue

        # Add New Point to Existing Kernel
        # 1. Perform efficient graph search to find neighbors
        start_nodes = np.random.choice(len(kernel), size=min(len(kernel), SEARCH_START_SUBSET), replace=False)
        neighbors_idx, distances = graph_search(
            x, k2n, np.array(kernel), start_nodes, SEARCH_ROUNDS, SEARCH_CLOSEST_PER_ROUND
        )
        
        min_dist = distances[0]
        closest_neighbor_idx = neighbors_idx[0]

        # 2. Decide how to handle the new point `x`
        if min_dist < MIN_DISTANCE_CUTOFF:
            # Point is too close to an existing node. Don't add it, but increase the weight
            # of all nearby nodes.
            valid_neighbors = neighbors_idx[distances < MAX_DISTANCE_CUTOFF]
            weights[valid_neighbors] += deposit_weight(distances[distances < MAX_DISTANCE_CUTOFF])
        else:
            # Point is far enough away to be added as a new node.
            new_node_idx = len(kernel)
            kernel.append(x)
            weights = np.append(weights, INITIAL_WEIGHT)
            
            if min_dist < MAX_DISTANCE_CUTOFF:
                # The new node is within the neighborhood of existing nodes.
                # Connect it to all valid neighbors.
                valid_neighbors = neighbors_idx[distances < MAX_DISTANCE_CUTOFF]
                k2n[new_node_idx] = set(valid_neighbors)
                for neighbor in valid_neighbors:
                    k2n[neighbor].add(new_node_idx)
                # Add weight to neighbors
                weights[valid_neighbors] += deposit_weight(distances[distances < MAX_DISTANCE_CUTOFF])
            else:
                # The new node is "remote". Connect it only to the single closest node.
                k2n[new_node_idx] = {closest_neighbor_idx}
                k2n[closest_neighbor_idx].add(new_node_idx)
                weights[closest_neighbor_idx] += deposit_weight(min_dist)

        # 3. Periodically rebuild the graph to maintain sparsity
        if t % CLEAN_GRAPH_EVERY == 0:
            k2n = build_graph_from_kernel(np.array(kernel), MAX_DISTANCE_CUTOFF)
        
        # 4. Apply weight decay
        weights *= WEIGHT_DECAY_FACTOR

    print(f"Simulation finished. Kernel size: {len(kernel)} points.")
    return np.array(kernel), weights, k2n, np.array(all_samples)


def visualize_results(kernel, weights, k2n, all_samples):
    """
    Generates and saves plots to visualize the simulation results.
    """
    print("Generating visualizations...")
    # Plot 1: The Kernel Graph
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot all original samples as a faint background
    ax.scatter(
        all_samples[:, 0], all_samples[:, 1],
        marker="x", s=5, alpha=0.1, color='gray', label='Original Samples'
    )
    
    # Plot the kernel points, with size and color scaled by weight
    normalized_weights = weights / weights.max()
    ax.scatter(
        kernel[:, 0], kernel[:, 1],
        s=normalized_weights * 200 + 10,
        c=normalized_weights,
        cmap='viridis',
        edgecolors='k',
        linewidth=0.5,
        zorder=3,
        label='Kernel Points'
    )

    # Plot the graph edges
    for node, neighbors in k2n.items():
        for neighbor in neighbors:
            if node < neighbor: # Avoid drawing edges twice
                p1 = kernel[node]
                p2 = kernel[neighbor]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=0.3, alpha=0.5, zorder=2)

    ax.set_title("Kernel Graph Representation of 2D Gaussian Distribution")
    ax.legend()
    plt.savefig("kernel_graph.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 12))
    # Plot 2: Reconstructed Distribution
    # Sample from the kernel by picking points based on their weight and adding small noise
    num_recon_samples = len(all_samples)
    chosen_indices = np.random.choice(
        len(kernel), size=num_recon_samples, p=weights / weights.sum(), replace=True
    )
    means = kernel[chosen_indices]
    reconstructed_samples = means + np.random.randn(*means.shape) * 0.1 # Add noise

    # Create a KDE plot to compare distributions
    plot_data = {
        "type": ["Reconstructed"] * num_recon_samples + ["Original"] * len(all_samples),
        "x": np.concatenate([reconstructed_samples[:, 0], all_samples[:, 0]]),
        "y": np.concatenate([reconstructed_samples[:, 1], all_samples[:, 1]]),
    }
    
    g = sns.jointplot(data=plot_data, x="x", y="y", hue="type", kind="kde", fill=False, alpha=0.6)
    g.fig.suptitle(f"Original vs. Reconstructed Distribution ({len(kernel)} kernel points)")
    g.fig.tight_layout()
    plt.savefig("reconstruction_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    kernel, weights, k2n, all_samples = run_simulation()
    visualize_results(kernel, weights, k2n, all_samples)