
import jax
import jax.numpy
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, preprocessing
from tqdm import tqdm


def get_distributions(
    data, domain_dim, time_dim, detailed_image_bins: int, animation_bins: int | None = None
):
    def _expand_batch_dims(batch_dims, right_dim):
        return (batch_dims + (right_dim,)) if batch_dims is not None else (right_dim,)

    def prior_dist(key, batch_dims=None):
        return jax.random.normal(key, _expand_batch_dims(batch_dims, domain_dim))

    def time_dist(key, batch_dims=None):
        return jax.random.uniform(key, _expand_batch_dims(batch_dims, time_dim))

    if data == "moons":
        n_samples = 10_000
        x, _ = datasets.make_moons(n_samples=n_samples, noise=0.06)
        scaler = preprocessing.StandardScaler()
        xs = scaler.fit_transform(x)
    elif data.startswith("homer"):
        if data == "homer1d":
            dim = 1
            _apx_homer = [
                [691, -200, -235, -159, 12, -67, -69, -1, -10, -19],
                [25, 1, -20, 12, 12, 5, -4, -4, 20, 12],
                [1, 8, 2, 10, 14, -2, -8, 1, 4, -9],
                [-9, -7, -10, -3, -1, -5, -4, -1, 3, 2],
            ]
            invert = False
        elif data == "homer2d":
            dim = 2
            # Homer data
            _apx_homer = [
                [842, -11, 114, 19, 40, 1, 6, -11, 1, -10, -2, 6, 2, 7, 0, 2, -2, 0, 1, -2],
                [55, 102, -79, -60, 2, -46, 21, 0, 9, -3, -1, -3, -5, 8, -3, -6, 0, 2, 0, -1],
                [26, -3, 18, 17, -45, 7, -4, -25, -2, 5, 1, 9, 0, -14, 0, 0, 7, -9, 1, 7],
                [-8, -34, 1, -26, 7, 14, -10, 54, -9, 15, 15, -8, -3, -1, 0, -9, 21, -7, -4, 6],
                [35, 12, -26, -27, 20, 9, -8, 8, -8, 6, -27, 0, -3, -6, 16, -1, -11, -1, 13, -8],
                [0, -14, 21, 26, -10, 5, 1, -23, -4, -11, -9, 6, -1, 14, -3, 5, 3, 2, -1, -4],
                [8, -8, -18, -21, -11, 12, 6, 16, 6, -7, 25, -3, 3, 15, -10, -1, -10, 6, -12, 7],
                [-5, -18, 13, 35, 11, -3, -8, -8, -22, 0, 10, -2, 15, -19, -12, 13, -8, 5, -4, 5],
                [12, 0, -17, -3, 19, 8, -25, 0, 19, -2, -7, 0, -6, -4, 0, -5, 4, 2, -2, 6],
                [20, 4, -25, -13, -12, 5, 10, -15, -7, 18, 13, -2, -10, 6, 13, 3, 0, 2, 11, -9],
                [-13, -4, 26, 18, -10, -13, -1, 10, 2, 1, -4, -4, 2, -12, -5, 3, 0, -2, -1, -2],
                [13, 3, -16, -14, -2, 9, 8, 12, -5, 1, -2, -12, 3, 3, 3, -4, 4, -4, -5, -4],
                [0, -17, -5, 25, 11, -13, -12, 0, 1, 0, -13, 6, 6, -2, 12, -1, 8, -2, 0, 6],
                [4, 2, -5, 5, 10, -3, -6, -3, -2, 3, -5, 0, 2, 1, 9, 0, -12, -13, 2, -2],
                [0, -3, -9, -7, 0, 14, 8, -8, -2, 13, 10, -9, 1, 0, -11, -4, -6, 4, 2, 4],
                [-3, 0, 14, 16, -13, -19, 0, 2, -1, -8, -1, 1, 10, 0, 0, 0, -5, 14, 2, 4],
                [16, 13, -29, -18, 6, 3, 12, 13, 2, -1, -1, -15, -1, -1, -2, 0, 0, 11, 0, 0],
                [-9, -8, 13, 26, -5, -27, -8, 7, 10, -3, -1, 8, 13, -8, -4, 9, -8, 0, -4, -2],
                [0, -1, -9, -2, 13, 16, -2, -15, -3, 1, 0, -8, 0, 2, -3, 3, 1, 6, -5, -2],
                [-3, -1, -2, 2, 7, 0, 7, 1, -14, -5, 6, 6, 7, 0, -8, 1, -4, -2, 1, -3],
            ]
            invert = True
        else:
            raise ValueError
        apx_homer = np.array(_apx_homer).flatten() / 1000

        def cosine_basis(X, basis, n, dim):
            ix = np.array(np.unravel_index(basis, [n] * dim))
            vals = np.where(ix != 0, np.cos(X * ix[None, :] * np.pi), np.ones_like(X))
            return np.prod(vals, axis=-1)

        def generate_image(shape, weights, n, dim):
            x = np.stack(
                np.meshgrid(*[np.linspace(0, 1, res) for res in shape], indexing="ij"), axis=-1
            )
            image = np.zeros(shape)
            for i in tqdm(list(range(n**dim))):
                image += weights[i] * cosine_basis(x, i, n, dim)
            return image

        if dim == 1:

            def sample_img(image, num_samples):
                w = image.shape[0]
                if invert:
                    image = image.max() - image
                else:
                    image -= image.min()
                p = image.flatten() / image.sum()
                p = image.flatten() / image.sum()
                samples = np.random.choice(len(p), p=p, size=num_samples, replace=True)
                samples = samples.astype(np.float32) / w
                samples = samples[..., None]
                return (samples * 4) - 2
        else:

            def sample_img(image, num_samples):
                w = image.shape[0]
                if invert:
                    image = image.max() - image
                else:
                    image -= image.min()
                p = image.flatten() / image.sum()
                i = np.random.choice(len(p), p=p, size=num_samples, replace=True)
                x, y = np.divmod(i, w)
                samples = np.stack([y / w, (w - x) / w], axis=-1)
                return (samples * 4) - 2

        n = int(apx_homer.shape[0] ** (1 / dim))
        detailed_img = generate_image((detailed_image_bins,) * dim, apx_homer, n, dim)

        xs = jax.numpy.array(sample_img(detailed_img, 10_000_000))

        if animation_bins:
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
            if dim == 1:
                ax1.plot(detailed_img)
                ax2.hist(xs, bins=animation_bins)
            else:
                ax1.imshow(detailed_img, cmap="gray")
                ax2.hist2d(*xs.T, cmap="gray_r", bins=40)
            ax1.set_axis_off()
            ax2.set_axis_off()
            plt.tight_layout()
            plt.show()
    else:
        raise NotImplementedError(f"{data} is not a valid choice.")

    def target_dist(key, batch_dims=None):
        return xs[jax.random.choice(key, xs.shape[0], shape=batch_dims)]

    return prior_dist, target_dist, time_dist


import random


def get_neighbors(u, n):
    """
    Given a vertex u = (x, y) on an n x n grid, 
    return all valid neighbors (up, down, left, right).
    """
    x, y = u
    neighbors = []
    if x > 0:
        neighbors.append((x - 1, y))
    if x < n - 1:
        neighbors.append((x + 1, y))
    if y > 0:
        neighbors.append((x, y - 1))
    if y < n - 1:
        neighbors.append((x, y + 1))
    return neighbors

def wilson_ust(n):
    """
    Generate a uniform spanning tree on an n x n grid using Wilson's algorithm.

    Returns:
      A set of edges, where each edge is a tuple ((x1, y1), (x2, y2)).
      You can interpret this as an undirected edge between those grid points.
    """
    all_vertices = [(x, y) for x in range(n) for y in range(n)]
    
    # Choose an arbitrary root and mark it in the tree
    root = (0, 0)
    in_tree = {root}
    edges = set()
    
    # For each vertex (except the root) we perform the random walk until it hits the tree
    for v in all_vertices:
        if v in in_tree:
            continue
        
        path = [v]
        visited_in_walk = {v: 0}  # Map from vertex -> index in the path
        
        # Continue random-walking until we encounter a vertex in the tree
        while path[-1] not in in_tree:
            current = path[-1]
            neighbors = get_neighbors(current, n)
            # Choose a random neighbor to walk to
            nxt = random.choice(neighbors)
            
            if nxt in visited_in_walk:
                # We have a loop. Remove the loop from the path
                loop_start_idx = visited_in_walk[nxt]
                # Erase the loop
                for removed_vertex in path[loop_start_idx + 1:]:
                    visited_in_walk.pop(removed_vertex, None)
                path = path[:loop_start_idx + 1]
            else:
                visited_in_walk[nxt] = len(path)
                path.append(nxt)
        
        # Now path[-1] is in in_tree, so we link all vertices in path to the tree
        # Each consecutive pair in path is an edge we want to add
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            # Add to the tree
            edges.add(tuple(sorted([a, b])))  # sorted so edge is order-independent
            in_tree.add(a)
            in_tree.add(b)
    
    # Edges were stored as sorted((x1,y1),(x2,y2)) tuples. Let's keep that consistent.
    # Return them in a canonical (smallest-first) form for convenience
    return edges

def sample_uniform_spanning_tree(n):
    """
    Convenience function that calls Wilson's algorithm 
    and returns a uniform spanning tree's edges for an n x n grid.
    """
    return wilson_ust(n)


def sample_points_on_edges(edges, m):
    """
    Given a set of edges and an integer m, return a set of (x, y) coordinates
    corresponding to taking m evenly spaced 'steps' along each edge.
    
    - edges is a set of undirected edges, where each edge is 
      ((x1, y1), (x2, y2)) in sorted order or any consistent order.
    - m is the number of segments (so we will produce m+1 points per edge 
      if we include both endpoints).
    
    Returns:
      A set of (x, y) points.
    """
    points = set()
    for edge in edges:
        # Edge might be stored in a sorted tuple, e.g. ((x1, y1), (x2, y2))
        (x1, y1), (x2, y2) = edge
        
        # We will generate m+1 points if we include endpoints
        for t in range(m + 1):
            frac = t / m  if m > 0 else 0  # handle the case m=0 gracefully
            # Interpolate
            x = x1 + frac * (x2 - x1)
            y = y1 + frac * (y2 - y1)
            points.add((x, y))
    return points


def ust_points_sampler(n=5, m=3):
    # 1. Generate a uniform spanning tree for an n x n grid
    ust_edges = sample_uniform_spanning_tree(n)

    # 2. Sample m evenly spaced points along each edge in the tree
    sampled_points = sample_points_on_edges(ust_edges, m)
    return np.stack(list(sampled_points))


if __name__ == "__main__":
    # Example: generate a 5x5 UST and sample 3 points on each edge
    n = 5
    m = 3

    # 1. Generate a uniform spanning tree for an n x n grid
    ust_edges = sample_uniform_spanning_tree(n)

    # 2. Sample m evenly spaced points along each edge in the tree
    sampled_points = sample_points_on_edges(ust_edges, m)
    
    plt.scatter(*np.stack(list(sampled_points)).T)
    plt.show()
    
    # Let's print or inspect some results
    print(f"Number of edges in the spanning tree: {len(ust_edges)}")
    print(f"First few edges (up to 10): {list(ust_edges)[:10]}")
    print(f"Number of sampled points: {len(sampled_points)}")
    print(f"First few sampled points (up to 10): {list(sampled_points)[:10]}")
