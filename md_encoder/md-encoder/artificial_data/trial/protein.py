import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax.random import PRNGKey
from md_encoder.atom_modules.modules import meshgrid
from tqdm import tqdm

TRAIN = ['AABBC', 'AABCC', 'AABBB', 'AACCB', 'AACBC', 'AABCB']
TEST = ['AACCC', 'AACBB']
A2I = dict(zip("ABC", range(3)))

spatial = 2
bond_length = 0.8
seq_len = 5


def generate_sequence(key):
    tail = random.choice(
        key, jnp.arange(2), p=jnp.array([0.7, 0.3]), replace=True, shape=(seq_len - 2,)
    )
    return "AA" + "".join([["B", "C"][t] for t in tail])


def sample_rotation_directions(key, sequence):
    keys = iter(random.split(key, num=seq_len - 2))
    map_ = {
        "B": (lambda: 0),
        "C": (
            lambda: random.choice(
                next(keys), jnp.array([-1, 1]), p=jnp.array([0.95, 0.05])
                )
            ),
    }
    return [map_[char]() for char in sequence[2:]]


def get_rotation(orientation):
    # clockwise
    return jnp.array([
        [jnp.cos(orientation), jnp.sin(orientation)],
        [-jnp.sin(orientation), jnp.cos(orientation)],
    ])


def generate_example(key):
    key, *subkeys = random.split(key, num=4)
    sequence = generate_sequence(subkeys[0])
    directions = sample_rotation_directions(key, sequence)
    initial_coord = random.normal(subkeys[1], (spatial,))
    initial_orientation = random.uniform(subkeys[2], (1,))[0] * jnp.pi * 2
    rotation_matrix = get_rotation(initial_orientation)
    global_rotation = rotation_matrix
    current_position = initial_coord
    coordinates = [initial_coord]
    bond = jnp.array([bond_length, 0])
    difference = global_rotation.dot(bond[:, None])
    current_position += difference[:, 0]
    coordinates.append(current_position)
    # print("-" * 80)
    for d in directions:
        angle = jnp.pi * 0.25 * d
        local_rotation_matrix = get_rotation(angle)
        global_rotation = local_rotation_matrix.dot(global_rotation)
        # print((d, angle, local_rotation_matrix, global_rotation))
        difference = global_rotation.dot(bond[:, None])
        current_position += difference[:, 0]
        coordinates.append(current_position)
    conf_hash = tuple(list(np.array(directions)))
    return jnp.stack(coordinates), sequence, conf_hash


def create_dataset(num=100):
    import collections
    dataset = collections.defaultdict(
        lambda: collections.defaultdict(lambda : ([], 0))
    )
    for i in tqdm(range(num)):
        key = PRNGKey(seed=i)
        coordinates, sequence, conf_hash = generate_example(key)
        example_list, count = dataset[sequence][conf_hash]
        example_list.append(coordinates)
        count += 1
        dataset[sequence][conf_hash] = example_list, count
    return {k: dict(v) for k, v in dataset.items()}


def generate_array_of_proteins(length_scale, g_width, n=100, num_attempts=10, size=100):
    """Repel each protein to fit in a box."""
    threshold = 1.1
    lattice = np.zeros((size, size))
    mg = np.array(meshgrid(np.arange(size), num_dimensions=2).reshape(-1, 2))
    mg = mg.astype(np.float64) * length_scale / size
    test = set(TEST)

    data = []
    i = 0
    for _ in tqdm(range(n)):
        while True:
            key = PRNGKey(seed=i)
            coordinates, sequence, conf_hash = generate_example(key)
            i += 1
            if sequence not in test:
                break
        p = (1e-3 + np.max(lattice) - lattice).reshape(-1).astype(np.float64)
        normed = p / p.sum()
        for _ in range(num_attempts):
            move = mg[np.random.choice(np.arange(mg.shape[0]), p=normed)]
            coords = coordinates + move[None]
            dists = np.linalg.norm(mg[:, None, :] - coords[None, :, :], axis=-1) ** 2
            delta = np.exp(-dists / g_width).sum(-1)
            future = lattice.reshape(-1) + delta
            # import pdb;pdb.set_trace()
            valid_position = not np.any(future > threshold)
            mask = (coords < length_scale) * (0. < coords)
            mask = mask.prod(-1)
            if valid_position and np.any(mask):
                lattice += delta.reshape(*lattice.shape)
                data.append((coords, sequence, mask))
                break
    return data, lattice


def generate_array_of_proteins_pbc(length_scale, g_width, n=100, num_attempts=10, size=100):
    """Repel each protein to fit in a box."""
    threshold = 1.1
    lattice = np.zeros((size, size, len(A2I)))
    mg = np.array(meshgrid(np.arange(size), num_dimensions=2).reshape(-1, 2))
    mg = mg.astype(np.float64) * length_scale / size
    test = set(TEST)
    replicates = meshgrid(np.array([-1, 0, 1]), num_dimensions=2).reshape(-1, 2)
    replicates = length_scale * np.array(replicates)

    data = []
    i = 0
    for _ in tqdm(range(n)):
        while True:
            key = PRNGKey(seed=i)
            coordinates, sequence, conf_hash = generate_example(key)
            i += 1
            if sequence not in test:
                break
        anylat = lattice.sum(-1)
        p = (1e-3 + np.max(anylat) - anylat).reshape(-1).astype(np.float64)
        normed = p / p.sum()
        for _ in range(num_attempts):
            move = mg[np.random.choice(np.arange(mg.shape[0]), p=normed)]
            coords = coordinates + move[None]
            over_the_boundary = np.mod(coords, length_scale)
            new = np.abs(over_the_boundary - coords).sum(-1) > 1e-5

            coords = (coords[:, None, :] + replicates[None, :, :]).reshape(-1, 2)

            dists = np.linalg.norm(mg[:, None, :] - coords[None, :, :], axis=-1) ** 2
            atom = np.zeros((len(sequence), len(A2I))) 
            for j, c in enumerate(sequence):
                atom[j, A2I[c]] = 1.
            atom = np.repeat(atom, 9, axis=0)
            delta = (np.exp(-dists / g_width)[:, :, None] * atom[None, :, :]).sum(1)
            future = anylat.reshape(-1) + delta.sum(-1)
            # import pdb;pdb.set_trace()
            valid_position = not np.any(future > threshold)
            if valid_position:
                lattice += delta.reshape(*lattice.shape)
                data.append((over_the_boundary, sequence, new))
                break
    return data, lattice


def tidy_data():
    # keep the border one voxel adjacent to the grid
    # remove anything outside of this
    # OR JUST GENERATE DATA WITH PERIODIC BOUNDARY CONDITION
    return


def plot():
    import matplotlib.pyplot as plt
    for i in range(10):
        key = PRNGKey(seed=i)
        coordinates, sequence, _ = generate_example(key)
        plt.plot(*coordinates.T)
        for char, (x, y) in zip(sequence, coordinates):
            plt.text(x, y, s=char)
    plt.show()


if __name__ == '__main__':
    # data = create_dataset(num=1000)
    # np.save("data.npy", data)

    import matplotlib.pyplot as plt

    size = 40
    length_scale = 10.0

    data, lattice = generate_array_of_proteins_pbc(
        length_scale=length_scale, g_width=0.2, n=50, num_attempts=5, size=size
    )

    plt.rcParams["figure.figsize"] = (10, 10)
    cm = ["Purples", "Oranges", "Greens"]
    ax = plt.subplot()
    for i in range(lattice.shape[-1]):
        ax.imshow(lattice[..., i].T, alpha=0.4, cmap=cm[i])
    for coordinates, sequence, new in data:
        coordinates *= size / length_scale
        plt.plot(*coordinates[new == True].T, "k")
        plt.plot(*coordinates[new == False].T, "k")
        for char, (x, y) in zip(sequence, coordinates):
            plt.text(x, y, s=char)
    plt.title("PBC")
    plt.show()

    # data, lattice = generate_array_of_proteins(
    #     length_scale=length_scale, g_width=0.2, n=50, num_attempts=50, size=size
    # )

    # plt.rcParams["figure.figsize"] = (10, 10)
    # plt.matshow(lattice.T)
    # for coordinates, sequence, mask in data:
    #     coordinates *= size / length_scale
    #     plt.plot(*coordinates.T)
    #     for char, (x, y), m in zip(sequence, coordinates, mask):
    #         plt.text(x, y, s=char if m else char.lower())
    # plt.show()
