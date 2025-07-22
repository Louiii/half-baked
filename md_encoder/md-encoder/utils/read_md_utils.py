import groio
import numpy as np
from pytrr import GroTrrReader


def read_frames(path):
    frames = []
    with GroTrrReader(path) as trrfile:
        for frame in trrfile:
            frames.append({**frame, **trrfile.get_data()})

    title, atoms, box = groio.parse_file(path.replace(".trr", ".gro"))
    box = np.array([float(box[i * 10:(i + 1) * 10]) for i in [0, 1, 2]])
    resname = [a["resname"] for a in atoms]
    atom_name = [a["atom_name"] for a in atoms]
    print(f"Read {title}")
    X = np.stack([f["x"] for f in frames])
    return X, resname, atom_name, box
