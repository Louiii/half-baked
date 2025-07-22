"""
Very tricky program to write; requirements:
- Jax-md hybrid QM/MM method.
- Dynamic datastructues / many buffers to control chemical bonds changing.

Self-supervised MD:
- Predict time dynamics.

Self-supervised DFT tuning:
- 'mask' region = select small volume / cluster of atoms to perform QM with.
- run the hybrid functions around it.
- in the lattice, mask these points (make a new channel of masking and place gaussians
on that channel, maybe wider gaussians?).
- embed and predict the ground truth lattices for multiple steps into the future.

Downstream:
- Predict all conformations of protein pairs?
  - Generative model sampling conformations.
- Embed huge atomic systems, step through their dynamics.
"""
