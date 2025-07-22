## MD data encoder-decoders

Philosophy/Methodology:
- Have a high resolution lattice grid in 3D.
- Associate each atom with an embedding vector, map all atom embeddings through radially
  local Gaussian to each lattice point.
- Similarly with edge embeddings for covalent bonds.
- (The previous two steps can be made efficient with locality hashing.)
- We can immediately determine the neighbour lattice indices for a central lattice
  index.
- Then we can compute relative attention embeddings for the surrounding neighbourhoods.
- The scope of the neighbourhood should probably be small to limit the initial
  complexity; I don't think there should be any rotationally invariance baked in. This
  is because the following layer will force the relative positions at this initial
  layer.
- The attention op is relative to a fixed size lattice calibrated by the density of a
  liquid; this can be strided with a step size within the scope of the positional
  embeddings.
- 

[Untitled.webm](https://user-images.githubusercontent.com/118839731/203433908-703cbd5b-9137-42f0-b80b-849d04a8c452.webm)

- Define a fixed length L_lattice for the length in angstrom between adjacent lattice
  points.
- Given a set of atoms X = (x_1, ..., x_n) in box B = ((0, w), (0, h), (0, d)), impose a
  lattice on top of this:

```
   - Data: boundary wraps round when similating.
         |--|--|--|--|--|--|--|--|-      lattice points
     ___/ \___               ___/ \___   gaussian scope
     wxyzabcdefghijklmnopqrstuvwxyzabc   data: boundary wraps round when similating
   - Data: assume data is unbounded / from a very large spatial region.
         1  1  1  1            1  1  1  1            1  1  1  1           1  1  1  0     mask
         |--|--|--|  '      '  |--|--|--|  '      '  |--|--|--|  '     '  |--|--|--|     lattice
     ___/ \______/ \___    ___/ \______/ \___    ___/ \______/ \___   ___/ \___/ \___    gaussian scope
     a b c d e f g h i j   g h i j k l m n o     m n o p q r s t u     t u v w x y z
     ^               ^ ^   ^ ^           ^ ^     ^ ^           ^ ^     ^ ^ 
     start            overlap              overlap               overlap
```
Let's say on average in a box (L_lattice)^3 we have 10 atoms, then for N_atoms we would
need n = ((N_atoms) ^ (1/3) / 10) lattice points in a single spatial dimension, so n ^ 3
lattice points. This is roughly the same number of lattice points to atoms in our
system. Using a N_conv convolutions with stride 2, decreases this by (2 ^ 3) ^ N_conv so
if we use 5 convolutions we summarise 8^5 = 32768 lattice points into 1 (8^4 = 4096, 8^3
= 512). This means we can summarise N_atoms into many less points, e.g. for 4
convolutions we would have N_atoms / ((10^(1/3)) * 8^4) = N_atoms / 8824 points to
operate on. It is reasonable to work with sets of size 1M, so we could handle 10B atoms.
For 5 convolutions it would be more like 100B, 6 convolutions around 1T, box side
density around 10K atoms.

With a convolutional architecture we can make a fully invertible architecture:
```
mlp_layer(x) = (linear_{m->m}(x)).^3
mlp(x) = linear_{m->m}(mlp_layer(...[n_layer]...mlp_layer(x)))
encoder_layer(x) = mlp(stride2_conv(x))
encoder(x) = encoder_layer(...[n_layer]...encoder_layer(x))
```
However, it is likely that this is restrictive as we can't do the classic transition
architecture of {n -> 4 * n}.
Also, this is may require convolutional kernels to have certain shapes to be fully
invertible, otherwise we would be able to map n points to < n points, which is of course
not invertible. It's also weird to think that an invertible architecture encoder, could
be initialised with random weights, not require any training, and just immediately work.

Alternatively, could only mirror the convolutional kernels in the decoder conv-transpose
layers, and not update them with their gradients.

NEXT TODO: -- implement atomtype channels into the latticify function. --> Could write
infer the mask from a datastructure which has the atom type represented as an integer,
and mask is defined as atom_type > 0.

Work out how to reconstruct the atom types and coordinates from the voxels efficiently.



----

# these can be done in parallel


Next step (1) is doing encoders based on the 3d image data. The signals in each channel
have a maximum which is constrained by density of lattice / radius of atoms. But it
depends on the atom type etc. This makes it difficult to determine what to normalise by.
This will need some calibration.
  --> Convolutions and Conv-Transpose works well, could just use some standardised conv
  architecture and then develop step (2/4)

One simple / hacky fix would be to normalise wrt to the batch.

Next step (2) can do full lattice -> point decoder.

Next step (3) allow wrapping boundary condition in neighbour voxels.

Next step (4) develop the multi-conformation experimental set-up
`artificial_data/protein.py`. The tricky bit is designing a model architecture to solve
this problem.
- It is actually not obvious of how to mask a patch of an image.
- Problem; x = system of molecules, y = set of conformations of these molecules