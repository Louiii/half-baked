# -*- coding: utf-8 -*-
"""
Tweak the code to:
- Define a relaxation time / prob dist? for each chain to flatten.
- Randomly select left / right to fold?
- Scale with neighbour code.
- jit with ML model in the loop!

Define ML tasks:
(1) Time dynamics:
  - If the saved data takes ages to load only save the first 5 frames.
  - Load in the data.
  - Make an inverse CNN encoder one layer deep.
  - Play about with resolution.

Original file is located at
    https://colab.research.google.com/drive/17h2Bj078YiKQoIrVcsdY1KxKE9KV207C
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# #@title NPT_TUTORIAL Imports & Utils
# 
# !pip install -q jax-md

from jax.config import config ; config.update('jax_enable_x64', True)
import os
import time
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax import jit, lax, random, vmap
from jax_md import dataclasses, energy, minimize, quantity, simulate, smap, space, util

# from jax_md.colab_tools import renderer
from jax_md.smap import _kwargs_to_bond_parameters, high_precision_sum

# set_matplotlib_formats('pdf', 'svg')

background_color = [56 / 256] * 3


sns.set_style(style='white')

def format_plot(x, y):  
  plt.xlabel(x, fontsize=20)
  plt.ylabel(y, fontsize=20)


def finalize_plot(shape=(1, 1)):
  plt.gcf().set_size_inches(
    shape[0] * 1.5 * plt.gcf().get_size_inches()[1], 
    shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
  plt.tight_layout()

#@title chain data

def make_chains(box_size, num_1d, prng_key):
  colours = jnp.array([
      [125, 148, 181],
      [200, 183, 166],
      [194, 149, 145],
      [112, 63, 55],
      [182, 193, 153],
  ]) / 255

  chain_length = 5
  train_chain_probabilities = [0.1, 0.1, 0.2, 0.2, 0.1, 0.3]
  train_data_split = ['AABBC', 'AABCC', 'AABBB', 'AACCB', 'AACBC', 'AABCB']
  test_data_split = ['AACCC', 'AACBB']
  all_molecule_types = train_data_split + test_data_split
  a2i = dict(zip("ABC", range(3)))
  molecule_type_atom = jnp.array(
      [[a2i[c] for c in s] for s in all_molecule_types], dtype=jnp.int32
  )

  def make_theta_state(seq):
    """
    Args:
        seq[str]: "AA" followed by three chars, from {'B', 'C'}

    Returns:
        states[jnp.array[num_states, num_angles]]
    """
    assert seq in all_molecule_types
    states = [[0, 0, 0]]
    for i, c in enumerate(seq[2:]):
      prev = list(states[-1])
      if c == "C":
        prev[i] = jnp.pi / 4
        states.append(prev)
    states += [states[-1]] * (4 - len(states))
    return jnp.array(states)

  # [n_molecule_types, n_stages, chain_length - 2)]
  theta_stages = jnp.stack([make_theta_state(seq=s) for s in all_molecule_types])


  _1d_pos = jnp.linspace(0, box_size, num_1d + 1)[:num_1d]
  _1d_pos += 0.5 * box_size / num_1d
  num_chains_lengthwise = num_1d // chain_length
  total_chains = num_1d * num_chains_lengthwise


  def chain_data(atom_index, start_x, start_y):
    # [chain_length,]
    chain_positions_x = start_x + jnp.linspace(
        0, box_size / num_chains_lengthwise, chain_length + 1
    )[:-1]
    # [chain_length,]
    chain_position_y = jnp.ones(chain_length) * start_y
    # [chain_length, 2]
    chain_positions = jnp.stack([chain_positions_x, chain_position_y]).T
    # [chain_length - 1 = n_bonds_per_chain, 2] : ixs
    chain_bonds = jnp.stack([
        atom_index + jnp.arange(chain_length - 1),
        atom_index + jnp.arange(1, chain_length),
    ]).T
    num_chain_angle_bonds = chain_length - 2
    # [chain_length, num_chain_angle_bonds] : ixs
    chain_angle_bonds = jnp.stack([
        atom_index + jnp.arange(0, chain_length - 2),
        atom_index + jnp.arange(1, chain_length - 1),
        atom_index + jnp.arange(2, chain_length - 0),
    ]).T

    molecule_ix = atom_index // chain_length
    atom_types = molecule_type_atom[molecule_type[molecule_ix]]
    species = atom_types[2:]
    # # [num_chain_angle_bonds,] : ixs
    # chain_angle_species = jnp.array(species, dtype=jnp.int32)
    # [chain_length,] : ixs
    atom_species = jnp.array(atom_types, dtype=jnp.int32)
    # [chain_length, 3]
    chain_atom_colors = colours[atom_species]
    # [2,] : ixs
    chain_ends = jnp.array([atom_index, atom_index + chain_length - 1], dtype=jnp.int32)
    return (
        jnp.mod(chain_positions, box_size),
        chain_bonds,
        chain_angle_bonds,
        # chain_angle_species,
        chain_atom_colors,
        chain_ends
    )

  n_training = len(train_data_split)
  molecule_type = jax.random.choice(
      prng_key,
      jnp.arange(n_training, dtype=jnp.int32),
      p=jnp.array(train_chain_probabilities),
      shape=(total_chains,)
  )

  def make_row_of_chains(atom_index, start_y):
    start_y = start_y.repeat(num_chains_lengthwise)
    atom_indices = atom_index + jnp.arange(num_chains_lengthwise) * chain_length
    start_x = _1d_pos[::chain_length] + 2 * start_y
    *out, chain_ends = jax.vmap(chain_data)(atom_indices, start_x, start_y)
    reshape = lambda arr: arr.reshape(np.prod(arr.shape[:2]), *arr.shape[2:])
    return *jax.tree_map(reshape, out), chain_ends

  *rows, chain_ends = jax.vmap(make_row_of_chains)(num_1d * jnp.arange(num_1d), _1d_pos)
  reshape = lambda arr: arr.reshape(np.prod(arr.shape[:2]), *arr.shape[2:])

  positions, bonds, bonds_for_angle, colors = jax.tree_map(reshape, rows)

  dynamic_theta_args = (theta_stages, chain_ends.reshape(-1, 2), molecule_type)
  return positions, bonds, bonds_for_angle, colors, dynamic_theta_args


def plot_chains(pos, rad, colors, size=800):
  return renderer.render(
      box_size,
      {
          'chains': renderer.Disk(
              pos,
              jnp.ones(positions.shape[0]) * rad,
              color=colors,
          ),
      },
      resolution=(size, size),
  )

# box_size = 60.0
# chain_length = 5
# positions, bonds, bonds_for_angle, colors, dynamic_theta_args = make_chains(
#     box_size=box_size, num_1d=40, prng_key=jax.random.PRNGKey(seed=0)
# )
# plot_chains(positions, 1, colors, size=800)

#@title SMAP Bond Angles

Array = util.Array

DisplacementOrMetricFn = space.DisplacementOrMetricFn

f32 = util.f32


# LR: edited jax_md.smap.bond, this is intended to work on an array with all
# atoms with two bonds
def bond_angles(fn: Callable[..., Array],
         displacement_or_metric: DisplacementOrMetricFn,
         static_bonds: Optional[Array]=None,
         static_bond_types: Optional[Array]=None,
         ignore_unused_parameters: bool=False,
         **kwargs) -> Callable[..., Array]:
  """Promotes a function that acts on a single pair to one on a set of bonds.

  TODO(schsam): It seems like bonds might potentially have poor memory access.
  Should think about this a bit and potentially optimize.

  Args:
    fn: A function that takes an ndarray of pairwise distances or displacements
      of shape `[n, m]` or `[n, m, d_in]` respectively as well as kwargs
      specifying parameters for the function. `fn` returns an ndarray of
      evaluations of shape `[n, m, d_out]`.
    metric: A function that takes two ndarray of positions of shape
      `[spatial_dimension]` and `[spatial_dimension]` respectively and returns
      an ndarray of distances or displacements of shape `[]` or `[d_in]`
      respectively. The metric can optionally take a floating point time as a
      third argument.
    static_bonds: An ndarray of integer pairs wth shape `[b, 2]` where each
      pair specifies a bond. `static_bonds` are baked into the returned compute
      function statically and cannot be changed after the fact.
    static_bond_types: An ndarray of integers of shape `[b]` specifying the
      type of each bond. Only specify bond types if you want to specify bond
      parameters by type. One can also specify constant or per-bond parameters
      (see below).
    ignore_unused_parameters: A boolean that denotes whether dynamically
      specified keyword arguments passed to the mapped function get ignored
      if they were not first specified as keyword arguments when calling
      `smap.bond(...)`.
    kwargs: Arguments providing parameters to the mapped function. In cases
      where no bond type information is provided these should be either

        1. a scalar
        2. an ndarray of shape `[b]`.

      If bond type information is provided then the parameters should be
      specified as either

        1. a scalar
        2. an ndarray of shape `[max_bond_type]`.
        3. a ParameterTree containing a PyTree of parameters and a mapping. See
           ParameterTree for details.
  Returns:
    A function `fn_mapped`. Note that `fn_mapped` can take arguments bonds and
    `bond_types` which will be bonds that are specified dynamically. This will
    incur a recompilation when the number of bonds changes. Improving this
    state of affairs I will leave as a TODO until someone actually uses this
    feature and runs into speed issues.
  """

  # Each call to vmap adds a single batch dimension. Here, we would like to
  # promote the metric function from one that computes the distance /
  # displacement between two vectors to one that acts on two lists of vectors.
  # Thus, we apply a single application of vmap.

  merge_dicts = partial(util.merge_dicts,
                        ignore_unused_parameters=ignore_unused_parameters)

  def compute_fn(R, bonds, bond_types, static_kwargs, dynamic_kwargs):
    Ra = R[bonds[:, 0]]
    Rb = R[bonds[:, 1]]
    Rc = R[bonds[:, 2]]
    _kwargs = merge_dicts(static_kwargs, dynamic_kwargs)
    _kwargs = _kwargs_to_bond_parameters(bond_types, _kwargs)
    # NOTE(schsam): This pattern is needed due to JAX issue #912.
    d = jax.vmap(partial(displacement_or_metric, **dynamic_kwargs), 0, 0)
    dr = d(Ra, Rb, Rc)
    return high_precision_sum(fn(dr, **_kwargs))

  def mapped_fn(R: Array,
                bonds: Optional[Array]=None,
                bond_types: Optional[Array]=None,
                **dynamic_kwargs) -> Array:
    accum = f32(0)
    if bonds is not None:
      accum = accum + compute_fn(R, bonds, bond_types, kwargs, dynamic_kwargs)

    if static_bonds is not None:
      accum = accum + compute_fn(
          R, static_bonds, static_bond_types, kwargs, dynamic_kwargs)


    return accum
  return mapped_fn

#@title Chain Energy

def chain_energy_components(displacement_fn, chain_hyp):
  def bond(r):
    return chain_hyp.k_r * (r - chain_hyp.r_eq) ** 2

  def bond_angle_energy(theta, target_angle):
    return chain_hyp.k_theta * (theta - target_angle) ** 2

  # eps = 1e-4
  def angle_metric(ra, rb, rc, **_kw_args):
    rab = displacement_fn(rb, ra)
    rbc = displacement_fn(rc, rb)
    rab /= jnp.linalg.norm(rab)
    rbc /= jnp.linalg.norm(rbc)
    sin = rab[0] * rbc[1] - rbc[0] * rab[1]
    cos = rab[0] * rbc[0] + rab[1] * rbc[1]
    return jnp.arctan2(sin, cos)

  def non_bonded(rs):
    return energy.soft_sphere(
        rs, sigma=chain_hyp.radius, epsilon=chain_hyp.strength
    )

  return {
      "bond": {"energy": bond, "metric": displacement_fn},
      "angle": {"energy": bond_angle_energy, "metric": angle_metric},
      "non_bonded": {"energy": non_bonded, "metric": displacement_fn}
  }


def create_chain_energy(energy_functions, bonds, bonds_for_angle):
  """
  Args:
    energy_functions: dict
    bonds: [num_bonds, 2] indices of atoms in positions array, two atoms needed
      for a spring bond.
    bonds_for_angle: [num_angle_bonds, 3] indices of atoms in positions array,
      three atoms needed for a bond angle.

  Returns:
    total_chain_energy[function[
      positions,    ; []
      target_theta, ; [num_angle_bonds,] corresponds to axis 0 of `bonds_for_angle`
    ]->[energy: float]]
  """
  def soft_sphere(pos):
    non_bonded_functions = energy_functions["non_bonded"]
    return smap.pair(
      non_bonded_functions["energy"],
      space.canonicalize_displacement_or_metric(non_bonded_functions["metric"]),
    )(pos)

  def bond_energy(pos):
    bond_functions = energy_functions["bond"]
    bond_type = None
    return smap.bond(
      bond_functions["energy"],
      space.canonicalize_displacement_or_metric(bond_functions["metric"]),
      bonds,
      bond_type,
    )(pos)

  # def angle_energy(pos):
  #   angle_functions = energy_functions["angle"]
  #   static_bond_types = None
  #   # change the following to have dynamic bond angle targets
  #   energy_fn = partial(angle_functions["energy"], species=bond_species)
  #   return bond_angles(
  #     energy_fn,
  #     angle_functions["metric"],
  #     bonds_for_angle,
  #     static_bond_types,
  #   )(pos)

  def dynamic_angle_energy(pos, target_theta):
    angle_functions = energy_functions["angle"]
    _static_bond_types = None
    return bond_angles(
      angle_functions["energy"],
      angle_functions["metric"],
      bonds_for_angle,
      _static_bond_types,
    )(pos, target_angle=target_theta)

  # def total_chain_energy(positions):
  #   return (
  #       soft_sphere(positions) + bond_energy(positions) + angle_energy(positions)
  #   )

  def total_chain_energy(positions, target_theta):
    return (
        soft_sphere(positions)
        + bond_energy(positions)
        + dynamic_angle_energy(positions, target_theta)
    )
  return total_chain_energy

"""```
# bonds:  [num_spring_bonds, 3]
# bonds_for_angle:  [num_bond_angles, 3]
total_chain_energy = create_chain_energy(energy_functions, bonds, bonds_for_angle)
total_chain_energy(
    positions, # [n_atoms, 2]
    target_theta, # [num_bond_angles, ]
)
```
"""

# """# NVT"""

# class ChainHyp:
#   strength = 100.0#1000.0
#   radius = jnp.array(1.0)
#   k_r = 10.0
#   k_theta = 10.0
#   r_eq = 1.1224594 * 1.000001
#   theta_eq = jnp.array([0.0, jnp.pi * 0.5])


# box_size = 60.0
# displacement_fn, shift_fn = space.periodic(box_size)
# positions, bonds, bonds_for_angle, colors, dynamic_theta_args = make_chains(
#     box_size=box_size, num_1d=40, prng_key=jax.random.PRNGKey(0)
# )
# chain_hyp = ChainHyp()
# energy_functions = chain_energy_components(displacement_fn, chain_hyp)
# total_chain_energy = create_chain_energy(energy_functions, bonds, bonds_for_angle)

# # keep theta_target static atm
# theta_target = jnp.zeros((bonds_for_angle.shape[0],))
# total_chain_energy = partial(total_chain_energy, target_theta=theta_target)

# def plot_energy_info():
#   #@title Bond Energies
#   from jax_md import energy

#   rs = jnp.linspace(0.5, 2.5)
#   plt.plot(rs, energy.lennard_jones(rs))
#   plt.plot(rs, energy_functions["non_bonded"]["energy"](rs))
#   plt.plot(rs, energy_functions["bond"]["energy"](rs))

#   plt.ylim([-1, 10])
#   plt.xlim([0, 2.5])
#   plt.xlabel('$r_{ij}$')
#   plt.ylabel('$\\epsilon$')

#   #@title Angle Metric
#   t = jnp.linspace(-jnp.pi, jnp.pi, 100)
#   ra = jnp.array([0, 0])
#   rb = jnp.array([1, 0])
#   rc = jnp.array([1, 0])[None, :] + jnp.stack([jnp.cos(t), jnp.sin(t)]).T

#   plt.plot(t, [energy_functions["angle"]["metric"](ra, rb, rc_) for rc_ in rc])
#   plt.xlabel('true theta')
#   plt.ylabel('recovered theta')


# def run_minimization(energy_fn, R_init, shift, num_steps=50):
#   """Simple EM, no record"""
#   dt_start = 0.001
#   dt_max   = 0.004
#   init, apply = minimize.fire_descent(
#       jit(energy_fn), shift, dt_start=dt_start, dt_max=dt_max
#   )
#   apply = jit(apply)

#   @jit
#   def scan_fn(state, i):
#     return apply(state), 0.

#   state = init(R_init)
#   state, _ = jax.lax.scan(scan_fn,state,np.arange(num_steps))

#   return state.position, np.amax(np.abs(-jax.grad(energy_fn)(state.position)))

# k = jax.random.PRNGKey(seed=0)
# # add noise to the positions
# position_noise = jax.random.normal(k, shape=positions.shape) * 0.3
# noisy_positions = positions + position_noise
# # relax everything
# positions_relaxed, max_force_component = run_minimization(
#     total_chain_energy, noisy_positions, shift_fn
# )

# #@title Standard NVT-Langevin

# @dataclasses.dataclass
# class Chains:
#   chains: simulate.NVTLangevinState

# simulation_steps = 10000
# write_every = 50
# write_steps = simulation_steps // write_every

# init_fn, step_fn = simulate.nvt_langevin(total_chain_energy, shift_fn, dt=5e-3, kT=0.0)


# def simulation_fn(i, state_trajectory):
#   state, traj = state_trajectory
#   traj = Chains(traj.chains.at[i].set(state.chains.position))

#   def total_step_fn(_, state): return Chains(step_fn(state.chains))

#   state = lax.fori_loop(0, write_every, total_step_fn, state)
#   return state, traj

# n = positions_relaxed.shape[0]

# key = jax.random.PRNGKey(seed=0)

# position_start = jnp.mod(positions_relaxed, box_size)

# state = Chains(init_fn(key, position_start),)
# trajectory = Chains(jnp.zeros((write_steps, n, 2)),)

# state, trajectory = lax.fori_loop(0, write_steps, simulation_fn, (state, trajectory))
# print(
#     f"original positions potential energy: {total_chain_energy(positions)};\n"
#     f"noisy positions potential energy: {total_chain_energy(noisy_positions)};\n"
#     f"relaxed positions potential energy: {total_chain_energy(position_start)};\n"
#     f"final positions potential energy: {total_chain_energy(state.chains.position)};\n"
# )

# plot_chains(trajectory.chains, chain_hyp.radius, colors, size=800)

# """...trying the deterministic NVT simulation routine..."""


# #@title Deterministic NVT-Nose-Hoover


# class DeterministicChainHyp:
#   dt = 5e-5
#   kT  = 0.1
#   tau = 100
# dchyp = DeterministicChainHyp()

# @dataclasses.dataclass
# class Chains:
#   chains: simulate.NVTNoseHooverState

# simulation_steps = 30_000 # ~3 min for 100K steps
# write_every = 100
# write_steps = simulation_steps // write_every

# init_fn, step_fn = simulate.nvt_nose_hoover(
#     total_chain_energy,
#     shift_fn,
#     dt=dchyp.dt,
#     kT=dchyp.kT,
#     tau=dchyp.tau,
# )


# def simulation_fn(i, state_trajectory):
#   state, traj = state_trajectory
#   traj = Chains(traj.chains.at[i].set(state.chains.position))

#   def total_step_fn(_, state):
#     return Chains(step_fn(state.chains))
#   state = lax.fori_loop(0, write_every, total_step_fn, state)
#   return state, traj


# def run(start_positions):
#   state, traj = lax.fori_loop(
#       0,
#       write_steps,
#       simulation_fn,
#       (
#           Chains(init_fn(key, start_positions)),
#           Chains(jnp.zeros((write_steps, n, 2))),
#       ),
#   )
#   return traj

# n = positions_relaxed.shape[0]
# key = jax.random.PRNGKey(seed=0)

# noisy_trajectory = run(noisy_positions)

# positions_relaxed, max_force_component = run_minimization(
#     total_chain_energy, positions, shift_fn, num_steps=5000,
# )
# position_start = jnp.mod(positions_relaxed + 0.742 * box_size, box_size)

# relaxed_trajectory = run(position_start)

# """```
# Chains(chains=NVTNoseHooverState(position, momentum, force, mass, chain=NoseHooverChain(position=DeviceArray([0., 0., 0., 0., 0.], dtype=float32), momentum=DeviceArray([0., 0., 0., 0., 0.], dtype=float32), mass=DeviceArray([3.2e+06, 1.0e+03, 1.0e+03, 1.0e+03, 1.0e+03], dtype=float32), tau=DeviceArray(100., dtype=float32), kinetic_energy=DeviceArray(157.94955, dtype=float32), degrees_of_freedom=3200)))
# ```


# """

# plot_chains(noisy_trajectory.chains, chain_hyp.radius, colors, size=800)

# plot_chains(relaxed_trajectory.chains, chain_hyp.radius, colors, size=800)

# _state = init_nvt(key, positions)

# _state.chain

# state.chain

# jax.tree_map(lambda x: x.shape, state.chain)

# #@title Full NVT Simulation Deterministic (changing temperature)

# # create system
# box_size = 60.0 # 56.377 # final size from npt
# displacement_fn, shift_fn = space.periodic(box_size)
# positions, bonds, bonds_for_angle, colors, dynamic_theta_args = make_chains(
#     box_size=box_size, num_1d=40, prng_key=jax.random.PRNGKey(0)
# )

# # create energy function
# class ChainHyp:
#   strength = 1000.0
#   radius = jnp.array(1.0)
#   k_r = 10.0
#   k_theta = 10.0
#   r_eq = 1.1224594 * 1.000001
#   theta_eq = jnp.array([0.0, jnp.pi * 0.5])
# chain_hyp = ChainHyp()
# energy_functions = chain_energy_components(displacement_fn, chain_hyp)
# total_chain_energy = create_chain_energy(energy_functions, bonds, bonds_for_angle)


# # ! keep theta_target static atm
# theta_target = jnp.zeros((bonds_for_angle.shape[0],))
# total_chain_energy = partial(total_chain_energy, target_theta=theta_target)


# class DeterministicHyp:
#   dt = 5e-3
#   kT_initial = 0.1
#   kT_final = 0.01

# dhyp = DeterministicHyp()

# # create dynamic temperature, as our thermostat can adjust to this temperature
# kT = lambda t: jnp.where(t < 5000.0 * dhyp.dt, dhyp.kT_initial, dhyp.kT_final)

# # create the simulation function


# def wrap_energy(pos, **var_kw): return total_chain_energy(pos)
# init_nvt, apply_nvt = simulate.nvt_nose_hoover(wrap_energy, shift_fn, dhyp.dt, kT(0.))
# key = jax.random.PRNGKey(seed=0)
# state = init_nvt(key, positions)


# # minimise energy

# def run_minimization(energy_fn, R_init, shift, num_steps=50):
#   """Simple EM, no record"""
#   dt_start = 0.001
#   dt_max   = 0.004
#   init_em, apply_em = minimize.fire_descent(
#       jit(energy_fn), shift, dt_start=dt_start, dt_max=dt_max
#   )
#   apply_em = jit(apply_em)

#   @jit
#   def scan_fn(state, i):
#     return apply_em(state), 0.
#   state = init_em(R_init)
#   state, _ = jax.lax.scan(scan_fn,state,np.arange(num_steps))
#   return state.position, np.amax(np.abs(-jax.grad(energy_fn)(state.position)))
# position, max_force_component = run_minimization(total_chain_energy, state.position, shift_fn, num_steps=5000)

# # create some random velocities for each chain
# state = simulate.NVTNoseHooverState(
#     position=position,
#     momentum=state.momentum,
#     force=state.force,
#     mass=state.mass,
#     chain=state.chain
# )

# # simulate NVT
# write_every = 100

# def step_fn_nvt(i, state_and_log):
#   state, log = state_and_log
#   t = i * dhyp.dt

#   # Log information about the simulation.
#   T = quantity.temperature(momentum=state.momentum)
#   log['kT'] = log['kT'].at[i].set(T)
#   H = simulate.nvt_nose_hoover_invariant(total_chain_energy, state, kT(t))
#   log['H'] = log['H'].at[i].set(H)
#   # Record positions every `write_every` steps.
#   log['position'] = lax.cond(i % write_every == 0,
#                              lambda p: \
#                              p.at[i // write_every].set(state.position),
#                              lambda p: p,
#                              log['position'])
#   state = apply_nvt(state, kT=kT(t))
#   return state, log

# steps = 10000
# log = {
#     'kT': jnp.zeros((steps,)),
#     'H': jnp.zeros((steps,)),
#     'position': jnp.zeros((steps // write_every,) + state.position.shape) 
# }

# state, log = lax.fori_loop(0, steps, step_fn_nvt, (state, log))

# plot_chains(log["position"], chain_hyp.radius, colors, size=800)

# plt.plot(log["H"])

# plt.plot(log["kT"])



"""# Dynamic Theta Test"""


# test data


# def make(a):
#   chain1 = jnp.stack([jnp.linspace(1, 5, 5), jnp.zeros(5)]).T
#   chain2 = jnp.stack([jnp.linspace(1, 5, 5), jnp.ones(5) * a]).T
#   chain3 = jnp.stack([jnp.linspace(1, 5, 5), jnp.ones(5) * 30]).T
#   chains = [chain1, chain2, chain3]
#   chains = jnp.concatenate(chains, axis=0)
#   return chains

# n_molecules = 3

# dists = [3.5, 2.5, 1.5, 1.5] + [2.5] * 13
# positions = jnp.stack([make(a) for a in dists])
# chain_ends_ix = jnp.array([[i * 5, ((i + 1) * 5) - 1] for i in range(n_molecules)], jnp.int32)
# molecule_type = jnp.zeros(n_molecules, jnp.int32)
# theta_stages = jnp.array([[[0,0,0,0], [0.1,0.1,0.2,0.3]]])
# theta = jnp.zeros((n_molecules, 4))
# positions
# def displacement(a, b):
#   return b-a

# chain_ends_ix

TRAIN = ['AABBC', 'AABCC', 'AABBB', 'AACCB', 'AACBC', 'AABCB']
TEST = ['AACCC', 'AACBB']
ALL_MOLECULE_TYPES = TRAIN + TEST
A2I = dict(zip("ABC", range(3)))

def make_theta_state(seq):
  """
  Args:
      seq[str]: "AA" followed by three chars, from {'B', 'C'}

  Returns:
      states[jnp.array[num_states, num_angles]]
  """
  assert (seq in (TRAIN + TEST))
  states = [[0, 0, 0]]
  for i, c in enumerate(seq[2:]):
    prev = list(states[-1])
    if c == "C":
      prev[i] = jnp.pi / 4
      states.append(prev)
  states += [states[-1]] * (4 - len(states))
  return jnp.array(states)

# # static
# # n_molecules = ...
# n_molecule_types = 2 ** 3
# n_angles_per_chain = 3
# chain_length = 5
# # chain_ends_ix = ... # [n_molecules, 2]
# reaction_threshold = 2.0
# n_transition_steps = 10
# # molecule_type = ... # [n_molecules]
# theta_stages = jnp.stack([make_theta_state(seq=s) for s in ALL_MOLECULE_TYPES]) # [n_molecule_types, n_stages, chain_length - 2)]
# # dynamic
# molecule_stage = jnp.zeros((n_molecules,), dtype=jnp.int32)
# transitioning = jnp.zeros((n_molecules,), dtype=jnp.int32)
# theta = jnp.zeros((n_molecules, n_angles_per_chain))

# @jax.jit
# def step_theta(positions, theta, molecule_stage, transitioning):
#   # compute molecules which are close to one another, by ends being close to one
#   # another for the same molecules.
#   start_positions = positions[chain_ends_ix[:, 0]]
#   end_positions = positions[chain_ends_ix[:, 1]]
#   # print(f"start_positions: {start_positions}")
#   # print(f"end_positions: {end_positions}")

#   def chain_ends_dists(pos_from, pos_to):
#     (pos1, pos2), (pos3, pos4) = pos_from, pos_to
#     # print(f"displacement(pos1, pos3): {displacement(pos1, pos3)}")
#     # print(f"displacement(pos2, pos4): {displacement(pos2, pos4)}")
#     active1 = jnp.linalg.norm(displacement(pos1, pos3)) < reaction_threshold
#     active2 = jnp.linalg.norm(displacement(pos2, pos4)) < reaction_threshold
#     return active1 * active2

#   dist_pair = vmap(vmap(chain_ends_dists, in_axes=(None, 0)), in_axes=(0, None))
#   # start <-> start, end <-> end
#   original_labeling = dist_pair(
#       (start_positions, end_positions), (start_positions, end_positions)
#   )
#   # start <-> end, end <-> start
#   alt_labeling = dist_pair(
#       (start_positions, end_positions), (end_positions, start_positions)
#   )
#   # print((original_labeling, alt_labeling))
#   active_pairs = original_labeling + alt_labeling > 0  # labeling OR alt_labeling
#   # remove self-interaction
#   active_pairs = active_pairs - jnp.eye(active_pairs.shape[0])
#   active = jnp.any(active_pairs, axis=0) * jnp.any(active_pairs, axis=1)
#   # print(f"active: {active.astype(int)}")

#   # get molecules which are available to start transitioning to the next angle.
#   available = transitioning == 0
#   # print(f"available: {available.astype(int)}")

#   # update the molecules which are transitioning
#   activate = (available * active).astype(jnp.int32)
#   transitioning += n_transition_steps * activate
#   # print(f"transitioning: {transitioning}")

#   # take a transitioning step
#   transitioning = jnp.maximum(0, transitioning - 1)

#   # update the next theta
#   def get_theta(stage, mol_type, transition):
#     """Update molecule stage and compute the current theta value"""
#     mol_theta_stage = theta_stages[mol_type]  # [n_stages, chain_length - 1]
#     current_theta = mol_theta_stage[stage]  # [chain_length - 1]
#     # we only have a new theta for the transitioning molecules.
#     next_molecule_stage = stage + (transition > 0)
#     next_theta = mol_theta_stage[next_molecule_stage]
#     progress = (n_transition_steps - transition) / n_transition_steps
#     inter_theta = current_theta * (1 - progress) + progress * next_theta
#     # if we are on a final transition step, update the theta stage
#     new_stage = stage + (transition == 1)
#     return new_stage, inter_theta

#   molecule_stage, theta = vmap(get_theta)(
#       molecule_stage, molecule_type, transitioning
#   )
#   return theta, molecule_stage, transitioning

# for p, d in zip(positions, dists):
#   print(f"dist: {d}")
#   print(theta)
#   theta, molecule_stage, transitioning = step_theta(p, theta, molecule_stage, transitioning)
# print(theta)

"""# NPT Chains"""

def create_theta_update(
    displacement,
    theta_stages,  # [n_molecule_types, n_stages, chain_length - 2)]
    chain_ends_ix,  # [n_molecules, 2]
    molecule_type,  # [n_molecules]
    n_transition_steps,  # int
    reaction_threshold,  # float
):
  def step_theta(positions, target_theta, molecule_stage, transitioning):
    # compute molecules which are close to one another, by ends being close to one
    # another for the same molecules.
    start_positions = positions[chain_ends_ix[:, 0]]
    end_positions = positions[chain_ends_ix[:, 1]]

    def chain_ends_dists(pos_from, pos_to):
      (pos1, pos2), (pos3, pos4) = pos_from, pos_to
      active1 = jnp.linalg.norm(displacement(pos1, pos3)) < reaction_threshold
      active2 = jnp.linalg.norm(displacement(pos2, pos4)) < reaction_threshold
      return active1 * active2

    dist_pair = vmap(vmap(chain_ends_dists, in_axes=(None, 0)), in_axes=(0, None))
    # start <-> start, end <-> end
    original_labeling = dist_pair(
        (start_positions, end_positions), (start_positions, end_positions)
    )
    # start <-> end, end <-> start
    alt_labeling = dist_pair(
        (start_positions, end_positions), (end_positions, start_positions)
    )
    active_pairs = original_labeling + alt_labeling > 0  # labeling OR alt_labeling
    # remove self-interaction
    active_pairs = active_pairs - jnp.eye(active_pairs.shape[0])
    active = jnp.any(active_pairs, axis=0) * jnp.any(active_pairs, axis=1)

    # get molecules which are available to start transitioning to the next angle.
    available = transitioning == 0

    # update the molecules which are transitioning
    activate = (available * active).astype(jnp.int32)
    transitioning += n_transition_steps * activate

    # take a transitioning step
    transitioning = jnp.maximum(0, transitioning - 1)

    # update the next theta
    def get_theta(stage, mol_type, transition):
      """Update molecule stage and compute the current theta value"""
      mol_theta_stage = theta_stages[mol_type]  # [n_stages, chain_length - 1]
      current_theta = mol_theta_stage[stage]  # [chain_length - 1]
      # we only have a new theta for the transitioning molecules.
      next_molecule_stage = stage + (transition > 0)
      next_theta = mol_theta_stage[next_molecule_stage]
      progress = (n_transition_steps - transition) / n_transition_steps
      inter_theta = current_theta * (1 - progress) + progress * next_theta
      # if we are on a final transition step, update the theta stage
      new_stage = stage + (transition == 1)
      return new_stage, inter_theta

    molecule_stage, target_theta = vmap(get_theta)(
        molecule_stage, molecule_type, transitioning
    )
    return target_theta, molecule_stage, transitioning
  return step_theta


def run_minimization(energy_fn, R_init, shift, num_steps=50):
  init_em, apply_em = minimize.fire_descent(
      jit(energy_fn), shift, dt_start=0.001, dt_max=0.004
  )
  apply_em = jit(apply_em)

  @jit
  def scan_fn(state, i):
    return apply_em(state), 0.
  state = init_em(R_init)
  state, _ = jax.lax.scan(scan_fn, state, np.arange(num_steps))
  return state.position, np.amax(np.abs(-jax.grad(energy_fn)(state.position)))


def random_init_momenta_per_chain(key, state, chain_length, kT):
  """Try making a state with random momenta per chain...."""
  num_chains = state.position.shape[0] // chain_length

  # def initialize_momenta(state: T, key: Array, kT: float) -> T:
  #   """Initialize momenta with the Maxwell-Boltzmann distribution."""
  #   R, mass = state.position, state.mass

  #   R, treedef = tree_flatten(R)
  #   mass, _ = tree_flatten(mass)
  #   keys = random.split(key, len(R))

  #   def initialize_fn(k, r, m):
  #     p = jnp.sqrt(m * kT) * random.normal(k, r.shape, dtype=r.dtype)
  #     # If simulating more than one particle, center the momentum.
  #     if r.shape[0] > 1:
  #       p = p - jnp.mean(p, axis=0, keepdims=True)
  #     return p

  #   P = [initialize_fn(k, r, m) for k, r, m in zip(keys, R, mass)]

  #   return state.set(momentum=tree_unflatten(treedef, P))

  key, split = jax.random.split(key)
  r = state.position
  m = state.mass
  per_chain_p = jax.random.normal(split, (num_chains,) + r.shape[1:], dtype=r.dtype)
  p = jnp.sqrt(m * kT) * jnp.repeat(per_chain_p, chain_length, axis=0)
  p = p - jnp.mean(p, axis=0, keepdims=True)

  state = state.set(momentum=p)
  return state


def npt_log_fn(log, i, t, state, target_theta, write_every, wrap_energy, pressure_fn, kT):
  unit_pos = state.position
  box = simulate.npt_box(state)

  pos = space.transform(box, unit_pos)

  log['PE'] = log['PE'].at[i].set(
      wrap_energy(pos, target_theta=target_theta.reshape(-1)) / pos.shape[0]
  )

  KE = 0.5 * (state.mass ** -1) * (state.momentum ** 2).sum(-1)
  log['KE'] = log['KE'].at[i].set(jnp.mean(KE))

  # Log information about the simulation.
  T = quantity.temperature(momentum=state.momentum)
  log['kT'] = log['kT'].at[i].set(T)
  log["box"] = log['box'].at[i].set(box)

  KE = quantity.kinetic_energy(momentum=state.momentum)
  P_measured = quantity.pressure(
    wrap_energy, state.position, box, KE, target_theta=target_theta.reshape(-1)
  )
  log['P'] = log['P'].at[i].set(P_measured)

  H = simulate.npt_nose_hoover_invariant(
    wrap_energy, state, pressure_fn(t), kT, target_theta=target_theta.reshape(-1)
  )
  log['H'] = log['H'].at[i].set(H)

  log['position'] = lax.cond(i % write_every == 0,
                             lambda p: p.at[i // write_every].set(pos),
                             lambda p: p,
                             log['position'])
  return log


def npt_step_fn(
  i,
  state_log_dynamic_theta,
  pressure_fn,
  write_every,
  step_theta_fn,
  apply,
  dt,
  wrap_energy,
  kT,
):
  state, log, dynamic_theta_vars = state_log_dynamic_theta
  target_theta, molecule_stage, transitioning = dynamic_theta_vars
  t = i * dt

  log = npt_log_fn(
    log, i, t, state, target_theta, write_every, wrap_energy, pressure_fn, kT
  )

  # update target theta
  target_theta, molecule_stage, transitioning = step_theta_fn(
      state.position, target_theta, molecule_stage, transitioning
  )

  # Take a simulation step.
  state = apply(state, pressure=pressure_fn(t), target_theta=target_theta.reshape(-1))

  return state, log, (target_theta, molecule_stage, transitioning)


def run_npt(
  state,
  apply,
  step_theta_fn,
  target_theta,
  molecule_stage,
  transitioning,
  pressure_fn,
  write_every,
  steps,
  dt,
  wrap_energy,
  kT,
):
  z = lambda: jnp.zeros((steps,))
  log = {
      'PE': z(),
      'KE': z(),
      'kT': z(),
      'P': z(),
      'H': z(),
      'position': jnp.zeros((steps // write_every,) + state.position.shape),
      'box': jnp.zeros((steps, 2, 2)),
  }

  dynamic_theta_vars = (target_theta, molecule_stage, transitioning)
  step_fn = partial(
    npt_step_fn,
    pressure_fn=pressure_fn,
    write_every=write_every,
    step_theta_fn=step_theta_fn,
    apply=apply,
    dt=dt,
    wrap_energy=wrap_energy,
    kT=kT,
  )
  state, log, dynamic_theta_vars = lax.fori_loop(
    0,
    steps,
    jax.jit(step_fn),
    (state, log, dynamic_theta_vars)
  )
  return state, log, dynamic_theta_vars


def setup_system(num_steps_minimisation):
  box_size = 60.0 # 56.377 # final size from npt
  chain_length = 5
  positions, bonds, bonds_for_angle, colors, dynamic_theta_args = make_chains(
      box_size=box_size, num_1d=40, prng_key=jax.random.PRNGKey(seed=0)
  )

  # create energy function
  class ChainEnergyHyp:
    strength = 1000.0
    radius = jnp.array(1.0)
    k_r = 10.0
    k_theta = 10.0
    r_eq = 1.1224594 * 1.000001
    theta_eq = jnp.array([0.0, jnp.pi * 0.5])


  # generate initial positions
  displacement_fn, shift_fn = space.periodic(box_size)
  chain_energy_hyp = ChainEnergyHyp()
  energy_functions = chain_energy_components(displacement_fn, chain_energy_hyp)
  _total_chain_energy = create_chain_energy(energy_functions, bonds, bonds_for_angle)

  # initially set theta_target to 0
  target_theta = jnp.zeros((bonds_for_angle.shape[0],))
  total_chain_energy = partial(_total_chain_energy, target_theta=target_theta)

  position, max_force_component = run_minimization(
      total_chain_energy, positions, shift_fn, num_steps=num_steps_minimisation
  )
  positions_in_unit_box = position / box_size
  return positions_in_unit_box, colors, dynamic_theta_args, chain_energy_hyp, chain_length, box_size, bonds, bonds_for_angle


def run_npt_chain_system(
  box_size,
  chain_energy_hyp,
  positions_in_unit_box,
  bonds,
  bonds_for_angle,
  write_every,
  steps,
  dt,
  kT,
  pressure_fn,
  chain_length,
  dynamic_theta_args,
):
  displacement, shift = space.periodic_general(box_size, fractional_coordinates=True) 
  energy_functions = chain_energy_components(displacement, chain_energy_hyp)
  total_chain_energy = create_chain_energy(energy_functions, bonds, bonds_for_angle)


  # create theta update rule
  theta_stages, chain_ends_ix, molecule_type = dynamic_theta_args
  step_theta_fn = create_theta_update(
      displacement=displacement,
      theta_stages=theta_stages,
      chain_ends_ix=chain_ends_ix,
      molecule_type=molecule_type,
      n_transition_steps=10,
      reaction_threshold=2.0,
  )
  n_molecules = positions_in_unit_box.shape[0] // chain_length
  n_angles_per_chain = chain_length - 2
  molecule_stage = jnp.zeros((n_molecules,), dtype=jnp.int32)
  transitioning = jnp.zeros((n_molecules,), dtype=jnp.int32)
  target_theta = jnp.zeros((n_molecules, n_angles_per_chain))

  def wrap_energy(pos, target_theta=target_theta, **var_kw):
    # if box.ndim > 0:## hoping this box scaling will fix everything, but will be slightly confused if it does.
    #   box = jnp.diag(box)
    # return total_chain_energy(pos * box) * 1e-1 #1e-3
    return total_chain_energy(pos, target_theta=target_theta) * 1e-1 #1e-3

  init, apply = simulate.npt_nose_hoover(wrap_energy, shift, dt, pressure_fn(0.), kT)
  key = jax.random.PRNGKey(0)
  state = init(key, positions_in_unit_box, box_size, target_theta=target_theta.reshape(-1))

  state = random_init_momenta_per_chain(key, state, chain_length, kT)

  state, log, dynamic_theta_vars = run_npt(
    state,
    apply,
    step_theta_fn,
    target_theta,
    molecule_stage,
    transitioning,
    pressure_fn=pressure_fn,
    write_every=write_every,
    steps=steps,
    dt=dt,
    wrap_energy=wrap_energy,
    kT=kT,
  )

  # plot_chains(log["position"], chain_energy_hyp.radius, colors, size=800)
  return state, log, dynamic_theta_vars


def create_nvt_state_from_npt(key, npt_state, box, init_nvt, target_theta):
  # create some random velocities for each chain
  pos = space.transform(box, npt_state.position)
  key, subkey = jax.random.split(key)
  nvt_state = simulate.NVTNoseHooverState(
      position=pos,
      momentum=npt_state.momentum,
      force=npt_state.force,
      mass=npt_state.mass,
      chain=init_nvt(subkey, pos, target_theta=target_theta.reshape(-1)).chain,
      # chain=simulate.NoseHooverChain(
      #     position=,
      #     momentum=,
      #     mass=,
      #     tau=,
      #     kinetic_energy=,
      #     degrees_of_freedom=
      # )
  )
  return key, nvt_state


"""# FINAL

Box := 56.377

P := 0.15

kT := 0.09641
"""


def setup_nvt_system(
  key, npt_state, box, dt, kT, chain_hyp, bonds, bonds_for_angle, dynamic_theta_args, dynamic_theta_vars
):  
  displacement, shift = space.periodic_general(box, fractional_coordinates=False) 
  energy_functions = chain_energy_components(displacement, chain_hyp)
  total_chain_energy = create_chain_energy(energy_functions, bonds, bonds_for_angle)

  # create theta update rule
  theta_stages, chain_ends_ix, molecule_type = dynamic_theta_args
  step_theta_fn = create_theta_update(
      displacement=displacement,
      theta_stages=theta_stages,
      chain_ends_ix=chain_ends_ix,
      molecule_type=molecule_type,
      n_transition_steps=10,
      reaction_threshold=2.0,
  )

  # try to change the target theta
  # n_molecules = positions.shape[0] // chain_length
  # n_angles_per_chain = chain_length - 2
  target_theta, molecule_stage, transitioning = dynamic_theta_vars

  def wrap_energy(pos, target_theta=target_theta, **var_kw):
    return total_chain_energy(pos, target_theta=target_theta) * 1e-1 #1e-3


  init_nvt, apply_nvt = simulate.nvt_nose_hoover(wrap_energy, shift, dt, kT)

  key, nvt_state = create_nvt_state_from_npt(key, npt_state, box, init_nvt, target_theta)
  return key, nvt_state, apply_nvt, target_theta, molecule_stage, transitioning, step_theta_fn, wrap_energy


def log_nvt_fn(log, i, nvt_state, target_theta, molecule_stage, transitioning, kT, wrap_energy, write_every):
  # unit_pos = state.position
  # box = simulate.npt_box(state)
  # Record positions every `write_every` steps.
  # pos = space.transform(box, unit_pos)
  # edge_size = 0.1
  # central_mask = jnp.prod(
  #     (unit_pos > edge_size) * (unit_pos < 1 - edge_size), axis=-1
  # )  # indicator of whether we are at a central atom
  # inc, ninc = debug_compute_potential_energy_components(
  #     pos, include_mask=central_mask, theta_target=target_theta.reshape(-1)
  # )

  # c = log['PE']['components']

  # cnew = {
  #     'central': {k: v.at[i].set(inc[k]) for k, v in c['central'].items()},
  #     'outer': {k: v.at[i].set(ninc[k]) for k, v in c['outer'].items()}
  # }
  # cnew['total'] = {
  #     k: v.at[i].set(cnew['central'][k][i] + cnew['outer'][k][i])
  #     for k, v in c['total'].items()
  # }
  # log['PE']['components'] = cnew
  # log['PE']['total_full_function'] = log['PE']['total_full_function'].at[i].set(
  #     total_chain_energy(pos, target_theta=target_theta.reshape(-1)) / pos.shape[0]
  # )

  # KE = 0.5 * (nvt_state.mass ** -1) * (nvt_state.momentum ** 2).sum(-1)
  # log['KE'] = {
  #     'overall_average': log['KE']['overall_average'].at[i].set(jnp.mean(KE)),
  # }


  # # Log information about the simulation.
  T = quantity.temperature(momentum=nvt_state.momentum)
  log['kT'] = log['kT'].at[i].set(T)
  # log["box"] = log['box'].at[i].set(box)

  # KE = quantity.kinetic_energy(momentum=nvt_state.momentum)
  # P_measured = quantity.pressure(wrap_energy, nvt_state.position, box, KE, target_theta=target_theta.reshape(-1))
  # log['P'] = log['P'].at[i].set(P_measured)

  H = simulate.nvt_nose_hoover_invariant(
    wrap_energy, nvt_state, kT, target_theta=target_theta.reshape(-1)
  )
  log['H'] = log['H'].at[i].set(H)

  log['position'] = lax.cond(i % write_every == 0,
                             lambda p: p.at[i // write_every].set(nvt_state.position),
                             lambda p: p,
                             log['position'])
  log['molecule_stage'] = lax.cond(i % write_every == 0,
                             lambda p: p.at[i // write_every].set(molecule_stage),
                             lambda p: p,
                             log['molecule_stage'])
  log['transitioning'] = lax.cond(i % write_every == 0,
                             lambda p: p.at[i // write_every].set(transitioning),
                             lambda p: p,
                             log['transitioning'])
  return log


def run_nvt(
  nvt_state,
  apply_nvt,
  target_theta,
  molecule_stage,
  transitioning,
  step_theta_fn,
  write_every,
  steps,
  kT,
  wrap_energy,
):
  # simulate NVT
  def step_fn_nvt(i, state_log_dynamic_theta, step_theta_fn):
    nvt_state, log, dynamic_theta_vars = state_log_dynamic_theta
    target_theta, molecule_stage, transitioning = dynamic_theta_vars
    # t = i * dhyp.dt

    log = log_nvt_fn(
      log, i, nvt_state, target_theta, molecule_stage, transitioning, kT, wrap_energy, write_every
    )

    # update target theta
    target_theta, molecule_stage, transitioning = step_theta_fn(
        nvt_state.position, target_theta, molecule_stage, transitioning
    )

    # Take a simulation step.
    nvt_state = apply_nvt(nvt_state, kT=kT, target_theta=target_theta.reshape(-1))

    return nvt_state, log, (target_theta, molecule_stage, transitioning)

  log = {
      'kT': jnp.zeros((steps,)),
      'H': jnp.zeros((steps,)),
      'position': jnp.zeros((steps // write_every,) + nvt_state.position.shape),
      'molecule_stage': jnp.zeros((steps // write_every,) + molecule_stage.shape),
      'transitioning': jnp.zeros((steps // write_every,) + transitioning.shape),
  }
  dynamic_theta_vars = target_theta, molecule_stage, transitioning

  state, log, dynamic_theta_vars = lax.fori_loop(
    0,
    steps,
    jax.jit(partial(step_fn_nvt, step_theta_fn=step_theta_fn)),
    (nvt_state, log, dynamic_theta_vars),
  )

  # plot_chains(log["position"], chain_hyp.radius, colors, size=800)
  return state, log, dynamic_theta_vars


def main():
  key = jax.random.PRNGKey(seed=0)
  num_steps_minimisation = 100
  positions_in_unit_box, colors, dynamic_theta_args, chain_energy_hyp, chain_length, box_size, bonds, bonds_for_angle = setup_system(num_steps_minimisation)
  print("Ran minimisation.")
  write_every = 5 # 50
  steps = 100 # 4_000
  dt = 5e-3
  kT = jnp.float32(0.1)
  pressure_fn = lambda t: jnp.where(t < 100.0, 0.05, 0.15)

  npt_state, log, dynamic_theta_vars = run_npt_chain_system(
    box_size,
    chain_energy_hyp,
    positions_in_unit_box,
    bonds,
    bonds_for_angle,
    write_every,
    steps,
    dt,
    kT,
    pressure_fn,
    chain_length,
    dynamic_theta_args,
  )
  print("Ran NPT simulation.")
  dt = 5e-3
  box = 56.377
  kT = 0.09641
  steps = 100 # 400_000
  write_every = 5 # 50
  (
    key, nvt_state, apply_nvt, target_theta, molecule_stage, transitioning, step_theta_fn, wrap_energy
  ) = setup_nvt_system(
    key, npt_state, box, dt, kT, chain_energy_hyp, bonds, bonds_for_angle, dynamic_theta_args, dynamic_theta_vars
  )
  state, log, dynamic_theta_vars = run_nvt(
    nvt_state,
    apply_nvt,
    target_theta,
    molecule_stage,
    transitioning,
    step_theta_fn,
    write_every,
    steps,
    kT,
    wrap_energy,
  )
  print("Ran NVT simulation.")

if __name__ == "__main__":
  main()

# log["box"][-1]


# path = "data"


# data = {
#     "radius": chain_hyp.radius,
#     "colors": colors,
#     "log": log,
#     "positions": positions,
#     "bonds": bonds,
#     "bonds_for_angle": bonds_for_angle,
#     "colors": colors,
#     "molecule_type": molecule_type,
#     "chain_ends_ix": chain_ends_ix,
#     "theta_stages": theta_stages
# }
# state_path = os.path.join(path, "npt_full_chains.npy")
# np.save(state_path, data)

# data_loaded = np.load(state_path, allow_pickle=True).item()

# for k, v in data_loaded.items():
#   print(f"{k}: {v.shape if hasattr(v, 'shape') else v.keys()}")

# log['box'][0], log['box'][-1]

# """Okay, so a few things are at play here, firstly, the starting position is geometrically perfect so has 0 force, but shifting it off slightly brings the force in, which can be too big causing Nans. Also, I'm not sure why I have to use the unit coords- because the only way I could get it to work was scaling it back to the origiinal scale in the energy function."""

# def area(m):
#   """m: row vectors which define a basis"""
#   m_2 = m.T.dot(m)
#   det = jnp.linalg.det(m_2)
#   return det ** 0.5


# areas = jax.vmap(area)(log['box'])
# plt.plot(areas)


# def main():
#   # create lattice of atoms in chains linked by covalent bonds
#   initial_positions, bonds, bond_angle_indices = ...
#   # create energy function
#   energy_fn = ...
#   # relax
#   positions = ...
#   # simulate NPT, find the appropriate box size
#   npt_state, npt_log = ...
#   # run the main simulation as a deterministic NVT-Nose-Hoover routine
#   nvt_state, nvt_log = ...
#   # record
#   data = {
#     "initial_nvt_vars": [],
#     "trajectory": [],
#     "fixed_datastructures": [],
#   }
#   return data